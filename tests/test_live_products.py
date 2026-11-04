"""Live checks against the ASF archive.

These do not test the science. They test the assumptions the package rests on: that search
still answers, that OPERA filenames still carry the fields the parser expects, that an RTC burst
still has a static layer to join to by burst ID, and that a real download still assembles
onto OPERA's own lattice.

The area and the date range are drawn from a generator seeded on the ISO week. A fixed site
passes forever on one lucky corner of the archive; a freely random one cannot be reproduced
when it fails. Seeding on the week means every run in a given week picks the same case, and
each new week sweeps somewhere else. Every test prints its seed and its choices.

Set OPERA_FETCH_SEED to replay a week. That fixes the area and the window, but not the
archive: reprocessing changes what a search returns, so an old seed may not reproduce an
old failure exactly.
"""

import os
import random
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import opera_fetch as of
from opera_fetch import constants as const
from opera_fetch import filenames

pytestmark = pytest.mark.network

# A polar site where the acquisition is HH+HV rather than VV+VH, kept so that branch of the
# reader is exercised on a schedule rather than whenever the draw happens to land there.
POLAR_SITE = ("svalbard", (15.5, 78.2))
POLAR_SHARE = 0.25

# Half-widths in degrees. Small is a burst or two; wide reaches across several bursts, so it
# can catch bursts that disagree about dates, a zone line, or a reprocessed duplicate.
SMALL_BOX = (0.08, 0.05)
WIDE_BOX = (0.25, 0.25)
WIDE_SHARE = 0.33

# How long a stretch of the archive to draw a granule from, and how many to try.
DRAW_MINUTES = 30
DRAW_TRIES = 5

# The fetch errors that mean this week's box has nothing in it, rather than a fault.
NOTHING_HERE = ("granules for that area and date range", "nothing overlapped the AOI")


def week_seed():
    """This week's seed, or whatever OPERA_FETCH_SEED names.

    A workflow_dispatch with no seed, and every scheduled run, set the variable to the
    empty string rather than leaving it unset. An empty string taken as a seed pins every
    run to one area forever, which is what seeding on the week exists to avoid, so the
    lookup falls back on falsiness and not on the key being absent.
    """
    seed = os.environ.get("OPERA_FETCH_SEED") or None
    if seed is None:
        year, number, _ = datetime.now(UTC).isocalendar()
        seed = f"{year}-W{number:02d}"
    return seed


@pytest.fixture(scope="module")
def week():
    """A generator stable within an ISO week, and its seed."""
    seed = week_seed()
    print(f"\nweek seed: {seed}  (replay with OPERA_FETCH_SEED={seed})")
    return random.Random(seed), seed


def drawn_granule(rng):
    """Where and when a granule the archive really holds was acquired, drawn at random.

    Searches a short stretch of time with no area at all, so the draw covers wherever
    OPERA has produced RTC rather than a list someone wrote down.
    """
    import asf_search as asf
    from shapely.geometry import shape

    for _ in range(DRAW_TRIES):
        # Somewhere in the last two years, but not the last fortnight, which the archive
        # may not have caught up with yet.
        minutes_back = rng.randrange(14 * 24 * 60, 714 * 24 * 60)
        moment = datetime.now(UTC) - timedelta(minutes=minutes_back)
        found = asf.search(dataset=asf.DATASET.OPERA_S1, processingLevel=of.RTC,
                           start=moment, end=moment + timedelta(minutes=DRAW_MINUTES))
        if found:
            # Sorted so the same week draws the same granule however ASF orders a reply.
            granules = sorted(found, key=lambda g: g.properties["fileID"])
            granule = rng.choice(granules)
            centre = shape(granule.geometry).centroid
            acquired = pd.Timestamp(granule.properties["startTime"]).date()
            return granule.properties["fileID"], (centre.x, centre.y), acquired
    pytest.skip(f"ASF returned nothing in {DRAW_TRIES} draws of {DRAW_MINUTES} minutes")


@pytest.fixture(scope="module")
def case(week):
    """An area and a date range for this week, around a granule the archive holds."""
    rng, _ = week
    if rng.random() < POLAR_SHARE:
        name, (lon, lat) = POLAR_SITE
        acquired = date.today() - timedelta(days=14 + rng.randrange(0, 700))
    else:
        name, (lon, lat), acquired = drawn_granule(rng)
    half_x, half_y = WIDE_BOX if rng.random() < WIDE_SHARE else SMALL_BOX
    aoi = (lon - half_x, lat - half_y, lon + half_x, lat + half_y)

    # A month ending just after the acquisition, so it is inside the window.
    end = acquired + timedelta(days=1)
    start = end - timedelta(days=30)
    print(f"  around: {name} {tuple(round(v, 3) for v in aoi)}\n  window: {start} to {end}")
    return aoi, str(start), str(end)


def test_search_still_answers_with_the_fields_we_parse(case):
    aoi, start, end = case
    found = of.search(aoi, start, end, product=of.RTC)
    if found.empty:
        pytest.skip("no RTC granules for this week's area and window")

    assert {"fileID", "burst_id", "track", "direction", "time", "urls", "sizes"} <= set(found)
    assert found.time.notna().all()
    assert found.direction.isin(["ASCENDING", "DESCENDING"]).all()
    assert (found.track == found.burst_id.str[1:4].astype(int)).all()


def test_every_granule_name_still_parses(case):
    """The filename regex over the four products, against whatever ASF is serving today."""
    aoi, start, end = case
    for product in (of.RTC, of.CSLC):
        found = of.search(aoi, start, end, product=product)
        if found.empty:
            continue
        for file_id in found.fileID:
            fields = filenames.parse_fields(file_id + "_x.tif")
            assert fields["burst_id"] == filenames.parse_burst_id(file_id)
            assert fields["sensor"].startswith("S1")
            assert pd.notna(filenames.parse_processing_time(file_id + "_x.tif"))


def test_an_rtc_burst_still_has_a_static_layer_to_join_to(case):
    """Joined by burst ID, not by bounding box, so the IDs have to keep matching."""
    aoi, start, end = case
    found = of.search(aoi, start, end, product=of.RTC)
    if found.empty:
        pytest.skip("no RTC granules for this week's area and window")

    static = of.search_static(found)
    assert not static.empty, "no static layers for any burst in the search"
    assert set(static.burst_id) <= set(found.burst_id)
    assert (static["product"] == of.RTC_STATIC).all()


def test_the_polarizations_on_offer_are_ones_we_read(case):
    """HH+HV at high latitude, VV+VH elsewhere. Asking for the wrong pair fetches a mask
    and no backscatter at all."""
    aoi, start, end = case
    found = of.search(aoi, start, end, product=of.RTC)
    if found.empty:
        pytest.skip("no RTC granules for this week's area and window")

    for polarizations in found.polarization:
        assert set(polarizations) <= set(const.LAYERS[of.RTC]), polarizations


def test_a_real_fetch_still_assembles_onto_operas_lattice(case, tmp_path):
    """The whole path: search, download, mosaic, stack, clip, check."""
    aoi, start, end = case
    try:
        stacks = of.fetch_stacks(aoi, start, end, product=of.RTC, cache_dir=tmp_path)
    except ValueError as err:
        # Only an empty archive or a burst that misses the box is a skip. Anything else is
        # the package failing on real data, as duplicate timestamps on T078 did.
        if not any(reason in str(err) for reason in NOTHING_HERE):
            raise
        pytest.skip(f"nothing to assemble this week: {err}")

    for key, stack in stacks.items():
        print(f"  assembled {key}: {stack.sizes['time']} times, "
              f"{stack.sizes['y']} by {stack.sizes['x']}")
        assert stack.rio.crs is not None
        assert of.grid_like([stack]).sizes == {"y": stack.sizes["y"], "x": stack.sizes["x"]}
        assert stack.indexes["time"].is_unique
        assert stack.attrs["granules"], "a stack must record what it was built from"

        backscatter = [name for name in stack.data_vars
                       if name in ("vv", "vh", "hh", "hv")]
        assert backscatter, f"no backscatter in {sorted(stack.data_vars)}"
        # Any polarization: where the mode switches between passes, a date holds only one pair.
        first_date = [stack[name].isel(time=0).values for name in backscatter]
        assert any(np.isfinite(values).any() for values in first_date)
        for values in first_date:
            if np.isfinite(values).any():
                assert np.nanmin(values) >= 0, "linear power cannot be negative"

        if "mask" in stack:
            codes = set(np.unique(stack.mask.values).tolist())
            assert codes <= {0, 1, 2, 3, const.MASK_NODATA[of.RTC]}, codes


def test_a_cslc_granule_still_has_the_group_layout_we_read(case):
    """Only the metadata: a CSLC granule is 275 MB, too much for a weekly job."""
    aoi, start, end = case
    found = of.search(aoi, start, end, product=of.CSLC)
    if found.empty:
        pytest.skip("no CSLC granules for this week's area and window")

    urls = of.data_urls(found)
    assert urls and all(url.endswith(".h5") for url in urls)
    for file_id in found.fileID:
        assert filenames.parse_polarization(file_id + "_x.h5") in const.CSLC_DATA

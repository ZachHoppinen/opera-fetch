"""Live checks against the ASF archive.

These do not test the science. They test the assumptions the package rests on: that search
still answers, that OPERA filenames still carry the fields we parse, that an RTC burst
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
import pytest

import opera_fetch as of
from opera_fetch import constants as const
from opera_fetch import filenames

pytestmark = pytest.mark.network

# Mountain areas with dependable Sentinel-1 coverage, plus one polar site where the
# acquisition is HH+HV rather than VV+VH.
SITES = {
    "alaska range": (-149.5, 62.0),
    "wind rivers": (-109.7, 43.1),
    "sierra nevada": (-119.3, 37.8),
    "colorado rockies": (-107.0, 39.0),
    "north cascades": (-121.0, 48.6),
    "sawtooths": (-114.9, 44.0),
    "svalbard": (15.5, 78.2),
}


@pytest.fixture(scope="module")
def week():
    """A generator stable within an ISO week, and its seed."""
    seed = os.environ.get("OPERA_FETCH_SEED")
    if seed is None:
        year, number, _ = datetime.now(UTC).isocalendar()
        seed = f"{year}-W{number:02d}"
    print(f"\nweek seed: {seed}  (replay with OPERA_FETCH_SEED={seed})")
    return random.Random(seed), seed


@pytest.fixture(scope="module")
def case(week):
    """An area and a date range for this week."""
    rng, _ = week
    name = rng.choice(sorted(SITES))
    lon, lat = SITES[name]
    # A small box: enough for a burst or two, not enough to be a slow download.
    aoi = (lon - 0.08, lat - 0.05, lon + 0.08, lat + 0.05)

    # A month, somewhere in the last two years but not the last fortnight, which the
    # archive may not have caught up with yet.
    end = date.today() - timedelta(days=14 + rng.randrange(0, 700))
    start = end - timedelta(days=30)
    print(f"  site: {name} {aoi}\n  window: {start} to {end}")
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
    """Our regex over the four products, against whatever ASF is serving today."""
    aoi, start, end = case
    for product in (of.RTC, of.CSLC):
        found = of.search(aoi, start, end, product=product)
        if found.empty:
            continue
        for file_id in found.fileID:
            fields = filenames.parse_fields(file_id + "_x.tif")
            assert fields["burst_id"] == filenames.parse_burst_id(file_id)
            assert fields["sensor"].startswith("S1")
            assert filenames.parse_processing_time(file_id + "_x.tif") is not None


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
        values = stack[backscatter[0]].isel(time=0).values
        assert np.isfinite(values).any()
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

"""Fixtures, including whatever real OPERA granules are on this machine.

The granule fixtures skip themselves when the data is not there, so the suite runs
anywhere. The tests that use them are the ones that check this package against products as
ASF actually ships them rather than against something written to be convenient.

They look in the directories the example scripts download into. Point them somewhere else
with OPERA_FETCH_TEST_RTC and OPERA_FETCH_TEST_CSLC, which is worth doing for CSLC: the
example fetches a fortnight, and the time-series tests want a burst with more than one
acquisition in it.
"""

import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from opera_fetch import filenames

RTC_DATA = Path(os.environ.get("OPERA_FETCH_TEST_RTC", "data/raw/east_river"))
CSLC_DATA = Path(os.environ.get("OPERA_FETCH_TEST_CSLC", "data/raw/cslc_example"))


def _busiest_burst(paths):
    """The burst with the most acquisitions among these files, and its files."""
    acquisitions = Counter()
    for path in paths:
        if filenames.parse_fields(path)["processed"]:          # not a static layer
            acquisitions[filenames.parse_burst_id(path)] += 1
    if not acquisitions:
        return None, []
    burst, _ = acquisitions.most_common(1)[0]
    return burst, sorted(p for p in paths if filenames.parse_burst_id(p) == burst)


@pytest.fixture(scope="session")
def rtc_paths():
    """Every RTC and RTC-STATIC file of whichever burst has the most acquisitions."""
    if not RTC_DATA.exists():
        pytest.skip(f"no RTC granules at {RTC_DATA}")
    _, paths = _busiest_burst(sorted(RTC_DATA.glob("*.tif")))
    if not paths:
        pytest.skip(f"no RTC acquisitions at {RTC_DATA}")
    return paths


@pytest.fixture(scope="session")
def rtc_two_bursts():
    """Several bursts of one track, which is what a mosaic is for."""
    if not RTC_DATA.exists():
        pytest.skip(f"no RTC granules at {RTC_DATA}")
    paths = sorted(RTC_DATA.glob("*.tif"))
    tracks = Counter(filenames.parse_burst_id(p)[:4] for p in paths)
    if not tracks:
        pytest.skip(f"no RTC granules at {RTC_DATA}")

    # One track only: a mosaic may not span two, so neither may the fixture.
    track, _ = tracks.most_common(1)[0]
    on_track = [p for p in paths if filenames.parse_burst_id(p).startswith(track)]
    if len({filenames.parse_burst_id(p) for p in on_track}) < 2:
        pytest.skip("need two bursts of one track")
    return sorted(on_track)


@pytest.fixture(scope="session")
def cslc_paths():
    """CSLC granules of whichever burst has the most acquisitions."""
    if not CSLC_DATA.exists():
        pytest.skip(f"no CSLC granules at {CSLC_DATA}")
    _, paths = _busiest_burst(sorted(CSLC_DATA.glob("*.h5")))
    acquisitions = [p for p in paths if filenames.parse_fields(p)["processed"]]
    if len(acquisitions) < 2:
        pytest.skip(f"need a burst with two acquisitions at {CSLC_DATA}; "
                    "set OPERA_FETCH_TEST_CSLC to a longer series")
    return paths


def make_burst(west, north, columns=8, rows=6, times=2, track=49, direction="ASCENDING",
               spacing=(30.0, 30.0), epsg=32612, fill=None, static=True):
    """A synthetic burst.

    fill defaults to a value per cell rather than one constant, so a test that says a value
    did not move is testing that and not the fill: a constant survives being shifted,
    transposed, or replaced by any other constant. Pass a number for a flat burst where the
    arithmetic is the point.

    static adds a once-per-burst (y, x) layer, which is the shape the static layers arrive
    in and the one nothing else in the fixtures has.
    """
    dx, dy = (spacing, spacing) if isinstance(spacing, (int, float)) else spacing
    x = west + (np.arange(columns) + 0.5) * dx
    y = north - (np.arange(rows) + 0.5) * dy
    stamps = np.array([np.datetime64("2024-10-04T01:10:54") + np.timedelta64(12 * i, "D")
                       for i in range(times)])
    shape = (times, rows, columns)
    values = (np.full(shape, fill, dtype="float32") if fill is not None
              else np.arange(times * rows * columns, dtype="float32").reshape(shape) + 1)
    layers = {
        "vv": (("time", "y", "x"), values),
        "mask": (("time", "y", "x"), np.zeros(shape, dtype="uint8")),
    }
    if static:
        layers["local_incidence_angle"] = (
            ("y", "x"), np.arange(rows * columns, dtype="float32").reshape(rows, columns))
    stack = xr.Dataset(layers, coords={"time": stamps, "y": y, "x": x})
    stack.attrs = {"product": "RTC", "burst_id": "T049-103327-IW3", "track": track,
                   "direction": direction, "spacing": (dx, dy),
                   "footprint": ""}
    return stack.rio.write_crs(epsg)

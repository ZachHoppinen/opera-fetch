"""Against real OPERA RTC granules, which is the only way to know the reader is right."""

import re

import numpy as np
import pytest

from opera_fetch import constants as const
from opera_fetch.grid import grid_like, place, spacing_of
from opera_fetch.rtc import read_burst
from opera_fetch.stack import assemble

pytestmark = pytest.mark.data


def test_a_burst_reads_into_a_time_series(rtc_paths):
    burst = read_burst(rtc_paths)

    assert re.fullmatch(r"T\d{3}-\d{6}-IW[1-3]", burst.attrs["burst_id"])
    assert burst.attrs["track"] == int(burst.attrs["burst_id"][1:4])
    assert burst.attrs["direction"] in ("ASCENDING", "DESCENDING")
    assert {"vv", "vh", "mask"} <= set(burst.data_vars)
    assert burst.sizes["time"] > 1
    assert burst.indexes["time"].is_monotonic_increasing
    assert burst.indexes["time"].is_unique


def test_it_lands_on_operas_own_30_m_lattice(rtc_paths):
    burst = read_burst(rtc_paths)
    assert spacing_of(burst) == (30.0, 30.0)
    assert 32600 < burst.rio.crs.to_epsg() < 32800
    # The grid built from it is its own coordinates back, so a placement loses nothing.
    assert np.array_equal(place(burst, grid_like([burst])).vv.values, burst.vv.values,
                          equal_nan=True)


def test_values_are_linear_gamma0_and_nothing_has_been_masked(rtc_paths):
    burst = read_burst(rtc_paths)
    scene = burst.vv.isel(time=0).values

    assert burst.vv.attrs["units"] == "1"
    assert np.nanmin(scene) >= 0, "linear power cannot be negative; this looks like dB"
    assert burst.mask.dtype == np.uint8
    # Layover and shadow pixels are still there to be looked at, not blanked on the way in.
    assert np.isfinite(scene[burst.mask.isel(time=0).values != const.MASK_CLEAR]).any()


def test_static_layers_come_in_as_radians_on_the_burst_grid(rtc_paths):
    burst = read_burst(rtc_paths)
    if "local_incidence_angle" not in burst:
        pytest.skip("no static layers cached for this burst")

    angle = burst.local_incidence_angle
    assert angle.attrs["units"] == "radians"
    assert 0 <= np.nanmedian(angle.values) <= np.pi
    assert angle.sizes == {"y": burst.sizes["y"], "x": burst.sizes["x"]}


def test_two_bursts_assemble_into_one_stack(rtc_two_bursts):
    stacks = assemble(rtc_two_bursts)
    assert len(stacks) == 1

    # Keyed by UTM zone, and whichever zone the fixture found is the one that matters.
    (epsg, stack), = stacks.items()
    assert 32600 < epsg < 32800, "OPERA delivers in UTM"
    assert stack.rio.crs.to_epsg() == epsg
    assert set(np.unique(stack.direction.values)) <= {"ASCENDING", "DESCENDING"}
    assert stack.attrs["bursts"] >= 2
    assert spacing_of(stack) == (30.0, 30.0)
    assert stack.indexes["time"].is_unique, "the bursts should share pass timestamps"
    assert np.isfinite(stack.vv.isel(time=0).values).any()


def test_assembling_to_an_aoi_gives_only_that_area(rtc_two_bursts):
    import shapely.wkt

    whole = assemble(rtc_two_bursts)
    (_, big), = whole.items()

    # An area inside the burst footprint, not inside the bounding box: the corners of a
    # burst's bounding box hold no data, so an AOI there is rightly refused.
    footprint = shapely.wkt.loads(big.attrs["footprint"])
    centre = footprint.centroid
    aoi = centre.buffer(0.02).envelope

    small = assemble(rtc_two_bursts, aoi=aoi)
    (_, cut), = small.items()

    assert cut.sizes["x"] < big.sizes["x"] and cut.sizes["y"] < big.sizes["y"]
    # Same lattice, so the cells that survive the cut are the very same cells.
    assert set(np.round(cut.x.values, 3)) <= set(np.round(big.x.values, 3))
    assert set(np.round(cut.y.values, 3)) <= set(np.round(big.y.values, 3))
    assert np.isfinite(cut.vv.isel(time=0).values).any()


def test_a_burst_whose_box_touches_but_whose_data_does_not_is_left_out(rtc_two_bursts):
    """ASF returns granules that graze an area; their bounding box lies about the rest."""
    import shapely.wkt

    from opera_fetch.stack import _overlapping, read_bursts

    bursts = read_bursts(rtc_two_bursts)
    footprint = shapely.wkt.loads(bursts[0].attrs["footprint"])
    west, _, _, north = footprint.bounds

    # The northwest corner of the bounding box, which the rotated footprint never reaches.
    corner = shapely.geometry.box(west, north - 0.01, west + 0.01, north)
    assert not footprint.intersects(corner)
    # Only this burst: the neighbour up the track really does cover that corner.
    assert _overlapping(bursts[:1], corner) == []


def test_what_varies_between_acquisitions_rides_on_the_time_axis(rtc_paths):
    burst = read_burst(rtc_paths)

    # Which satellite and which orbit change from one acquisition to the next, so they
    # cannot be attributes of the burst.
    assert set(burst.platform.dims) == {"time"}
    assert set(burst.absolute_orbit.dims) == {"time"}
    assert all(str(p).startswith("S1") for p in burst.platform.values)
    assert (burst.absolute_orbit.values > 0).all()
    assert burst.absolute_orbit.to_index().is_unique

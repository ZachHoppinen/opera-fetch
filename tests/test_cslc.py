"""Against real OPERA CSLC granules: complex, geocoded, and 5 by 10 m rather than square."""

import re

import numpy as np
import pytest

from opera_fetch import constants as const
from opera_fetch import filenames
from opera_fetch.cslc import read_burst
from opera_fetch.grid import grid_like, spacing_of

pytestmark = pytest.mark.data


def an_acquisition(paths):
    """One CSLC acquisition out of the fixture.

    Not paths[0]: the fixture is sorted, and "CSLC-S1-STATIC" sorts before "CSLC-S1_", so
    the first path is the static layer whenever one has been downloaded.
    """
    return next(p for p in paths if filenames.parse_product(p) == const.CSLC)


def test_a_burst_reads_into_a_complex_time_series(cslc_paths):
    burst = read_burst(cslc_paths)

    assert re.fullmatch(r"T\d{3}-\d{6}-IW[1-3]", burst.attrs["burst_id"])
    assert burst.attrs["track"] == int(burst.attrs["burst_id"][1:4])
    assert burst.sizes["time"] >= 2
    assert np.issubdtype(burst.vv.dtype, np.complexfloating)


def test_the_phase_survives_being_read(cslc_paths):
    burst = read_burst(cslc_paths)
    block = burst.vv.isel(time=0, y=slice(2000, 2004), x=slice(9000, 9004)).values

    assert np.isfinite(block).any()
    assert np.any(block.imag != 0), "an all-real result means the phase was thrown away"
    assert np.any(np.abs(np.angle(block[np.isfinite(block)])) > 0)


def test_it_lands_on_the_5_by_10_m_lattice(cslc_paths):
    burst = read_burst(cslc_paths)
    assert spacing_of(burst) == (5.0, 10.0)
    assert 32600 < burst.rio.crs.to_epsg() < 32800
    assert np.array_equal(grid_like([burst]).x.values, burst.x.values)


def test_the_big_phase_screens_are_left_out_unless_asked_for(cslc_paths):
    one = [an_acquisition(cslc_paths)]
    lean = read_burst(one)
    assert "flattening_phase" not in lean

    loaded = read_burst(one, extra=("flattening_phase",))
    assert "flattening_phase" in loaded


def test_nothing_is_read_into_memory_on_open(cslc_paths):
    burst = read_burst(cslc_paths)
    # 800 MB an acquisition: a reader that loaded eagerly would be unusable.
    assert burst.vv.chunks is not None


def test_bursts_from_two_places_are_refused(cslc_paths):
    from pathlib import Path

    one = an_acquisition(cslc_paths)
    burst = filenames.parse_burst_id(one)
    other = str(one).replace(burst, burst[:-1] + ("2" if burst[-1] == "1" else "1"))
    with pytest.raises(ValueError, match="span 2 bursts"):
        read_burst([one, Path(other)])


def test_what_varies_between_acquisitions_rides_on_the_time_axis(cslc_paths):
    burst = read_burst(cslc_paths)

    assert set(burst.platform.dims) == {"time"}
    assert set(burst.absolute_orbit.dims) == {"time"}
    assert all(str(p).startswith("S1") for p in burst.platform.values)
    assert (burst.absolute_orbit.values > 0).all()

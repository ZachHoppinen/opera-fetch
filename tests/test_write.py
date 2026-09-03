import numpy as np
import pytest
from tests.conftest import make_burst

from opera_fetch.write import read, write


@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
def test_a_stack_comes_back_exactly(tmp_path, suffix):
    burst = make_burst(west=500_010, north=4_332_210)
    burst["vv"][:] = np.random.default_rng(0).random(burst.vv.shape).astype("float32")

    path = write(burst, tmp_path / f"stack{suffix}")
    back = read(path)

    assert np.array_equal(back.vv.values, burst.vv.values)
    assert back.indexes["time"].equals(burst.indexes["time"])
    assert back.rio.crs == burst.rio.crs


@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
def test_complex_survives_the_round_trip(tmp_path, suffix):
    burst = make_burst(west=500_010, north=4_332_210)
    rng = np.random.default_rng(0)
    values = (rng.random(burst.vv.shape) + 1j * rng.random(burst.vv.shape)).astype("complex64")
    burst["vv"] = (burst.vv.dims, values)

    back = read(write(burst, tmp_path / f"cslc{suffix}"))
    assert back.vv.dtype == np.complex64
    assert np.array_equal(back.vv.values, values)


def test_several_zones_become_several_groups(tmp_path):
    """An AOI straddling a zone boundary gives two grids, which cannot share coordinates."""
    stacks = {
        32612: make_burst(west=500_010, north=4_332_210, fill=1.0),
        32613: make_burst(west=500_010, north=4_332_210, epsg=32613, fill=2.0),
    }
    back = read(write(stacks, tmp_path / "both.nc"))

    assert set(back) == {"EPSG32612", "EPSG32613"}
    assert float(back["EPSG32613"].vv.isel(time=0, y=0, x=0)) == 2.0


def test_attributes_a_file_cannot_hold_are_turned_into_text(tmp_path):
    burst = make_burst(west=500_010, north=4_332_210)
    burst.attrs["spacing"] = (30.0, 30.0)
    burst.attrs["nothing"] = None
    burst.attrs["awkward"] = {"a": 1}

    back = read(write(burst, tmp_path / "attrs.nc"))
    assert list(back.attrs["spacing"]) == [30.0, 30.0]
    assert "nothing" not in back.attrs
    assert isinstance(back.attrs["awkward"], str)


def test_an_unknown_suffix_says_what_it_takes(tmp_path):
    with pytest.raises(ValueError, match=r"\.nc, \.h5 or \.zarr"):
        write(make_burst(west=500_010, north=4_332_210), tmp_path / "stack.tif")

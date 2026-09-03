import numpy as np
import pytest
from shapely.geometry import box
from tests.conftest import make_burst

from opera_fetch.grid import clip, grid_like, place, spacing_of


def test_a_grid_of_one_burst_is_that_burst_own_coordinates():
    burst = make_burst(west=500_010, north=4_332_210, columns=8, rows=6)
    grid = grid_like([burst])

    assert np.array_equal(grid.x.values, burst.x.values)
    assert np.array_equal(grid.y.values, burst.y.values)


def test_the_lattice_comes_from_the_product_not_from_multiples_of_the_spacing():
    """The reason the grid is anchored on a burst rather than snapped to a rule.

    OPERA happens to snap to multiples of the posting today, in the zones anyone has
    checked. A grid that assumes it would put this burst off its own lattice.
    """
    burst = make_burst(west=500_017, north=4_332_217, columns=8, rows=6)
    grid = grid_like([burst])

    assert np.array_equal(grid.x.values, burst.x.values)
    assert float(grid.x[0]) % 30 != 0
    # And the burst still places on it losslessly, which is the whole point.
    assert np.array_equal(place(burst, grid).vv.values, burst.vv.values)


def test_a_grid_of_two_bursts_covers_both_on_one_lattice():
    a = make_burst(west=500_010, north=4_332_210, columns=8)
    b = make_burst(west=500_010 + 4 * 30, north=4_332_210, columns=8)
    grid = grid_like([a, b])

    assert grid.sizes["x"] == 12
    assert set(np.round(a.x.values, 6)) <= set(np.round(grid.x.values, 6))
    assert set(np.round(b.x.values, 6)) <= set(np.round(grid.x.values, 6))


def test_bounds_widen_outward_so_rounding_does_not_move_the_grid():
    burst = make_burst(west=500_010, north=4_332_210, columns=20, rows=20)
    # Both requests fall inside the same lattice cells, so both must give the same grid.
    tight = grid_like([burst], bounds=(500_075, 4_332_045, 500_245, 4_332_195))
    loose = grid_like([burst], bounds=(500_072, 4_332_048, 500_248, 4_332_192))

    assert np.array_equal(tight.x.values, loose.x.values)
    assert np.array_equal(tight.y.values, loose.y.values)
    assert set(np.round(tight.x.values, 6)) <= set(np.round(burst.x.values, 6))


def test_a_cslc_lattice_is_not_square():
    burst = make_burst(west=500_000, north=4_000_000, columns=20, rows=10, spacing=(5.0, 10.0))
    assert spacing_of(grid_like([burst])) == (5.0, 10.0)


def test_placing_a_burst_moves_no_value():
    burst = make_burst(west=500_010, north=4_332_210, columns=8, rows=6, fill=7.0)
    grid = grid_like([burst], bounds=(500_010 - 60, 4_332_210 - 300, 500_010 + 480, 4_332_210))
    placed = place(burst, grid)

    assert np.array_equal(placed.sel(x=burst.x, y=burst.y).vv.values, burst.vv.values)
    # Cells the burst never reached stay empty rather than being interpolated into.
    assert np.isnan(placed.vv.isel(time=0, x=0, y=-1))


def test_a_burst_off_the_grid_lattice_is_refused_and_named():
    grid = grid_like([make_burst(west=500_010, north=4_332_210)])
    off = make_burst(west=500_017, north=4_332_210)      # 7 m off that lattice
    off.attrs["burst_id"] = "T049-103328-IW3"

    with pytest.raises(ValueError, match="T049-103328-IW3 is not on the same lattice"):
        place(off, grid)


def test_another_projection_is_refused_rather_than_reprojected():
    grid = grid_like([make_burst(west=500_010, north=4_332_210)])
    with pytest.raises(ValueError, match="nothing here reprojects"):
        place(make_burst(west=500_010, north=4_332_210, epsg=32613), grid)


def test_a_different_spacing_is_refused():
    grid = grid_like([make_burst(west=500_010, north=4_332_210)])
    with pytest.raises(ValueError, match="spacing"):
        place(make_burst(west=500_010, north=4_332_210, spacing=(60.0, 60.0)), grid)


def test_clipping_keeps_the_values_it_keeps():
    burst = make_burst(west=500_010, north=4_332_210, columns=20, rows=20)
    burst["vv"][:] = np.arange(burst.vv.size, dtype="float32").reshape(burst.vv.shape)
    cut = clip(burst, box(*_lonlat(burst, 5, 15, 5, 15)))

    assert cut.sizes["x"] < burst.sizes["x"]
    assert np.array_equal(cut.vv.values, burst.sel(x=cut.x, y=cut.y).vv.values)


def test_clipping_somewhere_else_says_so():
    with pytest.raises(ValueError, match="does not overlap"):
        clip(make_burst(west=500_010, north=4_332_210), (10.0, 45.0, 10.1, 45.1))


def _lonlat(burst, x0, x1, y0, y1):
    """The lon/lat box around a block of a burst's own pixels."""
    from pyproj import CRS, Transformer

    to_lonlat = Transformer.from_crs(CRS.from_user_input(burst.rio.crs), CRS.from_epsg(4326),
                                     always_xy=True)
    west, north = to_lonlat.transform(float(burst.x[x0]), float(burst.y[y0]))
    east, south = to_lonlat.transform(float(burst.x[x1]), float(burst.y[y1]))
    return west, south, east, north


def test_mosaicking_bursts_off_one_lattice_is_refused():
    """The grid is anchored on the first burst, so the others have to be on its lattice."""
    from opera_fetch.mosaic import mosaic

    anchor = make_burst(west=500_010, north=4_332_210)
    stray = make_burst(west=500_017, north=4_332_210)
    stray.attrs["burst_id"] = "T049-103328-IW3"

    with pytest.raises(ValueError, match="T049-103328-IW3 is not on the same lattice"):
        mosaic([anchor, stray])

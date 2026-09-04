"""The grid the products are already on.

OPERA delivers every burst on a fixed lattice in its UTM zone, so bursts of one track and
reprocessings of one burst share their pixel centres exactly. Combining and clipping are
therefore lookups: nothing is interpolated, averaged into a neighbour, or moved.

The lattice is read off a product rather than rebuilt from a rule about where OPERA snaps,
which keeps it exact in a zone or posting nobody has checked.
"""

import logging

import numpy as np
import rioxarray  # noqa: F401  registers the .rio accessor
import xarray as xr
from pyproj import CRS, Transformer
from shapely.ops import transform as shapely_transform

from opera_fetch import constants as const
from opera_fetch.aoi import WGS84, as_geometry

log = logging.getLogger(__name__)


def spacing_of(obj):
    """The (x, y) posting of a grid, positive, in its own units.

    From the attribute where the object carries one, because a one-cell grid has no
    coordinate difference to read and a single-pixel time series is an ordinary thing to
    ask this package for.

    Where the coordinates can be read too and disagree, the attribute still wins and the
    disagreement is a warning: something re-gridded the object and left the attribute
    behind, and everything that reads a footprint from here is then out by half the
    difference.
    """
    spacing = obj.attrs.get("spacing")
    measured = measured_spacing(obj)
    if spacing is not None:
        spacing = (float(spacing[0]), float(spacing[1]))
        if measured is not None and not np.allclose(spacing, measured):
            log.warning("the spacing attribute says %s but the coordinates are %s apart; "
                        "the attribute is being used. Whatever re-gridded this did not "
                        "update it.", spacing, measured)
        return spacing

    if measured is None:
        raise ValueError(
            "this grid is one pixel across and carries no spacing attribute, so there is "
            "nothing to read the posting from")
    return measured


def measured_spacing(obj):
    """The posting read off the coordinates, or None where there is nothing to read.

    For updating a stale ``spacing`` attribute after something has re-gridded an object.
    ``spacing_of`` deliberately trusts the attribute, so whatever moves a grid has to say
    so; every footprint in the package is worked out from that number.
    """
    if obj.sizes.get("x", 0) < 2 or obj.sizes.get("y", 0) < 2:
        return None
    return (float(abs(obj.x[1] - obj.x[0])), float(abs(obj.y[1] - obj.y[0])))


def bounds_of(obj):
    """The outer pixel edges of a grid, as (west, south, east, north).

    Not ``rio.bounds()``: that reads the resolution off a coordinate difference, which a
    one-pixel grid has not got.
    """
    dx, dy = spacing_of(obj)
    return (float(obj.x.min()) - dx / 2, float(obj.y.min()) - dy / 2,
            float(obj.x.max()) + dx / 2, float(obj.y.max()) + dy / 2)


def grid_like(bursts, bounds=None):
    """An empty grid covering the bursts, on the lattice the bursts are themselves on.

    bounds narrows it to an area, given in the bursts' projection; without it the grid
    spans every burst, which is a convenience for a handful of neighbours rather than
    something to run a track through. A whole track at 30 m is 286 million cells, of
    which the bursts are a thin diagonal ribbon.

    Edges move outward to reach the lattice, so the same area gives the same pixels
    however the request was rounded.
    """
    # The first burst supplies the lattice; place() refuses any burst not on it.
    bursts = list(bursts)
    anchor = bursts[0]
    dx, dy = spacing_of(anchor)
    west, south, east, north = bounds if bounds is not None else _union(bursts)

    # Anchored on the burst's own outer pixel edges, so both runs land on its lattice.
    west_edge = float(anchor.x[0]) - dx / 2
    north_edge = float(anchor.y[0]) + dy / 2
    x = _centres(west_edge, dx, west, east)
    # y is built the same way and then turned round, because a north-up grid descends.
    y = _centres(north_edge, dy, south, north)[::-1]
    if x.size == 0 or y.size == 0:
        raise ValueError(f"bounds {(west, south, east, north)} are under one pixel across")

    grid = xr.DataArray(np.zeros((y.size, x.size), dtype="int8"),
                        dims=("y", "x"), coords={"y": y, "x": x}, name="grid",
                        attrs={"spacing": (dx, dy)})
    return grid.rio.write_crs(anchor.rio.crs)


def place(obj, grid):
    """Put a burst on a grid by looking its coordinates up.

    Both sit on the same lattice, so this is a lookup and not a resample. Cells the burst
    does not reach come back empty, which is what lets two bursts be combined without
    either one moving.
    """
    dx, dy = spacing_of(grid)
    if obj.rio.crs != grid.rio.crs:
        raise ValueError(f"{obj.rio.crs} is not the grid's {grid.rio.crs}; nothing here "
                         "reprojects, so assemble one UTM zone at a time")
    if not np.allclose(spacing_of(obj), (dx, dy)):
        raise ValueError(f"{_name(obj)} has spacing {spacing_of(obj)}, not the grid's {(dx, dy)}")
    if not (_aligned(obj.x[0], grid.x[0], dx) and _aligned(obj.y[0], grid.y[0], dy)):
        raise ValueError(f"{_name(obj)} is not on the same lattice as the grid, so it cannot "
                         "be placed without resampling")

    # nearest with a tiny tolerance is exact matching that survives float noise in the
    # coordinates, not a search for something nearby.
    placed = obj.reindex(y=grid.y, x=grid.x, method="nearest", tolerance=min(dx, dy) / 100)
    return placed.rio.write_crs(grid.rio.crs)


def clip(obj, aoi, crs=None, mask=False):
    """Cut a stack down to an area, keeping whole pixels of the grid it is already on.

    The AOI is reprojected to the data's projection and the data is placed on the grid
    that area asks for, so every value that comes out is a value that went in.

    mask additionally blanks pixels outside the polygon rather than outside its bounding
    box. That moves nothing either; it only sets the corners to nodata.
    """
    # The outline moves to the data, never the data to the outline.
    target = CRS.from_user_input(obj.rio.crs)
    geometry = reproject(as_geometry(aoi, crs), target)

    left, bottom, right, top = bounds_of(obj)
    west, south, east, north = geometry.bounds
    if left >= east or right <= west or bottom >= north or top <= south:
        raise ValueError(
            f"the AOI does not overlap the data: AOI {_round(geometry.bounds)} against "
            f"data {_round(bounds_of(obj))} in {target.to_string()}")

    cut = place(obj, grid_like([obj], bounds=geometry.bounds))
    if mask:
        # Told what no observation means, rioxarray blanks the corners with it. Left to
        # work it out, it filled the layover mask with 0, which is the code for clear.
        cut = _mask_nodata_declared(cut).rio.clip([geometry], target, drop=False)
        cut = mask_codes(cut)

    log.debug("clipped to %d by %d cells", cut.sizes["y"], cut.sizes["x"])
    return cut


def mask_codes(obj):
    """A copy whose layover masks are integers again.

    Anything that fills floats an integer mask, and a class code is not a float. Those
    cells are no observation, which the mask already has a code for.
    """
    if not isinstance(obj, xr.Dataset):
        return obj
    product = obj.attrs.get("product", const.RTC)
    out = obj.copy()
    for name, array in out.data_vars.items():
        if name.endswith("mask") and array.dtype.kind == "f":
            out[name] = (array.fillna(const.MASK_NODATA[product])
                         .astype(const.MASK_DTYPE[product]))
    return out


def _mask_nodata_declared(obj):
    """A copy whose layover masks say which code means no observation."""
    if not isinstance(obj, xr.Dataset):
        return obj
    nodata = const.MASK_NODATA[obj.attrs.get("product", const.RTC)]
    out = obj.copy()
    for name, array in out.data_vars.items():
        if name.endswith("mask") and array.rio.nodata is None:
            out[name] = array.rio.write_nodata(nodata)
    return out


def reproject(geometry, crs):
    """A lon/lat geometry in another projection. The only reprojection in the package,
    and it moves an outline rather than any data."""
    crs = CRS.from_user_input(crs)
    if crs == WGS84:
        return geometry
    return shapely_transform(Transformer.from_crs(WGS84, crs, always_xy=True).transform,
                             geometry)


def _centres(edge, spacing, low, high):
    """Ascending pixel centres on a lattice, covering low to high.

    edge is any pixel edge on that lattice; the run is placed relative to it rather than
    to zero, which is what keeps it on the product's own grid. It widens outward to whole
    cells, so it starts in the cell holding low and ends in the cell holding high.
    """
    steps_back = np.floor((low - edge) / spacing)
    first_edge = edge + steps_back * spacing

    # Rounded before the ceiling: a bound that lands exactly on an edge comes out of the
    # division a hair over the integer, and would otherwise add an empty cell.
    cells = int(np.ceil(round((high - first_edge) / spacing, 6)))
    return first_edge + (np.arange(max(cells, 0)) + 0.5) * spacing


def _aligned(a, b, step):
    """Whether two coordinates sit on the same lattice."""
    offset = (float(a) - float(b)) % step
    return min(offset, step - offset) < step / 100


def _union(bursts):
    """Bounds covering every burst."""
    corners = np.array([bounds_of(burst) for burst in bursts])
    west, south = corners[:, 0].min(), corners[:, 1].min()
    east, north = corners[:, 2].max(), corners[:, 3].max()
    return west, south, east, north

def _round(bounds):
    return tuple(round(value) for value in bounds)


def _name(burst):
    return burst.attrs.get("burst_id") or "a burst"

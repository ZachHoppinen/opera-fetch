"""An area of interest, from whatever you have to the one lon/lat polygon ASF wants."""

import logging
from pathlib import Path

import shapely.geometry
import shapely.wkt
from pyproj import CRS, Transformer
from shapely.ops import transform as shapely_transform

log = logging.getLogger(__name__)

WGS84 = CRS.from_epsg(4326)


def as_geometry(aoi, crs=None):
    """An area as a shapely polygon in lon/lat, or several where the input has several.

    Separate polygons stay separate, because that is the area to clip to. ``one_polygon``
    is what turns them into the single outline ASF's search takes.

    Takes WKT, a path to any vector file geopandas can read, a shapely geometry, a
    GeoDataFrame or GeoSeries, a GeoJSON-like mapping, a (west, south, east, north) box,
    or a list of (x, y) coordinates.

    crs names the projection the coordinates are in when it is not lon/lat. Inputs that
    carry their own, such as a file or a GeoDataFrame, are reprojected from that instead.
    """
    # A projection the input carries beats the one the caller named.
    geometry, native = _to_shapely(aoi)
    crs = CRS.from_user_input(native if native is not None else (crs or WGS84))

    if not geometry.is_valid:
        # A self-intersecting ring reaches ASF as a complaint about the request, not the shape.
        geometry = geometry.buffer(0)

    if crs != WGS84:
        geometry = shapely_transform(
            Transformer.from_crs(crs, WGS84, always_xy=True).transform, geometry)

    if geometry.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(f"AOI must be a polygon, not a {geometry.geom_type}")
    if geometry.is_empty or geometry.area == 0:
        raise ValueError("AOI has no area")

    _in_range(geometry, native is None and crs == WGS84)

    # A box across 180 comes out as its complement: the whole world but the box. Silently
    # searching the globe is worse than refusing, and splitting it is the caller's call
    # because the two halves land in different UTM zones and cannot share a grid anyway.
    west, _, east, _ = geometry.bounds
    if east - west > 180:
        raise ValueError(
            f"the AOI spans {east - west:.0f} degrees of longitude, which means it was "
            "wrapped the long way round the antimeridian. Split it at 180 and assemble "
            "each side: they fall in different UTM zones and cannot share one grid.")

    return geometry


def one_polygon(geometry):
    """One polygon covering an area, which is all ASF's search takes.

    Separate polygons become their convex hull, which is wider than what was asked for.
    Only the search sees this: clipping keeps the outline the caller gave, or the product
    would cover ground nobody asked about.
    """
    if geometry.geom_type != "MultiPolygon":
        return geometry
    log.warning("AOI is %d separate polygons; searching their convex hull. What comes "
                "back is still clipped to the polygons themselves", len(geometry.geoms))
    return geometry.convex_hull


def _in_range(geometry, assumed_lonlat):
    """Refuse coordinates that are not lon/lat, having been read as lon/lat.

    ASF clamps a latitude past 90 rather than complaining, which degenerates the polygon
    and returns nothing: the archive gets blamed for a transposed pair or a forgotten
    aoi_crs. The message says which value is wrong, and what it probably was.
    """
    west, south, east, north = geometry.bounds
    bad = ([f"longitude {value:g}" for value in (west, east) if abs(value) > 180]
           + [f"latitude {value:g}" for value in (south, north) if abs(value) > 90])
    if not bad:
        return

    hint = ""
    if assumed_lonlat and abs(south) <= 180 and abs(north) <= 180 and abs(west) <= 90:
        hint = (". These look like (south, west, north, east): a box is "
                "(west, south, east, north)")
    elif assumed_lonlat and max(abs(west), abs(east), abs(south), abs(north)) > 1000:
        hint = ". These look like projected metres, so pass aoi_crs with the zone they are in"
    raise ValueError(f"AOI is not in lon/lat: {', '.join(bad)}{hint}")


def _to_shapely(aoi):
    """The input as (geometry, its own crs or None)."""
    if isinstance(aoi, shapely.geometry.base.BaseGeometry):
        return aoi, None

    # A GeoDataFrame or a GeoSeries, without importing geopandas to find out. Both answer
    # .geometry with a GeoSeries; on the series it is the object itself.
    if hasattr(aoi, "geometry") and hasattr(aoi, "crs"):
        if len(aoi.geometry) == 0:
            raise ValueError("AOI has no geometries in it")
        return aoi.geometry.union_all(), aoi.crs

    if isinstance(aoi, (str, Path)):
        if Path(aoi).exists():
            import geopandas as gpd

            frame = gpd.read_file(aoi)
            if frame.empty:
                raise ValueError(f"{aoi} has no geometries in it")
            return frame.geometry.union_all(), frame.crs
        try:
            return shapely.wkt.loads(str(aoi)), None
        except Exception as err:
            raise ValueError(
                f"AOI is neither WKT nor a path that exists: {str(aoi)[:60]!r}") from err

    if isinstance(aoi, dict):
        return shapely.geometry.shape(aoi), None

    if isinstance(aoi, (list, tuple)):
        if len(aoi) == 4 and all(isinstance(value, (int, float)) for value in aoi):
            west, south, east, north = aoi
            if west > east:
                raise ValueError(
                    f"west {west} is east of east {east}, so this box crosses the "
                    "antimeridian. Split it at 180 and assemble each side separately.")
            # shapely.box quietly swaps these, so a transposed pair would come out as a
            # valid box somewhere the caller never asked about.
            if south > north:
                raise ValueError(
                    f"south {south} is north of north {north}. A box is "
                    "(west, south, east, north).")
            return shapely.geometry.box(*aoi), None
        if len(aoi) >= 3 and all(hasattr(pair, "__len__") and len(pair) == 2 for pair in aoi):
            return shapely.geometry.Polygon(aoi), None
        raise ValueError(
            "AOI sequence must be (west, south, east, north) or at least three (x, y) pairs")

    raise TypeError(f"cannot read an AOI from a {type(aoi).__name__}")

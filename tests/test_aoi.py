import geopandas as gpd
import pytest
from shapely.geometry import Polygon, box

from opera_fetch.aoi import as_geometry

BOX = (-107.0, 38.85, -106.85, 38.95)


def test_reads_every_shape_of_aoi_the_same_way():
    forms = [
        BOX,
        list(BOX),
        box(*BOX),
        box(*BOX).wkt,
        box(*BOX).__geo_interface__,
        list(box(*BOX).exterior.coords),
        gpd.GeoDataFrame(geometry=[box(*BOX)], crs="EPSG:4326"),
    ]
    for form in forms:
        assert as_geometry(form).equals_exact(box(*BOX), 1e-9), form


def test_reprojects_from_a_given_crs():
    utm = box(779_910, 4_283_790, 873_150, 4_332_210)
    lonlat = as_geometry(utm, crs="EPSG:32612")
    assert -108 < lonlat.centroid.x < -106
    assert 38 < lonlat.centroid.y < 39


def test_takes_the_crs_off_a_geodataframe_rather_than_the_argument():
    frame = gpd.GeoDataFrame(geometry=[box(779_910, 4_283_790, 873_150, 4_332_210)],
                             crs="EPSG:32612")
    assert -108 < as_geometry(frame, crs="EPSG:4326").centroid.x < -106


def test_reads_a_vector_file(tmp_path):
    path = tmp_path / "aoi.geojson"
    gpd.GeoDataFrame(geometry=[box(*BOX)], crs="EPSG:4326").to_file(path)
    assert as_geometry(path).bounds == pytest.approx(BOX)


def test_repairs_a_self_intersecting_polygon():
    bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])
    assert as_geometry(bowtie).is_valid


def test_a_multipart_aoi_becomes_its_hull_with_a_warning(caplog):
    parts = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1), box(3, 3, 4, 4)], crs="EPSG:4326")
    geometry = as_geometry(parts)
    assert geometry.geom_type == "Polygon"
    assert "convex hull" in caplog.text


def test_rejects_what_is_not_an_area():
    with pytest.raises(ValueError):
        as_geometry("not wkt and not a path")
    with pytest.raises(TypeError):
        as_geometry(42)
    with pytest.raises(ValueError):
        as_geometry([1, 2, 3])


def test_a_box_across_the_antimeridian_is_refused_not_wrapped():
    """Left alone it comes out as the whole world but the box, and searches the globe."""
    with pytest.raises(ValueError, match="antimeridian"):
        as_geometry((179.5, 51.5, -179.5, 52.0))


def test_a_geometry_that_wrapped_the_long_way_is_caught_too():
    from shapely.geometry import Polygon

    wrapped = Polygon([(-179.5, 51.5), (179.5, 51.5), (179.5, 52.0), (-179.5, 52.0)])
    with pytest.raises(ValueError, match="wrapped the long way"):
        as_geometry(wrapped)


def test_an_aoi_with_no_area_is_refused():
    with pytest.raises(ValueError, match="no area"):
        as_geometry([(0, 0), (1, 1), (2, 2)])


@pytest.mark.parametrize(("box", "says"), [
    ((38.88, -107.0, 38.95, -106.92), "latitude -107"),
    ((-107.0, 120.0, -106.9, 130.0), "latitude 120"),
    ((400.0, 38.0, 401.0, 39.0), "longitude 400"),
    ((326_500, 4_302_000, 339_000, 4_312_000), "aoi_crs"),
])
def test_coordinates_that_are_not_lon_lat_are_refused(box, says):
    """ASF clamps a latitude past 90 rather than complaining, degenerates the polygon and
    returns nothing, and the archive gets blamed for a transposed pair."""
    with pytest.raises(ValueError, match=says):
        as_geometry(box)


def test_a_box_with_south_above_north_is_refused():
    """shapely.box quietly swaps them, so it would come out valid somewhere else."""
    with pytest.raises(ValueError, match=r"south 39\.0 is north of north 38\.0"):
        as_geometry((-107.0, 39.0, -106.9, 38.0))


def test_a_projected_box_with_its_crs_is_still_ordinary():
    """The range check is about coordinates read as lon/lat, not about projected ones."""
    geometry = as_geometry((326_500, 4_302_000, 339_000, 4_312_000), "EPSG:32613")
    assert -108 < geometry.bounds[0] < -106

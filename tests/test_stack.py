import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_burst

from opera_fetch.stack import Pass, _one_zone, align_passes, group_paths

RTC = "OPERA_L2_RTC-S1_T049-103327-IW3_20241004T011054Z_20241004T043235Z_S1A_30_v1.0_VV.tif"
CSLC = "OPERA_L2_CSLC-S1_T049-103327-IW3_20241004T011054Z_20241005T000000Z_S1A_VV_v1.1.h5"
STATIC = ("OPERA_L2_RTC-S1-STATIC_T049-103327-IW3_20140403_S1A_30_v1.0"
          "_local_incidence_angle.tif")


def test_files_group_by_burst_and_by_product_family():
    groups = group_paths([RTC, STATIC, CSLC])
    assert set(groups) == {("RTC", "T049-103327-IW3"), ("CSLC", "T049-103327-IW3")}
    # The static layer belongs with the RTC acquisitions it describes.
    assert len(groups[("RTC", "T049-103327-IW3")]) == 2


def test_browse_and_checksums_are_not_mistaken_for_data():
    assert group_paths([RTC + ".md5", RTC.replace("_VV.tif", "_BROWSE.png")]) == {}


def test_bursts_of_one_overpass_get_one_timestamp():
    first = make_burst(west=500_010, north=4_332_210)
    # The next burst down the track is acquired a couple of seconds later.
    second = make_burst(west=500_010, north=4_332_210 - 180)
    second = second.assign_coords(time=second.indexes["time"] + pd.Timedelta("3s"))

    aligned = align_passes([first, second])
    assert aligned[0].indexes["time"].equals(aligned[1].indexes["time"])
    # Stamped with the earliest of the pass, not an invented average.
    assert aligned[1].indexes["time"][0] == first.indexes["time"][0]


def test_separate_acquisitions_stay_separate():
    burst = make_burst(west=500_010, north=4_332_210, times=3)
    aligned = align_passes([burst])
    assert len(aligned[0].indexes["time"].unique()) == 3


def test_the_tolerance_is_what_decides():
    first = make_burst(west=500_010, north=4_332_210)
    second = first.assign_coords(time=first.indexes["time"] + pd.Timedelta("5min"))

    assert align_passes([first, second], tolerance="1min")[1].indexes["time"][0] != \
        first.indexes["time"][0]
    assert align_passes([first, second], tolerance="10min")[1].indexes["time"][0] == \
        first.indexes["time"][0]


def test_a_pass_prints_as_something_readable():
    assert str(Pass(49, "ASCENDING", 32612)) == "T049 ascending EPSG:32612"


def test_passes_sort_without_complaint():
    assert sorted([Pass(49, "ASCENDING", 32612), Pass(27, "DESCENDING", 32612)])[0].track == 27


def test_a_stack_records_what_it_was_built_from():
    """A reprocessing changes the numbers, so a saved stack has to name its granules."""
    from opera_fetch import constants as const
    from opera_fetch.mosaic import mosaic

    a = make_burst(west=500_010, north=4_332_210)
    b = make_burst(west=500_010 + 4 * 30, north=4_332_210)
    a.attrs["granules"] = "OPERA_L2_RTC-S1_T049-103327-IW3_A"
    b.attrs["granules"] = "OPERA_L2_RTC-S1_T049-103328-IW3_B"
    b.attrs["burst_id"] = "T049-103328-IW3"

    merged = mosaic([a, b])
    assert merged.attrs["granules"].split("\n") == [
        "OPERA_L2_RTC-S1_T049-103327-IW3_A", "OPERA_L2_RTC-S1_T049-103328-IW3_B"]
    assert const.__version__


def test_a_misaligned_burst_is_not_swallowed_as_an_empty_one():
    """read_bursts skips a burst with nothing in it. A burst whose acquisitions do not
    share a grid is a different thing, and must not vanish with one warning."""
    import pytest

    from opera_fetch.errors import NoAcquisitions
    from opera_fetch.stack import read_bursts

    assert issubclass(NoAcquisitions, ValueError), "callers catching ValueError still work"

    class Misaligned(ValueError):
        pass

    def explode(paths, **kwargs):
        raise Misaligned("acquisitions are not on the same grid")

    import opera_fetch.rtc as rtc_module
    original = rtc_module.read_burst
    rtc_module.read_burst = explode
    try:
        with pytest.raises(Misaligned):
            read_bursts([
                "OPERA_L2_RTC-S1_T049-103327-IW3_20241004T011054Z_"
                "20241004T043235Z_S1A_30_v1.0_VV.tif"])
    finally:
        rtc_module.read_burst = original


def test_a_utm_box_can_be_given_as_the_aoi_in_that_zone():
    """There is no separate bounds argument: an exact box is an aoi with aoi_crs set."""
    from opera_fetch.aoi import as_geometry
    from opera_fetch.grid import reproject

    box = (326_500, 4_302_000, 339_000, 4_312_000)
    back = reproject(as_geometry(box, "EPSG:32613"), 32613).bounds
    # Within a cell of what was asked for, which the lattice widens out to anyway.
    assert all(abs(a - b) < 30 for a, b in zip(box, back, strict=True))


def _zone_bursts():
    """Two tracks in one zone, and one in another: what a boundary AOI looks like."""
    a = make_burst(west=500_010, north=4_332_210, track=49, direction="ASCENDING")
    b = make_burst(west=500_010, north=4_332_210, track=56, direction="DESCENDING")
    b = b.assign_coords(time=b.indexes["time"] + pd.Timedelta("6h"))
    b.attrs.update(track=56, direction="DESCENDING", burst_id="T056-118980-IW2")
    return a, b


def test_tracks_in_one_zone_share_one_dataset():
    """OPERA's grid is constant within a zone, so a track is not a reason to split."""
    from opera_fetch.mosaic import mosaic

    a, b = _zone_bursts()
    zone = _one_zone([_stamped(mosaic([a]), 49, "ASCENDING"),
                      _stamped(mosaic([b]), 56, "DESCENDING")], 32612)

    assert zone.sizes["time"] == a.sizes["time"] + b.sizes["time"], "no padding"
    assert sorted(int(t) for t in np.unique(zone.track.values)) == [49, 56]
    assert zone.indexes["time"].is_monotonic_increasing
    # Selecting a track is a comparison on the time axis, not a dictionary lookup.
    assert zone.sel(time=zone.track == 56).sizes["time"] == b.sizes["time"]


def test_a_zone_records_every_track_it_holds():
    from opera_fetch.mosaic import mosaic

    a, b = _zone_bursts()
    zone = _one_zone([_stamped(mosaic([a]), 49, "ASCENDING"),
                      _stamped(mosaic([b]), 56, "DESCENDING")], 32612)
    assert zone.attrs["tracks"] == [49, 56]
    assert zone.attrs["epsg"] == 32612


def _stamped(stack, track, direction):
    """What assemble puts on a pass before the zone concatenates them."""
    times = stack.sizes["time"]
    return stack.assign_coords(track=("time", [track] * times),
                               direction=("time", [direction] * times))


def test_reproject_to_gives_one_dataset_and_says_so(caplog):
    """The only resampling in the package, so it has to be asked for and has to be loud."""
    import logging

    from opera_fetch.stack import _onto_one_crs

    a = make_burst(west=500_010, north=4_332_210, epsg=32612)
    b = make_burst(west=500_010, north=4_332_210, epsg=32613)
    for stack, track in ((a, 49), (b, 56)):
        stack.coords["track"] = ("time", [track] * stack.sizes["time"])

    with caplog.at_level(logging.WARNING, logger="opera_fetch.stack"):
        joined = _onto_one_crs({32612: a, 32613: b}, "EPSG:32613")

    assert not isinstance(joined, dict), "one CRS means one Dataset"
    assert joined.rio.crs.to_epsg() == 32613
    assert joined.attrs["reprojected_from"] == [32612, 32613]
    assert "moves a value" in caplog.text


def test_reprojection_does_not_invent_a_mask_code():
    """Untold which code is nodata, GDAL rewrote 255 to 254, which OPERA does not define."""
    from opera_fetch.stack import _onto_one_crs

    a = make_burst(west=500_010, north=4_332_210, epsg=32612)
    b = make_burst(west=500_010, north=4_332_210, epsg=32613)
    a["mask"][:] = 255

    joined = _onto_one_crs({32612: a, 32613: b}, "EPSG:32613")
    codes = set(np.unique(joined.mask.values).tolist())
    assert codes <= {0, 1, 2, 3, 255}, codes


def _complex_zones():
    a = make_burst(west=500_010, north=4_332_210, epsg=32612)
    b = make_burst(west=500_010, north=4_332_210, epsg=32613)
    for stack, track in ((a, 49), (b, 56)):
        values = (stack.vv.values + 1j).astype("complex64")
        stack["vv"] = (stack.vv.dims, values)
        stack.coords["track"] = ("time", [track] * stack.sizes["time"])
    return {32612: a, 32613: b}


def test_complex_survives_being_reprojected():
    """Nearest moves a sample. Anything else would average two phases into a third."""
    from opera_fetch.stack import _onto_one_crs

    joined = _onto_one_crs(_complex_zones(), "EPSG:32613")
    assert np.issubdtype(joined.vv.dtype, np.complexfloating)

    finite = joined.vv.values[np.isfinite(joined.vv.values)]
    assert finite.size
    assert np.any(finite.imag != 0), "the imaginary part was dropped somewhere"


def test_complex_is_oversampled_before_it_is_moved():
    """Not a kernel choice: resampling a CSLC directly costs coherence whatever the kernel,
    because speckle fills the band up to Nyquist. Oversampling first is the fix."""
    from opera_fetch import resample
    from opera_fetch.stack import _onto_one_crs

    joined = _onto_one_crs(_complex_zones(), "EPSG:32613")
    assert np.issubdtype(joined.vv.dtype, np.complexfloating)
    assert resample.FACTOR >= 8, "eight is where the coherence curve flattens"


def test_the_oversampling_factor_gives_way_on_a_big_scene():
    """The transform costs the square of the factor, so a large scene takes a smaller one
    rather than the machine taking the hit."""
    from opera_fetch import resample

    small = make_burst(west=500_010, north=4_332_210, columns=64, rows=64)
    assert resample.affordable_factor(small.vv) == resample.FACTOR
    assert resample.affordable_factor(small.vv, budget=1_000) == 1
    # The budget is spent against the real peak, which is three times the wide array.
    assert resample.affordable_factor(
        small.vv, budget=small.vv.nbytes * 16 * resample.OVERHEAD) == 4


def test_oversampling_reproduces_the_samples_it_started_from():
    """Zero padding the spectrum is exact for a bandlimited signal, which is the whole
    reason to do it rather than interpolate twice."""
    from opera_fetch import resample

    rng = np.random.default_rng(0)
    values = (rng.random((64, 64)) + 1j * rng.random((64, 64))).astype("complex64")
    coarse = make_burst(west=500_010, north=4_332_210, columns=64, rows=64).vv.isel(time=0)
    coarse = coarse.copy(data=values)

    fine = resample.oversample(coarse, 4)
    assert fine.shape == (256, 256)
    assert np.allclose(fine.values[::4, ::4], values, atol=1e-5)
    assert float(fine.x[0]) == float(coarse.x[0])


def test_the_kernel_choice_applies_to_real_layers():
    """resampling= is for the real layers. A complex one is oversampled either way."""
    from opera_fetch.stack import _onto_one_crs

    joined = _onto_one_crs(_complex_zones(), "EPSG:32613", resampling="bilinear")
    assert np.issubdtype(joined.vv.dtype, np.complexfloating)


def test_the_mask_moves_by_nearest_whatever_the_data_does():
    """A class code interpolated with its neighbours is a code nobody observed."""
    from opera_fetch.stack import _onto_one_crs

    a = make_burst(west=500_010, north=4_332_210, epsg=32612)
    b = make_burst(west=500_010, north=4_332_210, epsg=32613)
    a["mask"][:] = 2
    b["mask"][:] = 0

    joined = _onto_one_crs({32612: a, 32613: b}, "EPSG:32613", resampling="bilinear")
    codes = set(np.unique(joined.mask.values).tolist())
    assert codes <= {0, 1, 2, 3, 255}, codes
    assert joined.mask.dtype == np.uint8


def test_a_complex_mosaic_keeps_its_coordinate_types():
    """combine_first aligns and fills, which floats an int coordinate and objects a string."""
    from opera_fetch.mosaic import mosaic

    a, b = two_complex_bursts()
    merged = mosaic([a, b])
    assert merged.track.dtype == np.int64
    assert merged.platform.dtype.kind == "U", merged.platform.dtype


def two_complex_bursts():
    a = make_burst(west=500_010, north=4_332_210, columns=8)
    b = make_burst(west=500_010 + 4 * 30, north=4_332_210, columns=8)
    b.attrs["burst_id"] = "T049-103328-IW3"
    for stack in (a, b):
        stack["vv"] = (stack.vv.dims, (stack.vv.values + 1j).astype("complex64"))
        stack.coords["platform"] = ("time", ["S1A"] * stack.sizes["time"])
        stack.coords["track"] = ("time", np.full(stack.sizes["time"], 49, dtype="int64"))
    return a, b


def test_the_memory_estimate_is_the_one_the_budget_uses():
    """It is knowable before anything runs: the grid comes from the area asked for."""
    from opera_fetch import resample

    scene = make_burst(west=500_010, north=4_332_210, columns=64, rows=64).vv.isel(time=0)
    # Three times the wide array, not one: the transform leaves temporaries behind.
    assert resample.peak_bytes(scene, 8) == scene.nbytes * 64 * resample.OVERHEAD

    # And the budget is spent against that estimate, not against the naive one.
    budget = scene.nbytes * 64 * resample.OVERHEAD
    assert resample.affordable_factor(scene, budget=budget) == 8
    assert resample.affordable_factor(scene, budget=budget - 1) == 4


def test_oversampling_handles_an_odd_number_of_samples():
    """Splitting the spectrum at n//2 would drop a row of an odd sized scene."""
    from opera_fetch import resample

    rng = np.random.default_rng(0)
    values = (rng.random((31, 33)) + 1j * rng.random((31, 33))).astype("complex64")
    coarse = make_burst(west=500_010, north=4_332_210, columns=33, rows=31).vv.isel(time=0)
    coarse = coarse.copy(data=values)

    fine = resample.oversample(coarse, 4)
    assert fine.shape == (124, 132)
    assert np.allclose(fine.values[::4, ::4], values, atol=1e-4)


def test_both_products_describe_themselves_the_same_way():
    """RTC keeps its identity in GeoTIFF tags and CSLC in an HDF5 group. They used to build
    the attributes separately, which is how two products drift apart."""
    from opera_fetch import cslc, metadata, rtc

    tags = {"BURST_ID": "t049_103327_iw3", "TRACK_NUMBER": "49",
            "ORBIT_PASS_DIRECTION": "ascending", "BOUNDING_POLYGON": "POLYGON ((0 0))",
            "PRODUCT_VERSION": "1.0"}
    group = {"burst_id": b"t049_103327_iw3".decode(), "track_number": 49,
             "orbit_pass_direction": "Ascending", "bounding_polygon": "POLYGON ((0 0))",
             "product_version": "1.0"}
    assert set(rtc.identify(tags)) == set(cslc.identify(group)) == set(metadata.FIELDS)

    one = metadata.describe(make_burst(west=500_010, north=4_332_210), "RTC",
                            rtc.identify(tags), ["a"]).attrs
    two = metadata.describe(make_burst(west=500_010, north=4_332_210), "RTC",
                            cslc.identify(group), ["a"]).attrs
    assert one == two, "the same burst described two ways must come out the same"
    assert one["burst_id"] == "T049-103327-IW3"
    assert one["direction"] == "ASCENDING"
    assert isinstance(one["track"], int)


def test_a_granule_with_no_identity_is_refused():
    from opera_fetch import metadata

    with pytest.raises(ValueError, match="names no burst_id"):
        metadata.describe(make_burst(west=500_010, north=4_332_210), "RTC",
                          {"track": 49, "direction": "ASCENDING"}, [])


def test_a_zone_already_in_the_target_crs_is_not_resampled():
    """Choosing the busiest zone as the reference resampled data that was already right,
    and put the result on a grid nobody delivered."""
    from opera_fetch.stack import _onto_one_crs

    quiet = make_burst(west=500_010, north=4_332_210, epsg=32612, times=2)
    busy = make_burst(west=500_010, north=4_332_210, epsg=32613, times=4)
    for stack, track in ((quiet, 49), (busy, 56)):
        stack.coords["track"] = ("time", [track] * stack.sizes["time"])

    # Ask for the quieter zone's own CRS: its grid must survive untouched.
    joined = _onto_one_crs({32612: quiet, 32613: busy}, "EPSG:32612")
    assert np.array_equal(joined.x.values, quiet.x.values)
    assert np.array_equal(joined.y.values, quiet.y.values)


def test_a_crs_no_zone_is_in_says_everything_moves(caplog):
    import logging

    from opera_fetch.stack import _onto_one_crs

    a = make_burst(west=500_010, north=4_332_210, epsg=32612)
    b = make_burst(west=500_010, north=4_332_210, epsg=32613)
    for stack, track in ((a, 49), (b, 56)):
        stack.coords["track"] = ("time", [track] * stack.sizes["time"])

    with caplog.at_level(logging.WARNING, logger="opera_fetch.stack"):
        _onto_one_crs({32612: a, 32613: b}, "EPSG:32611")
    assert "no zone is in" in caplog.text


def test_auto_picks_the_zone_that_leaves_the_fewest_values_to_move():
    """Nobody knows their UTM zone offhand, so 'auto' picks the one holding the most data.

    Coverage alone and acquisition count alone both get this wrong, so each case below is
    one where the two disagree and only their product gives the answer that resamples less.
    """
    from opera_fetch.stack import _best_zone

    # Wide but rarely imaged, 48 cells x 2, against a sliver imaged often, 12 cells x 9.
    wide = make_burst(west=500_010, north=4_332_210, epsg=32612, times=2)
    often = make_burst(west=500_010, north=4_332_210, epsg=32613, times=9)
    often["vv"][:, :, 2:] = np.nan
    assert _best_zone({32612: wide, 32613: often}) == 32613, "96 values would move, not 108"

    # Same shapes, but now the sliver is imaged only three times: 48 x 2 beats 12 x 3.
    seldom = make_burst(west=500_010, north=4_332_210, epsg=32613, times=3)
    seldom["vv"][:, :, 2:] = np.nan
    assert _best_zone({32612: wide, 32613: seldom}) == 32612, "36 values would move, not 96"


def test_auto_is_repeatable_when_two_zones_hold_the_same():
    """Same AOI, same answer: a tie must not fall to whichever zone was read first."""
    from opera_fetch.stack import _best_zone

    a = make_burst(west=500_010, north=4_332_210, epsg=32612)
    b = make_burst(west=500_010, north=4_332_210, epsg=32613)
    assert _best_zone({32612: a, 32613: b}) == 32612
    assert _best_zone({32613: b, 32612: a}) == 32612


def test_auto_leaves_the_chosen_zone_on_operas_own_grid():
    """The point of choosing the busiest zone is that most of the data then never moves."""
    from opera_fetch.stack import _best_zone, _onto_one_crs

    winner = make_burst(west=500_010, north=4_332_210, epsg=32612)
    other = make_burst(west=500_010, north=4_332_210, epsg=32613)
    other["vv"][:, :, 4:] = np.nan          # half the coverage, same number of acquisitions

    zones = {32612: winner, 32613: other}
    joined = _onto_one_crs(zones, f"EPSG:{_best_zone(zones)}")

    assert joined.rio.crs.to_epsg() == 32612
    assert np.array_equal(joined.x.values, winner.x.values), "the winner's grid moved"
    assert np.array_equal(joined.y.values, winner.y.values), "the winner's grid moved"


def test_auto_on_a_single_zone_resamples_nothing(caplog):
    """The common case: one zone, and 'auto' just unwraps the dict without touching it."""
    import logging

    from opera_fetch.stack import _best_zone, _onto_one_crs

    only = make_burst(west=500_010, north=4_332_210, epsg=32612)
    with caplog.at_level(logging.WARNING, logger="opera_fetch.stack"):
        joined = _onto_one_crs({32612: only}, f"EPSG:{_best_zone({32612: only})}")

    assert np.array_equal(joined.vv.values, only.vv.values)
    assert "moves a value" not in caplog.text

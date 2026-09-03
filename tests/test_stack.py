import numpy as np
import pandas as pd
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


def test_an_aoi_and_bounds_together_are_refused():
    """Both say what area to deliver, and bounds used to lose without a word."""
    import pytest

    from opera_fetch.stack import assemble

    with pytest.raises(ValueError, match="not both"):
        assemble(["OPERA_L2_RTC-S1_T049-103327-IW3_20241004T011054Z_"
                  "20241004T043235Z_S1A_30_v1.0_VV.tif"],
                 aoi=(-107.0, 38.8, -106.9, 38.9), bounds=(0, 0, 1, 1))


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

import pandas as pd
from tests.conftest import make_burst

from opera_fetch.stack import Pass, align_passes, group_paths

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

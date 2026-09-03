import pandas as pd

from opera_fetch import constants as const
from opera_fetch.filenames import (
    keep_latest_processing,
    parse_burst_id,
    parse_layer,
    parse_processing_time,
    parse_product,
)

RTC = "OPERA_L2_RTC-S1_T049-103327-IW3_20241004T011054Z_20241004T043235Z_S1A_30_v1.0_VV.tif"
STATIC = ("OPERA_L2_RTC-S1-STATIC_T049-103327-IW3_20140403_S1A_30_v1.0"
          "_local_incidence_angle.tif")
CSLC = "OPERA_L2_CSLC-S1_T094-200132-IW2_20250111T032103Z_20250123T000352Z_S1A_VV_v1.1.h5"
CSLC_STATIC = "OPERA_L2_CSLC-S1-STATIC_T049-103327-IW3_20140403_S1A_v1.0.h5"


def test_static_is_not_read_as_its_time_varying_sibling():
    assert parse_product(RTC) == const.RTC
    assert parse_product(STATIC) == const.RTC_STATIC
    assert parse_product(CSLC) == const.CSLC
    assert parse_product(CSLC_STATIC) == const.CSLC_STATIC


def test_burst_id_comes_out_hyphenated():
    assert parse_burst_id(RTC) == "T049-103327-IW3"
    assert parse_burst_id(CSLC) == "T094-200132-IW2"


def test_a_layer_whose_name_holds_underscores_is_read_whole():
    # Splitting on the last underscore would call this layer "angle".
    assert parse_layer(STATIC, const.LAYERS[const.RTC_STATIC]) == "local_incidence_angle"
    assert parse_layer(RTC, const.LAYERS[const.RTC]) == "VV"
    assert parse_layer("something_else.tif", const.LAYERS[const.RTC]) is None


def test_processing_stamp_orders_reprocessed_versions():
    later = RTC.replace("20241004T043235Z", "20250101T000000Z")
    assert parse_processing_time(later) > parse_processing_time(RTC)
    assert parse_processing_time(RTC) == pd.Timestamp("2024-10-04T04:32:35")


def test_a_reprocessed_granule_supersedes_the_one_it_replaces():
    """Both readers rely on this: the archive keeps every processing of an acquisition."""
    original = RTC
    reprocessed = RTC.replace("20241004T043235Z", "20250101T000000Z")
    other_day = RTC.replace("20241004T011054Z", "20241016T011054Z")

    keys = ["2024-10-04", "2024-10-04", "2024-10-16"]
    latest = keep_latest_processing(keys, [original, reprocessed, other_day])

    assert latest == {"2024-10-04": 1, "2024-10-16": 2}


def test_nothing_is_dropped_when_no_acquisition_repeats():
    granules = [RTC, RTC.replace("20241004T011054Z", "20241016T011054Z")]
    assert len(keep_latest_processing(["a", "b"], granules)) == 2

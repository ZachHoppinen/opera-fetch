"""Live ASF queries. Deselect with -m 'not network'."""

import pytest

from opera_fetch import constants as const
from opera_fetch.search import data_urls, search, search_static

pytestmark = pytest.mark.network

AOI = (-107.0, 38.85, -106.85, 38.95)
WINDOW = ("2024-11-01", "2024-11-15")


def test_rtc_search_returns_a_frame_worth_looking_at():
    found = search(aoi=AOI, start=WINDOW[0], end=WINDOW[1], product=const.RTC)

    assert not found.empty
    assert set(found.burst_id.str.slice(0, 1)) == {"T"}
    assert found.time.notna().all()
    assert found.direction.isin(["ASCENDING", "DESCENDING"]).all()


def test_the_urls_it_gives_are_the_data_layers_and_nothing_else():
    found = search(aoi=AOI, start=WINDOW[0], end=WINDOW[1], product=const.RTC)
    urls = data_urls(found)

    assert urls, "no URLs came back"
    assert all(url.endswith(("_VV.tif", "_VH.tif", "_mask.tif")) for url in urls)
    assert not any("BROWSE" in url or url.endswith(".md5") for url in urls)


def test_static_layers_are_found_by_burst_rather_than_by_date():
    found = search(aoi=AOI, start=WINDOW[0], end=WINDOW[1], product=const.RTC)
    static = search_static(found)

    assert set(static.burst_id) <= set(found.burst_id)
    assert (static["product"] == const.RTC_STATIC).all()
    # The looks layer comes too: it is what a mosaic weights its bursts by.
    assert all(url.endswith(("_local_incidence_angle.tif", "_number_of_looks.tif"))
               for url in data_urls(static))


def test_cslc_search_finds_one_h5_per_granule():
    found = search(aoi=AOI, start=WINDOW[0], end=WINDOW[1], product=const.CSLC)
    urls = data_urls(found)

    assert len(urls) == len(found)
    assert all(url.endswith(".h5") for url in urls)


def test_a_frame_carries_what_it_takes_to_group_by_pass():
    found = search(aoi=AOI, start=WINDOW[0], end=WINDOW[1], product=const.RTC)
    grouped = found.groupby(["track", "direction"]).burst_id.nunique()

    assert len(grouped) >= 1
    assert found.burst_id.nunique() == grouped.sum()


def test_a_track_filter_narrows_the_search():
    everything = search(aoi=AOI, start=WINDOW[0], end=WINDOW[1], product=const.RTC)
    one = everything.track.iloc[0]
    narrowed = search(aoi=AOI, start=WINDOW[0], end=WINDOW[1], product=const.RTC, track=one)

    assert set(narrowed.track) == {one}
    assert len(narrowed) <= len(everything)


def test_the_size_of_the_job_is_reported_before_it_is_started(caplog):
    import logging

    found = search(aoi=AOI, start=WINDOW[0], end=WINDOW[1], product=const.CSLC)
    with caplog.at_level(logging.INFO, logger="opera_fetch.search"):
        data_urls(found)

    # One CSLC burst is a few hundred MB an acquisition, and the .iso.xml beside it is a
    # few hundred KB, so counting the wrong files would be off by three orders.
    reported = float(caplog.text.split(" GB")[0].split()[-1])
    assert reported > 0.1 * len(found)


def test_a_search_with_no_results_is_an_empty_frame_not_a_crash():
    """A year OPERA does not cover has to read like any other empty result."""
    found = search(aoi=AOI, start="2016-01-01", end="2016-01-03", product=const.RTC)
    assert list(found.columns)
    assert found.empty or len(found) > 0
    assert "burst_id" in found.columns


@pytest.mark.parametrize("retries", [0, -1])
def test_a_retry_count_under_one_is_refused(retries):
    """range(1, 1) is empty, so the search was never made and the result never bound."""
    with pytest.raises(ValueError, match="at least 1"):
        search(aoi=AOI, start=WINDOW[0], end=WINDOW[1], retries=retries)

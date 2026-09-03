"""Date parsing. Lives with search, because a date range is a search parameter and the
static products that ignore one are a search rule."""

import pandas as pd
import pytest

from opera_fetch.search import as_dates


def test_dates_read_from_anything_pandas_knows():
    start, end = as_dates("2024-10-01", "2025-06-30")
    assert start == pd.Timestamp("2024-10-01") and end == pd.Timestamp("2025-06-30")


def test_an_open_ended_range_is_allowed_because_static_products_have_no_time():
    assert as_dates(None, None) == (None, None)
    assert as_dates("2024-10-01", None)[1] is None


def test_a_backwards_date_range_is_an_error():
    with pytest.raises(ValueError, match="after"):
        as_dates("2025-06-30", "2024-10-01")


def test_something_that_is_not_a_date_says_which_end_it_was():
    with pytest.raises(ValueError, match="start"):
        as_dates("not a date", "2025-06-30")


def test_a_range_entirely_before_the_opera_archive_says_so():
    """Sentinel-1 launched in 2014, but OPERA's archive starts years later, so this
    returns nothing whatever the area."""
    with pytest.raises(ValueError, match="OPERA archive starts"):
        as_dates("2014-01-01", "2015-06-30")


def test_a_range_that_only_starts_early_is_allowed():
    start, end = as_dates("2015-01-01", "2020-01-01")
    assert start is not None and end is not None

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_burst

from opera_fetch.mosaic import align_passes, mosaic


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


def two_overlapping_bursts(fill_a=2.0, fill_b=4.0):
    """Two bursts of one track, sharing four columns."""
    a = make_burst(west=500_010, north=4_332_210, columns=8, fill=fill_a)
    b = make_burst(west=500_010 + 4 * 30, north=4_332_210, columns=8, fill=fill_b)
    b.attrs["burst_id"] = "T049-103328-IW3"
    return a, b


def test_overlaps_average_and_the_rest_is_carried_through():
    a, b = two_overlapping_bursts()
    merged = mosaic([a, b])

    assert merged.sizes["x"] == 12
    assert merged.vv.isel(time=0, x=0).values == pytest.approx(2.0)     # only a
    assert merged.vv.isel(time=0, y=0, x=6).values == pytest.approx(3.0)     # both, averaged
    assert merged.vv.isel(time=0, x=-1).values == pytest.approx(4.0)    # only b


def test_a_mask_takes_the_worst_code_rather_than_the_mean_of_codes():
    """Both bursts fed the averaged value, so the overlap carries the worse of the two."""
    a, b = two_overlapping_bursts()
    a["mask"][:] = 2          # layover in a
    b["mask"][:] = 0          # clear in b
    merged = mosaic([a, b])

    assert merged.mask.dtype == np.uint8
    # The mean would be 1, the code for shadow, which is a class nobody observed.
    assert int(merged.mask.isel(time=0, y=0, x=6)) == 2
    assert int(merged.mask.isel(time=0, y=0, x=0)) == 2      # only a
    assert int(merged.mask.isel(time=0, y=0, x=-1)) == 0     # only b


def test_bursts_are_weighted_by_their_looks_where_they_overlap():
    """OPERA mosaics its own bursts this way: a burst counts for the looks behind it."""
    a, b = two_overlapping_bursts(fill_a=2.0, fill_b=4.0)
    a["number_of_looks"] = (("y", "x"), np.full((a.sizes["y"], a.sizes["x"]), 3.0))
    b["number_of_looks"] = (("y", "x"), np.full((b.sizes["y"], b.sizes["x"]), 1.0))

    merged = mosaic([a, b])
    # Flat, the overlap would be 3.0; weighted 3:1 toward a it is 2.5.
    assert float(merged.vv.isel(time=0, y=0, x=6)) == pytest.approx(2.5)
    # And the looks behind the mosaicked pixel are all of them.
    assert float(merged.number_of_looks.isel(y=0, x=6)) == pytest.approx(4.0)


def test_without_the_looks_layer_the_average_is_flat():
    a, b = two_overlapping_bursts(fill_a=2.0, fill_b=4.0)
    merged = mosaic([a, b])
    assert float(merged.vv.isel(time=0, y=0, x=6)) == pytest.approx(3.0)


def test_complex_bursts_are_not_averaged():
    a, b = two_overlapping_bursts()
    for burst, value in ((a, 1 + 1j), (b, -1 - 1j)):
        burst["vv"] = burst.vv.astype("complex64") * 0 + np.complex64(value)
    merged = mosaic([a, b])

    # Averaging these would cancel them to zero, which is what must not happen.
    assert merged.vv.isel(time=0, y=0, x=6).values == np.complex64(1 + 1j)


def test_mixing_ascending_with_descending_is_refused():
    a, b = two_overlapping_bursts()
    b.attrs["direction"] = "DESCENDING"
    with pytest.raises(ValueError, match="direction"):
        mosaic([a, b])


def test_mixing_tracks_is_refused():
    a, b = two_overlapping_bursts()
    b.attrs["track"] = 27
    with pytest.raises(ValueError, match="track"):
        mosaic([a, b])


def test_mixing_utm_zones_is_refused_with_the_reason():
    a, _ = two_overlapping_bursts()
    b = make_burst(west=500_010, north=4_332_210, epsg=32613)
    with pytest.raises(ValueError, match="one UTM zone at a time"):
        mosaic([a, b])


def test_unaligned_times_are_warned_about(caplog):
    a, b = two_overlapping_bursts()
    b = b.assign_coords(time=b.indexes["time"] + np.timedelta64(3, "s"))
    mosaic([a, b])
    assert "align_passes" in caplog.text


def test_the_footprint_of_a_mosaic_is_the_union_of_its_bursts():
    """Taking the first burst's understated a real three-burst mosaic threefold."""
    import shapely.wkt

    a, b = two_overlapping_bursts()
    a.attrs["footprint"] = "POLYGON ((0 0, 2 0, 2 1, 0 1, 0 0))"
    b.attrs["footprint"] = "POLYGON ((1 0, 3 0, 3 1, 1 1, 1 0))"

    merged = mosaic([a, b])
    assert shapely.wkt.loads(merged.attrs["footprint"]).area == pytest.approx(3.0)


def test_a_mosaic_keeps_what_varies_between_acquisitions():
    a, b = two_overlapping_bursts()
    for burst in (a, b):
        burst.coords["platform"] = ("time", ["S1A"] * burst.sizes["time"])
        burst.coords["absolute_orbit"] = ("time", list(range(burst.sizes["time"])))

    merged = mosaic([a, b])
    assert "platform" in merged.coords and "absolute_orbit" in merged.coords
    assert list(merged.platform.values) == ["S1A"] * merged.sizes["time"]


def test_an_unknown_look_count_does_not_drop_the_burst_from_that_cell():
    """A weight of zero would mean "contributed nothing", not "we do not know"."""
    a, b = two_overlapping_bursts(fill_a=2.0, fill_b=4.0)
    for burst, count in ((a, 3.0), (b, 1.0)):
        burst["number_of_looks"] = (("y", "x"),
                                    np.full((burst.sizes["y"], burst.sizes["x"]), count))
    a["number_of_looks"][:] = np.nan          # a's looks are missing everywhere

    merged = mosaic([a, b])
    # a still counts: unknown falls back to what b weighs, so the overlap is a flat mean.
    assert float(merged.vv.isel(time=0, y=0, x=6)) == pytest.approx(3.0)
    # and a cell only a reaches keeps its value rather than going empty.
    assert float(merged.vv.isel(time=0, y=0, x=0)) == pytest.approx(2.0)


def test_a_single_burst_mosaic_leaves_the_caller_alone():
    """A mosaic of one burst is still a new object, not the burst that went in."""
    one = make_burst(west=500_010, north=4_332_210)
    before = dict(one.attrs)

    merged = mosaic([one])
    assert merged is not one
    assert one.attrs == before, "the caller's burst was mutated"
    assert merged.attrs["bursts"] == 1
    assert np.array_equal(merged.vv.values, one.vv.values, equal_nan=True)


def test_looks_nobody_knows_stay_unknown():
    """Summing nothing gives zero, and zero looks means something else entirely."""
    a, b = two_overlapping_bursts()
    for burst in (a, b):
        burst["number_of_looks"] = (("y", "x"),
                                    np.full((burst.sizes["y"], burst.sizes["x"]), np.nan))

    merged = mosaic([a, b])
    assert np.isnan(float(merged.number_of_looks.isel(y=0, x=6)))


def test_taking_the_first_burst_leaves_the_mask_an_integer():
    """combine_first fills, and a filled mask is a float one. This is the default for
    complex data, so every CSLC mosaic carried 127.0 and NaN at the same time."""
    from opera_fetch.mosaic import mosaic

    a = make_burst(west=500_010, north=4_332_210, columns=8)
    b = make_burst(west=500_010 + 8 * 30, north=4_332_210, columns=8)
    b.attrs["burst_id"] = "T049-103328-IW3"
    b["mask"][:] = 1

    merged = mosaic([a, b], how="first")

    assert merged["mask"].dtype == np.uint8
    assert set(np.unique(merged["mask"].values).tolist()) <= {0, 1, 2, 3, 255}

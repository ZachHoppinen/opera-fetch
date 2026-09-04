import numpy as np
import pytest
from tests.conftest import make_burst

from opera_fetch.validate import check_files, quicklook, report, summary


def test_a_sound_stack_reports_what_is_in_it():
    burst = make_burst(west=500_010, north=4_332_210, times=3)
    found = report(burst)

    assert found["times"] == 3
    assert found["crs"] == "EPSG:32612"
    assert found["monotonic"]
    assert found["median_coverage"] == 1.0
    assert found["damage"] == []


def test_coverage_is_counted_over_every_finite_pixel():
    burst = make_burst(west=500_010, north=4_332_210, times=2)
    burst["vv"][0, :3, :] = np.nan
    found = report(burst)
    assert found["coverage"][0] == pytest.approx(0.5)
    assert found["coverage"][1] == 1.0


def test_a_repeated_timestamp_is_damage():
    burst = make_burst(west=500_010, north=4_332_210, times=2)
    burst = burst.assign_coords(time=[burst.indexes["time"][0]] * 2)
    with pytest.raises(ValueError, match="more than once"):
        report(burst)


def test_an_empty_acquisition_is_reported_but_not_fatal(caplog):
    burst = make_burst(west=500_010, north=4_332_210, times=2)
    burst["vv"][0] = np.nan
    found = report(burst)

    assert len(found["empty_times"]) == 1
    assert "entirely empty" in caplog.text


def test_summary_is_a_few_lines_a_person_can_read():
    text = summary(make_burst(west=500_010, north=4_332_210))
    assert "grid" in text and "times" in text and "coverage" in text
    assert len(text.splitlines()) <= 8


def test_a_truncated_file_is_caught(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    path = tmp_path / "burst_VV.tif"
    with rasterio.open(path, "w", driver="GTiff", height=64, width=64, count=1,
                       dtype="float32", crs="EPSG:32612",
                       transform=from_origin(500_010, 4_332_210, 30, 30)) as out:
        out.write(np.ones((64, 64), dtype="float32"), 1)

    assert check_files([path]) == []
    path.write_bytes(path.read_bytes()[: len(path.read_bytes()) // 2])
    assert check_files([path]) == [path]


def test_a_quicklook_gets_written(tmp_path):
    pytest.importorskip("matplotlib")
    burst = make_burst(west=500_010, north=4_332_210, times=3)
    burst["vv"][:] = np.random.default_rng(0).random(burst.vv.shape).astype("float32")

    path = quicklook(burst, tmp_path / "figures" / "stack.png")
    assert path.exists() and path.stat().st_size > 0


def test_a_complex_quicklook_plots_its_amplitude(tmp_path):
    pytest.importorskip("matplotlib")
    burst = make_burst(west=500_010, north=4_332_210, times=2)
    burst["vv"] = (burst.vv.dims, (burst.vv.values + 1j).astype("complex64"))
    assert quicklook(burst, tmp_path / "cslc.png").exists()


def test_a_single_acquisition_still_plots(tmp_path):
    """A stack of one date is ordinary, and a line of one point is not something to fail on."""
    pytest.importorskip("matplotlib")
    burst = make_burst(west=500_010, north=4_332_210, times=1)
    assert quicklook(burst, tmp_path / "one.png").exists()


def test_a_mask_code_opera_does_not_define_is_damage():
    """MASK_MEANINGS is the forecaster-facing flag_meanings, so a code outside it is not
    a class anyone can act on."""
    from opera_fetch.validate import report

    stack = make_burst(west=500_010, north=4_332_210)
    stack["mask"][:] = 42
    with pytest.raises(ValueError, match="42, which OPERA does not define"):
        report(stack)


def test_a_stack_of_nothing_but_the_nodata_code_is_not_fully_covered():
    """255 is finite, and astype(bool) on it is True, so a stack holding no observation
    at all reported 100% coverage and no damage."""
    from opera_fetch.validate import primary_variable, report

    stack = make_burst(west=500_010, north=4_332_210).drop_vars(
        ["vv", "local_incidence_angle"])
    stack["mask"][:] = 255

    assert primary_variable(stack) == "mask", "there is nothing else to report on"
    with pytest.raises(ValueError, match="no acquisition has a single finite pixel"):
        report(stack)


def test_a_grid_that_misses_the_aoi_entirely_is_damage():
    """aoi_covered was computed correctly from the first version and never looked at."""
    from shapely.geometry import box

    from opera_fetch.validate import report

    stack = make_burst(west=500_010, north=4_332_210)
    with pytest.raises(ValueError, match="do not overlap"):
        report(stack, aoi=box(-100.0, 30.0, -99.9, 30.1))

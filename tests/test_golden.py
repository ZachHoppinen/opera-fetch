"""End-to-end runs pinned to a recorded result.

Every other test says one thing about one function. These run ``assemble`` over files on
disk, the way a caller does, and compare everything that comes out against a digest
checked in beside them. A change nobody meant to make shows up here as a diff rather than
as a wrong number in somebody's melt map a season later.

The granules are synthetic and deterministic, so these run in CI, where the tests marked
``data`` do not.

Regenerate after a change you meant:

    OPERA_FETCH_GOLDEN=update pytest tests/test_golden.py

and read the diff before committing it. A digest that changed for a reason you cannot
name is the finding, not the noise.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from tests.granules import write_granule, write_static

from opera_fetch.grid import bounds_of, spacing_of
from opera_fetch.stack import assemble

GOLDEN = Path(__file__).parent / "golden"

# Written afresh every run, so they can never agree.
VOLATILE = ("created", "opera_fetch_version")


def digest(stack, exact):
    """Everything about a stack that ought not to change by accident.

    exact adds a hash of the values themselves. It is for the paths where the package
    promises it moves nothing: a hash is what tells a shifted or transposed grid from an
    intact one, where a sum or a count cannot. The reprojection path takes exact=False,
    because what GDAL's kernels return is its business and its version's, and a digest
    that fails on somebody's upgrade teaches everyone to ignore it.
    """
    found = {
        "sizes": {k: int(v) for k, v in sorted(stack.sizes.items())},
        "crs": stack.rio.crs.to_epsg(),
        "spacing": [round(v, 6) for v in spacing_of(stack)],
        "bounds": [round(v, 3) for v in bounds_of(stack)],
        "attrs": {k: _plain(v) for k, v in sorted(stack.attrs.items())
                  if k not in VOLATILE},
        "coords": {},
        "variables": {},
    }
    for name in ("time", "track", "direction", "platform", "absolute_orbit"):
        if name in stack.coords:
            found["coords"][name] = _plain(stack[name].values)

    for name, array in sorted(stack.data_vars.items()):
        values = np.asarray(array.values)
        # The variable's own attributes too: flag_meanings is what a forecaster reads a
        # class code by, and swapping two of them changes nothing else about the numbers.
        entry = {"dims": list(array.dims), "dtype": str(values.dtype),
                 "attrs": {k: _plain(v) for k, v in sorted(array.attrs.items())}}
        if values.dtype.kind in "iu":
            # A class code has no range worth recording, and how many cells hold each code
            # is what says whether something filled or interpolated one.
            codes, counts = np.unique(values, return_counts=True)
            entry["codes"] = {str(code): int(count)
                              for code, count in zip(codes, counts, strict=True)}
        else:
            finite = values[np.isfinite(values)]
            entry["observed"] = int(finite.size)
            entry["range"] = ([round(float(finite.min()), 4), round(float(finite.max()), 4)]
                              if finite.size else None)
        if exact:
            entry["hash"] = _hash(values)
        found["variables"][name] = entry
    return found


def _hash(values):
    """A hash of the values as laid out, so a shift or a transposition changes it."""
    rounded = np.round(np.nan_to_num(values.astype("float64"), nan=-9999.0), 6)
    return hashlib.sha256(np.ascontiguousarray(rounded).tobytes()).hexdigest()[:16]


def _plain(value):
    """JSON's idea of a value, out of numpy's."""
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value]
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.datetime64):
        # .item() gives a datetime at second resolution and a bare integer count of
        # nanoseconds at nanosecond resolution, so an instant has to be written out here
        # rather than left to numpy to describe.
        return str(value.astype("datetime64[s]"))
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float):
        # NaN is never equal to itself, and json writes and reads it happily, so a digest
        # holding one would fail against a copy of itself.
        return "nan" if np.isnan(value) else round(value, 6)
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def compare(name, found):
    """The digest against the one on record, or written down when there is none."""
    path = GOLDEN / f"{name}.json"
    if os.environ.get("OPERA_FETCH_GOLDEN") == "update" or not path.exists():
        path.write_text(json.dumps(found, indent=2, sort_keys=True) + "\n")
        if os.environ.get("OPERA_FETCH_GOLDEN") != "update":
            pytest.fail(f"no golden file for {name}; wrote one. Read it, then commit it.")
        return

    expected = json.loads(path.read_text())
    if found != expected:
        pytest.fail(f"{name} no longer matches {path.name}:\n"
                    + "\n".join(_differences(expected, found)))


def _differences(expected, found, path=""):
    """Where two digests disagree, one line each, rather than two walls of JSON."""
    if isinstance(expected, dict) and isinstance(found, dict):
        lines = []
        for key in sorted(set(expected) | set(found)):
            where = f"{path}.{key}" if path else key
            if key not in expected:
                lines.append(f"  + {where} = {found[key]!r}")
            elif key not in found:
                lines.append(f"  - {where} = {expected[key]!r}")
            else:
                lines += _differences(expected[key], found[key], where)
        return lines
    if expected != found:
        return [f"  ! {path}: was {expected!r}, now {found!r}"]
    return []


def written_and_read(stack, path):
    """A stack through write and back, which is how anyone uses one twice."""
    from opera_fetch.write import read, write

    return read(write(stack, path))


ACQUISITIONS = [("20241004T011054", "20241004T043235"),
                ("20241016T011054", "20241016T043235")]


@pytest.fixture
def two_zones(tmp_path):
    """One AOI straddling the zone boundary, which is what forces a reprojection."""
    paths = []
    for acquired, processed in ACQUISITIONS:
        paths += write_granule(tmp_path, "T049-103327-IW3", acquired, processed, epsg=32612)
        paths += write_granule(tmp_path, "T056-118980-IW2", acquired, processed, epsg=32613,
                               track=56, direction="DESCENDING")
    paths.append(write_static(tmp_path, "T049-103327-IW3", epsg=32612))
    return [str(p) for p in paths]


@pytest.fixture
def one_track(tmp_path):
    """Two bursts of one track abutting on one lattice, twice over, with their statics.

    The ordinary case, and the one the no-resampling promise is about.
    """
    paths = []
    for acquired, processed in ACQUISITIONS:
        paths += write_granule(tmp_path, "T049-103327-IW3", acquired, processed)
        paths += write_granule(tmp_path, "T049-103328-IW3", acquired, processed, column=1)
    paths.append(write_static(tmp_path, "T049-103327-IW3"))
    paths.append(write_static(tmp_path, "T049-103328-IW3", column=1))
    return [str(p) for p in paths]


def test_one_track_assembles_the_same_way_it_always_has(one_track):
    stacks = assemble(one_track)
    assert list(stacks) == [32612]
    compare("one_track", digest(stacks[32612], exact=True))


def test_taking_the_first_burst_gives_the_same_mosaic_it_always_has(one_track):
    """how="first" is the default for complex data, where two phases share no datum, so
    the branch is not exotic. It fills where bursts do not overlap, and a fill is what
    floats a class code."""
    stack = assemble(one_track, how="first")

    assert stack[32612]["mask"].dtype == np.uint8
    compare("first", digest(stack[32612], exact=True))


def test_a_reprocessed_granule_replaces_the_one_it_supersedes(tmp_path):
    """The later processing wins, the earlier one contributes nothing, and the stack names
    only what it used."""
    paths = write_granule(tmp_path, "T049-103327-IW3", "20241004T011054", "20241004T043235")
    paths += write_granule(tmp_path, "T049-103327-IW3", "20241004T011054", "20241121T000000")
    paths += write_granule(tmp_path, "T049-103327-IW3", "20241016T011054", "20241016T043235")

    stack = assemble([str(p) for p in paths])[32612]
    assert stack.sizes["time"] == 2, "one acquisition, not one per processing"
    assert "20241121" in stack.attrs["granules"], "the reprocessing is what was used"
    assert "20241004T043235" not in stack.attrs["granules"], "the superseded one is not"
    compare("reprocessed", digest(stack, exact=True))


def test_two_zones_land_on_one_grid_the_same_way_they_always_have(two_zones):
    """The path a package downstream takes unconditionally, and where the provenance and
    the mask codes both went wrong before."""
    joined = assemble(two_zones, reproject_to="auto")
    assert not isinstance(joined, dict), "one CRS means one Dataset"
    compare("two_zones", digest(joined, exact=False))


def test_a_stack_off_utm_describes_its_own_grid(two_zones):
    """Onto a CRS whose units are not metres, where an attribute left saying 30 makes
    every footprint in the package 340 times too big. Nobody asks for degrees, but it is
    the one target that catches a spacing that did not follow its grid."""
    joined = assemble(two_zones, reproject_to="EPSG:4326")

    assert joined.attrs["spacing"][0] < 0.001, "degrees, not metres"
    assert bounds_of(joined) == pytest.approx(joined.rio.bounds(), abs=1e-9)
    compare("degrees", digest(joined, exact=False))


def test_an_aoi_cuts_the_same_cells_it_always_has(one_track):
    """clip moves nothing, so this stays exact even with mask=True: the values that come
    out are the values that went in, and the corners are the no-observation code."""
    from tests.granules import ANCHOR, COLUMNS, ROWS, SPACING

    west, north = ANCHOR[32612]
    # A box over the middle of the two bursts, cutting both, given in their own zone.
    box = (west + 6 * SPACING, north - (ROWS - 2) * SPACING,
           west + (2 * COLUMNS - 6) * SPACING, north - 2 * SPACING)

    stack = assemble(one_track, aoi=box, aoi_crs="EPSG:32612", mask=True)[32612]
    assert stack.sizes["x"] < 2 * COLUMNS, "the cut is narrower than the mosaic"
    compare("clipped", digest(stack, exact=True))


@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
@pytest.mark.parametrize("reprojected", [False, True], ids=["as-delivered", "reprojected"])
def test_a_stack_survives_being_written_and_read(one_track, two_zones, tmp_path, suffix,
                                                 reprojected):
    """Nothing about a stack may change by going to disk and back. Stated against the
    same digest the golden files hold, so a format that starts quietly re-encoding a
    value, a dtype or a class code fails here rather than in somebody's next season.
    """
    # Both, because only the reprojected path declares a nodata on the mask, and a
    # declared nodata is what a format re-encodes on the way back in.
    stack = (assemble(two_zones, reproject_to="auto") if reprojected
             else assemble(one_track)[32612])
    back = written_and_read(stack, tmp_path / f"stack{suffix}")
    # Twice, because a reader leaves its own encoding on every variable and the second
    # write is where a backend rejects a key it has never heard of. The package could not
    # re-write its own shipped file until it stopped carrying that encoding forward.
    written_and_read(back, tmp_path / f"again{suffix}")

    before, after = digest(stack, exact=True), digest(back, exact=True)
    changed = _differences(before, after)

    # _FillValue is the one attribute that is meant to be gone. rioxarray leaves it where
    # something declared a nodata before filling, and CF decoding would read it back as a
    # gap: a mask that returns as float32 with NaN where the class code was. Write strips
    # it on purpose, so it is missing afterwards and everything else is not.
    changed = [line for line in changed if not line.startswith("  - variables.")
               or "._FillValue" not in line]

    if suffix == ".nc":
        # netCDF cannot hold a one-element list attribute: it reads back as a scalar,
        # whatever it was written as, so a single-track stack comes back tracks=49 rather
        # than [49]. Nothing else may change, and this one is stated rather than hidden
        # so that it fails here if it ever spreads to another attribute.
        collapsed = [line for line in changed if line.startswith("  ! attrs.tracks:")]
        assert changed == collapsed, changed
        return
    assert changed == []


def test_netcdf_cannot_hold_a_one_element_list(tmp_path):
    """The exception the round trip above allows, on its own, so that a reader of these
    tests can see it is the format and not the package. Anything that consumes tracks off
    a stack read from netCDF has to cope with a bare int, which is what as_list is for."""
    import xarray as xr

    from opera_fetch.search import as_list

    one = xr.Dataset({"a": ("x", [1.0])}, coords={"x": [0]})
    one.attrs["tracks"] = [49]
    one.to_netcdf(tmp_path / "one.nc", engine="h5netcdf")
    back = xr.open_dataset(tmp_path / "one.nc", engine="h5netcdf").attrs["tracks"]

    assert back == 49 and not isinstance(back, list)
    assert as_list(back) == [49]

    two = xr.Dataset({"a": ("x", [1.0])}, coords={"x": [0]})
    two.attrs["tracks"] = [49, 56]
    two.to_netcdf(tmp_path / "two.nc", engine="h5netcdf")
    assert list(xr.open_dataset(tmp_path / "two.nc",
                                engine="h5netcdf").attrs["tracks"]) == [49, 56]

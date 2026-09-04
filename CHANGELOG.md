# Changelog

## 0.2.2

Found by reading the same granules with two packages that share no code with this one.
sarvalanche reprojects every granule onto a reference grid and stacks the files; spicy_snow
does the same and takes an acquisition's instant and its track out of the filename where
this package reads the tags. Put on one grid over a real four-track cache, all three agree
bit for bit on the backscatter, on every mask code, on which cells are clear ground, and
on the mean over a three-burst overlap. Two disagreements came out of it, one a bug here.

- A burst counts its looks towards a mosaicked pixel only where it observed something. A
  burst's static layers span its whole footprint while the acquisition carries nodata over
  layover, shadow and the edge of the swath, and `number_of_looks` was summed over every
  burst that reached a cell. Over a real three-burst T056 pass that came out a median of
  two times too high on 53% of the mosaic, up to three times, which is a noise floor
  computed from those looks reading too low by the square root of that. The backscatter
  was always right: the weighted mean renormalizes over what is finite.
- The README said every value that comes out is a value that went in, without saying that
  a cell reached by one burst still goes through the weighted average, where `(w * x) / w`
  loses a last bit in float32 about a tenth of the time: a relative 2e-7, measured. No
  value moves and nothing is interpolated; it is the arithmetic of averaging one number.
- `align_passes` said times closer than the tolerance become one pass. Each acquisition
  joins the pass before it when it falls within the tolerance of the one *before it*, so
  five acquisitions nine minutes apart are one pass thirty-six minutes long. That is what
  a pass is, since a track is acquired as a continuous sweep whose ends can be further
  apart than any tolerance worth setting, so the docstring is what changed. It now also
  says the cost: a tolerance set too wide chains rather than rounding the edges.

Nothing here changes an interface. A stack's `number_of_looks` will read lower than it did
wherever bursts overlap, which is the correction; anything using backscatter alone is
unaffected.

## 0.2.1

Fixes a crash in 0.2.0 on the path a downstream package takes unconditionally:
`assemble(..., reproject_to=...)` over an AOI spanning two UTM zones, where one of those
zones holds more than one track.

```
ValueError: Dimension time already exists.
```

Two tracks see the ground from different angles, so a zone holding both gives its
incidence angle a time axis while the zone beside it keeps one band. Concatenating them,
that layer was classified per zone rather than once, went down both branches, and the
branch for a once-per-burst layer expanded a time dimension the layer already had. A
layer is now timed if any zone has it timed, and the zones that hold it once are
broadcast along their own acquisitions, which is what it means.

Only reachable with `reproject_to` set, two zones, and two tracks in one of them. A single
zone, or two zones of one track each, was never affected.

The synthetic granules the tests build gave every burst the same incidence angle, so the
layer stayed one band however many tracks a zone held and this shape never turned up.
They vary it per track now, and `tests/golden/mixed_zones.json` is that arrangement.

## 0.2.0

### What a caller has to know

- A multi-zone stack no longer carries `track` or `direction` in its attributes, because
  it had more than one of each and was naming one. Read `tracks` for the set, or the
  per-acquisition `track` and `direction` coordinates on the time axis, which were always
  right. A single-zone stack is unchanged.
- `as_geometry` returns a MultiPolygon where the input has several parts, rather than
  their convex hull. Pass it through `aoi.one_polygon` if you need the single outline
  ASF's search takes; clipping wants the parts.
- `report` raises on three states it used to pass over: a grid that does not meet the AOI,
  a mask code OPERA does not define, and a stack holding nothing but the no-observation
  code. A stack that reported clean before and raises now was already wrong.
- Variables written to netCDF and Zarr no longer carry `_FillValue`. Written down, CF
  decoding read it back as a gap and returned the layover mask as float with NaN where the
  class code was. `flag_meanings` still names the codes, the nodata among them.
- `search._listed` is `search.as_list`, since two modules use it and it was never private.

New, all additive: `assemble(track=..., direction=...)` to pick a pass out of a cache,
`search.file_sizes`, `aoi.one_polygon`, `grid.mask_codes` and `grid.measured_spacing`.

### Known limitation

netCDF cannot hold a one-element list attribute: it reads back as a scalar whatever it was
written as, so a single-track stack saved as `.nc` returns `tracks=49` rather than `[49]`.
Zarr keeps the list. Anything reading `tracks` off a file has to cope with both, which is
what `search.as_list` is for. `tests/test_golden.py` pins this so it cannot spread to
another attribute unnoticed.

### Fixed

Multi-zone stacks kept only one zone's provenance. A stack assembled from T049 in
EPSG:32612 and T056 in EPSG:32613 came back labelled `track: 56`, `direction: DESCENDING`
and listing only the T056 granules, on data that was half ascending and from two tracks.
Anything reading those attributes to work out an overpass time got a confident wrong
answer.

- `assemble(reproject_to=...)` now pools the attributes of every zone instead of copying
  the reference zone's. `tracks`, `bursts`, `burst_id`, `granules` and `footprint` cover
  all of the data; `track` and `direction` are dropped where the zones disagree, the same
  way they are already dropped when passes of one zone are concatenated. Per-acquisition
  track and direction are unchanged, on the time axis, which is where they were always
  correct.
- `assemble` takes `track` and `direction`, matching `fetch_stacks`. A cache holds
  whatever has ever been downloaded into it, and this is the only way to pick one pass out
  of it short of filtering paths by filename. A filter that matches nothing raises and
  names the passes that are there.
- `spacing_of` warns when a `spacing` attribute disagrees with the coordinates it can also
  read. The attribute still wins, because a one-cell grid has nothing else, but a
  re-gridded object that kept a stale attribute puts every footprint out by half the
  difference.
- `xr.concat` in the reprojection path pins `data_vars="all"`, which is what it needs and
  what xarray is about to stop defaulting to.
- `rtc.read_burst` checks the burst ID over the static layers too, not only over the
  acquisitions. A static joins by grid, and the neighbouring burst of the same track is on
  that grid, so the wrong burst's incidence angle was attached under the right name and
  40% NaN. `cslc.read_burst` already had the check.
- `clip(mask=True)` declares the mask's no-observation code before rioxarray blanks the
  corners with it. Undeclared it filled with 0, which is the code for clear, relabelling
  layover, shadow and no-observation pixels as good ground.
- `write` carries only the grid mapping out of a variable's encoding. Everything else a
  reader leaves there is either rejected by the backend, which made a stack from `read`
  unwritable, or silently re-encodes the values. `_FillValue` is stripped from the
  attributes for the same reason: written down, CF decoding reads 255 back as a gap and
  the mask returns as float with NaN where the class code was.
- The mask-code convention is one function, `grid.mask_codes`, rather than a loop repeated
  wherever something fills.
- The `spacing` attribute follows the grid it describes. Reprojecting left it at 30 on a
  degree grid, and `bounds_of` reads it rather than the coordinates, so a stack on
  EPSG:4326 reported a footprint 30 degrees square. `clip`, `grid_like` and the validation
  coverage check all go through `bounds_of`.
- A once-per-burst layer stays `(y, x)` through the reprojection path where the zones
  agree on it, instead of being broadcast along time. A real incidence angle was 42 times
  the bytes and a shape nothing positional expects.
- Concatenating passes takes the variables the passes have rather than the first pass's. A
  cache that grew one track at a time holds static layers for one track and not the other,
  and that raised a bare `KeyError`. What is missing from a pass is named in a warning.
- A pass is empty only when every acquisition of it is. Read from the first acquisition
  alone, a pass whose first date happened to be blank was dropped with all 41 acquisitions
  behind it, and then the AOI was blamed for covering nothing.
- An AOI whose coordinates are not lon/lat is refused rather than sent to ASF, which clamps
  a latitude past 90, degenerates the polygon and returns nothing. A transposed pair and a
  forgotten `aoi_crs` each get told what they look like. A box with south above north is
  refused too, where `shapely.box` quietly swapped them.
- The convex hull of a multipart AOI is what ASF is asked and nothing more. `as_geometry`
  returns the polygons themselves now and `search` takes the hull, so what comes back is
  clipped to the area the caller gave rather than to a hull that can be several times
  larger.
- A stack names the granules that contributed to it. Both readers passed the granule list
  from before the reprocessing dedupe, so a superseded granule was named alongside the one
  that replaced it.
- `report` fires on three things it was computing and ignoring: a grid that does not meet
  the AOI at all, a mask code OPERA does not define, and a stack of nothing but the
  no-observation code, which came back as fully covered because 255 is finite.
- `download` compares a cached file against the length ASF declares, given
  `search.file_sizes`. A transfer killed partway through left a file with the right name
  and the right shape, and it was a cache hit for good.
- `mosaic(how="first")` and a pass short of a layer leave the mask an integer. Both fill,
  and this is the default for complex data, so every CSLC mosaic carried 127.0 and NaN at
  the same time.
- Pooling zone attributes copes with values a round trip has turned into arrays and
  scalars, where it raised on the comparison.

### Tests

`tests/test_golden.py` runs `assemble` over synthetic OPERA granules written to disk and
compares everything that comes out against a digest in `tests/golden/`. It needs no network
and no cache, so it runs in CI, and it is what catches a change to the shape of the output
that no single unit test is looking at. Regenerate with `OPERA_FETCH_GOLDEN=update`.

The synthetic fixtures also fill with a value per cell rather than a constant, and carry a
`(y, x)` layer. A constant fill survives being shifted, transposed, or replaced by another
constant, and the tests that promised nothing had moved were passing on that.

## 0.1.0

First release.

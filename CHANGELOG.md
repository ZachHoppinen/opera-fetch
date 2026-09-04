# Changelog

## 0.2.0

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

## 0.1.0

First release.

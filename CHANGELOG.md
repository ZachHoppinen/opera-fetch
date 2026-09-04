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

## 0.1.0

First release.

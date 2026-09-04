"""Downloaded files to one analysis-ready stack per UTM zone.

Bursts are read on their own lattice, given a shared time axis, and mosaicked last, so one
burst can fill what another is missing: layover and shadow fall in different places for
different look geometries.

A pass, one track and direction, is the unit bursts are mosaicked over. It is not the unit
results come back in: within a zone OPERA's grid is constant, so every pass lands on it and
one Dataset per zone holds them all, with track and direction on the time axis.
"""

import logging
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import xarray as xr
from pyproj import CRS

from opera_fetch import constants as const
from opera_fetch import cslc, filenames, rtc
from opera_fetch.aoi import as_geometry
from opera_fetch.errors import NoAcquisitions
from opera_fetch.grid import clip, mask_codes, measured_spacing, reproject
from opera_fetch.mosaic import TOLERANCE, align_passes, mosaic
from opera_fetch.search import _listed

log = logging.getLogger(__name__)


class Pass(NamedTuple):
    """One track, one direction, one zone: as much as a single mosaic may span.

    The unit bursts are averaged over, not the unit results come back in. Two tracks of one
    zone share a grid and end up in one Dataset; what they must not share is a mosaic.
    """

    track: int
    direction: str
    epsg: int

    def __str__(self):
        return f"T{self.track:03d} {self.direction.lower()} EPSG:{self.epsg}"


def group_paths(paths):
    """Downloaded files grouped by the burst they belong to, as {(family, burst): paths}.

    The family is RTC or CSLC, so a cache holding both products for one burst comes back as
    two groups rather than one unreadable pile.
    """
    # Data files only: a cache also holds browse images and checksums.
    groups = defaultdict(list)
    for path in sorted(Path(p) for p in paths):
        if path.suffix not in (".tif", ".h5"):
            continue
        product = filenames.parse_product(path)
        family = const.CSLC if product in (const.CSLC, const.CSLC_STATIC) else const.RTC
        groups[(family, filenames.parse_burst_id(path))].append(path)
    return dict(groups)


def read_bursts(paths, chunks=None, extra=()):
    """Every burst among the given files, each on its own native grid."""
    bursts = []
    for (family, burst_id), group in sorted(group_paths(paths).items()):
        reader = cslc if family == const.CSLC else rtc
        options = {"extra": extra} if family == const.CSLC else {}
        if chunks is not None:
            options["chunks"] = chunks
        try:
            bursts.append(reader.read_burst(group, **options))
        except NoAcquisitions as err:
            # A burst with static layers cached but no acquisitions in this date range is
            # ordinary, and should not take the other bursts down with it. Anything else a
            # reader raises means the data is wrong, and is left to travel.
            log.warning("skipping %s: %s", burst_id, err)
    if not bursts:
        raise NoAcquisitions("no readable OPERA bursts among the given paths")
    return bursts


def assemble(paths, aoi=None, aoi_crs=None, how=None, tolerance=TOLERANCE,
             chunks=None, extra=(), mask=False, reproject_to=None, resampling=None,
             track=None, direction=None):
    """Read, mosaic and stack every burst among the given files, one Dataset per UTM zone.

    Parameters
    ----------
    paths
        Downloaded RTC, RTC-STATIC, CSLC or CSLC-STATIC files, in any mixture. They are
        grouped by burst, and anything that is not a data file is ignored.
    aoi, aoi_crs
        The area to cut down to, and the projection its coordinates are in. Worth giving:
        without it the grid spans every burst, which for a whole track is mostly empty
        space. For an exact UTM box, pass it as the aoi with aoi_crs set to that zone.
    how
        ``"mean"`` or ``"first"`` where bursts overlap, defaulting to ``"mean"`` for real
        data and ``"first"`` for complex, whose phases share no datum.
    tolerance
        How close two acquisitions must be to count as one overpass.
    chunks
        Dask chunk size in pixels, passed to the reader.
    extra
        Names of the large per-acquisition CSLC phase screens to carry along as well.
    mask
        Also blank what falls outside the AOI polygon, rather than outside its bounding box.
    reproject_to
        A CRS to put every zone on, returning a single Dataset instead of one per zone.
        ``"auto"`` picks the zone the AOI lies in, so nobody has to know their EPSG and
        the same AOI lands on the same grid whatever window is asked for. This is the only
        resampling the package does, which is why it has to be asked for.
    resampling
        How the real layers are interpolated: any name from ``rasterio.enums.Resampling``,
        nearest by default. A mask is categorical and always moves by nearest. A complex
        layer takes neither: it is oversampled first and then read at nearest, which is
        the only way to move one without giving up coherence.

        Nearest for real layers is a trade. Neighbouring UTM lattices do not align, so it
        displaces a value by a median 12 m on a 30 m grid, where bilinear places it right
        and costs 42% of the variance. It is the default because it invents nothing.
        ``cubic`` and ``lanczos`` produce negative gamma0, which is a power ratio.
    track, direction
        Keep only bursts on these relative orbits, or on ``"ASCENDING"`` or
        ``"DESCENDING"``. track takes one track or several. A cache holds whatever has ever
        been downloaded into it, and this is the only way to pick a pass back out of it
        short of filtering paths by filename.

    Returns
    -------
    dict of {int: xarray.Dataset}
        One Dataset per UTM zone, keyed by EPSG, each on the grid OPERA delivered its
        bursts on. Usually one entry; an AOI straddling a zone boundary gives two, because
        those bursts genuinely cannot share a grid.

        Every acquisition is one step on the time axis, whichever track it came from, with
        ``track`` and ``direction`` as coordinates alongside it. Selecting a track is
        ``stack.sel(time=stack.track == 49)``.

        With reproject_to, a single Dataset on that CRS instead.
    """
    # A pass is what may be averaged together: one track, one direction, one zone.
    # Ascending and descending land on the same day and must not be mixed in a mosaic.
    found = defaultdict(list)
    for burst in read_bursts(paths, chunks=chunks, extra=extra):
        found[Pass(burst.attrs["track"], burst.attrs["direction"],
                   CRS.from_user_input(burst.rio.crs).to_epsg())].append(burst)

    passes = {key: group for key, group in found.items() if _asked_for(key, track, direction)}
    if not passes:
        # Only reachable through the filter: read_bursts has already refused an empty set.
        raise NoAcquisitions(
            f"no pass matches {_filter_text(track, direction)}; these files hold "
            f"{sorted(str(key) for key in found)}")
    if len(passes) < len(found):
        log.info("%d of %d passes match %s", len(passes), len(found),
                 _filter_text(track, direction))

    if aoi is not None:
        aoi = as_geometry(aoi, aoi_crs)
    mosaicked = defaultdict(list)
    for key, group in sorted(passes.items()):
        wanted = _overlapping(group, aoi) if aoi is not None else group
        if not wanted:
            log.info("%s: no burst reaches the AOI", key)
            continue

        # Sizing the grid from the AOI keeps a whole track from being mostly empty space.
        # Every pass of a zone gets the same bounds, and they are already on one lattice,
        # so the grids come out identical and the passes concatenate without resampling.
        area = None if aoi is None else reproject(aoi, key.epsg).bounds
        stack = mosaic(align_passes(wanted, tolerance), bounds=area, how=how)
        if aoi is not None and mask:
            stack = clip(stack, aoi, mask=True)

        if not _has_data(stack):
            # ASF returns granules that graze the edge of an area, and their footprints do
            # touch, but the data stops short.
            log.warning("%s: its bursts touch the AOI but hold no data over it; left out", key)
            continue

        # Which pass an acquisition came from belongs on the time axis, not in a key:
        # every acquisition is one row, and selecting a track is a comparison.
        times = stack.sizes["time"]
        stack = stack.assign_coords(track=("time", [key.track] * times),
                                    direction=("time", [key.direction] * times))
        mosaicked[key.epsg].append(stack)
        log.info("%s: %d times, %d by %d cells, %d bursts", key, times,
                 stack.sizes["y"], stack.sizes["x"], len(wanted))

    if not mosaicked:
        raise ValueError("nothing overlapped the AOI")

    # One Dataset per zone. Within a zone OPERA's grid is constant, so every pass lands on
    # it and they differ only in which acquisitions they contribute.
    stacks = {epsg: _one_zone(group, epsg) for epsg, group in sorted(mosaicked.items())}
    if len(stacks) > 1:
        log.info("this AOI spans %d UTM zones, %s, which cannot share a grid. Pass "
                 "reproject_to to put them on one.", len(stacks), sorted(stacks))
    if reproject_to == "auto":
        reproject_to = f"EPSG:{_best_zone(stacks, aoi)}"
    if reproject_to is not None:
        return _onto_one_crs(stacks, reproject_to, resampling)
    return stacks


def _asked_for(key, track, direction):
    """Whether a pass is one the caller wants, with None meaning no restriction."""
    if track is not None and key.track not in {int(t) for t in _listed(track)}:
        return False
    return direction is None or key.direction.upper() == str(direction).upper()


def _filter_text(track, direction):
    """The track and direction filter as something to put in a message."""
    asked = [f"track {track}" if track is not None else None,
             str(direction).lower() if direction is not None else None]
    return " ".join(part for part in asked if part) or "no filter"


def _stacked_in_time(parts, join):
    """Several stacks of one grid as one time series.

    A once-per-burst layer stays (y, x) where the parts agree on it. Concatenating
    everything together instead broadcasts it along time, which for a real incidence angle
    is 42 times the bytes and a shape nothing positional expects.

    The variables are the union over the parts rather than the first part's: a cache that
    grew one track at a time holds static layers for one track and not the other, and
    VV+VH alongside HH+HV is a configuration constants.LAYERS names.
    """
    timed, static = [], []
    for part in parts:
        for name, array in part.data_vars.items():
            into = timed if "time" in array.dims else static
            if name not in into:
                into.append(name)

    absent = sorted({name for name in timed + static
                     for part in parts if name not in part.data_vars})
    if absent:
        log.warning("%s not in every pass; where absent the stack holds no observation",
                    ", ".join(absent))

    stack = xr.concat([part[[name for name in timed if name in part.data_vars]]
                       for part in parts],
                      dim="time", join=join, data_vars="all",
                      combine_attrs="drop_conflicts")

    for name in static:
        holding = [part for part in parts if name in part.data_vars]
        layers = [part[name] for part in holding]
        # Compared per pass rather than per acquisition: a handful of passes, one band each.
        same = all(np.array_equal(layers[0].values, other.values, equal_nan=True)
                   for other in layers[1:])
        if same and len(holding) == len(parts):
            # One track, or tracks whose geometry agrees: one layer for the whole zone
            # rather than a copy per acquisition.
            stack[name] = layers[0]
        else:
            # Two tracks see the ground from different angles, so this is not one layer.
            per_time = xr.concat(
                [layer.expand_dims(time=part.time)
                 for layer, part in zip(layers, holding, strict=True)], dim="time")
            stack[name] = per_time.reindex(time=stack.time)

    return stack


def _one_zone(passes, epsg):
    """Every pass of one zone as a single Dataset, concatenated in time.

    They are already on the same grid, so join="exact" is a check rather than a
    constraint: anything else means a pass was built on a different lattice.
    """
    stack = _stacked_in_time(passes, join="exact").sortby("time")

    stack.attrs.update(
        epsg=epsg,
        tracks=sorted({int(t) for t in np.unique(stack.track.values)}),
        bursts=sum(p.attrs.get("bursts", 1) for p in passes),
        burst_id=", ".join(sorted({b for p in passes for b in p.attrs["burst_id"].split(", ")})),
        granules="\n".join(sorted({g for p in passes
                                   for g in p.attrs.get("granules", "").split("\n") if g})),
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        opera_fetch_version=const.__version__,
    )
    log.info("EPSG:%d: %d acquisitions over tracks %s on one %d by %d grid", epsg,
             stack.sizes["time"], stack.attrs["tracks"], stack.sizes["y"], stack.sizes["x"])
    return stack


def _best_zone(stacks, aoi=None):
    """The EPSG to reproject onto when the caller would rather not name one.

    The zone the AOI actually lies in, which is the one that fits it with the least
    distortion. Found as the delivered zone whose central meridian is nearest the middle of
    the AOI: zones are six degree bands about those meridians, so the nearest one is the
    zone containing the AOI whenever OPERA delivered that zone, and the next best fit when
    it did not.

    Some of the data still has to move and this does not try to make it the smaller half.
    Which zone holds more acquisitions is an accident of the tracks that happened to be
    scheduled, so choosing on it would put the same AOI on a different grid from one season
    to the next.
    """
    if aoi is None:
        log.info("no AOI to place, so reprojecting onto EPSG:%d, the lowest zone delivered",
                 min(stacks))
        return min(stacks)

    centre = aoi.centroid
    hemisphere = 326 if centre.y >= 0 else 327

    def fit(epsg):
        # Zone n is centred on 6n - 183: zone 13 on -105, zone 33 on 15.
        meridian = (epsg % 100) * 6 - 183
        return epsg // 100 == hemisphere, -abs(centre.x - meridian)

    # Sorted first, so two zones that fit equally well go to the lower EPSG rather than to
    # whichever was read first.
    best = max(sorted(stacks), key=fit)
    if len(stacks) > 1:
        log.info("reprojecting onto EPSG:%d, the zone the AOI lies in; EPSG:%s has to move",
                 best, [epsg for epsg in sorted(stacks) if epsg != best])
    return best


def _onto_one_crs(stacks, crs, resampling=None):
    """Every zone onto the grid the caller asked for, as a single Dataset.

    A zone already in that CRS is the reference and is not touched, so the result stays on
    the grid OPERA delivered and only the other zones move. Ask for a CRS no zone is in and
    everything moves, onto a grid derived from the busiest zone, which is nobody's lattice.

    How each layer moves depends on what its numbers mean. A mask is categorical, so
    nearest. A complex layer is sinc interpolated, in two steps: oversampled by zero padding
    its spectrum, which is exact, then read at the nearest fine sample. Everything else
    takes the caller's kernel.
    """
    from rasterio.enums import Resampling

    how = Resampling[resampling or const.DEFAULT_RESAMPLING["real"]]
    target = CRS.from_user_input(crs)

    already = [s for s in stacks.values() if CRS.from_user_input(s.rio.crs) == target]
    if already:
        reference = max(already, key=lambda s: s.sizes["time"])
    else:
        log.warning("no zone is in %s, so every one of them is resampled onto a grid "
                    "derived from the busiest rather than one OPERA delivered",
                    target.to_string())
        reference = max(stacks.values(), key=lambda s: s.sizes["time"])
        reference = reference.rio.reproject(target, resampling=how)

    moved = []
    for epsg, stack in sorted(stacks.items()):
        if CRS.from_user_input(stack.rio.crs) == CRS.from_user_input(reference.rio.crs):
            moved.append(stack)
            continue
        log.warning("resampling EPSG:%d onto %s, which is the one place this package "
                    "moves a value", epsg, target.to_string())
        # Three kinds of layer, three ways of moving them. A mask is categorical, so it
        # goes by nearest. A complex layer is oversampled first, which is the only way to
        # move it without giving up coherence. Everything else takes the caller's kernel.
        masks = [name for name in stack.data_vars if name.endswith("mask")]
        complex_layers = [name for name in stack.data_vars
                          if np.issubdtype(stack[name].dtype, np.complexfloating)]
        plain = stack.drop_vars(masks + complex_layers)
        matched = (plain.rio.reproject_match(reference, resampling=how) if plain.data_vars
                   else stack[[]].rio.reproject_match(reference, resampling=how))
        log.info("moved %d layer(s) with %s, %d mask(s) with nearest, %d complex "
                 "oversampled first", len(plain.data_vars), how.name, len(masks),
                 len(complex_layers))

        # Told which code means no observation GDAL leaves it alone; left to work it out
        # it rewrote 255 as 254, which OPERA does not define.
        nodata = const.MASK_NODATA[stack.attrs.get("product", const.RTC)]
        for name in masks:
            matched[name] = stack[name].rio.write_nodata(nodata).rio.reproject_match(
                reference, resampling=Resampling.nearest)
        for name in complex_layers:
            matched[name] = _oversampled_reproject(stack[name], reference)
        moved.append(matched)

    joined = _stacked_in_time(moved, join="outer").sortby("time")

    # Reprojecting floats a mask wherever a cell has no source.
    joined = mask_codes(joined)

    joined.attrs = _across_zones([stacks[epsg] for epsg in sorted(stacks)])
    joined.attrs.update(epsg=target.to_epsg(), reprojected_from=sorted(stacks))

    # The grid just moved, and every footprint in the package is worked out from this
    # number rather than from the coordinates. Left at 30 on a degree grid, bounds_of
    # answers 30 degrees square.
    spacing = measured_spacing(joined)
    if spacing is not None:
        joined.attrs["spacing"] = spacing
    return joined


def _across_zones(stacks):
    """The attributes of several zones on one grid, which are not the reference zone's.

    Taking one zone's dict wholesale labelled the whole stack with that zone's track and
    direction, and dropped the granules every other zone contributed. Anything the zones
    disagree on goes, the same way concatenating passes within a zone drops it, and what
    the zones each hold a share of is pooled.
    """
    import shapely
    import shapely.wkt

    shared = {key: value for key, value in stacks[0].attrs.items()
              if all(other.attrs.get(key) == value for other in stacks[1:])}

    outlines = [shapely.wkt.loads(s.attrs["footprint"]) for s in stacks
                if s.attrs.get("footprint")]
    return shared | {
        "tracks": sorted({t for s in stacks
                          for t in (s.attrs.get("tracks") or [s.attrs.get("track")]) if t}),
        "bursts": sum(s.attrs.get("bursts", 1) for s in stacks),
        "burst_id": ", ".join(sorted({b for s in stacks
                                      for b in s.attrs.get("burst_id", "").split(", ") if b})),
        "granules": "\n".join(sorted({g for s in stacks
                                      for g in s.attrs.get("granules", "").split("\n") if g})),
        "footprint": shapely.union_all(outlines).wkt if outlines else "",
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _oversampled_reproject(field, reference):
    """One complex layer onto the reference grid, oversampled on the way.

    Done a slice at a time, so the four times the memory the transform costs is four times
    one scene rather than four times the series.
    """
    from rasterio.enums import Resampling

    from opera_fetch import resample

    timed = "time" in field.dims
    slices = [field.isel(time=i) for i in range(field.sizes["time"])] if timed else [field]

    moved = []
    for scene in slices:
        factor = resample.affordable_factor(scene)
        if factor == 1:
            log.warning("%s is too large to oversample, so it is resampled directly and "
                        "gives up some coherence to the kernel", field.name)
            # lanczos rather than nearest here: without a fine grid to read from, the
            # widest kernel available is the least bad of the options.
            moved.append(scene.rio.reproject_match(reference, resampling=Resampling.lanczos))
            continue

        # Nearest on the fine grid, not the caller's kernel: the oversampling has already
        # done the interpolation exactly, and filtering again is what costs coherence.
        log.info("oversampling %s %dx before reprojecting, about %.0f MB for this scene",
                 field.name, factor, resample.peak_bytes(scene, factor) / 1e6)
        fine = resample.oversample(scene, factor).rio.reproject_match(
            reference, resampling=Resampling.nearest)

        # Where the data is travels separately: the transform fills the gaps with zero,
        # and zero is an ordinary value for a complex scene to hold.
        was_there = resample.validity(scene).rio.reproject_match(
            reference, resampling=Resampling.nearest)
        moved.append(fine.where(was_there > 0))

    if not timed:
        return moved[0]
    return xr.concat(moved, dim="time").assign_coords(time=field.time)


def _overlapping(bursts, geometry):
    """The bursts that actually reach the AOI, so the others cost nothing.

    Tested against the granule's own footprint. A burst is a rotated parallelogram inside a
    bounding box about a third larger, and the corner of that box is empty, so a box test
    keeps bursts that have nothing over the area.
    """
    import shapely.wkt

    keep = [burst for burst in bursts
            if not burst.attrs.get("footprint")
            or shapely.wkt.loads(burst.attrs["footprint"]).intersects(geometry)]
    if len(bursts) > len(keep):
        log.info("%d burst(s) do not reach the AOI and were left out", len(bursts) - len(keep))
    return keep


def _has_data(stack):
    """Whether a pass has a single finite pixel over the area, in any acquisition.

    Over the whole series rather than the first acquisition: mosaic outer-joins the times
    it is given, so a pass whose first date happens to be empty can have every later date
    behind it. Read from time=0 alone that dropped 41 acquisitions and then blamed the AOI.
    """
    from opera_fetch.validate import primary_variable

    return bool(np.isfinite(stack[primary_variable(stack)]).any())

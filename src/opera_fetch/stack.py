"""Downloaded files to one analysis-ready stack per pass.

Bursts are read on their own lattice, given a shared time axis, and mosaicked last, so one
burst can fill what another is missing: layover and shadow fall in different places for
different look geometries.

One Dataset comes back per track, pass direction and UTM zone, which are the boundaries a
single grid cannot cross.
"""

import logging
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS

from opera_fetch import constants as const
from opera_fetch import cslc, filenames, rtc
from opera_fetch.aoi import as_geometry
from opera_fetch.errors import NoAcquisitions
from opera_fetch.grid import clip, reproject
from opera_fetch.mosaic import mosaic

log = logging.getLogger(__name__)

TOLERANCE = "10min"


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

    @property
    def name(self):
        """The pass as a group name a file format will accept."""
        return f"T{self.track:03d}_{self.direction.upper()}_EPSG{self.epsg}"


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


def align_passes(bursts, tolerance=TOLERANCE):
    """Give the bursts of one overpass the same timestamp, so they can be mosaicked.

    Neighbouring bursts are acquired a couple of seconds apart, which is enough to make
    every burst's time axis unique and a mosaic of them an empty diagonal ribbon. Times
    closer than the tolerance become one pass, stamped with the earliest of them.
    """
    every_time = set()
    for burst in bursts:
        every_time.update(burst.indexes["time"])
    stamps = sorted(every_time)
    gap = pd.Timedelta(tolerance)

    # Walk the timestamps in order. Each one either follows close enough on the last to be
    # the same overpass, or opens a new one. Either way it is stamped with when that
    # overpass began.
    passes = {}
    pass_started = previous = stamps[0]
    for stamp in stamps:
        if stamp - previous > gap:
            pass_started = stamp
        passes[stamp] = pass_started
        previous = stamp

    log.debug("%d acquisitions across %d bursts fall into %d passes",
              len(stamps), len(bursts), len(set(passes.values())))

    aligned = []
    for burst in bursts:
        stamped = [passes[stamp] for stamp in burst.indexes["time"]]
        aligned.append(burst.assign_coords(time=stamped))
    return aligned


def assemble(paths, aoi=None, aoi_crs=None, bounds=None, how=None, tolerance=TOLERANCE,
             chunks=None, extra=(), mask=False, reproject_to=None, resampling="nearest"):
    """Read, mosaic and stack every burst among the given files, one Dataset per pass.

    Parameters
    ----------
    paths
        Downloaded RTC, RTC-STATIC, CSLC or CSLC-STATIC files, in any mixture. They are
        grouped by burst, and anything that is not a data file is ignored.
    aoi, aoi_crs
        The area to cut down to, and the projection its coordinates are in. Worth giving:
        without it the grid spans every burst, which for a whole track is mostly empty
        space.
    bounds
        The same thing as (west, south, east, north) already in the bursts' projection.
        An alternative to aoi, not a companion: giving both is an error, and bounds cannot
        be used when the bursts span more than one UTM zone.
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
        This is the only resampling the package does, which is why it has to be asked for.
    resampling
        How that reprojection interpolates: any name from ``rasterio.enums.Resampling``.
        Nearest by default, and the only one allowed for complex data, where averaging
        neighbours averages their phases.

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
    # Checked before anything is read, so a bad call costs nothing.
    if bounds is not None and aoi is not None:
        raise ValueError("give an aoi or bounds, not both: they both say what area to "
                         "deliver, and the aoi would win")

    # A pass is what may be averaged together: one track, one direction, one zone.
    # Ascending and descending land on the same day and must not be mixed in a mosaic.
    passes = defaultdict(list)
    for burst in read_bursts(paths, chunks=chunks, extra=extra):
        passes[Pass(burst.attrs["track"], burst.attrs["direction"],
                    CRS.from_user_input(burst.rio.crs).to_epsg())].append(burst)

    if aoi is not None:
        aoi = as_geometry(aoi, aoi_crs)
    zones = {key.epsg for key in passes}
    if bounds is not None and len(zones) > 1:
        raise ValueError(
            f"bounds are in one projection, but these bursts span {sorted(zones)}. "
            "Give an aoi instead, which is reprojected per zone, or assemble one zone "
            "at a time.")

    mosaicked = defaultdict(list)
    for key, group in sorted(passes.items()):
        wanted = _overlapping(group, aoi) if aoi is not None else group
        if not wanted:
            log.info("%s: no burst reaches the AOI", key)
            continue

        # Sizing the grid from the AOI keeps a whole track from being mostly empty space.
        # Every pass of a zone gets the same bounds, and they are already on one lattice,
        # so the grids come out identical and the passes concatenate without resampling.
        area = bounds if aoi is None else reproject(aoi, key.epsg).bounds
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
    if reproject_to is not None:
        return _onto_one_crs(stacks, reproject_to, resampling)
    return stacks


def _one_zone(passes, epsg):
    """Every pass of one zone as a single Dataset, concatenated in time.

    They are already on the same grid, so join="exact" is a check rather than a
    constraint: anything else means a pass was built on a different lattice.
    """
    timed = [name for name in passes[0].data_vars if "time" in passes[0][name].dims]
    static = [name for name in passes[0].data_vars if "time" not in passes[0][name].dims]
    stack = xr.concat([one[timed] for one in passes], dim="time", join="exact",
                      combine_attrs="drop_conflicts")

    for name in static:
        layers = [one[name] for one in passes]
        if _all_equal(layers):
            # One track, or tracks whose geometry agrees: one layer for the whole zone
            # rather than a copy per acquisition.
            stack[name] = layers[0]
        else:
            # Two tracks see the ground from different angles, so this is not one layer.
            stack[name] = xr.concat(
                [layer.expand_dims(time=one.time) for layer, one in zip(layers, passes,
                                                                        strict=True)],
                dim="time")

    stack = stack.sortby("time")

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


def _all_equal(layers):
    """Whether every one of these layers holds the same values.

    Compared per pass, not per acquisition: there are a handful of passes and the layers
    are one band each, so this is cheap next to what it saves.
    """
    first = layers[0].values
    return all(np.array_equal(first, other.values, equal_nan=True) for other in layers[1:])


def _onto_one_crs(stacks, crs, resampling="nearest"):
    """Every zone resampled onto one grid, as a single Dataset.

    The only resampling in the package, and the caller has to ask for it by name. The
    reference is the zone with the most acquisitions, so the largest part of the result is
    still on a grid OPERA delivered.

    Nearest by default, which moves values without inventing any. Anything else averages
    neighbours, and for complex data that destroys the measurement: interpolating the real
    and imaginary parts of two pixels a fringe apart gives a number that is not a phase
    either of them had. So a complex stack takes nearest and nothing else.
    """
    from rasterio.enums import Resampling

    complex_data = any(np.issubdtype(stack[name].dtype, np.complexfloating)
                       for stack in stacks.values() for name in stack.data_vars)
    if complex_data and resampling != "nearest":
        raise ValueError(
            f"complex data can only be reprojected with nearest, not {resampling!r}: "
            "averaging neighbouring pixels averages their phases, which is not a phase "
            "anything measured. Take the amplitude first if you want a smooth result.")

    how = Resampling[resampling]
    target = CRS.from_user_input(crs)
    reference = max(stacks.values(), key=lambda s: s.sizes["time"])
    if CRS.from_user_input(reference.rio.crs) != target:
        reference = reference.rio.reproject(target, resampling=how)

    moved = []
    for epsg, stack in sorted(stacks.items()):
        if CRS.from_user_input(stack.rio.crs) == CRS.from_user_input(reference.rio.crs):
            moved.append(stack)
            continue
        log.warning("resampling EPSG:%d onto %s, which is the one place this package "
                    "moves a value", epsg, target.to_string())
        # The mask is categorical, so it moves by nearest whatever the data does.
        masks = [name for name in stack.data_vars if name.endswith("mask")]
        ready = _declare_mask_nodata(stack)
        matched = ready.drop_vars(masks).rio.reproject_match(reference, resampling=how)
        for name in masks:
            matched[name] = ready[[name]].rio.reproject_match(
                reference, resampling=Resampling.nearest)[name]
        moved.append(matched)

    joined = xr.concat(moved, dim="time", join="outer").sortby("time")

    # Reprojecting floats a mask wherever a cell has no source, and a class code is not a
    # float. Those cells are no observation, which the mask already has a code for.
    product = joined.attrs.get("product", const.RTC)
    for name in joined.data_vars:
        if name.endswith("mask") and joined[name].dtype.kind == "f":
            joined[name] = (joined[name].fillna(const.MASK_NODATA[product])
                            .astype(const.MASK_DTYPE[product]))

    joined.attrs = dict(reference.attrs)
    joined.attrs.update(epsg=target.to_epsg(), reprojected_from=sorted(stacks))
    return joined


def _declare_mask_nodata(stack):
    """Tell rasterio which mask code means no observation, before it guesses.

    Left unsaid, GDAL sees the fill value in the data, decides it must not be mistaken for
    nodata, and rewrites it: 255 came out as 254, a code OPERA does not define.
    """
    product = stack.attrs.get("product", const.RTC)
    stack = stack.copy()
    for name in stack.data_vars:
        if name.endswith("mask"):
            stack[name] = stack[name].rio.write_nodata(const.MASK_NODATA[product])
    return stack


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
    """Whether a pass has a single finite pixel over the area.

    Read from the first acquisition alone: what a burst covers is fixed by its geometry, so
    an acquisition empty here means every acquisition is.
    """
    from opera_fetch.validate import primary_variable

    scene = stack[primary_variable(stack)]
    if "time" in scene.dims:
        scene = scene.isel(time=0)
    return bool(np.isfinite(scene).any())

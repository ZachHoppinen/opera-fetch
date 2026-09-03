"""Downloaded files to one analysis-ready stack per pass.

Bursts are read on their own lattice, given a shared time axis, and mosaicked last, so one
burst can fill what another is missing: layover and shadow fall in different places for
different look geometries.

One Dataset comes back per track, pass direction and UTM zone, which are the boundaries a
single grid cannot cross.
"""

import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from pyproj import CRS

from opera_fetch import constants as const
from opera_fetch import cslc, filenames, rtc
from opera_fetch.aoi import as_geometry
from opera_fetch.grid import clip, reproject
from opera_fetch.mosaic import mosaic

log = logging.getLogger(__name__)

TOLERANCE = "10min"


class Pass(NamedTuple):
    """One track, one pass direction and one UTM zone: as far as one grid can reach."""

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
        except ValueError as err:
            # A burst with static layers cached but no acquisitions in this date range is
            # ordinary, and should not take the other bursts down with it.
            log.warning("skipping %s: %s", burst_id, err)
    if not bursts:
        raise ValueError("no readable OPERA bursts among the given paths")
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
             chunks=None, extra=(), mask=False):
    """Read, mosaic and stack every burst among the given files, one Dataset per pass.

    aoi cuts the result down to an area, and is worth giving: without it the grid spans
    every burst, which for a whole track is mostly empty space. mask additionally blanks
    what falls outside the AOI polygon rather than outside its bounding box.

    Nothing is resampled, so each Dataset comes back on OPERA's own lattice in the
    projection its bursts were delivered in.
    """
    # Split the bursts along the boundaries one grid cannot cross.
    grouped = defaultdict(list)
    for burst in read_bursts(paths, chunks=chunks, extra=extra):
        grouped[Pass(burst.attrs["track"], burst.attrs["direction"],
                     CRS.from_user_input(burst.rio.crs).to_epsg())].append(burst)

    if aoi is not None:
        aoi = as_geometry(aoi, aoi_crs)
    if bounds is not None and len({key.epsg for key in grouped}) > 1:
        raise ValueError(
            f"bounds are in one projection, but these bursts span "
            f"{sorted(k.epsg for k in grouped)}. "
            "Give an aoi instead, which is reprojected per pass, or assemble one zone at a time.")

    stacks = {}
    for key, group in sorted(grouped.items()):
        wanted = _overlapping(group, aoi) if aoi is not None else group
        if not wanted:
            log.info("%s: no burst reaches the AOI", key)
            continue

        # Sizing the grid from the AOI keeps a whole track from being mostly empty space.
        area = bounds if aoi is None else reproject(aoi, key.epsg).bounds
        stack = mosaic(align_passes(wanted, tolerance), bounds=area, how=how).sortby("time")
        if aoi is not None and mask:
            stack = clip(stack, aoi, mask=True)
        stack.attrs.update(track=key.track, direction=key.direction, epsg=key.epsg,
                           created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           opera_fetch_version=const.__version__)

        if not _has_data(stack):
            # ASF returns granules that graze the edge of an area, and their footprints do
            # touch, but the data stops short.
            log.warning("%s: its bursts touch the AOI but hold no data over it; left out", key)
            continue
        stacks[key] = stack
        log.info("%s: %d times, %d by %d cells, %d bursts", key, stack.sizes["time"],
                 stack.sizes["y"], stack.sizes["x"], len(wanted))

    if not stacks:
        raise ValueError("nothing overlapped the AOI")
    return stacks


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

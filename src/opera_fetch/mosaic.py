"""Collapse the bursts of one overpass onto a shared time axis, then onto one grid.

Weighting the overlap by looks is what OPERA's own mosaic does (opera-adt/RTC,
``mosaic_geobursts.py``): it accumulates value*nlooks and divides by sum(nlooks).

Where bursts overlap they are independent looks at the same ground, so averaging real
backscatter is a free extra look and hides the seam.

Complex data is averaged at its peril: two bursts have different squint and reference
functions, so their phases share no datum and averaging cancels signal. A complex mosaic
takes the first burst covering a cell instead, and the seam stays a seam.
"""

import logging
from functools import reduce

import numpy as np
import pandas as pd
import xarray as xr

from opera_fetch import constants as const
from opera_fetch.grid import grid_like, mask_codes, place

log = logging.getLogger(__name__)


TOLERANCE = "10min"


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


def mosaic(bursts, bounds=None, how=None):
    """Put every burst on one grid and combine where they overlap.

    bursts share a track, a pass direction, a projection and a time axis, which
    ``stack.assemble`` is what arranges. bounds says what area to deliver, in the bursts'
    own projection.

    how is "mean" or "first", and defaults to whichever suits the data.
    """
    bursts = list(bursts)
    if not bursts:
        raise ValueError("no bursts to mosaic")
    _one_pass(bursts)

    # What the data is decides how overlaps combine, unless the caller says otherwise.
    complex_data = any(np.issubdtype(burst[name].dtype, np.complexfloating)
                       for burst in bursts for name in burst.data_vars)
    how = how or ("first" if complex_data else "mean")
    if how == "mean" and complex_data:
        log.warning("averaging complex bursts: their phases are not on a common datum")

    grid = grid_like(bursts, bounds)
    placed = [place(burst, grid) for burst in bursts]
    # A layer only some bursts carry still mosaics, but covers only their part of it.
    layers = [set(burst.data_vars) for burst in placed]
    in_any, in_every = set.union(*layers), set.intersection(*layers)
    for missing in in_any - in_every:
        log.warning("%s is not in every burst, so it covers only part of the mosaic", missing)

    # Same for a date. An acquisition the archive is missing for one burst still makes a
    # time step, covering only where the other bursts reached, which is a hole in the
    # scene rather than a gap in the series.
    times = [set(burst.indexes["time"]) for burst in placed if "time" in burst.dims]
    if times:
        every_acquisition = set.union(*times)
        in_every_burst = set.intersection(*times)
        partial = len(every_acquisition - in_every_burst)
        if partial:
            log.warning("%d of %d acquisitions are missing from at least one burst, so "
                        "they cover only part of the mosaic",
                        partial, len(every_acquisition))

    combined = _mean(placed) if how == "mean" else _first(placed)
    combined.attrs = _shared_attrs(bursts)

    log.debug("mosaicked %d bursts onto %d by %d cells",
              len(placed), grid.sizes["y"], grid.sizes["x"])
    return combined.rio.write_crs(grid.rio.crs)


def _mean(placed):
    """Average the overlaps, weighted by looks, and take the worst mask code.

    Weighted because that is how OPERA mosaics its own bursts: a burst contributes in
    proportion to the looks behind each of its pixels. Without the number_of_looks layer
    this falls back to a flat mean.
    """
    stacked = xr.concat(placed, dim="burst", join="outer")
    masks = [name for name in stacked.data_vars if name.endswith("mask")]
    data = stacked.drop_vars(masks)

    if const.LOOKS in data:
        looks = data[const.LOOKS]
        # A cell whose looks are unknown must not weigh zero, or the burst is dropped from
        # it entirely. Fall back to what the other bursts weigh at that cell, and where
        # none of them knows either, weigh them all the same.
        typical = looks.mean(dim="burst", skipna=True)
        weights = looks.fillna(typical).fillna(1.0)
        combined = data.drop_vars(const.LOOKS).weighted(weights).mean(dim="burst")
        # The looks behind a mosaicked pixel are the looks of the bursts that delivered a
        # value there, not of every burst whose static layer reaches it. A burst covers
        # more ground than it observes, and counted by footprint the layer came out a
        # median of two times too high over a real three-burst pass, which is a noise
        # floor half what it should be.
        # min_count so a cell no burst knows stays unknown: summing nothing gives 0, and
        # zero looks is a real value that means something else entirely.
        combined[const.LOOKS] = looks.where(_delivered(data)).sum(
            dim="burst", skipna=True, min_count=1)
    else:
        log.debug("no %s layer, so bursts are averaged flat", const.LOOKS)
        combined = data.mean(dim="burst", skipna=True)

    # Placing a burst introduces NaN, which floats the mask; it goes back to its own
    # integer type once every burst has had its say.
    product = placed[0].attrs.get("product", const.RTC)
    for name in masks:
        # Averaging class codes invents classes: shadow (1) and both (3) average to
        # layover (2), which nobody saw. And since every burst contributed to the value
        # above, the worst thing any of them saw is what the mosaicked pixel carries.
        # Unobserved becomes -1, below every real code, so the max never meets a slice
        # with nothing in it and a cell nobody saw comes back as -1 to be relabelled.
        nodata = const.MASK_NODATA[product]
        codes = stacked[name].where(stacked[name] != nodata).fillna(-1)
        worst = codes.max(dim="burst")
        combined[name] = worst.where(worst >= 0, nodata).astype(const.MASK_DTYPE[product])
    return _keep_attrs(combined, placed[0])


def _delivered(data):
    """Where each burst actually observed something, as (burst, y, x).

    Read from the per-acquisition layers, since a static layer spans the whole burst
    whether or not the radar returned anything from a given cell. Collapsed over time
    because the looks layer does not vary in time either: what a burst can see is fixed
    by its geometry, so the dates of one burst agree about this to within nothing.
    """
    timed = [name for name in data.data_vars
             if name != const.LOOKS and "time" in data[name].dims]
    if not timed:
        return True
    seen = np.isfinite(data[timed[0]])
    for name in timed[1:]:
        seen = seen | np.isfinite(data[name])
    return seen.any("time") if "time" in seen.dims else seen


def _first(placed):
    """Take the first burst covering each cell, leaving the others where it does not.

    First is by burst ID, which is the order ``stack.assemble`` reads them in. Arbitrary,
    but fixed: the same bursts always give the same mosaic.
    """
    combined = reduce(lambda a, b: a.combine_first(b), placed)

    # combine_first aligns and fills, which promotes an integer coordinate to float and a
    # string one to object. Every burst carries the same ones, so put them back as they were.
    kept = {name: placed[0][name] for name in placed[0].coords if name not in combined.dims}
    # combine_first fills too, and a filled mask is a float one. This is the default for
    # complex data, so a CSLC mosaic would carry NaN and the nodata code at once.
    combined = mask_codes(combined.assign_coords(kept))
    return _keep_attrs(combined, placed[0])


def _keep_attrs(combined, reference):
    for name in combined.data_vars:
        combined[name].attrs = reference[name].attrs if name in reference else {}
    return combined


def _shared_attrs(bursts):
    """The mosaic's attributes, which are not simply the first burst's.

    A mosaic covers all of its bursts, so its footprint is their union, and that outline is
    what decides whether a burst is worth reading at all.
    """
    import shapely
    import shapely.wkt

    attrs = dict(bursts[0].attrs)
    attrs["burst_id"] = ", ".join(sorted(b.attrs.get("burst_id", "?") for b in bursts))
    attrs["bursts"] = len(bursts)

    outlines = [shapely.wkt.loads(b.attrs["footprint"]) for b in bursts
                if b.attrs.get("footprint")]
    attrs["footprint"] = shapely.union_all(outlines).wkt if outlines else ""

    granules = sorted(set().union(*(b.attrs.get("granules", "").split("\n") for b in bursts)))
    attrs["granules"] = "\n".join(g for g in granules if g)
    return attrs

def _one_pass(bursts):
    """Refuse to mosaic across a boundary a mosaic has no business crossing.

    Ascending and descending land on the same day, and averaging a 6 am pass with a 6 pm
    one destroys the diurnal difference that keeping them apart exists to measure.

    Unreachable through ``stack.assemble``, which groups bursts into passes before it gets
    here. It is for everyone else: ``mosaic`` is public, and two bursts of different tracks
    average into something plausible looking rather than failing.
    """
    for field in ("track", "direction", "product"):
        seen = {burst.attrs.get(field) for burst in bursts}
        if len(seen) > 1:
            raise ValueError(f"bursts span more than one {field}: {sorted(map(str, seen))}")

    projections = {burst.rio.crs for burst in bursts}
    if len(projections) > 1:
        raise ValueError(
            f"bursts span more than one projection: {sorted(map(str, projections))}. Nothing "
            "here resamples, so mosaic one UTM zone at a time and reproject at the end.")

    times = [burst.indexes["time"] for burst in bursts if "time" in burst.dims]
    if len(times) < 2:
        return
    union = pd.DatetimeIndex(sorted(set().union(*(set(t) for t in times))))
    # Neighbouring bursts are acquired seconds apart. Two timestamps that close are one
    # overpass that was never collapsed, and mosaicking them gives an empty diagonal ribbon.
    if len(union) > 1 and union.to_series().diff().min() < pd.Timedelta("60s"):
        log.warning("%d timestamps across %d bursts, some seconds apart: these look like one "
                    "overpass that did not go through align_passes first",
                    len(union), len(times))

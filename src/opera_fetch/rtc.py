"""Read one OPERA RTC burst's GeoTIFFs into a time series on the burst's own 30 m grid.

Values come out as delivered: linear gamma0, not dB, and unmasked. The layover/shadow mask
rides along as its own variable, so applying it stays the caller's decision:
``stack.where(stack.mask == 0)``.
"""

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray
import xarray as xr

from opera_fetch import constants as const
from opera_fetch import filenames, metadata
from opera_fetch.errors import NoAcquisitions
from opera_fetch.grid import place

log = logging.getLogger(__name__)

CHUNK = 512
ACQUIRED = "ZERO_DOPPLER_START_TIME"

# Sentinel-1 acquires VV+VH over most land and HH+HV at high latitude, so a polar burst
# carries neither of the two everyone expects.
POLARIZATIONS = ("vv", "vh", "hh", "hv")


def read_burst(paths, chunks=CHUNK):
    """One burst's acquisitions as a Dataset of vv, vh, mask and any static layers.

    paths are the downloaded RTC and RTC-STATIC files for a single burst. Files of other
    products are ignored; files of another burst are an error rather than a quiet drop.
    """
    paths = [p for p in paths if filenames.parse_product(p) in (const.RTC, const.RTC_STATIC)]
    granules = _granules(paths)
    if not granules:
        raise NoAcquisitions("no OPERA RTC acquisitions among the given paths")

    found = {layer.lower() for files in granules.values() for layer in files}
    if not found & set(POLARIZATIONS):
        raise ValueError(
            f"these granules carry {sorted(found)} and no backscatter. Sentinel-1 acquires "
            "HH+HV at high latitude, so ask for those layers as well as VV+VH")

    # One time step per acquisition, built from whichever layers that granule has.
    slices, times, platforms, orbits, tags = [], [], [], [], None
    for files in granules.values():
        layers = {layer.lower(): _open(path, chunks) for layer, path in files.items()}
        first = next(iter(layers.values()))
        tags = tags or first.attrs

        slices.append(xr.Dataset(layers))
        times.append(pd.Timestamp(first.attrs[ACQUIRED].rstrip("Z")))
        # Satellite and orbit change between acquisitions of one burst, so they belong on
        # the time axis rather than in the attributes.
        platforms.append(first.attrs.get("PLATFORM", "").replace("Sentinel-1", "S1"))
        orbits.append(int(first.attrs.get("ABSOLUTE_ORBIT_NUMBER", -1)))

    # Drop reprocessed duplicates, then put what survives in time order.
    keep = sorted(filenames.keep_latest_processing(times, list(granules)).values())
    slices = [slices[i] for i in keep]
    platforms = [platforms[i] for i in keep]
    orbits = [orbits[i] for i in keep]
    times = pd.DatetimeIndex([times[i] for i in keep])
    order = times.argsort()

    # join="exact" so an acquisition on a different grid is an error. The default outer
    # join pads every date out to the union instead, giving a silently misaligned stack.
    stack = xr.concat([slices[i] for i in order], dim="time", join="exact")
    stack = stack.assign_coords(
        time=times[order],
        platform=("time", [platforms[i] for i in order]),
        absolute_orbit=("time", [orbits[i] for i in order]))
    # Identity and units from the tags, then the once-per-burst layers.
    stack = _describe(stack, tags, granules)
    stack = _add_static(stack, paths, chunks)

    log.info("burst %s: %d acquisitions on a %s grid",
             stack.attrs["burst_id"], stack.sizes["time"], stack.rio.crs)
    return stack.chunk({"time": -1})


def _open(path, chunks):
    """One layer as a 2-D DataArray, chunked on the COG blocks.

    Unmasked on purpose: masking would upcast the uint8 layover mask to float and turn its
    255 nodata into a gap, and the float layers already carry NaN nodata.
    """
    return rioxarray.open_rasterio(
        path, chunks={"x": chunks, "y": chunks}, masked=False).squeeze("band", drop=True)


def _granules(paths):
    """RTC files grouped by acquisition, as {granule: {layer: path}}."""
    granules = defaultdict(dict)
    for path in sorted(paths):
        if filenames.parse_product(path) != const.RTC:
            continue
        layer = filenames.parse_layer(path, const.LAYERS[const.RTC])
        if layer is None:
            log.debug("%s is not an RTC layer; skipped", Path(path).name)
            continue
        # The granule is the name with "_<layer>" taken off the end.
        granule = Path(path).stem.removesuffix(f"_{layer}")
        granules[granule][layer] = path

    bursts = {filenames.parse_burst_id(granule) for granule in granules}
    if len(bursts) > 1:
        raise ValueError(f"paths span {len(bursts)} bursts: {sorted(bursts)}. Group them first")
    return dict(granules)

def identify(tags):
    """The burst's identity out of an RTC granule's tags, in the shape metadata wants."""
    return {
        "burst_id": tags.get("BURST_ID", ""),
        "track": tags.get("TRACK_NUMBER"),
        "direction": tags.get("ORBIT_PASS_DIRECTION"),
        "footprint": tags.get("BOUNDING_POLYGON"),
        "product_version": tags.get("PRODUCT_VERSION"),
    }


def _describe(stack, tags, granules):
    """Carry the burst's identity, and the units of its values, onto the stack."""
    metadata.describe(stack, const.RTC, identify(tags), granules)
    for polarization in POLARIZATIONS:
        if polarization in stack:
            stack[polarization].attrs = {
                "units": "1", "long_name": f"{polarization.upper()} gamma0, linear power"}
    if "mask" in stack:
        stack["mask"].attrs = {
            "units": "1", "long_name": "layover and shadow mask",
            "flag_meanings": f"{const.MASK_MEANINGS}, "
                             f"{const.MASK_NODATA[const.RTC]} no observation"}
    return stack


def _add_static(stack, paths, chunks):
    """Attach the RTC-STATIC layers, put on the burst's own grid.

    They were made from a reference pass, so their extent can differ from any one
    acquisition even though the lattice does not.
    """
    for path in sorted(paths):
        if filenames.parse_product(path) != const.RTC_STATIC:
            continue
        layer = filenames.parse_layer(path, const.LAYERS[const.RTC_STATIC])
        if layer is None:
            continue
        placed = place(_open(path, chunks), stack)
        if layer.endswith("incidence_angle"):
            # Radians at ingest, so everything downstream shares one unit system.
            placed = np.deg2rad(placed)
            placed.attrs = {"units": "radians", "long_name": layer.replace("_", " ")}
        # The static mask is the same field as the per-acquisition one, so it needs its
        # own name to sit beside it.
        stack["static_mask" if layer == "mask" else layer] = placed
    return stack

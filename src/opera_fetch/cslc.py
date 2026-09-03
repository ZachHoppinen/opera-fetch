"""Read one OPERA CSLC burst's HDF5 files into a time series on the burst's own grid.

Values stay complex throughout: for anything interferometric the phase is the measurement.
The two phase screens in each granule are float64 at full resolution, so as large as the
data itself, and are left out unless asked for.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from opera_fetch import constants as const
from opera_fetch import filenames
from opera_fetch.errors import NoAcquisitions
from opera_fetch.grid import place

log = logging.getLogger(__name__)

CHUNK = 1024
DIMS = {"y_coordinates": "y", "x_coordinates": "x"}


def read_burst(paths, chunks=CHUNK, extra=()):
    """One burst's acquisitions as a Dataset of complex vv or vh, and any static layers.

    paths are the downloaded CSLC and CSLC-STATIC files for a single burst. extra names
    any of the large per-acquisition phase screens to carry along as well.
    """
    paths = [p for p in paths if filenames.parse_product(p) in (const.CSLC, const.CSLC_STATIC)]
    acquisitions = [p for p in sorted(paths) if filenames.parse_product(p) == const.CSLC]
    if not acquisitions:
        raise NoAcquisitions("no OPERA CSLC acquisitions among the given paths")

    bursts = {filenames.parse_burst_id(path) for path in paths}
    if len(bursts) > 1:
        raise ValueError(f"paths span {len(bursts)} bursts: {sorted(bursts)}. Group them first")

    # A polarization arrives as its own granule, so an acquisition is time plus that.
    keys = [_acquired_and_polarization(path) for path in acquisitions]
    latest = filenames.keep_latest_processing(keys, acquisitions)

    # One entry per acquisition, holding a granule for each polarization of it.
    by_time, identities = {}, {}
    for (time, _), index in sorted(latest.items()):
        path = acquisitions[index]
        by_time.setdefault(time, []).append(_read_one(path, chunks, extra))
        identities[time] = _identification(path)

    times = pd.DatetimeIndex(sorted(by_time))
    # A burst acquired in both polarizations arrives as two granules of one instant, so
    # they merge into a single time step rather than becoming two.
    slices = [xr.merge(by_time[time], join="exact") for time in times]
    stack = xr.concat(slices, dim="time", join="exact").assign_coords(
        time=times,
        # Both change between acquisitions of one burst, so they go on the time axis. The
        # orbit number is what a baseline is worked out from.
        platform=("time", [identities[t]["mission_id"] for t in times]),
        absolute_orbit=("time", [int(identities[t]["absolute_orbit_number"]) for t in times]))

    identity = identities[times[0]]
    stack.attrs = {
        "product": const.CSLC,
        "burst_id": bursts.pop(),
        "track": int(identity["track_number"]),
        "direction": identity["orbit_pass_direction"].upper(),
        "spacing": const.SPACING[const.CSLC],
        # The outline of the data itself, which the bounding box overstates by a third.
        "footprint": identity.get("bounding_polygon", ""),
        # Which granules these values came from, for the same reason as RTC.
        "granules": "\n".join(sorted(Path(p).stem for p in acquisitions)),
        "product_version": str(identity.get("product_version", "")),
    }
    stack = _add_static(stack, paths, chunks)

    log.info("burst %s: %d acquisitions on a %s grid",
             stack.attrs["burst_id"], stack.sizes["time"], stack.rio.crs)
    return stack.chunk({"time": -1})


def _open(path, group, chunks):
    """One HDF5 group as a georeferenced Dataset on x and y, chunked so nothing loads whole.

    The coordinate arrays are real dimension scales in these files, so xarray picks them up
    on its own and only the names need changing.

    The three helper variables go: they describe the grid rather than measure anything, and
    leaving ``projection`` in place gives rioxarray a second thing that looks like a grid
    mapping, after which it refuses to report a projection at all.
    """
    dataset = xr.open_dataset(path, engine="h5netcdf", group=group,
                              chunks={dim: chunks for dim in DIMS})
    dataset = dataset.rename({old: new for old, new in DIMS.items() if old in dataset.dims})

    # Take the projection off the helper variable before dropping it.
    epsg = int(dataset["projection"])
    dataset = dataset.drop_vars([name for name in ("projection", "x_spacing", "y_spacing")
                                 if name in dataset.variables])
    # These files are CF-compliant and point every layer at the "projection" variable just
    # dropped. Clearing the pointer lets the CRS be written back under rioxarray's own name.
    for variable in dataset.variables.values():
        variable.attrs.pop("grid_mapping", None)
        variable.encoding.pop("grid_mapping", None)
    return dataset.rio.write_crs(epsg)


def _read_one(path, chunks, extra):
    """One CSLC granule as a 2-D Dataset with its layers named in lower case."""
    data = _open(path, "data", chunks)
    wanted = [name for name in const.CSLC_DATA if name in data]
    wanted += [name for name in extra if name in data]
    if not wanted:
        raise ValueError(f"{path} holds none of the {const.CSLC_DATA} layers")

    granule = data[wanted].rename({name: name.lower() for name in wanted})
    for name in wanted:
        granule[name.lower()].attrs = {
            "units": "1", "long_name": f"{name} geocoded single-look complex"}
    return granule.rio.write_crs(data.rio.crs)


def _identification(path):
    """The identification group as a plain dict of Python values."""
    identity = {}
    with xr.open_dataset(path, engine="h5netcdf", group="identification") as group:
        for name, variable in group.variables.items():
            value = variable.values.item() if variable.ndim == 0 else variable.values
            # Everything textual in these files is stored as fixed-width bytes.
            if isinstance(value, bytes):
                value = value.decode()
            identity[name] = value
    return identity


def _acquired_and_polarization(path):
    """What makes one CSLC acquisition distinct: when, and in which polarization."""
    return filenames.parse_acquisition_time(path), filenames.parse_polarization(path)


def _add_static(stack, paths, chunks):
    """Attach the CSLC-STATIC layers, put on the burst's own grid.

    These were computed once from a reference pass years before the acquisitions, so their
    extent differs while the lattice does not.
    """
    for path in sorted(paths):
        if filenames.parse_product(path) != const.CSLC_STATIC:
            continue
        data = _open(path, "data", chunks)
        for layer in const.CSLC_STATIC_DATA:
            if layer not in data:
                continue
            placed = place(data[layer], stack)
            name = layer
            if layer == "local_incidence_angle":
                # Radians at ingest, so everything downstream shares one unit system.
                placed = np.deg2rad(placed)
                placed.attrs = {"units": "radians", "long_name": "local incidence angle"}
            elif layer == "layover_shadow_mask":
                # Called "mask" as in RTC, though CSLC's is per burst rather than per
                # acquisition: it is the only one OPERA ships for this product.
                name = "mask"
                placed.attrs = {"units": "1", "long_name": "layover and shadow mask",
                                "flag_meanings": f"{const.MASK_MEANINGS}, "
                                                 f"{const.MASK_NODATA[const.CSLC]} no observation"}
            else:
                placed.attrs = {"units": "1", "long_name": layer.replace("_", " ")}
            stack[name] = placed
    return stack

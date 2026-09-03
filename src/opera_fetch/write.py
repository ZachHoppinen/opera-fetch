"""Put a stack on disk, as netCDF, HDF5 or Zarr.

Passes are not on one grid, so they cannot share x and y coordinates; each becomes a group.

netCDF-4 has no complex type, so a CSLC stack goes through h5netcdf with ``invalid_netcdf``:
valid HDF5 that xarray reads back, but a strict netCDF reader will not open it. Prefer Zarr
for complex.
"""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

log = logging.getLogger(__name__)


def write(obj, path, complevel=4):
    """Write a stack, or the several passes of one, to path.

    The format follows the suffix: .nc and .h5 go through h5netcdf, .zarr through zarr.
    """
    path = Path(path)
    if path.suffix not in (".nc", ".h5", ".zarr"):
        raise ValueError(f"cannot write {path.suffix!r}; use .nc, .h5 or .zarr")
    path.parent.mkdir(parents=True, exist_ok=True)

    # A lone stack writes to the root; passes each get their own group.
    stacks = obj if isinstance(obj, dict) else {None: obj}
    mode = "w"
    for key, stack in stacks.items():
        stack = _clean(stack)
        group = key.name if key is not None else None
        if path.suffix == ".zarr":
            stack.to_zarr(path, group=group, mode=mode, consolidated=True)
        else:
            has_complex = any(np.issubdtype(stack[name].dtype, np.complexfloating)
                              for name in stack.variables)
            stack.to_netcdf(path, group=group, mode=mode, engine="h5netcdf",
                            invalid_netcdf=has_complex, encoding=_encoding(stack, complevel))
        mode = "a"

    log.info("wrote %d pass(es) to %s", len(stacks), path)
    return path


def read(path, group=None):
    """Read back what ``write`` wrote: one Dataset, or every group as a dict."""
    path = Path(path)
    # decode_coords="all" is what turns the stored grid mapping back into a coordinate.
    # Without it the projection comes back as an ordinary variable and .rio.crs is None.
    engine = "zarr" if path.suffix == ".zarr" else "h5netcdf"
    options = {"engine": engine, "decode_coords": "all"}
    if group is not None:
        return xr.open_dataset(path, group=group, **options)

    groups = _groups_in(path)
    if not groups:
        return xr.open_dataset(path, **options)
    return {name: xr.open_dataset(path, group=name, **options) for name in groups}


def _groups_in(path):
    if path.suffix == ".zarr":
        import zarr

        # A zarr array is a directory as much as a group is, so this has to ask zarr rather
        # than look at the filesystem.
        return sorted(name for name, _ in zarr.open_group(path, mode="r").groups())

    import h5py

    with h5py.File(path, "r") as handle:
        return sorted(name for name, item in handle.items() if isinstance(item, h5py.Group))


STORABLE = (str, bytes, int, float, np.number, np.ndarray)


def _clean(stack):
    """A copy whose attributes a file format can hold."""
    stack = stack.copy()
    stack.attrs = _storable(stack.attrs)
    for name in stack.variables:
        stack[name].attrs = _storable(stack[name].attrs)
    return stack


def _storable(attrs):
    """One set of attributes, with anything a file cannot hold turned into text."""
    out = {}
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = list(value)
        elif not isinstance(value, STORABLE):
            value = str(value)
        out[key] = value
    return out


def _encoding(stack, complevel):
    """Deflate every array, which roughly halves an RTC stack and costs little to read.

    Built on top of the encoding each variable already carries: that is where rioxarray
    keeps the grid mapping, and replacing it wholesale writes a file whose projection
    cannot be read back.
    """
    if not complevel:
        return {}
    return {name: {**array.encoding, "compression": "gzip", "compression_opts": complevel}
            for name, array in stack.data_vars.items() if array.ndim >= 2}

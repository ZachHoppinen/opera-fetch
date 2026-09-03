"""Checks worth running before trusting a stack, and a picture worth looking at.

``report`` says what a stack holds and raises only on damage, never on a value it thinks is
too low. ``check_files`` catches downloads that died halfway, which leave a file that looks
fine until something reads it weeks later.
"""

import logging
from pathlib import Path

import numpy as np

from opera_fetch.grid import reproject, spacing_of

log = logging.getLogger(__name__)


def check_files(paths):
    """The files that do not open and read, out of the ones given.

    Reads the last row of each raster rather than the whole thing: a truncated download is
    short at the end, so that is where the damage is.
    """
    broken = []
    for path in paths:
        try:
            _read_the_end(path)
        except Exception as err:
            log.debug("%s does not read: %s", path, err)
            broken.append(path)
    return broken


def _read_the_end(path):
    """Open a file and read the part a half-finished download would be missing."""
    name = str(path)
    if name.endswith((".tif", ".tiff")):
        import rasterio

        with rasterio.open(path) as source:
            last_row = ((source.height - 1, source.height), (0, source.width))
            source.read(1, window=last_row)

    elif name.endswith(".h5"):
        import h5py

        with h5py.File(path, "r") as handle:
            if "data" not in handle:
                raise OSError("no data group")


def report(stack, aoi=None, strict=True):
    """What an assembled stack contains, as a dict. ``summary`` prints it.

    Raises on a stack that is structurally broken: no projection, repeated timestamps, no
    finite data at all. Everything else is reported and left alone.
    """
    times = stack.indexes.get("time")
    found = {
        "crs": str(stack.rio.crs) if stack.rio.crs else None,
        "spacing": spacing_of(stack),
        "shape": (stack.sizes.get("y"), stack.sizes.get("x")),
        "variables": list(stack.data_vars),
        "bursts": stack.attrs.get("bursts", 1),
    }

    # Coverage over every finite pixel, with no mask or threshold applied.
    if times is not None:
        finite = np.isfinite(stack[primary_variable(stack)]).mean(dim=("y", "x"))
        if hasattr(finite.data, "compute"):
            finite = finite.compute()
        finite = np.asarray(finite)
        found.update(
            times=len(times), first=times.min(), last=times.max(),
            duplicated=sorted(times[times.duplicated()].unique()),
            monotonic=bool(times.is_monotonic_increasing),
            coverage=finite,
            empty_times=[str(t) for t, f in zip(times, finite) if f == 0],
            median_coverage=float(np.median(finite)) if len(finite) else float("nan"),
        )
    if aoi is not None:
        found["aoi_covered"] = _aoi_fraction(stack, aoi)

    # Damage is what makes a stack unusable, not what makes it disappointing.
    found["damage"] = damage = []
    if found["crs"] is None:
        damage.append("the stack carries no projection")
    if found.get("duplicated"):
        damage.append(f"{len(found['duplicated'])} timestamps appear more than once")
    if times is not None and not found["monotonic"]:
        damage.append("the time axis is not in order")
    if times is not None and found["median_coverage"] == 0:
        damage.append("no acquisition has a single finite pixel")

    if found.get("empty_times"):
        log.warning("%d acquisition(s) are entirely empty over this area",
                    len(found["empty_times"]))
    if damage and strict:
        raise ValueError("; ".join(damage))
    return found


def summary(stack, aoi=None):
    """The report as a few lines of text, for printing between steps."""
    found = report(stack, aoi=aoi, strict=False)
    lines = [
        f"grid       {found['shape'][0]} by {found['shape'][1]} at {found['spacing']} m, "
        f"{found['crs']}",
        f"variables  {', '.join(found['variables'])}",
        f"bursts     {found['bursts']}",
    ]
    if "times" in found:
        lines += [
            f"times      {found['times']} from {found['first']:%Y-%m-%d} "
            f"to {found['last']:%Y-%m-%d}",
            f"coverage   {100 * found['median_coverage']:.0f}% of cells finite, "
            f"median over time",
        ]
    if found.get("aoi_covered") is not None:
        lines.append(f"aoi        {100 * found['aoi_covered']:.0f}% of the AOI is inside the grid")
    if found["damage"]:
        lines.append("damage     " + "; ".join(found["damage"]))
    return "\n".join(lines)


def quicklook(stack, path, variable=None, figsize=(6, 4), dpi=150):
    """Three panels of a stack: one scene, the coverage, the scene means.

    Complex data is shown as its amplitude, which the axis label says.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variable = variable or primary_variable(stack)
    field = stack[variable]
    label = variable
    if np.issubdtype(field.dtype, np.complexfloating):
        field, label = np.abs(field), f"|{variable}|"

    fig, axes = plt.subplots(1, 3, figsize=(figsize[0] * 3, figsize[1]), dpi=dpi)
    timed = "time" in field.dims

    scene = field.isel(time=0) if timed else field
    scene.plot(ax=axes[0], robust=True, cbar_kwargs={"label": label})
    axes[0].set_title(label)

    if timed:
        np.isfinite(field).sum("time").plot(ax=axes[1], cbar_kwargs={"label": "acquisitions"})
        axes[1].set_title("observations per cell")
        # Drawn through matplotlib because xarray refuses a series of one point, and a
        # stack of a single acquisition is an ordinary thing to look at.
        series = field.mean(dim=("y", "x"), skipna=True)
        axes[2].plot(series["time"].values, np.asarray(series.values),
                     marker=".", linestyle="none")
        axes[2].set(title=f"{label}, scene mean", xlabel="time", ylabel=label)
        fig.autofmt_xdate()

    for axis in axes[:2]:
        axis.set(xlabel="x (m)", ylabel="y (m)")
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    log.info("wrote %s", path)
    return path


def primary_variable(stack):
    """The variable a report is about: the first real data layer, not a static one."""
    for name in ("vv", "hh", "vh", "hv"):
        if name in stack.data_vars:
            return name
    for name, array in stack.data_vars.items():
        if "time" in array.dims:
            return name
    if not stack.data_vars:
        raise ValueError("the stack holds no variables")
    return next(iter(stack.data_vars))


def _aoi_fraction(stack, aoi):
    """How much of the AOI polygon the stack's grid actually covers."""
    from shapely.geometry import box

    from opera_fetch.aoi import as_geometry

    geometry = reproject(as_geometry(aoi), stack.rio.crs)
    covered = geometry.intersection(box(*stack.rio.bounds()))
    return float(covered.area / geometry.area) if geometry.area else float("nan")

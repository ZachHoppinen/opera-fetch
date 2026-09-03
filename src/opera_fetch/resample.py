"""Oversampling complex data before it is resampled.

A CSLC is bandlimited to its own grid: OPERA geocoded it there with a sinc, so it carries
no energy above that grid's Nyquist. Speckle fills the band right up to it though, and half
the energy of a real burst sits beyond half of Nyquist, which is exactly where a truncated
sinc rolls off. Resampling one directly therefore costs coherence.

Zero padding the spectrum is the exact interpolator for such a signal, so the fix is to put
the same content in the lower part of a much wider band and then simply take the nearest
fine sample. Nearest is the right second step precisely because it filters nothing: on an
eight times finer grid the sample it picks is within a sixteenth of a pixel of the position
asked for, and nothing further is smoothed away.

Interpolating on the fine grid instead is worse, which is not obvious until measured.
Coherence against the analytic answer for real CSLC, worst case over a range of sub-pixel
shifts:

    lanczos, no oversampling      0.951
    oversample 2x, then lanczos   0.978
    oversample 8x, then nearest   0.996
    oversample 16x, then nearest  0.999

The 2x row is held back by GDAL filtering a second time as it decimates, which costs more
than it saves. ``scratch/oversample_before_reprojecting.py`` reproduces the numbers.
"""

import logging

import numpy as np
import rioxarray  # noqa: F401  registers the .rio accessor
import xarray as xr

log = logging.getLogger(__name__)

# Eight is where the curve flattens: sixteen buys 0.003 more coherence for four times the
# memory.
FACTOR = 8

# Oversampling costs the square of the factor, so a budget rather than a fixed factor: a
# scene too large for eight still gets four or two, and only a very large one gets none.
BUDGET = 2_000_000_000

# What it really peaks at, measured rather than assumed: the wide array, the transform's
# output, and one temporary inside it. Estimating the wide array alone was out by three.
OVERHEAD = 3


def oversample(field, factor=FACTOR):
    """One complex 2-D layer on a grid `factor` times finer, by zero padding its spectrum.

    Exact where the signal is bandlimited to the grid it arrives on, which is what makes it
    worth doing rather than interpolating twice: the fine samples that coincide with coarse
    ones reproduce them to float precision.

    Nodata is filled with zero for the transform, since one NaN would take the whole
    spectrum with it, and the caller is expected to put the gaps back afterwards.
    """
    dy = float(field.y[1] - field.y[0])
    dx = float(field.x[1] - field.x[0])
    values = np.asarray(field.values)
    fine = _pad_spectrum(np.where(np.isfinite(values), values, 0), factor)

    coords = {
        "y": float(field.y[0]) + np.arange(fine.shape[0]) * dy / factor,
        "x": float(field.x[0]) + np.arange(fine.shape[1]) * dx / factor,
    }
    out = xr.DataArray(fine.astype(values.dtype, copy=False), dims=("y", "x"),
                       coords=coords, name=field.name)
    return out.rio.write_crs(field.rio.crs)


def _pad_spectrum(values, factor):
    """Zero pad an unshifted spectrum by moving its four quadrants to the corners.

    The same thing as fftshift, pad, ifftshift, and a quarter less memory: those three
    each leave a copy of the wide array behind, and the wide array is the expensive one.
    """
    ny, nx = values.shape
    spectrum = np.fft.fft2(values)
    wide = np.zeros((factor * ny, factor * nx), dtype=spectrum.dtype)

    # The low half of each axis holds DC and the positive frequencies, the high half the
    # negative ones, and the zeros go between them where the signal has no content.
    low_y, low_x = (ny + 1) // 2, (nx + 1) // 2
    high_y, high_x = ny - low_y, nx - low_x
    wide[:low_y, :low_x] = spectrum[:low_y, :low_x]
    if high_x:
        wide[:low_y, -high_x:] = spectrum[:low_y, low_x:]
    if high_y:
        wide[-high_y:, :low_x] = spectrum[low_y:, :low_x]
    if high_y and high_x:
        wide[-high_y:, -high_x:] = spectrum[low_y:, low_x:]

    del spectrum
    fine = np.fft.ifft2(wide)
    fine *= factor ** 2
    return fine


def peak_bytes(field, factor=FACTOR):
    """Roughly what oversampling this scene will peak at.

    Knowable before anything runs: the grid comes from the area asked for and the product's
    posting, so a 500 by 900 CSLC scene is 4 MB and costs about 800 MB to oversample eight
    times over.
    """
    return field.nbytes * factor ** 2 * OVERHEAD


def affordable_factor(field, budget=BUDGET):
    """The largest oversampling this scene can afford, or 1 if even the smallest is too much."""
    for factor in (FACTOR, 4, 2):
        if peak_bytes(field, factor) <= budget:
            return factor
    return 1


def validity(field):
    """Where a layer has data, as something that survives being reprojected.

    Carried alongside rather than inferred afterwards: the transform fills the gaps with
    zero, and zero is a perfectly ordinary value for a complex scene to hold.
    """
    mask = xr.DataArray(np.isfinite(field.values).astype("uint8"), dims=("y", "x"),
                        coords={"y": field.y, "x": field.x}, name="valid")
    return mask.rio.write_crs(field.rio.crs).rio.write_nodata(0)

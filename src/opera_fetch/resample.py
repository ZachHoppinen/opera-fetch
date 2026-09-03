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


def oversample(field, factor=FACTOR):
    """One complex 2-D layer on a grid `factor` times finer, by zero padding its spectrum.

    Exact where the signal is bandlimited to the grid it arrives on, which is what makes it
    worth doing rather than interpolating twice: the fine samples that coincide with coarse
    ones reproduce them to float precision.

    Nodata is filled with zero for the transform, since a NaN would take the whole spectrum
    with it, and the caller is expected to put the gaps back afterwards.
    """
    dy = float(field.y[1] - field.y[0])
    dx = float(field.x[1] - field.x[0])
    values = np.asarray(field.values)
    valid = np.isfinite(values)

    padded = np.fft.fftshift(np.fft.fft2(np.where(valid, values, 0)))
    ny, nx = values.shape
    pad_y, pad_x = (factor * ny - ny) // 2, (factor * nx - nx) // 2
    wide = np.pad(padded, ((pad_y, pad_y), (pad_x, pad_x)))
    fine = np.fft.ifft2(np.fft.ifftshift(wide)) * factor ** 2

    coords = {
        "y": float(field.y[0]) + np.arange(fine.shape[0]) * dy / factor,
        "x": float(field.x[0]) + np.arange(fine.shape[1]) * dx / factor,
    }
    out = xr.DataArray(fine.astype(values.dtype), dims=("y", "x"), coords=coords,
                       name=field.name)
    return out.rio.write_crs(field.rio.crs)


def affordable_factor(field, budget=BUDGET):
    """The largest oversampling this scene can afford, or 1 if even the smallest is too much."""
    for factor in (FACTOR, 4, 2):
        if field.nbytes * factor ** 2 <= budget:
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

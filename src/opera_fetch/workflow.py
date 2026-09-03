"""The six steps in one call. Each of them is public and can be run on its own."""

import logging

from opera_fetch import constants as const
from opera_fetch.aoi import as_geometry
from opera_fetch.download import download
from opera_fetch.search import data_urls, search, search_static
from opera_fetch.stack import assemble
from opera_fetch.validate import report
from opera_fetch.write import write

log = logging.getLogger(__name__)


def fetch_stacks(aoi, start=None, end=None, product=const.RTC, cache_dir="data/raw/opera",
                 aoi_crs=None, static=True, layers=None, static_layers=None, track=None,
                 direction=None, out=None, mask=False, how=None, tolerance="10min",
                 chunks=None, extra=(), max_workers=10):
    """Search, download, mosaic, stack, clip and check. One Dataset per pass.

    Parameters
    ----------
    aoi
        The area, as WKT, a path to any vector file, a (west, south, east, north) box in
        lon/lat, a list of (x, y) pairs, a shapely geometry, or a GeoDataFrame.
    start, end
        The date range, as anything pandas reads as a date. Either may be left out for
        open-ended.
    product
        ``RTC`` for terrain-corrected backscatter, or ``CSLC`` for complex.
    cache_dir
        Where granules are kept. Nothing already there is downloaded again, so re-running
        with a wider date range only fetches the difference.
    aoi_crs
        The projection ``aoi`` is in, when it is not lon/lat and does not carry its own.
    static
        Fetch the matching static layers too: the local incidence angle for RTC, and the
        incidence angle, layover/shadow mask and line-of-sight vectors for CSLC. They are
        made once per burst, so they cost the same whatever the date range.
    layers, static_layers
        Which layers to fetch, defaulting to the useful ones. Pass
        ``opera_fetch.constants.LAYERS[product]`` for everything ASF publishes.
    track, direction
        Narrow the search to one relative orbit, or to ``"ASCENDING"``/``"DESCENDING"``.
    out
        Write the result here, in the format the suffix names: ``.nc``, ``.h5`` or
        ``.zarr``. Use ``.zarr`` for CSLC, because netCDF-4 has no complex type.
    mask
        Also blank pixels outside the AOI polygon, rather than outside its bounding box.
    how
        ``"mean"`` or ``"first"`` where bursts overlap. Defaults to ``"mean"`` for real
        data, which is a free extra look, and ``"first"`` for complex, whose phases are
        not on a common datum.
    tolerance
        How close two acquisitions must be to count as one overpass. Neighbouring bursts
        are seconds apart.
    chunks, extra
        The dask chunk size, and any of the large CSLC phase screens to carry along.
    max_workers
        Parallel downloads.

    Returns
    -------
    dict of {Pass: xarray.Dataset}
        One entry per track, pass direction and UTM zone, since those are the boundaries a
        single grid cannot cross. Each Dataset is on OPERA's own lattice, in the projection
        its bursts were delivered in. Nothing is resampled anywhere.

        RTC gives ``vv``, ``vh``, ``mask`` as linear gamma0, unmasked. CSLC gives complex
        ``vv`` or ``vh``. ``platform`` and ``absolute_orbit`` are coordinates on the time
        axis; ``track``, ``direction``, ``epsg`` and ``footprint`` are attributes.

    Examples
    --------
    A month of backscatter over the East River, written to netCDF::

        import opera_fetch as of

        stacks = of.fetch_stacks((-107.0, 38.85, -106.85, 38.95), "2024-11-01", "2024-11-30",
                                 cache_dir="data/raw/east_river",
                                 out="data/processed/east_river.nc")
        for key, stack in stacks.items():
            print(key, of.summary(stack))

    One descending track only, from a shapefile, with the layover/shadow mask applied::

        stacks = of.fetch_stacks("aoi/east_river.shp", "2024-10-01", "2025-06-30",
                                 track=56, direction="DESCENDING")
        stack = stacks[of.Pass(56, "DESCENDING", 32613)]
        clear = stack.where(stack.mask == 0)

    Complex data, which belongs in Zarr::

        stacks = of.fetch_stacks(aoi, "2024-11-01", "2024-11-15", product=of.CSLC,
                                 out="data/processed/cslc.zarr")

    Raises
    ------
    ValueError
        If ASF has nothing for that area and range, or nothing overlaps the AOI.

    Notes
    -----
    Downloads need an Earthdata login in ``~/.netrc``, or the data pool answers 403.
    """
    if product not in const.TIME_VARYING:
        raise ValueError(f"this takes {const.TIME_VARYING}; static layers come with them")

    aoi = as_geometry(aoi, aoi_crs)
    log.info("step 1/6  %s from %s to %s over %s", product, start, end,
             tuple(round(value, 3) for value in aoi.bounds))

    log.info("step 2/6  searching ASF")
    found = search(aoi=aoi, start=start, end=end, product=product, track=track,
                   direction=direction)
    if found.empty:
        raise ValueError(f"ASF has no {product} granules for that area and date range")
    # data_urls logs the size, so the volume is known before anything is fetched.
    urls = data_urls(found, layers)
    if static:
        urls += data_urls(search_static(found), static_layers)

    log.info("step 3/6  downloading %d files", len(urls))
    paths = download(urls, cache_dir, max_workers=max_workers)

    # assemble does both: bursts are mosaicked and the result stacked in time.
    log.info("step 4/6 and 5/6  mosaicking bursts and stacking them in time")
    stacks = assemble(paths, aoi=aoi, how=how, tolerance=tolerance, chunks=chunks,
                      extra=extra, mask=mask)

    log.info("step 6/6  clipped to the AOI; checking")
    for stack in stacks.values():
        report(stack, aoi=aoi)

    if out is not None:
        write(stacks, out)
    return stacks

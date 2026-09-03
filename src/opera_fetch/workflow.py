"""The six steps in one call. Each of them is public and can be run on its own."""

import logging
from pathlib import Path

from opera_fetch import constants as const
from opera_fetch.aoi import as_geometry
from opera_fetch.download import download
from opera_fetch.search import data_urls, search, search_static
from opera_fetch.stack import TOLERANCE, assemble
from opera_fetch.validate import check_files, report, summary
from opera_fetch.write import write

log = logging.getLogger(__name__)


def _cache_dir_path(cache_dir, url):
    """Where download puts a URL, so a re-fetch can be matched back to it."""
    return Path(cache_dir) / Path(url).name


def fetch_stacks(aoi, start=None, end=None, product=const.RTC, cache_dir="data/raw/opera",
                 aoi_crs=None, static=True, layers=None, static_layers=None, track=None,
                 direction=None, out=None, mask=False, how=None, tolerance=TOLERANCE,
                 chunks=None, extra=(), max_workers=10, reproject_to=None,
                 resampling=None):
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

    reproject_to
        A CRS to put every zone on, giving one Dataset instead of one per zone.
        ``"auto"`` picks it for you: the zone the AOI lies in, so nobody has to know their
        EPSG and the same AOI lands on the same grid whatever window is asked for. The only
        resampling in the package, which is why it has to be asked for.
    resampling
        How the real layers are interpolated: any name from ``rasterio.enums.Resampling``,
        nearest by default. A mask always moves by nearest, being categorical, and a
        complex layer is oversampled first and then read at nearest, which is the only way
        to move one without giving up coherence.

    Returns
    -------
    dict of {int: xarray.Dataset}
        One entry per UTM zone, keyed by EPSG. Usually one; an AOI straddling a zone
        boundary gives two, because OPERA assigns the zone per burst and no single grid
        holds both. Nothing is resampled unless reproject_to says so.

        Every acquisition is one step on the time axis, whichever track it came from, with
        ``track``, ``direction``, ``platform`` and ``absolute_orbit`` as coordinates
        alongside it. Selecting a track is ``stack.sel(time=stack.track == 49)``.

        RTC gives ``vv``, ``vh`` or ``hh``, ``hv``, plus ``mask``, as linear gamma0 and
        unmasked. CSLC gives complex ``vv`` or ``hh``.

        With reproject_to, a single Dataset on that CRS rather than a dict.

    Examples
    --------
    A month of backscatter over the East River, written to netCDF::

        import opera_fetch as of

        stacks = of.fetch_stacks((-107.0, 38.85, -106.85, 38.95), "2024-11-01", "2024-11-30",
                                 cache_dir="data/raw/east_river",
                                 out="data/processed/east_river.nc")
        for epsg, stack in stacks.items():
            print(epsg, of.summary(stack))

    Everything over one AOI as a single Dataset, without having to know the zone::

        stack = of.fetch_stacks(aoi, "2024-11-01", "2024-11-30", reproject_to="auto")

    One descending track only, from a shapefile, with the layover/shadow mask applied::

        stacks = of.fetch_stacks("aoi/east_river.shp", "2024-10-01", "2025-06-30",
                                 track=56, direction="DESCENDING")
        stack = stacks[32613]
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

    # The cache may hold a file some earlier run left half written, which the
    # exists-and-not-empty check reads as done. Opening them says otherwise.
    broken = check_files(paths)
    if broken:
        log.warning("%d cached file(s) do not read; fetching them again", len(broken))
        for path in broken:
            path.unlink(missing_ok=True)
        again = [url for url in urls if _cache_dir_path(cache_dir, url) in set(broken)]
        download(again, cache_dir, max_workers=max_workers)
        still_broken = check_files(broken)
        if still_broken:
            raise OSError(f"{len(still_broken)} files will not read after a second "
                          f"attempt: {', '.join(p.name for p in still_broken[:3])}")

    # assemble does both: bursts are mosaicked and the result stacked in time.
    log.info("step 4/6 and 5/6  mosaicking bursts and stacking them in time")
    stacks = assemble(paths, aoi=aoi, how=how, tolerance=tolerance, chunks=chunks,
                      extra=extra, mask=mask, reproject_to=reproject_to,
                      resampling=resampling)

    log.info("step 6/6  clipped to the AOI; checking")
    for key, stack in (stacks.items() if isinstance(stacks, dict) else [(None, stacks)]):
        # report raises on damage; summary is what makes the step visible.
        report(stack, aoi=aoi)
        log.info("EPSG:%s\n%s", key or stack.rio.crs.to_epsg(), summary(stack, aoi=aoi))

    if out is not None:
        write(stacks, out)
    return stacks

"""Ask ASF what OPERA granules exist over an area, in a date range, for a product.

Results come back as a DataFrame rather than as asf_search objects, so a search can be
looked at, filtered and costed before anything is downloaded.
"""

import logging
import time

import pandas as pd

from opera_fetch import constants as const
from opera_fetch import filenames
from opera_fetch.aoi import as_geometry, one_polygon

log = logging.getLogger(__name__)


def as_dates(start=None, end=None):
    """A date range as two timestamps, either of which may be None for open-ended.

    Takes anything pandas reads as a date. A static product has no acquisition time at
    all, which is why both ends are allowed to be missing.
    """
    def stamp(value, name):
        if value is None:
            return None
        try:
            read = pd.Timestamp(value)
        except Exception as err:
            raise ValueError(f"could not read {name} date from {value!r}") from err
        if pd.isna(read):
            # pandas reads "" as NaT rather than raising, and asf_search takes a NaT
            # without complaint: it reaches CMR as the literal string "NaT" in the
            # temporal parameter. Given the "" itself asf_search does raise, so it is this
            # conversion that launders it, and this conversion that has to catch it.
            raise ValueError(f"could not read {name} date from {value!r}. "
                             "Pass None for an open-ended range")
        return read

    start, end = stamp(start, "start"), stamp(end, "end")
    if start is not None and end is not None and start > end:
        raise ValueError(f"start {start} is after end {end}")

    archive = pd.Timestamp(const.ARCHIVE_START)
    if end is not None and end < archive:
        raise ValueError(
            f"the OPERA archive starts around {const.ARCHIVE_START} and this range ends "
            f"{end:%Y-%m-%d}, so it would return nothing whatever the area")
    if start is not None and start < archive:
        log.info("range starts before the OPERA archive does; nothing before %s exists",
                 const.ARCHIVE_START)
    return start, end


def search(aoi=None, start=None, end=None, product=const.RTC, burst_id=None, track=None,
           direction=None, retries=5, delay=5.0):
    """OPERA granules matching an area, a date range and a product, as a DataFrame.

    aoi is anything ``opera_fetch.aoi.as_geometry`` reads. burst_id narrows to one or more
    bursts, in either the hyphenated or the underscored form. A static product has no
    acquisition time, so start and end are ignored for those.
    """
    if product not in const.PRODUCTS:
        raise ValueError(f"{product!r} is not one of {const.PRODUCTS}")
    if aoi is None and burst_id is None:
        raise ValueError("give an aoi, a burst_id, or both")
    if retries < 1:
        raise ValueError(f"retries is how many attempts to make, so at least 1, not {retries}")

    import asf_search as asf

    # Built up rather than passed whole: ASF treats a None as a filter on nothing.
    query = {"dataset": asf.DATASET.OPERA_S1, "processingLevel": product}
    if aoi is not None:
        query["intersectsWith"] = one_polygon(as_geometry(aoi)).wkt
    if burst_id is not None:
        # ASF wants T049_103327_IW3; the hyphenated form in every filename returns nothing.
        query["operaBurstID"] = [str(b).replace("-", "_").upper() for b in as_list(burst_id)]
    if track is not None:
        query["relativeOrbit"] = [int(t) for t in as_list(track)]
    if direction is not None:
        query["flightDirection"] = str(direction).upper()
    if product in const.TIME_VARYING:
        start, end = as_dates(start, end)
        if start is not None:
            query["start"] = start.to_pydatetime()
        if end is not None:
            query["end"] = end.to_pydatetime()

    # ASF times out often enough that a search worth waiting for needs repeating.
    for attempt in range(1, retries + 1):
        try:
            results = asf.search(**query)
            break
        except Exception as err:
            if attempt == retries:
                raise
            log.warning("search failed (%s); retry %d of %d in %.0fs", err, attempt, retries, delay)
            time.sleep(delay)

    frame = _frame(results)
    log.info("%s: %d granules, %d bursts, tracks %s", product, len(frame),
             frame.burst_id.nunique(), sorted(frame.track.dropna().unique().tolist()))
    return frame


COLUMNS = ["fileID", "product", "burst_id", "track", "direction", "polarization",
           "time", "processed", "urls", "sizes"]


def search_static(frame, **kwargs):
    """The static-layer granules for every burst in a search result."""
    return search(burst_id=sorted(frame.burst_id.unique()),
                  product=const.STATIC_OF[_one(frame, "product")], **kwargs)


def data_urls(frame, layers=None):
    """The download URLs in a search result, for the layers worth having.

    layers defaults to the product's own: VV, VH and the mask for RTC, the local incidence
    angle for RTC-STATIC, the single HDF5 for either CSLC product. Pass
    ``opera_fetch.constants.LAYERS[product]`` for everything ASF publishes.

    The logged size is the point of doing this before downloading: one CSLC burst is about
    275 MB an acquisition, so a season of a track runs to hundreds of gigabytes.
    """
    if frame.empty:
        return []
    wanted = _wanted(frame, layers)
    product = _one(frame, "product")

    log.info("%s: %d files, %.1f GB, for %d granules",
             product, len(wanted), sum(wanted.values()) / 1e9, len(frame))
    if not wanted:
        raise ValueError(f"no {product} URLs matched layers {tuple(layers)}")
    return sorted(wanted)


def file_sizes(frame, layers=None):
    """The size ASF declares for each file in a search result, as {filename: bytes}.

    ``download`` reads it to tell a finished file from one an interrupted transfer left
    the right shape but the wrong length.
    """
    if frame.empty:
        return {}
    return {url.rsplit("/", 1)[-1]: size for url, size in _wanted(frame, layers).items()}


def _wanted(frame, layers):
    """The data URLs in a search result and the size ASF declares for each, as {url: bytes}."""
    product = _one(frame, "product")
    layers = const.DEFAULT_LAYERS[product] if layers is None else layers

    wanted = {}
    for urls, sizes in zip(frame["urls"], frame["sizes"], strict=True):
        for url in urls:
            name = url.rsplit("/", 1)[-1]
            if _is_data(name, product, layers):
                wanted[url] = (sizes or {}).get(name, 0)
    return wanted


def _is_data(name, product, layers):
    """Whether a filename is one of the requested data layers, not browse or a checksum."""
    if name.endswith((".md5", ".png", ".xml")):
        return False
    if product in (const.CSLC, const.CSLC_STATIC):
        return name.endswith(".h5")
    if not name.endswith(".tif"):
        return False
    # Whole-suffix match: several layer names contain underscores themselves, so the last
    # underscore is not where the layer starts.
    stem = name.removesuffix(".tif")
    return any(stem.endswith(f"_{layer}") for layer in layers)


def _frame(results):
    rows = []
    for granule in results:
        properties = granule.properties
        burst_id = filenames.parse_burst_id(properties["fileID"])
        rows.append({
            "fileID": properties["fileID"],
            "product": filenames.parse_product(properties["fileID"]),
            "burst_id": burst_id,
            "track": int(burst_id[1:4]),
            "direction": properties.get("flightDirection"),
            "polarization": properties.get("polarization"),
            "time": pd.to_datetime(properties.get("startTime")),
            "processed": pd.to_datetime(properties.get("processingDate")),
            "urls": sorted(set(granule.find_urls())),
            "sizes": {name: item.get("bytes", 0)
                      for name, item in (properties.get("bytes") or {}).items()},
        })
    # Named columns even when empty, so a search of a year OPERA does not cover reads
    # like any other empty result instead of raising on a missing column.
    frame = pd.DataFrame(rows, columns=COLUMNS)
    if frame.empty:
        return frame
    return frame.sort_values(["burst_id", "time"], ignore_index=True)


def as_list(value):
    """One value or several as a list, where a string counts as one and not as its characters."""
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        return [value]
    return list(value)


def _one(frame, column):
    values = set(frame[column].dropna().unique())
    if len(values) != 1:
        raise ValueError(f"expected one {column} in the frame, found {sorted(values)}")
    return values.pop()

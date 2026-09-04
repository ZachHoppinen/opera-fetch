"""What an OPERA filename says about the file.

Parsing by named groups follows ``opera_utils.constants.CSLC_S1_FILE_REGEX`` (checked
against opera-utils 0.25.6), widened here to cover all four products, including the static
ones that carry no processing time. The pattern itself is in ``constants.GRANULE``.

The examples below are real granules from ``data/raw``, not invented ones.

    OPERA_L2_RTC-S1_T049-103327-IW3_20241004T011054Z_20241004T043235Z_S1A_30_v1.0_VV.tif
    OPERA_L2_RTC-S1-STATIC_T049-103327-IW3_20140403_S1A_30_v1.0_local_incidence_angle.tif
    OPERA_L2_CSLC-S1_T094-200132-IW2_20250111T032103Z_20250123T000352Z_S1A_VV_v1.1.h5
"""

import logging
from pathlib import Path

import pandas as pd

from opera_fetch import constants as const

log = logging.getLogger(__name__)


def parse_product(path):
    """Which of the four OPERA products a file belongs to."""
    name = Path(path).name
    for token, product in const.PRODUCT_TOKEN.items():
        if f"_{token}_" in name:
            return product
    raise ValueError(f"{name} names no OPERA product")


def parse_burst_id(path):
    """The burst ID, hyphenated as it appears in filenames."""
    found = const.BURST_ID.search(Path(path).name)
    if not found:
        raise ValueError(f"{Path(path).name} carries no OPERA burst ID")
    track, number, swath = found.groups()
    return f"T{track}-{number}-{swath.upper()}"


def parse_layer(path, layers):
    """Which of the named layers a file holds, or None when it is not one of them.

    Matched as a whole suffix: several layer names contain underscores themselves, so
    splitting on the last one would call local_incidence_angle "angle".
    """
    stem = Path(path).stem
    matched = [layer for layer in layers if stem.endswith(f"_{layer}")]
    return max(matched, key=len) if matched else None


def parse_fields(path):
    """Every field an OPERA granule name carries, as a dict.

    Counting underscores works until it does not: static products have no processing time
    where the others do, and only CSLC names its polarization.
    """
    found = const.GRANULE.match(Path(path).name)
    if not found:
        raise ValueError(f"{Path(path).name} is not an OPERA granule name")
    return found.groupdict()


def parse_acquisition_time(path):
    """When the burst was acquired. A static product gives its reference date instead."""
    return pd.to_datetime(parse_fields(path)["acquired"].rstrip("Z"), errors="coerce")


def parse_polarization(path):
    """The polarization a CSLC granule names, or None for products that do not."""
    return parse_fields(path)["polarization"]


def parse_processing_time(path):
    """When the granule was processed, which orders reprocessed versions of one acquisition."""
    processed = parse_fields(path)["processed"]
    return pd.to_datetime(processed, format="%Y%m%dT%H%M%SZ", errors="coerce")


def keep_latest_processing(keys, granules):
    """The index of the most recently processed granule for each key.

    The archive holds reprocessed versions of an acquisition, sharing its zero-doppler time
    and differing only in the processing stamp. Keeping both double-weights that date and
    repeats a timestamp that nothing downstream expects.

    keys are what makes an acquisition distinct: its time for RTC, its time and
    polarization for CSLC, whose polarizations arrive as separate granules.
    """
    latest = {}
    for index, (key, granule) in enumerate(zip(keys, granules, strict=True)):
        if key in latest:
            best_so_far = parse_processing_time(granules[latest[key]])
            if parse_processing_time(granule) <= best_so_far:
                continue
        latest[key] = index

    dropped = len(granules) - len(latest)
    if dropped:
        log.info("dropped %d superseded granule(s), keeping the latest processing", dropped)
    return latest

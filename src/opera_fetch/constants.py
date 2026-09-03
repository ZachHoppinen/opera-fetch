"""Names, layers and grid numbers for the four OPERA Sentinel-1 products."""

import re
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("opera-fetch")
except PackageNotFoundError:
    # Running from a source tree that was never installed, as the tests do.
    __version__ = "0.0.0+source"

RTC = "RTC"
RTC_STATIC = "RTC-STATIC"
CSLC = "CSLC"
CSLC_STATIC = "CSLC-STATIC"
PRODUCTS = (RTC, RTC_STATIC, CSLC, CSLC_STATIC)
TIME_VARYING = (RTC, CSLC)

# Static layers are made once per burst, not once per acquisition, so they are searched
# by burst ID and fetched once however many seasons the stack covers.
STATIC_OF = {RTC: RTC_STATIC, CSLC: CSLC_STATIC}

# (x, y) posting in metres. CSLC is not square: 5 m across track, 10 m along it.
SPACING = {RTC: (30.0, 30.0), RTC_STATIC: (30.0, 30.0),
           CSLC: (5.0, 10.0), CSLC_STATIC: (5.0, 10.0)}

# Every layer ASF publishes, as the trailing field of the filename. Both CSLC products are
# one .h5 holding all of theirs, so there is nothing to choose.
LAYERS = {
    RTC: ("VV", "VH", "HH", "HV", "mask"),
    RTC_STATIC: ("local_incidence_angle", "incidence_angle", "mask", "number_of_looks",
                 "rtc_anf_gamma0_to_beta0", "rtc_anf_gamma0_to_sigma0"),
    CSLC: (),
    CSLC_STATIC: (),
}
# What comes down unless more is asked for.
DEFAULT_LAYERS = {
    RTC: LAYERS[RTC],
    # Not every static layer: the angle for geometry, the looks for weighting a mosaic.
    RTC_STATIC: ("local_incidence_angle", "number_of_looks"),
    CSLC: LAYERS[CSLC],
    CSLC_STATIC: LAYERS[CSLC_STATIC],
}

# The weight OPERA itself mosaics with, and the name it arrives under.
LOOKS = "number_of_looks"

# The layers inside the two HDF5 products. CSLC is co-pol only, and it is HH wherever
# Sentinel-1 acquires HH+HV, which is most of the high Arctic and Antarctic. opera-utils'
# ``get_dataset_name`` reads /data/VV or /data/HH for the same reason.
CSLC_DATA = ("VV", "HH")
CSLC_STATIC_DATA = ("local_incidence_angle", "layover_shadow_mask", "los_east", "los_north")

# How a reprojection interpolates, when one is asked for. Complex data gets a windowed
# sinc, which is what OPERA's own geocoding uses for it: measured against an exact
# half-pixel shift of a fringe, lanczos and bilinear both reproduce the phase exactly while
# nearest does not interpolate at all, and lanczos keeps the most amplitude. Real data
# keeps nearest, which moves values without inventing any. A mask is categorical and is
# always nearest, whatever the data does.
DEFAULT_RESAMPLING = {"complex": "lanczos", "real": "nearest", "mask": "nearest"}

# rms takes magnitudes, so it throws the phase away rather than moving it.
NO_PHASE = ("rms", "mode")

# Codes and fill values as OPERA defines them: opera-adt/RTC ``h5_prep.py`` for RTC,
# opera-adt/COMPASS ``s1_geocode_metadata.py`` for CSLC.
# Layover/shadow codes: 0 clear, 1 shadow, 2 layover, 3 both. RTC ships one mask per
# acquisition; CSLC ships none, and its only mask is the once-per-burst static one. Both
# are called "mask" here so `stack.where(stack.mask == 0)` reads the same either way.
MASK_CLEAR = 0
MASK_NODATA = {RTC: 255, CSLC: 127}
MASK_DTYPE = {RTC: "uint8", CSLC: "int8"}
MASK_MEANINGS = "0 clear, 1 shadow, 2 layover, 3 both"

# The OPERA archive starts well after Sentinel-1 did: asking for 2014 returns nothing at
# all, with no hint that the date rather than the area was the problem. Measured against
# ASF, not read from a document.
ARCHIVE_START = "2016-01-01"

# Sentinel-1B failed in December 2021 and 1C only reached operations in 2025, so a range
# inside that gap has roughly half the acquisitions a range either side of it does.
#TODO do we need this for some reason?
ONE_SATELLITE = ("2021-12-23", "2025-03-01")

# T049-103327-IW3 in a filename, t049_103327_iw3 inside an HDF5.
BURST_ID = re.compile(r"T(\d{3})[-_](\d{6})[-_](IW[1-3])", re.IGNORECASE)

# The fields of a granule name, so nothing has to count underscores to find one. Static
# products carry a reference date where the others carry an acquisition and a processing
# time, and only CSLC names its polarization; the rest name it in the layer suffix.
GRANULE = re.compile(
    r"OPERA_L2_(?P<product>[A-Z0-9-]+)_"
    r"(?P<burst_id>T\d{3}-\d{6}-IW[1-3])_"
    r"(?P<acquired>\d{8}(T\d{6}Z)?)_"
    r"((?P<processed>\d{8}T\d{6}Z)_)?"
    r"(?P<sensor>S1[A-E])"
    r"(_(?P<polarization>VV|VH|HH|HV))?",
    re.IGNORECASE)

# Matched as _<token>_, so RTC-S1 cannot be found inside RTC-S1-STATIC and the order of
# this mapping does not matter.
PRODUCT_TOKEN = {"RTC-S1-STATIC": RTC_STATIC, "CSLC-S1-STATIC": CSLC_STATIC,
                 "RTC-S1": RTC, "CSLC-S1": CSLC}

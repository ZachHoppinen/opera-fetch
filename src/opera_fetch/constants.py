"""Names, layers and grid numbers for the four OPERA Sentinel-1 products.

Every number here is either read off a delivered granule or taken from a published source,
and the comment says which. Checked against opera-utils 0.25.6 and against RTC v1.0 and
CSLC v1.1 granules in ``data/raw``, so a claim can be re-checked rather than believed.
"""

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
# Sentinel-1 acquires HH+HV, which is most of the high Arctic and Antarctic.
# opera_utils._cslc.get_dataset_name does the same, falling back to /data/HH when /data/VV
# is absent, and opera_utils.constants.OPERA_DATASET_NAME is "/data/VV".
CSLC_DATA = ("VV", "HH")
CSLC_STATIC_DATA = ("local_incidence_angle", "layover_shadow_mask", "los_east", "los_north")

# How a reprojection moves each kind of layer. Real data keeps nearest, which moves values
# without inventing any, and the caller can choose otherwise. A mask is categorical, so it
# is nearest whatever the data does. Complex is not a kernel choice at all: it is
# oversampled first and then taken at nearest, which is the only way to move it without
# giving up coherence. See opera_fetch.resample.
DEFAULT_RESAMPLING = {"real": "nearest", "mask": "nearest"}

# The codes are stated by the products themselves rather than inferred.
#
#   RTC, the mask GeoTIFF's LAYER_DESCRIPTION tag:
#     "Mask Layer. Values: 0: not masked; 1: shadow; 2: layover; 3: layover and shadow;
#      255: invalid/fill value"
#     and the band declares nodata=255, dtype uint8.
#   CSLC, the description attribute on /data/layover_shadow_mask:
#     "Layover shadow mask. 0=no layover, no shadow; 1=shadow; 2=layover;
#      3=shadow and layover."
#     dtype int8. The description names no fill, but 127 is present in real data. Do not
#     read the HDF5 fillvalue instead: it is 0, which here means clear ground.
#
# RTC ships one mask per acquisition; CSLC ships none, and its only mask is the
# once-per-burst static one. Both are called "mask" here so `stack.where(stack.mask == 0)`
# reads the same either way.
MASK_NODATA = {RTC: 255, CSLC: 127}
MASK_DTYPE = {RTC: "uint8", CSLC: "int8"}
MASK_MEANINGS = "0 clear, 1 shadow, 2 layover, 3 both"

# The OPERA archive starts well after Sentinel-1 did: asking for 2014 returns nothing at
# all, with no hint that the date rather than the area was the problem. Measured against
# ASF, not read from a document.
ARCHIVE_START = "2016-01-01"

# T049-103327-IW3 in a filename, t049_103327_iw3 inside an HDF5. The same pattern as
# opera_utils.constants.OPERA_BURST_RE, which is where the two spellings come from.
BURST_ID = re.compile(r"T(\d{3})[-_](\d{6})[-_](IW[1-3])", re.IGNORECASE)

# The fields of a granule name, so nothing has to count underscores to find one. Follows
# opera_utils.constants.CSLC_S1_FILE_REGEX, widened to the other three products: a static
# product carries a reference date where the others carry an acquisition and a processing
# time, and only CSLC names its polarization, the rest naming it in the layer suffix.
# S1[A-E] rather than S1[AB] for the same reason opera-utils uses S1[ABCDE].
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

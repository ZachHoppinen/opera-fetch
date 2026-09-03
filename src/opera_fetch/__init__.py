"""Get OPERA Sentinel-1 data for an area and a date range, on OPERA's own grid.

The six steps every project repeats:

    1. name an area and a date range     opera_fetch.aoi
    2. find what ASF has                 opera_fetch.search
    3. download it                       opera_fetch.download
    4. mosaic the bursts                 opera_fetch.mosaic
    5. stack them in time                opera_fetch.stack
    6. clip to the area and check        opera_fetch.grid, opera_fetch.validate

``fetch`` runs all six. Everything it uses is public, so any one step can be run alone.

Nothing here resamples. Products come back on the lattice OPERA delivers them on, in the
UTM zone of their bursts, and reprojecting is left to whatever comes next.
"""

from opera_fetch.aoi import as_geometry
from opera_fetch.constants import CSLC, CSLC_STATIC, RTC, RTC_STATIC, __version__
from opera_fetch.download import download
from opera_fetch.grid import clip, grid_like, place
from opera_fetch.mosaic import mosaic
from opera_fetch.search import as_dates, data_urls, search, search_static
from opera_fetch.stack import align_passes, assemble, read_bursts
from opera_fetch.validate import check_files, quicklook, report, summary
from opera_fetch.workflow import fetch_stacks
from opera_fetch.write import read, write

__all__ = [
    "CSLC", "CSLC_STATIC", "RTC", "RTC_STATIC", "__version__",
    "align_passes", "as_dates", "as_geometry", "assemble", "check_files", "clip",
    "data_urls", "download", "fetch_stacks", "grid_like", "mosaic", "place",
    "quicklook", "read", "read_bursts", "report", "search", "search_static",
    "summary", "write",
]

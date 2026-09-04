"""Synthetic OPERA granules on disk, for tests that want the whole pipeline.

Real granules are the only way to know the readers are right, and the tests marked
``data`` do that. These are for the other half: running ``assemble`` from files, in CI,
on a scene small enough to write down and fixed enough to compare against.

Everything is deterministic. A cell's value says which burst, which acquisition and where
in the grid it came from, so a stack that has been shifted, transposed or built from the
wrong granule does not come out looking the same.
"""

import numpy as np
import rasterio
from rasterio.transform import from_origin

from opera_fetch import constants as const

# Zone 12 and zone 13 over the same ground, near the boundary at -108. Two bursts of one
# track abut in x. The same easting in both zones would be the same ground 520 km apart.
ANCHOR = {32612: (759_420, 4_332_150), 32613: (240_570, 4_332_150)}

SPACING = 30.0
COLUMNS, ROWS = 12, 10


def write_granule(directory, burst_id, acquired, processed, layers=("VV", "mask"),
                  epsg=32612, column=0, track=49, direction="ASCENDING", orbit=55_893):
    """One acquisition of one burst, as the several files ASF delivers it in.

    column offsets the burst east by a whole grid width, which is how two bursts of one
    track sit next to each other on the same lattice.
    """
    written = []
    for layer in layers:
        name = (f"OPERA_L2_RTC-S1_{burst_id}_{acquired}Z_{processed}Z"
                f"_S1A_30_v1.0_{layer}.tif")
        written.append(_write(directory / name, layer, burst_id, acquired, epsg, column,
                              track, direction, orbit))
    return written


def write_static(directory, burst_id, layer="local_incidence_angle", epsg=32612, column=0,
                 track=49, direction="ASCENDING"):
    """The once-per-burst layer, which carries no acquisition time."""
    name = f"OPERA_L2_RTC-S1-STATIC_{burst_id}_20140403_S1A_30_v1.0_{layer}.tif"
    return _write(directory / name, layer, burst_id, "20140403T000000", epsg, column,
                  track, direction, 0)


def _write(path, layer, burst_id, acquired, epsg, column, track, direction, orbit):
    west, north = ANCHOR[epsg]
    west += column * COLUMNS * SPACING

    # A value that says where it came from: the burst, the day and the cell. Nothing here
    # is constant, so a value that has moved is a value that shows up as moved.
    cells = np.arange(ROWS * COLUMNS, dtype="float64").reshape(ROWS, COLUMNS)
    signature = int(burst_id[5:11]) + int(acquired[6:8]) * 1000 + column

    if layer == "mask":
        # Every code OPERA defines, laid out so a fill or an interpolation shows.
        values = (cells % 4).astype("uint8")
        values[0, :] = const.MASK_NODATA[const.RTC]
        dtype = "uint8"
    elif layer == "local_incidence_angle":
        # Per track, because two tracks see the ground from different angles. Made the
        # same for every burst, the layer is one band for a whole zone however many
        # tracks it holds, and the shape a mixed cache produces never turns up.
        values = (10.0 + track / 100.0 + cells / 10.0).astype("float32")
        dtype = "float32"
    else:
        values = (signature + cells / 1000.0).astype("float32")
        dtype = "float32"

    with rasterio.open(path, "w", driver="GTiff", height=ROWS, width=COLUMNS, count=1,
                       dtype=dtype, crs=f"EPSG:{epsg}",
                       transform=from_origin(west, north, SPACING, SPACING)) as handle:
        handle.write(values, 1)
        handle.update_tags(
            BURST_ID=burst_id,
            TRACK_NUMBER=str(track),
            ORBIT_PASS_DIRECTION=direction,
            ZERO_DOPPLER_START_TIME=f"{acquired[:4]}-{acquired[4:6]}-{acquired[6:8]}"
                                    f"T{acquired[9:11]}:{acquired[11:13]}:{acquired[13:15]}Z",
            PLATFORM="Sentinel-1A",
            ABSOLUTE_ORBIT_NUMBER=str(orbit),
            PRODUCT_VERSION="1.0",
            BOUNDING_POLYGON=_footprint(epsg, west, north))
    return path


def _footprint(epsg, west, north):
    """The granule's outline in lon/lat, which is what the tag carries."""
    from pyproj import Transformer

    to_lonlat = Transformer.from_crs(epsg, 4326, always_xy=True)
    east, south = west + COLUMNS * SPACING, north - ROWS * SPACING
    corners = [(west, south), (east, south), (east, north), (west, north), (west, south)]
    points = [to_lonlat.transform(x, y) for x, y in corners]
    return "POLYGON((" + ", ".join(f"{x} {y}" for x, y in points) + "))"

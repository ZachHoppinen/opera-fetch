"""The same six steps for CSLC, where the values are complex and stay that way.

Step by step rather than through fetch_stacks, because CSLC is heavy: about 275 MB per
burst per acquisition, so the search is worth reading before the download starts.

    conda run -n opera-fetch python scripts/example_cslc.py
"""

import logging

import numpy as np

import opera_fetch as of

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logging.getLogger("asf_search").setLevel(logging.WARNING)

AOI = (-107.0, 38.85, -106.95, 38.90)
START, END = "2024-11-01", "2024-11-15"
CACHE = "data/raw/cslc_example"


def main():
    found = of.search(AOI, START, END, product=of.CSLC)
    print(found.groupby(["track", "direction"]).burst_id.nunique())

    # data_urls logs how many gigabytes this is before anything is fetched.
    paths = of.download(of.data_urls(found), CACHE)
    paths += of.download(of.data_urls(of.search_static(found)), CACHE)

    stacks = of.assemble(paths, aoi=AOI)
    for key, stack in stacks.items():
        print(f"\n=== EPSG:{key}")
        print(of.summary(stack, aoi=AOI))

        # The phase is the measurement, so check it survived the trip.
        scene = stack.vv.isel(time=0).values
        finite = scene[np.isfinite(scene)]
        print(f"complex dtype {stack.vv.dtype}, "
              f"phase spread {np.std(np.angle(finite)):.2f} radians over {finite.size} pixels")

        of.quicklook(stack, f"figures/cslc_example/epsg{key}.png")

    # Zarr, not netCDF: complex has no place in the netCDF-4 standard.
    of.write(stacks, "data/processed/cslc_example.zarr")


if __name__ == "__main__":
    main()

"""The six steps end to end over the East River, Colorado. A month of RTC, about 0.3 GB.

    conda run -n opera-fetch python scripts/example_east_river.py
"""

import logging

import opera_fetch as of

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
# asf_search logs every query and page of results at INFO, burying everything else.
logging.getLogger("asf_search").setLevel(logging.WARNING)

AOI = (-107.0, 38.85, -106.85, 38.95)
START, END = "2024-11-01", "2024-11-30"


def main():
    stacks = of.fetch_stacks(AOI, START, END, product=of.RTC,
                             cache_dir="data/raw/east_river",
                             out="data/processed/east_river.nc")

    # One entry per track, pass direction and UTM zone.
    for key, stack in stacks.items():
        print(f"\n=== {key}")
        print(of.summary(stack, aoi=AOI))
        of.quicklook(stack, f"figures/east_river/{key.track:03d}_{key.direction.lower()}.png")


if __name__ == "__main__":
    main()

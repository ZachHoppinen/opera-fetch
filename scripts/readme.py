"""The README example, verbatim and runnable. A month of RTC, about 0.3 GB.

    conda run -n opera-fetch python scripts/readme.py
"""

import opera_fetch as of


def main():
    stacks = of.fetch_stacks((-107.0, 38.85, -106.85, 38.95),
                             start="2024-11-01", end="2024-11-30",
                             product=of.RTC, cache_dir="data/raw/east_river",
                             out="data/processed/east_river.nc")

    for epsg, stack in stacks.items():
        print(f"=== EPSG:{epsg}")
        print(of.summary(stack))


if __name__ == "__main__":
    main()

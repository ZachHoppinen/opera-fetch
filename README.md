# opera-fetch

Helper to build OPERA Sentinel-1 RTC and CSLC data stacks for a user defined area and a date range.


[OPERA](https://www.jpl.nasa.gov/go/opera/), Observational Products for End-Users from
Remote Sensing Analysis, is a NASA project at JPL that turns satellite data into products
you can use without doing the SAR processing yourself. Its two Sentinel-1 products are
[RTC-S1](https://www.jpl.nasa.gov/go/opera/products/rtc-product/), radiometrically terrain
corrected backscatter, and CSLC-S1, coregistered single-look complex, both geocoded to UTM
and delivered one burst at a time, free, through the
[ASF DAAC](https://asf.alaska.edu/datasets/daac/opera/). The processing code is open at
[github.com/opera-adt](https://github.com/opera-adt) and the granules are searchable in
[ASF Vertex](https://search.asf.alaska.edu).

This project uses these six steps to get OPERA data for scientific/operational analysis:

1. name an area and a date range
2. find what ASF has
3. download it
4. mosaic the bursts
5. stack them in time
6. clip to the area and check

```python
import opera_fetch as of

stacks = of.fetch_stacks((-107.0, 38.85, -106.85, 38.95),
                         start="2024-11-01", end="2024-11-30",
                         product=of.RTC, cache_dir="data/raw/east_river",
                         out="data/processed/east_river.nc")

for epsg, stack in stacks.items():
    print(f"=== EPSG:{epsg}")
    print(of.summary(stack))
```

That is `scripts/readme.py`, so it can be run as it stands:

```bash
conda run -n opera-fetch python scripts/readme.py
```

```
=== EPSG:32612
grid       390 by 451 at (30.0, 30.0) m, EPSG:32612
variables  vh, vv, mask, local_incidence_angle, number_of_looks
bursts     1
times      2 from 2024-11-09 to 2024-11-21
coverage   100% of cells finite, median over time

=== EPSG:32613
grid       381 by 442 at (30.0, 30.0) m, EPSG:32613
variables  vh, vv, mask, local_incidence_angle, number_of_looks
bursts     4
times      4 from 2024-11-09 to 2024-11-28
coverage   100% of cells finite, median over time
```

Two entries because this AOI straddles a UTM zone boundary. Usually there is one.

## Processing choices

**Nothing is resampled unless you ask.** OPERA delivers every burst on a fixed lattice in
the UTM zone it falls in, so two bursts of one track, and the same burst reprocessed next
year, share their pixel centres exactly. Mosaicking and clipping are therefore coordinate
lookups: every value that comes out is a value that went in. `reproject_to` is the only
resampling in the package, and it has to be asked for by name.

**The grid is OPERA's own.** The lattice is read off the products rather than rebuilt from
a rule about where OPERA snaps. OPERA assigns the UTM zone per burst, so an AOI near a zone
boundary is covered by bursts in two zones, and the same ground then has two sets of x and
y coordinates that no single grid holds. Rather than resample one of them away, the result
is a dictionary keyed by EPSG. Usually it has one entry; a cross-zone AOI is the only thing
that forces a second.

```
{32612: <xarray.Dataset>, 32613: <xarray.Dataset>}
```

**What comes back is xarray, and nothing else.** A `Dataset` per zone, dimensions
`(time, y, x)`, one layer per variable, dask-backed so a season is not read until it is
used, with the CRS where rioxarray keeps it. Track, direction, platform and orbit are
coordinates on the time axis rather than keys in a structure of ours, so selecting a track
is `stack.sel(time=stack.track == 49)`. There is no container of ours to learn.

### What it refuses

Four things raise rather than returning something quietly wrong:

- **An AOI whose coordinates are not lon/lat.** ASF clamps a latitude past 90 rather than
  complaining, which degenerates the polygon and returns nothing, so a transposed pair or a
  forgotten `aoi_crs` reads as an empty archive. A box with south above north is refused
  too, where `shapely.box` quietly swaps them.
- **An AOI across the antimeridian.** Left alone, `box(179.5, 51.5, -179.5, 52)` becomes
  the whole world except that box, and the search returns thousands of granules. Split it
  at 180 and assemble each side; they are in different zones anyway.
- **A date range ending before the OPERA archive starts**, which is around 2016, not 2014
  when Sentinel-1 launched. A range inside the December 2021 to 2025 single-satellite gap
  is allowed but noted: Sentinel-1B had failed and 1C was not yet operational, so there are
  about half the usual acquisitions.
- **A resampling name GDAL will not apply to complex data**, the quantiles among them,
  which have no meaning for a complex number. GDAL raises those itself.

## Reprojection, if you ask for it

```python
stack = of.fetch_stacks(aoi, start, end, reproject_to="auto")          # one Dataset
stack = of.fetch_stacks(aoi, start, end, reproject_to="EPSG:32613")    # if you care which
```

`"auto"` picks the zone the AOI lies in, found as the delivered zone whose central meridian
is nearest the middle of the AOI. Data already in that zone keeps the grid OPERA delivered
and only the other zones move. With one zone, the usual case, `"auto"` resamples nothing.

A mask is categorical and always moves by nearest. Real layers take `nearest` by default,
overridden with `resampling=` and any name from `rasterio.enums.Resampling`. A complex
layer takes neither: it is oversampled eight times over by zero padding its spectrum, which
is exact, and the fine sample read directly, the same sinc family OPERA geocodes complex
data with.

**Nearest for real layers is a trade, not a clear win.** Neighbouring UTM lattices do not
line up. Putting the East River zone 12 grid into zone 13 leaves every pixel centre a median
12 m, and up to 21 m, from the nearest source centre on a 30 m grid, as a rotation and a
slight scale change rather than a shift. Nearest returns a real observed gamma0 with its
speckle statistics intact and attributes it to a cell up to two thirds of a pixel away;
bilinear puts it in the right place and removes 42% of the variance. Nearest is the default
because it invents nothing, not because the error is small.

Two things before overriding it. `cubic` and `lanczos` have negative lobes and produce
negative gamma0, 0.12% and 0.35% of cells on that scene, and gamma0 is a power ratio, so
`10*log10` gives NaN there. And the oversampling used for CSLC does not transfer: gamma0 is
detected rather than complex, so it is not bandlimited to its own grid and zero padding its
spectrum also gives negatives.

Where bursts carry `number_of_looks` they are averaged in proportion to it, so a cell only
one burst reaches still goes through the weighted average, and `(w * x) / w` is not always
`x` in float32: about a tenth of such cells come back within one unit in the last place, a
relative 2e-7. No value moves and nothing is interpolated; that last bit is the arithmetic
of averaging a single number.

## Products

| product | posting | what comes back |
|---|---|---|
| `RTC` | 30 m | `vv`, `vh` or `hh`, `hv`, plus `mask`, linear gamma0 |
| `RTC_STATIC` | 30 m | `local_incidence_angle` in radians, `number_of_looks`, more on request |
| `CSLC` | 5 by 10 m | `vv` or `hh`, `complex64` |
| `CSLC_STATIC` | 5 by 10 m | `local_incidence_angle`, `mask`, `los_east`, `los_north` |

An RTC stack therefore comes back with `vv`, `vh`, `mask`, `local_incidence_angle` and
`number_of_looks`; a CSLC one with `vv` (or `hh`), `mask`, `local_incidence_angle`,
`los_east` and `los_north`.

What varies between acquisitions is a coordinate on the time axis: `track`,
`direction`, `platform` (S1A, S1B, S1C) and `absolute_orbit`, which is what a baseline is
worked out from.

What is fixed for the whole stack is an attribute: `epsg`, `tracks`, `spacing`, `burst_id`,
`bursts`, and `footprint`.

A once-per-burst layer stays `(y, x)` while one track covers the area, and gains a time
axis when a second track does, because the two see the ground from different angles.

The AOI is delivered whole: the grid covers it, widened outward to the lattice, so each
edge lands under one cell outside what you asked for and never inside it.

Every stack also records where it came from: `granules` lists the exact granule IDs it was
built from, with `product_version`, `created` and `opera_fetch_version` beside them.

Static layers are made once per burst rather than once per acquisition, so they are
searched by burst ID and downloaded once however many seasons the stack covers. They come
along automatically; pass `static=False` to skip them.

### Writing a CSLC: use `.zarr`

The netCDF-4 standard has no complex type, so there is nowhere in it to put a `complex64`
value. `write` does it anyway, through h5netcdf with `invalid_netcdf=True`, and no data is
lost, but what comes out is HDF5 that is not conforming netCDF.

What that costs, measured rather than assumed:

| reading it back | you get |
|---|---|
| `of.read`, or xarray with `engine="h5netcdf"` | `complex64`, bit for bit |
| `netCDF4`, or xarray with `engine="netcdf4"` | opens and reads, but the variable is a compound `{r, i}` pair of float32 rather than a complex type |

So the file is not unreadable elsewhere, which is what the h5netcdf warning implies. The
values survive; the type does not, and anything downstream has to reassemble it:

```python
import netCDF4, numpy as np

raw = netCDF4.Dataset("cslc.nc").variables["vv"][:]
values = (raw["r"] + 1j * raw["i"]).astype("complex64")     # exact
```

Zarr has a complex type, so none of this applies:

```python
of.fetch_stacks(aoi, start, end, product=of.CSLC, out="data/processed/cslc.zarr")
of.fetch_stacks(aoi, start, end, product=of.RTC,  out="data/processed/rtc.nc")
```

RTC is real valued and writes to either without any of this.

## Install

```bash
conda env create -f environment.yml
conda activate opera-fetch
```

`environment.yml` is the runtime only. Two features are extras, so nobody pays for what
they do not use, and the tests want both:

```bash
pip install -e ".[vector]"    # AOIs read from a shapefile or GeoJSON
pip install -e ".[plot]"      # quicklook
pip install -e ".[develop]"   # both, plus pytest, ruff, build, twine
```

A GeoDataFrame you already hold works without `[vector]`; only opening a file needs it.
netCDF4 is deliberately not a dependency: every read and write names `engine="h5netcdf"`
or `"zarr"`, so the netCDF C library is never touched.

Any environment with the dependencies will do; the one in `environment.yml` is only the
one known to work.

**On Python 3.14, writing netCDF ends in dozens of lines of `Error in sys.excepthook:` with
nothing after them.** The file is written correctly and the exit code is zero. It is an
interpreter-shutdown interaction between rasterio, dask and h5py, not this package: the
same code outside it does the same thing, and on 3.11 and 3.12 it does not happen at all.

## Earthdata login

Searching ASF needs nothing. **Downloading needs a free NASA Earthdata account**, and
without one the data pool answers `403` on every granule.

1. Register at [urs.earthdata.nasa.gov/users/new](https://urs.earthdata.nasa.gov/users/new).
   It is free and approval is immediate.
2. Put the credentials in `~/.netrc`, which is where `requests` and `asf_search` both look:

   ```
   machine urs.earthdata.nasa.gov login YOUR_USERNAME password YOUR_PASSWORD
   ```

3. Make it readable only by you, which is the convention for a file holding a password:

   ```bash
   chmod 600 ~/.netrc
   ```

On Windows the file is `~/_netrc`, which `requests` looks for alongside `.netrc`. NASA's own
walkthrough is
[How to Generate Earthdata Prerequisite Files](https://disc.gsfc.nasa.gov/information/howto?title=How%20to%20Generate%20Earthdata%20Prerequisite%20Files),
which covers the same thing for curl and wget.

Nothing in this package reads or stores the credentials itself: `asf_search` opens the
session and `requests` finds the file. In CI they come from repository secrets and are
written to `~/.netrc` by the workflow, never committed.

Downloading uses threads rather than processes, so there is nothing platform-specific in it
and it should run anywhere Python does. It has only been tested on macOS.

## Tests

```bash
pytest                      # everything
pytest -m "not network"     # no ASF queries
pytest -m "not data"        # no OPERA granules on this machine needed
```

The `data` tests read whatever granules the example scripts have downloaded, and skip when
there are none. They pick whichever burst has the most acquisitions rather than naming one,
so any cache will do. Point them elsewhere with `OPERA_FETCH_TEST_RTC` and `OPERA_FETCH_TEST_CSLC`, which is
worth doing for CSLC, where the tests want a burst with a longer time series than the
example scripts download.

Tests marked `data` run against real OPERA granules if they are on this machine and skip
themselves otherwise. They are the ones that check this package against products as ASF
actually ships them.

`tests/test_golden.py` runs `assemble` over synthetic granules written to disk and compares
everything that comes out against a digest in `tests/golden/`: sizes, coordinates, dtypes,
attributes, mask code counts and, where the package promises it moves nothing, a hash of
the values themselves. It needs no network and no cache, so it runs in CI. After a change
you meant to make:

```bash
OPERA_FETCH_GOLDEN=update pytest tests/test_golden.py
```

Read the diff before committing it. A digest that changed for a reason you cannot name is
the finding, not the noise.

## Sources

Very helpful repositories consulted:

- [opera-adt/RTC](https://github.com/opera-adt/RTC) (Apache-2.0) defines the
  layover/shadow codes and the layer names, and its `mosaic_geobursts.py` is where
  weighting an overlap by number of looks comes from.
- [opera-adt/COMPASS](https://github.com/opera-adt/COMPASS) (Apache-2.0) defines the CSLC
  mask codes and its 127 fill value.
- [opera-utils](https://github.com/opera-adt/opera-utils) (BSD-3-Clause or Apache-2.0) is
  where parsing filenames by named group comes from, and where the CSLC HH case is handled
  as `/data/VV` or `/data/HH`.
- [opera-adt/CSLC-S1_Specs](https://github.com/opera-adt/CSLC-S1_Specs) is the CSLC product
  specification, used for the identification field names.
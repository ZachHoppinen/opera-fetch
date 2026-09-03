# opera-fetch

OPERA Sentinel-1 RTC and CSLC for an area and a date range, on OPERA's own grid.

The six steps that every project repeats:

1. name an area and a date range
2. find what ASF has
3. download it
4. mosaic the bursts
5. stack them in time
6. clip to the area and check

```python
import opera_fetch as of

stacks = of.fetch_stacks((-107.0, 38.85, -106.85, 38.95), "2024-11-01", "2024-11-30",
                  product=of.RTC, cache_dir="data/raw/east_river",
                  out="data/processed/east_river.nc")
```

```
T049 ascending EPSG:32612
grid       390 by 451 at (30.0, 30.0) m, EPSG:32612
variables  vh, vv, local_incidence_angle, number_of_looks, mask
bursts     1
times      2 from 2024-11-09 to 2024-11-21
coverage   100% of cells finite, median over time
aoi        100% of the AOI is inside the grid
```

## Processing choices

**Nothing is resampled.** OPERA delivers every burst on a fixed lattice in the UTM zone it
falls in, so two bursts of one track, and the same burst reprocessed next year, share their
pixel centres exactly. Mosaicking and clipping are coordinate lookups, so every value that
comes out is a value that went in. Reprojecting is left to whatever comes next, which is
the step that knows what it wants.

The lattice is taken from the OPERA products themselves.

That is why a result is **one Dataset per UTM zone**, keyed by EPSG:

```python
{32612: <xarray.Dataset>, 32613: <xarray.Dataset>}
```

Usually one entry. The zone is the only thing that forces a second, because OPERA assigns
it per burst: over the East River two tracks come as 32612 and two as 32613, and no grid
holds both. It is not rare, either. Of four mountain sites, two straddled a boundary.

Within a zone OPERA's grid is constant, so every track lands on it. **Every acquisition is
one step on the time axis**, whichever track it came from, with `track` and `direction` as
coordinates alongside:

```python
stack.sel(time=stack.track == 49)          # one track
stack.sel(time=stack.direction == "ASCENDING")
```

Nothing is padded to make that work: four acquisitions across two tracks are four rows.

Ascending and descending still must not be *averaged* together, and neither must two
tracks. That is a mosaic rule, and mosaicking happens per pass before any of this.

If you would rather have one Dataset than two, ask for it by name:

```python
stacks = of.fetch_stacks(aoi, start, end, reproject_to="EPSG:32613")
```

That is the only resampling in the package, which is why it has to be requested. It moves
the smaller zone onto the larger one's grid, nearest neighbour, so no value is invented.

**Neighbouring bursts are found by a tolerance.** That is enough to make every burst's
time axis unique and a mosaic of them an empty diagonal ribbon. `align_passes` collapses
timestamps closer together than a tolerance (10 minutes by default) into one pass. `assemble`
does this for you.

**Overlapping bursts are averaged by their looks**, which is how OPERA mosaics its own,
rather than flat. In practice this changes almost nothing: `number_of_looks` is a function
of the terrain, and two bursts of one track view the same ground at nearly the same angle,
so their looks agree to four decimal places over 99.999% of the overlap. It is kept because
it is the right computation where they do differ, not because it moves the numbers.

Where two bursts disagree on the layover/shadow code the mosaic takes the worse of the two,
since both fed the averaged value. Averaging the codes themselves would be worse than
either: shadow (1) and both (3) average to layover (2), a class nobody saw.

**A mosaic's `time` is the earliest burst of that pass**, not each burst's own
zero-doppler start. Down a track that is a few seconds; it is not the instant any
particular pixel was measured. For per-burst timing, read the bursts and skip the mosaic:

```python
bursts = of.read_bursts(paths)      # each keeps its own acquisition times

**Nothing is masked.** Values come out as OPERA delivers them: linear gamma0 for RTC, not
dB; complex for CSLC, with the phase intact. The layover/shadow mask rides along as its own
variable rather than being applied, so whether a layover pixel counts stays is a future decision:

```python
clear = stack.where(stack.mask == 0)
```

That line works for either product, but the two masks are not the same thing. OPERA ships
an RTC mask **per acquisition**, so `mask` there has dims `(time, y, x)`. It ships no CSLC
mask at all; the only one is in CSLC-STATIC, made **once per burst**, so `mask` there is
`(y, x)` and broadcasts over time. Codes for both RTC and CSLC — 0 clear, 1 shadow, 2 layover, 3 both —
but no-observation is 255 for RTC (uint8) and 127 for CSLC (int8).

## Products

| product | posting | what comes back |
|---|---|---|
| `RTC` | 30 m | `vv`, `vh` or `hh`, `hv`, plus `mask`, linear gamma0 |
| `RTC_STATIC` | 30 m | `local_incidence_angle` in radians, `number_of_looks`, more on request |
| `CSLC` | 5 by 10 m | `vv` or `hh`, `complex64` |
| `CSLC_STATIC` | 5 by 10 m | `local_incidence_angle`, `mask`, `los_east`, `los_north` |

A CSLC stack therefore comes back with `vv` (or `hh`) alongside `local_incidence_angle`,
`mask`, `los_east` and `los_north`; an RTC one with `vv`, `vh`, `local_incidence_angle`,
`number_of_looks` and `mask`.

What varies between acquisitions rides on the time axis as a coordinate: `platform`
(S1A, S1B, S1C) and `absolute_orbit`, which is what a baseline is worked out from. What is
fixed for the whole stack is an attribute: `track`, `direction`, `epsg`, `spacing`,
`burst_id`, `bursts`, and `footprint`, the outline of the data rather than its bounding box.

The AOI is delivered whole: the grid covers it, widened outward to the lattice, so each
edge lands under one cell outside what you asked for and never inside it.

Every stack also records where it came from: `granules` lists the exact granule IDs it was
built from, with `product_version`, `created` and `opera_fetch_version` beside them. A
reprocessing changes the numbers, so without those a saved file cannot be told apart from
one built next year. The version is stored as text, because as a float v1.10 and v1.1 are
the same number.

Static layers are made once per burst rather than once per acquisition, so they are
searched by burst ID and downloaded once however many seasons the stack covers. They come
along automatically; pass `static=False` to skip them.

Complex data has nowhere standard to live: netCDF-4 has no complex type, so a CSLC stack
written to `.nc` goes through h5netcdf with `invalid_netcdf`. It is valid HDF5 and xarray
reads it straight back, but a strict netCDF reader will not open it. **Use `.zarr` for
CSLC.**

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

Downloads need an Earthdata login in `~/.netrc`; without it the ASF data pool answers 403.

```
machine urs.earthdata.nasa.gov login <user> password <password>
```

Downloading uses threads rather than processes, so there is nothing platform-specific in
it and it should run anywhere Python does. It has only been tested on macOS. On Windows the
credentials file is `~/_netrc`, which `requests` looks for alongside `.netrc`.

Any environment with the dependencies will do; the one in `environment.yml` is only the one
that is known to work. **On Python 3.14, writing netCDF ends in dozens of lines of `Error in
sys.excepthook:` with nothing after them.** The file is written correctly and the exit code
is zero. It is an interpreter-shutdown interaction between rasterio, dask and h5py, not this
package: the same code outside it does the same thing, and on 3.11 and 3.12 it does not
happen at all. Measured on the RTC example: 62 such lines on 3.14, none on 3.11.

## Tests

```bash
pytest                      # everything
pytest -m "not network"     # no ASF queries
pytest -m "not data"        # no OPERA granules on this machine needed
```

The `data` tests read whatever granules the example scripts have downloaded, and skip when
there are none. They pick whichever burst has the most acquisitions rather than naming one,
so any cache will do. Point them elsewhere with `OPERA_FETCH_TEST_RTC` and
`OPERA_FETCH_TEST_CSLC` — worth it for CSLC, where the time-series tests want a burst with
more than the fortnight the example fetches.

Tests marked `data` run against real OPERA granules if they are on this machine and skip
themselves otherwise. They are the ones that check this package against products as ASF
actually ships them.

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
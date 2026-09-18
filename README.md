# Reassessing U.S. Heatwave Trends with Bias-Corrected Observations

Analysis code for a study of how homogenization and station sampling affect apparent
U.S. heatwave trends, comparing the 1930s Dust Bowl with the present day.

The central question is how much of the difference between the 1930s and today comes
from the climate, and how much from two choices made in building a temperature record:
whether station data are bias-corrected, and which stations are in the network.

## What is here

| File | Contents |
| --- | --- |
| `figures/` | One script per figure — `figure1.py` … `figure6.py` — plus `common.py` (paths and shared machinery) and `prepare_data.py` (builds the inputs the archive does not carry). See `figures/README.md`. |
| `US-Heatwave-Trends-Analysis-Code.ipynb` | The original notebook the scripts were split out of. Every figure, shared code first, then one section per figure. Saved with its outputs, so the figures are visible without running anything. |
| `requirements.txt` | Python packages. |
| `.gitignore` | Keeps data, caches and generated figures out of the repository. |

Notebook or scripts, either produces the figures; the analysis code is identical, line
for line. Use the notebook to read the analysis in order, the scripts to rebuild one
figure. The scripts are what currently runs against the lab archive — the notebook still
carries the original author's paths.

Figure S1 of the paper is not in the notebook and has no script here.

## Running the scripts

```bash
pip install -r requirements.txt
```

Once, to build the derived inputs:

```bash
python figures/prepare_data.py
```

Then any figure, in any order:

```bash
python figures/figure3.py
```

Each script runs top to bottom — shared checks, its own settings and helpers, build, its
own `CHECK`, plot — and prints the paths it wrote. Everything lands in
`PAPER_FIGURES_FINAL/`. The first run of a figure fills `_cache_*/` and is slow, minutes
per dataset; later runs read the cache in seconds.

Cartopy and geopandas are the awkward dependencies. `conda install -c conda-forge
cartopy geopandas` is usually easier than pip. `pyarrow` must be a working install —
a conda `libarrow` built against a `libutf8proc` that is no longer present will import
but segfault; `pip install --force-reinstall --no-deps pyarrow` gets a self-contained
wheel.

### Running the notebook instead

```bash
jupyter lab US-Heatwave-Trends-Analysis-Code.ipynb
```

Run sections 1-4 once, then the figure you want; within a figure, run its cells in order.
Run section 7 (Figure 3) before section 10 (Figure 6): it builds the daily station cube
that the heat-wave figure reads. Update the paths in the Setup cell first — they point at
the original author's drive.

## Data

Nothing in this repository. The raw archives run to tens of GB and the caches to several
more, so all of it is regenerated rather than committed.

The scripts resolve every input against `/Volumes/adessler_lab` by default, with a
fallback to the layout the notebook used. Override with:

| Variable | What it sets | Default |
| --- | --- | --- |
| `HEATWAVE_ROOT` | raw archive | `/Volumes/adessler_lab` |
| `HEATWAVE_WORK` | caches, derived inputs, figure output | the repository root |
| `HEATWAVE_STAGE=0` | read the archive in place instead of staging it locally | unset |
| `HEATWAVE_SHOW=1` | draw interactively instead of headless | unset (`Agg`) |

Every script starts by printing which inputs it can see, so a missing drive or an unbuilt
input is reported before anything is computed.

### What the archive provides, and what is rebuilt

Provided directly: Berkeley Earth (`Processed/` and `RAW/`), ERA5 dailies,
NOAAGlobalTemp, CRUTEM5, nCLIMDIV, GHCN-Daily station metadata, and the USHCN v2.5
monthly raw / FLs.52j pair with its monthly offsets.

Three inputs the figures need are not on the archive in that form, and
`prepare_data.py` builds them into `$HEATWAVE_WORK/_data` (~870 MB, one-time):

1. **GHCN-Daily 24-50N band archive** — rebuilt from the raw by-year CSVs. The archive's
   ready-made daily cube is CONUS-only, and Figures 5 and 6 need stations worldwide.
2. **USHCN daily raw / bias-corrected pair** — *a reconstruction.* Only monthly USHCN is
   on the archive, so the daily pair is rebuilt by adding each station-month's v2.5
   offset to that station's daily GHCN-Daily values.
3. **2-degree global gridded GHCN-Daily TMAX** — Figure 6's bottom-row GHCN panel.

`figures/README.md` documents each of these, the local staging that works around
unreliable HDF5 reads over the network share, and `prepare_data.py prune` for reclaiming
the staged copies afterwards.

### Data sources

| Dataset | Source |
| --- | --- |
| GHCN-Daily | NOAA NCEI |
| USHCN v2.5 (raw and bias-corrected) | NOAA NCEI |
| nCLIMDIV | NOAA NCEI, statewide file, CONUS region code 110 |
| Berkeley Earth | Berkeley Earth daily products |
| NOAAGlobalTemp v6, CRUTEM5 | NOAA NCEI / Met Office Hadley Centre and CRU |
| ERA5 | ECMWF, via the Copernicus Climate Data Store |

## The figures

| Script | Notebook section | Figure | What it shows |
| --- | --- | --- | --- |
| `figure1.py` | 5 | Figure 1 | CONUS JJA anomalies, TMAX / TMIN / TAVG, eight datasets, 1900-2024 |
| `figure2.py` | 6 | Figure 2 | Seasonal bars, DJF-SON by element, Dust Bowl vs. modern |
| `figure3.py` | 7 | Figure 3 | CONUS daily TMAX record frequency, May-September |
| `figure4.py` | 8 | Figure 4 | Berkeley Earth exceedance maps, p95 from the full record |
| `figure5.py` | 9 | Figure 5 | Northern mid-latitude band, 24-50N, JJA |
| `figure6.py` | 10 | Figure 6 | Heat-wave days, CONUS vs. the global strip, Christy (2026) method |

Notebook sections 1-4 — setup, geometry, the anomaly-first pipeline, the gridded
loaders — are `figures/common.py`.

### The CHECK blocks

Every section ends with a `CHECK` that tests what was just defined, so a failure is
located before any figure is drawn. Most run on synthetic numbers and need no data at
all. They confirm, among other things, that a season is labelled by the year of its last
month; that a field equal to its own climatology gives exactly zero; that a station
sitting on a grid point is reproduced exactly by the interpolation; that a station
without enough baseline years is dropped; that ties in the record count are split rather
than awarded to the earliest year; that each record series integrates to exactly 153
May-September days; and that a five-day hot spell is not counted as a heat wave while a
six-day one is.

Figure 1's check is also an end-to-end validation: it prints the homogenization signal
(USHCN-BC minus USHCN-Daily) against the published run, and the current build reproduces
it to 0.01 degC.

## Methods, in one paragraph each

**Anomalies come first.** Each station is converted to an anomaly against its own
1951-1980 monthly climatology before it is gridded, and each grid cell against its own
before cells are averaged. Averaging absolute temperatures first lets a change in which
stations report move the series, because stations differ in elevation and exposure.

**One CONUS domain.** A Natural Earth 50m polygon of the United States, clipped to
24-50N / 125-66W and minus lakes, tests both stations and grid cells. Each cell is
weighted by its spherical band area times the fraction of the cell inside the polygon.

**Station products are gridded by IDW** onto a 0.5 degree CONUS grid (power 2, 8
neighbours, 150 km cutoff), except where a figure counts at stations on purpose.

**Records** (Figure 3) are counted per calendar day, May-September: for each station or
cell, the year holding the highest TMAX of 1900-2024 gets that day, and years sharing the
highest value split it equally. A location must have data in at least 100 of 125 years
and 80% of possible days.

**Heat waves** (Figure 6) follow Christy (2026): runs of six or more days above a
day-of-season 90th percentile computed over the full record in a +/-3 day window. A
station or cell is dropped from a year's average if fewer than 70% of that year's
May-September days have a valid TMAX.

## Caveats on the current reproduction

Three places where what the scripts produce is not bit-for-bit what the paper shows, all
of them consequences of what is and is not on the archive:

- **The USHCN daily pair is reconstructed**, not copied — see above. The paper's own
  `ushcn_daily_homog` was built by the original author and is not on the archive. The
  method is reproduced and the matched-pair invariant the figures assert holds, and
  Figure 1's homogenization signal matches the published value, but the USHCN-Daily and
  USHCN-BC lines are a rebuild.
- **nCLIMDIV is a later release.** The archive carries the `20260806` processing date;
  the paper used `20260406`, so nCLIMDIV numbers can differ slightly. The scripts resolve
  the filename rather than pinning a date.
- **Figure 4's embedded image in the .docx is stale.** Its colourbar is labelled
  "baseline 1951-1980" and its scale runs to ~47%, while its own title and the paper's
  caption both say the threshold comes from the full 1900-2024 record. The script follows
  the caption and the notebook, which puts the scale at ~25%. The spatial patterns are
  the same either way.

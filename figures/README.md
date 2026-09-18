# One script per figure

`US-Heatwave-Trends-Analysis-Code.ipynb` split up, so each figure of the paper
can be produced on its own.

| Script | Paper figure | Notebook section |
| --- | --- | --- |
| `figure1.py` | Figure 1 — CONUS JJA anomalies, TMAX/TMIN/TAVG, eight datasets | 5 |
| `figure2.py` | Figure 2 — seasonal bars, DJF–SON by element | 6 |
| `figure3.py` | Figure 3 — CONUS daily TMAX record frequency | 7 |
| `figure4.py` | Figure 4 — Berkeley Earth exceedance maps, 24–50°N | 8 |
| `figure5.py` | Figure 5 — northern mid-latitude band, JJA | 9 |
| `figure6.py` | Figure 6 — heat-wave days, CONUS vs. the global strip | 10 |
| `common.py` | not a figure: paths and the shared machinery | 1–4 |
| `prepare_data.py` | not a figure: builds the inputs the archive does not carry | — |

Figure S1 of the paper is not in the notebook and so has no script here.

## Running

Once, to build the derived inputs (see *Data* below):

```bash
python figures/prepare_data.py
```

Then any figure, in any order:

```bash
python figures/figure1.py
```

Each script runs top to bottom: shared checks, its own settings and helpers,
build, its own `CHECK`, plot. It writes its PNG (and the CSV or PDF the
notebook wrote) and prints the paths at the end. Unlike the notebook, Figure 6
no longer needs Figure 3 to have been run first.

The first run of a figure fills `_cache_*/` and is slow — minutes per dataset.
Later runs read the cache in seconds.

## Data

`common.py` resolves every input, defaulting to `/Volumes/adessler_lab` and
falling back to the layout the notebook used. Override with:

| Variable | What it sets | Default |
| --- | --- | --- |
| `HEATWAVE_ROOT` | raw archive | `/Volumes/adessler_lab` |
| `HEATWAVE_WORK` | caches, derived inputs, figure output | the repository root |
| `HEATWAVE_USREG` | Christy CONUS mask (optional, Figure 6) | unset |
| `HEATWAVE_SHOW=1` | draw interactively instead of headless | unset (`Agg`) |

Every script starts by printing which inputs it can see, so a missing drive or
an unbuilt input is reported before anything is computed.

### What the archive provides directly

Berkeley Earth (`Processed/` and `RAW/`), ERA5 dailies, NOAAGlobalTemp,
CRUTEM5, nCLIMDIV, the GHCN-Daily station metadata, and the USHCN v2.5 monthly
raw / FLs.52j pair with its monthly offsets.

Two names are resolved rather than pinned, because they carry a processing date
that changes between pulls: the NOAAGlobalTemp gridded file, and the nCLIMDIV
statewide files. The nCLIMDIV files on this archive are the `20260806`
processing date; the paper used `20260406`, so nCLIMDIV numbers can differ
slightly from the published version.

### Local staging (`_stage/`)

The archive lives on an SMB network share, and HDF5 — which every netCDF read
goes through — does many small random-access reads. Against this share that is
unreliable: it returns `OSError: Bad file descriptor`, or segfaults the HDF5
library outright, on files that a plain sequential `cp` copies perfectly. Which
file it hits varies from run to run, so it cannot be retried around.

`common.py` therefore wraps `xr.open_dataset` / `xr.open_mfdataset`: any file
under `HEATWAVE_ROOT` is copied to `$HEATWAVE_WORK/_stage/` once (with `/bin/cp`,
not Python's buffered `open`, which also fails on this share) and opened from
there. The figure code is untouched — it still calls xarray exactly as the
notebook did. Copies are kept, so later runs are much faster than the first.

Budget roughly 30 GB for `_stage/`, most of it ERA5 for Figure 6. Set
`HEATWAVE_STAGE=0` to read the archive in place, which is fine when the data
are on a local disk.

Copies are kept so re-runs are fast, but they are disposable — reclaim the
space once the figures have run:

```bash
python figures/prepare_data.py prune
```

That frees about 20 GB and keeps the ~220 MB that Figures 3 and 4 reopen on
every run (neither caches what it reads from those two files). `--all` removes
those too, `--dry-run` just reports. Pruning is always safe: staging is
automatic, so anything deleted is copied back the next time something opens it.
With `_cache_*/` intact, the big inputs are never reopened at all.

### What `prepare_data.py` builds

Three inputs the figures need that no archive carries in that form. All land
under `$HEATWAVE_WORK/_data` and are skipped if already present.

**1. `ghcnd_band/ghcn_global_YYYY.parquet` — the GHCN-Daily yearly archive.**
The drive has the raw global by-year CSVs and a QC'd CONUS-only daily cube.
Figure 5 and Figure 6's bottom row need stations outside the United States, so
the archive is rebuilt from the raw CSVs and restricted to 24–50°N at every
longitude — the widest domain any figure asks for. QC is the archive's own
documented rule, the same one its CONUS cube was built with: a non-blank
quality flag means the value failed, and the first observation of a
station-day-element wins. Values are converted from the archive's 0.1 °C.
About 600 MB and 10–30 minutes, mostly waiting on the network share.

**2. `ushcn_daily_homog/` — the USHCN daily raw / bias-corrected pair.**
*This is a reconstruction, not a copy.* The archive has USHCN v2.5 **monthly**
raw and FLs.52j, and their difference as `ushcn_offsets_1900_2024.nc`; it has
no daily adjusted product. The pair is rebuilt by taking each USHCN station's
daily GHCN-Daily values as the raw leg and adding that station-month's monthly
offset to get the adjusted leg. A day counts only if the raw value and that
month's offset both exist, so the adjustment is the only difference between the
legs — the invariant Figures 3 and 6 assert. The paper's own `ushcn_daily_homog`
was built by the original author and is not on this drive, so the USHCN-Daily
and USHCN-BC lines are reproduced in method rather than bit-for-bit.
Also writes the paired monthly table and the May–September pivots the figures read.

**3. `ghcnd_global_grid/ghcn_grid_YYYY.nc` — 2° global gridded GHCN-Daily TMAX.**
Figure 6's bottom-row GHCN panel reads a 2° gridded product that is not on the
drive. It is rebuilt from (1) by averaging stations within each 2° cell over
24–50°N. The heat-wave thresholds are per-cell and taken from each cell's own
record, so the grid's construction does not bias the comparison.

## What changed from the notebook

The figure code is the notebook's, line for line. The differences are all
outside the computation:

1. **Sections 1–4 live in `common.py`** and each script does `from common
   import *`.
2. **Paths are resolved, not hard-coded**, so the scripts work against either
   archive layout, and the nCLIMDIV / NOAAGlobalTemp filenames are globbed
   rather than pinned to one processing date.
3. **The May–September GHCN-Daily station cube moved into `common.py`.** In the
   notebook it was built by section 7 (Figure 3) and section 10 (Figure 6) read
   the `.npz` it left behind, so the notebook had to be run in order. It is now
   `ghcnd_daily_cube()`, cached at the same path, so either figure can run first.
   The USHCN May–September pivots moved to `prepare_data.py` for the same reason.
4. **`figure6.py` draws the title the paper's copy of Figure 6 carries**
   ("CONUS VS Northern Mid-latitude Band Heatwave Days"); the notebook's plot
   block drew no title. Set `SUPTITLE_F6 = None` for the bare panel grid.
5. **Figure 6 writes to `PAPER_FIGURES_FINAL/`** like every other figure. The
   notebook sent this one to the work directory instead.

### A note on Figure 4

The script reproduces the paper's Figure 4 *caption*, which specifies a
day-of-year 95th-percentile threshold "calculated over the full 1900-2024
record" — that is what the notebook computes and what the script draws.

The image actually embedded in the .docx is older: its colourbar is labelled
"baseline 1951-1980" even though its title says full record, and its absolute
scale runs to about 47% rather than 25%. A 1951-80 threshold puts a much larger
share of modern days above it, which is exactly that difference. The spatial
patterns are the same in both. So the script agrees with the caption and the
notebook, and the embedded image is the stale one.

## Bug fixes

One bug fix: the `area_mean` assertion in the pipeline `CHECK` called `float()`
on a size-1 1-D array, which numpy 2 rejects. It now takes the single time step
explicitly. Same numbers, and nothing that draws a figure was touched.

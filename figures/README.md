# One script per figure

`US-Heatwave-Trends-Analysis-Code.ipynb` split up, so each figure of the paper
can be produced on its own.

| Script | Paper figure | Notebook section |
| --- | --- | --- |
| `figure1.py` | Figure 1 — CONUS JJA anomalies, TMAX/TMIN/TAVG, eight datasets | 5 |
| `figure2.py` | Figure 2 — seasonal bars, DJF–SON by element | 6 |
| `figure3.py` | Figure 3 — CONUS daily TMAX record frequency (two layouts) | 7 |
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

Figure 3 aside — see below — the figure code is the notebook's, line for line,
and the differences are all outside the computation:

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

### Figure 6 now follows Christy's procedure

`figure6.py` claimed the Christy (2026) heat-wave method but departed from the
reference implementation — `us_dly_waves.py`, a Python port of J.R. Christy's
`us_dly_waves.f` — in four places. All four are now switches at the top of the
script, marked `(I1)`–`(I5)`, so the old behaviour is one edit away and the
CHECK block prices the difference on every run.

What was already identical, and is untouched: the May–September season and its
153-day index, the ±3-day pooled threshold window and its clipping at the season
edges, the ≥6-consecutive-day run rule with every day of a qualifying run
counted, and the 70% valid-day gate on a station-year. `_PCTILE = 90` and
`_MIN_RUN = 6` also stay; Christy's script is driven by `--percentile` /
`--min-run` and his own example is 95/3, but Figure 6 keeps its published 90/6.

1. **The spatial reduction, and the one that matters.** Christy counts wave days
   at each station, IDW-interpolates *the counts* onto a grid, and then takes a
   cos(lat) **area** average over the accepted cells. The script averaged over
   **stations** instead, which weights by where the observers are rather than by
   land. The IDW is transcribed exactly, Fortran quirks included: a 115 km strict
   search radius, weight `(115/d)²`, Christy's own great-circle formula with its
   6335.44 km Earth radius and its `cos(cell_lat)` in place of `cos(stn_lat)`,
   and his acceptance rule — a cell is used in a year only if ≥2 in-range
   stations reported, or exactly one within 115/1.5 = 77 km. This now applies to
   GHCN-Daily CONUS, both USHCN legs, and the Berkeley-sampled-at-USHCN
   comparator, which has to travel the same road or its ratio stops isolating
   sampling.

2. **The percentile estimator.** Christy sorts the pooled window and takes
   `sorted[int(n·p/100)]` — a nearest-rank rule, so the threshold is an
   *observed* value. `np.nanpercentile` interpolates between two of them. Note
   that this is *not* `np.percentile(method="inverted_cdf")`, which sits one rank
   low whenever `n·p/100` lands on an integer; with a 7-day window at the 90th
   percentile that product is `6.3·n_years`, an integer for every record whose
   length is a multiple of 5.

3. **The exceedance test, `>` → `>=`,** and this one is much larger than it
   looks. GHCN-Daily and USHCN daily values are whole degrees **Fahrenheit**
   converted to °C: a station's entire 125-year record holds only 54–73 distinct
   values. A nearest-rank threshold is one of those levels, so **3.7% of all
   station-days sit exactly on their threshold**. Against 8.3% strictly above,
   admitting the ties raises exceedance days by a factor of 1.44 — and because a
   wave needs six *consecutive* days, ties also bridge spells that `>` would have
   split. It is the correct choice as well as the faithful one: `>` was
   delivering an effective 91.7th percentile where the parameter says 90th.

4. **The threshold sample minimum,** `MIN_THRESH_N` 30 → 201 (Christy's
   `icnt > 200`). It never binds on full-record gridded data; on stations it is
   the de-facto record-length filter, roughly 29 near-complete seasons.

(2)–(4) are shared code, so they apply to **all six panels**, gridded lines
included. That is deliberate: the nearest-rank bias relative to interpolation
scales as 1/n, so forking the estimator would give the sparse products a
different *effective* percentile from the dense ones — precisely the
method-versus-data confound the figure exists to remove. On Berkeley the tie
fraction is 0.001, so `>=` is very nearly a no-op there; the station lines move
and the gridded lines barely do, which means any narrowing of the gap between
them is real rather than method.

**One deliberate departure from Christy, marked `(I7)`.** His loop records
`nval[yi] = total` for any year clearing the 70% data gate *even when the station
has no thresholds at all*, so a short record reports a hard **zero** heat-wave
days rather than "unknown". Among his 1,218 uniformly long USHCN records that can
never happen. Among the global band's stations most are in exactly that state,
and a hard zero is not a missing value — it would be interpolated, averaged, and
would drag the field down. A unit must now also *have* thresholds on 70% of the
season, or its years are NaN.

**The output grid is Berkeley's, not Christy's.** He grids to his own 0.5° CONUS
mask (`usreg_half.txt`, 116×50). Both rows here grid onto the Berkeley Earth grid
instead — 1°, 26×59 for CONUS, 26×360 for the 24–50°N band — using Berkeley's own
land mask, so every line in a panel sits on the same cells and the final cos(lat)
area mean is byte-identical to the one the Berkeley line goes through. The IDW's
radius and acceptance rule are properties of the search, not of the cell, so they
carry over untouched. (If a half-degree sensitivity test is ever wanted,
`common.g05_centers()` already reproduces Christy's grid exactly.)

**The bottom row changed too.** It read a 2° grid of daily *temperatures* built
by `prepare_data.py`, which grids first and counts second — the reverse of
Christy's order. It now counts at stations and grids the counts, like the top
row, reading the band station archive through the new `common.ghcnd_band_cube()`.
Because the network outside the US is far sparser, that row uses its own search
radius (`IDW_RADIUS_KM_GLOBAL`); set it from the coverage diagnostic the script
prints rather than by eye. The old path is still there under
`GHCND_GLOBAL_SOURCE = "grid2deg"`.

**A second Berkeley line on each row, `(I9)`.** Christy's drifting footprint is
kept — each year averages over the cells the IDW accepted that year — but on the
global band that footprint grows from 47% of the land in the 1930s to 84% today,
so an era comparison there is partly a comparison of coverage. Pinning the domain
to the never-missing cells would answer that by discarding three quarters of the
band. Instead each Berkeley panel gains a second trace: the *same* Berkeley
field, reduced each year over exactly the cells GHCN carried a value in that
year. Berkeley is the only dataset here that exists everywhere, so it is the only
one that can hold the climate fixed and vary the footprint alone; the gap between
the two green lines is what the drift is worth, and where they lie on top of each
other, coverage is not the story.

On CONUS they nearly coincide (ratio 0.80 vs 0.90), which is what the stable
0.86–0.92 coverage predicts. On the band they separate before about 1960: the
1930s go from 2.24 on the full grid to 3.44 on GHCN's cells, so the early station
network sat on the warmer part of the band. That number is worth dwelling on —
GHCN's own 1930s value is 3.42, so once the footprint is matched the two datasets
agree almost exactly in that decade. They do not agree recently: 5.49 against
11.71 on the same cells, which is a dataset difference and not a sampling one.

**Two further diagnostics, both printed rather than asserted.** Christy's
`ddd(0)/ddtot` coverage — the cos(lat) share of the masked domain whose cells
were accepted, per era — because the IDW follows the network as it opens and
closes, so a sparse era is being compared on a different footprint and not only
in a different climate. And an old-versus-new table: both reductions are computed
from the same pass over the counts, so the cost of the method change is visible
rather than inferred.

**Every cached series in `_cache_christy_hw/` is tagged with the method that
produced it**, because (2)–(4) reach the gridded lines as well and a figure drawn
half from old pickles and half from new ones would look entirely plausible. The
old filenames can never be matched, so they survive on disk as the previous
record. The masks, the Figure-3 cube and the USHCN pivots are *data*, not method,
and are deliberately not invalidated — wiping the cache directory on a method
change would cost a multi-hour rebuild for nothing.

### Figure 3 has diverged

`figure3.py` is no longer section 7. The notebook was left alone deliberately,
so the two now differ in what they compute and not merely in where the code
lives. Run the notebook and you get the original figure. What the script adds:

1. **The smoothed line reaches both ends of the record.** The notebook's
   centered 11-year mean required a full window, so it stopped in 2019 — five
   years short of the data. The script writes the same mean as an unweighted
   local linear fit: identical in the interior, because the least-squares line
   through a symmetric window read at its own center *is* that window's mean,
   which the script asserts to machine precision, and carrying the local trend
   to the endpoint over the outer five years at each end. `END_METHOD` also
   offers a shrinking mean, Savitzky–Golay, Mann (2004) minimum-roughness
   padding, and `none` for the notebook's behaviour.

   This smoother now lives in `common.py` and Figures 1 and 5 use it too, so
   there is one implementation rather than one per script. Those two pass
   `strict_interior=True`, which keeps the plain full-window rule inside the
   record and applies the local linear fit only at the record's own ends: a gap
   stays a gap, so ERA5, whose record starts in 1940, still has no smoothed
   value before 1945 rather than a six-point line drawn through the boundary.
2. **The endpoint's uncertainty is reported rather than asserted.**
   `end_uncertainty()` truncates a series at every year, smooths the
   truncation, and scores its estimate against the centered mean the full record
   eventually reports there. That RMSE — 0.6 to 0.8 records/yr at the last
   year, about 0.3 two years in — is printed as a table. It is no longer drawn:
   `END_SHADE` and `END_BAND` are both off, so the panels carry the smoothed
   line alone, and Figure 5's co-sample spread is likewise printed rather than
   shaded.
3. **Berkeley read at the USHCN sites**, built like the existing GHCN version:
   one unit per station at its Berkeley cell, duplicates kept so the line
   carries the network's density and not merely its footprint, on the 671
   stations that survive the completeness cut and therefore build the purple
   lines.
4. **A sampling correction for USHCN-BC.** Berkeley is the only dataset here
   that exists both on the full grid and on a station footprint, so their ratio
   is what sampling alone does to a record count with the climate held fixed:
   0.795 in the 1930s, 1.293 over 2015–24. USHCN-BC multiplied through by it
   goes from 3.12 to 2.48 in the 1930s and from 1.52 to 2.00 recently, taking
   its Dust Bowl ratio from 2.05 to 1.24 against Berkeley's own 1.13. The factor
   is taken from the smoothed Berkeley pair, not year by year: annually the
   ratio runs 0.58 to 1.99 and its denominator falls to 0.057 records/yr.
5. **A second layout.** `Fig3_panels_*` draws one series per panel in Figure 6's
   `FigHW` style — wide panel grid, shared y, bold panel titles — alongside the
   original single-axes `Fig3_final_*`. Both are drawn from the same computed
   series, so they cannot drift apart. The output filenames now carry the
   endpoint rule, so a run never overwrites a figure drawn with another one.

Two columns in the printed table and the CSV are new: `Berkeley Earth at USHCN
Stations` and `USHCN-BC × BE/BE-at-USHCN`. The second is a product rather than a
record count, so it is the one column that does not integrate to 153.

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

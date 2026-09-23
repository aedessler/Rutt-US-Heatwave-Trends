# One script per figure

`US-Heatwave-Trends-Analysis-Code.ipynb` split up, so each figure of the paper
can be produced on its own.

| Script | Paper figure | Notebook section |
| --- | --- | --- |
| `figure1.py` | Figure 1 — CONUS JJA anomalies, TMAX/TMIN/TAVG, eight datasets | 5 |
| `figure2.py` | Figure 2 — JJA bars by element | 6 |
| `figure3.py` | Figure 3 — CONUS daily TMAX record frequency (two layouts) | 7 |
| `figure4.py` | Figure 4 — Berkeley Earth exceedance maps, 24–50°N | 8 |
| `figure5.py` | Figure 5 — northern mid-latitude band, JJA | 9 |
| `figure6.py` | Figure 6 — heat-wave days, CONUS vs. the global strip | 10 |
| `common.py` | not a figure: paths and the shared machinery | 1–4 |
| `prepare_data.py` | not a figure: builds the inputs the archive does not carry | — |
| `build_derived.py` | not a figure: rebuilds the two archive-level derived products, which is what lets the record move past 2024 | — |

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
raw and FLs.52j, and their difference as `ushcn_offsets_<y0>_<y1>.nc`; it has
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

### What `build_derived.py` builds

`prepare_data.py` reads two things that are themselves derived, and the code
that produced the copies sitting on the archive is not in this repository. That
was a real dead end rather than an inconvenience: the raw ingredients for 2025
were on the drive the whole time, but nothing here could turn them into the two
files the pipeline actually opens, so the record could not leave 2024.

Both rules below were recovered — the first from the cube's own netCDF
attributes, the second partly from the offsets file's description and partly
from the data — and both are checked rather than trusted. `--verify` rebuilds
something that already exists and diffs it:

```bash
python figures/build_derived.py cube    --verify 2024
python figures/build_derived.py offsets --verify
```

The cube reproduces the archived 2024 file exactly: same station axis, same
missing pattern, `max|diff| = 0`. The offsets reproduce the archived 1900–2024
file exactly: identical counts (1,552,798 / 1,520,135 / 1,472,453 finite
station-months for TMAX / TMIN / TAVG), no cell present in one and not the
other, agreement to float32 rounding.

**`GHCND/station_daily/<year>.nc` — the QC'd CONUS daily cube.** The raw global
`by_year/<year>.csv.gz` cut down to the CONUS station set under the rules the
existing files record in their own attributes: a non-blank quality flag drops
the row, the first observation of a station-day-element wins, values stay in
GHCN's 0.1 °C integers, and the year is laid out on 366 leap-aligned slots with
slot 59 always 29 February. The station axis is *not* re-derived. Every
archived year shares one 18,128-station axis, so it is read off an existing file
and reused — which is both simpler than reconstructing the original selection
and the only way a new year stays aligned with the old ones. A station that
first reports in a new year is therefore not added; with a 100-year
completeness rule in front of it, nothing downstream would have used it.

**`USHCN_v2.5/derived/ushcn_offsets_<y0>_<y1>.nc` — the homogenization
offsets.** FLs.52j minus raw, per station/year/month, keyed by GHCN-Daily id
through `ushcn_crosswalk.csv`, scaled to °C. One rule is written down nowhere
and had to be recovered from the data: **a month whose FLs.52j value carries
measurement flag `E` is dropped.** `E` marks a value infilled from neighbours
rather than an adjustment of the station's own observation, so the difference
there is not a bias correction for that station's daily data. Keeping those
months adds about 87,000 spurious offsets to TMAX alone and does not reproduce
the archive. `common.py` resolves the newest `ushcn_offsets_*.nc` by glob, so a
longer span supersedes a shorter one without an edit.

### How far the record runs, and why

It depends on the season the figure uses, because the datasets end at different
dates. The archives were refreshed from source on 2026-09-20.

| Figure | Season | Runs to | Why it stops there |
| --- | --- | --- | --- |
| 1 | JJA | **2026** | June–August 2026 is complete in GHCN-Daily, USHCN and nCLIMDIV |
| 2 | JJA | 2025 | its bars are fixed years, so a 2026 bar would not be drawn |
| 3, 6 | May–Sep | 2025 | September 2026 is not finished |
| 5 | JJA | 2025 | ERA5 reaches JJA 2025; Berkeley, its other line, ends in 2024 |
| 4 | — | 2024 | Berkeley-only |

Only four datasets reach JJA 2026, so Figure 1's last point rests on nCLIMDIV,
GHCN-Daily, USHCN-Daily and USHCN-BC alone:

| Dataset | Ends | Reaches JJA 2026? |
| --- | --- | --- |
| GHCN-Daily (`by_year/2026.csv.gz`) | 2026-09-18 | yes |
| USHCN monthly offsets | 2026-08 | yes, thinly |
| nCLIMDIV (`-20260904`) | 2026-08 | yes |
| CRUTEM5 | 2026-07 | no, one month short |
| ERA5 | 2025-12 | no — not on the archive; a rebuild needs Copernicus |
| NOAAGlobalTemp gridded | 2025-12 | no — NCEI has published no newer file |
| Berkeley daily | 2024-08-31 | no |

**The 2026 point is provisional.** USHCN's most recent months are still filling
in — in the 2026-09-06 pull, June (three months old) had its usual ~710 stations
while July had 611 and August 460, so a month needs roughly three months to
settle. Figures 3 and 6 can take 2026 once September closes, in early October
2026, but the same argument says a stable value wants until about December.

### Refreshing USHCN moves the past, but not the answer

USHCN v2.5 reruns its pairwise homogenization from scratch on every build, so a
newer pull is not simply "the old file plus new months". Going from the
2026-09-06 to the 2026-09-19 tarballs changed **64% of historical station-month
offsets**, with a median shift of 0.03 °C and a maximum of 1.87 °C — PHA
re-detecting breakpoints across the whole record.

Almost all of it cancels in the average. The CONUS-mean TMAX adjustment moved
+0.004 °C in the 1930s and +0.004 °C over 2010–24, so the homogenization signal
the paper reports — the difference between those two — moved by **0.0007 °C**.
Figures 2, 3 and 6 came out numerically identical afterwards. Worth measuring
again on the next refresh rather than assumed, but not a reason to avoid
refreshing.

Extending the record is not only a matter of the year constants, because several
caches were keyed without a span and would have been handed back silently from
the shorter record. `TAG_F3`, `TAG_F5`, Figure 1's IDW grids and ERA5 field, and
the GHCN CONUS station-months table now all carry their year range in the cache
name, so a change of span misses the cache instead of quietly truncating.

**Berkeley Earth cannot follow.** Its daily release ends 2024-08-31: 2024 is
partial and blanked, and 2025 is absent outright. In Figure 3 that mattered more
than it looks — the record kernel counts, so a year with no data is not missing
but a hard **zero**, which would have dropped the Berkeley lines to the floor in
2025. Figure 3 now blanks every year past the last one Berkeley covers in full,
not just the partial year. Figure 6's Berkeley series are keyed by year, so an
absent year is simply not a key and needs nothing. Figure 4 is Berkeley-only and
still ends in 2024.

A dataset ending before the axis does also breaks the smoother, which is why
Figures 1 and 5 now cut each series to its own last year before calling `roll`.
`strict_interior` measures the end region from the end of the AXIS, so once the
record ran to 2025 a series stopping in 2024 had its last real year treated as
interior: it wanted a full 11-year window, reached into the empty year, and came
out as a **hole five years short of its own end with a detached loclin segment
after it** — clearly visible on Figure 5's Berkeley line and quietly present on
Figure 1's. A mid-record hole is still a hole. Figure 3 was never affected: it
runs `strict_interior` off, where every point is a local-linear fit and a
trailing gap costs nothing.

Figures 1 and 5 now trim the **head** the same way (`SMOOTH_FROM_RECORD_START`
and `SMOOTH_FROM_RECORD_START_F1`, 2026-09-23). The leading edge had the
mirror-image problem: ERA5's record starts in 1940 while the axis starts in
1900, so 1940 sat forty points into the array, was treated as interior, and the
orange line began at 1945 — five years of real data with no curve over them.
Cutting each series to its own first valid year puts the local-linear fit where
the record actually begins, and the ERA5 line starts at 1940 in both figures.

ERA5 is the only series in either figure that starts after the axis does, and
none of them carry interior gaps, so nothing else moved: Figure 1's annual CSV
is byte-identical across the change, and no smoothed value that existed before
changed by any amount. The new segment is worth looking at rather than skipping
over — ERA5's CONUS JJA TMAX over 1940–1945 sits roughly 1 °C below every
station-based dataset on the same panel, which is the largest disagreement
anywhere on Figure 1.

### Figure 5 averages all band land

Figure 5 used to draw Berkeley on the GHCN-Daily footprint: a 2° cell counted
only where **both** datasets reported it, and CONUS was randomly thinned each
year to the rest-of-band cell count so the United States could not dominate.
That made the station network, not the band, decide which cells entered the
average. It has been removed (2026-09-23). Both lines are now plain
cos(lat)-weighted means over **all land in 24–50°N**, on each dataset's own
native grid — Berkeley at 1°, ERA5 at 0.25° — with no co-sampling, no CONUS
thinning and no GHCN-Daily input at all.

Averaging natively rather than on the old 2° common grid matters little (0.01 °C
on the 1930s mean), because the 2° step gave a box that was one quarter land the
same weight as one that was all land. Dropping the co-sample matters a great
deal. Berkeley TMAX over 1930–1939 falls from **+0.57 °C to +0.19 °C**, while
2010–2024 barely moves, +1.23 to +1.27 °C: the old footprint was weighted toward
the United States, where the 1930s were exceptionally hot, so it carried a
Dust Bowl signal that the band as a whole does not have.

**ERA5 is now land-masked too.** It was previously a land+ocean band mean, which
is not comparable to a land-only record. Berkeley's daily TMAX/TMIN product
carries no data at all off its own `land_mask`, so that mask *is* Berkeley's
footprint, and it is carried onto ERA5's 0.25° grid by nearest cell centre
rather than a second, independent coastline being introduced. The two curves
then cover identical geography, and they agree: over their common 1940–2024
period, TMAX correlates at 0.979 with a +0.04 °C mean offset and a 0.13 °C RMS
difference.

Berkeley's coverage of the band's land area is reported per decade when the
script runs — 71% in the 1900s, 85% in the 1930s, 97.6% from the 1960s on — so
the part of the early record that rests on partial coverage is visible rather
than assumed. ERA5 covers 100% of that footprint from 1940.

## What changed from the notebook

Figures 2 and 3 aside — see below — the figure code is the notebook's, line for
line, and the differences are all outside the computation:

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

### Figure 2 draws JJA only

`figure2.py` is no longer section 6's full DJF/MAM/JJA/SON grid. The notebook
was left alone deliberately, so running it still gives the original four-season
figure. The script instead builds one panel per element (TMAX/TMIN/TAVG),
JJA only, and suppresses the GHCN-Daily and Berkeley-at-USHCN ("BE @ USHCN")
bars from the plot -- both series are still computed and appear in the
script's CHECK table and CSV output, they are just not drawn. The dataset-name
labels under each bar group sit on a single row rather than the notebook's
two-tier staggered layout, since JJA's smaller bar groups no longer need it.

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
   stays a gap rather than having a six-point line drawn through the boundary.
   What counts as "the record's own ends" depends on the series being trimmed to
   its own first and last valid year first — see *Smoothing to the ends of the
   record* above.
2. **The endpoint's uncertainty is reported rather than asserted.**
   `end_uncertainty()` truncates a series at every year, smooths the
   truncation, and scores its estimate against the centered mean the full record
   eventually reports there. That RMSE — 0.6 to 0.8 records/yr at the last
   year, about 0.3 two years in — is printed as a table. It is no longer drawn:
   `END_SHADE` and `END_BAND` are both off, so the panels carry the smoothed
   line alone. Figure 5 no longer has a co-sample spread to report: its two
   lines are now plain land averages (see *Figure 5 averages all band land*).
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

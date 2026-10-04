#!/usr/bin/env python3
"""Figure 5 -- heat-wave days, CONUS against the global 24-50N strip, drawn from the master annual table.

Plot-only: the columns conus_hw_days__<dataset> and band_hw_days__<dataset> of
PAPER_FIGURES_FINAL/master_annual_table.csv, which build_master_table.py writes (runs of six or more
days above a cell's own day-of-season 90th percentile, Christy (2026) method; CONUS mean, and the mean
over land in 24-50N). Only the LOWESS lines are computed here.

Columns: Berkeley Earth | USHCN (USHCN-Daily and USHCN-BC) | Reanalyses (ERA5, 20CR, ERA-20C on the same
axes). A column appears when one of its datasets is selected; USHCN has no global strip, so its bottom
panel is left empty. The top row is CONUS, the bottom row the 24-50N land strip.

    python figures/figure5_20cr_era20c.py                                  # every dataset
    python figures/figure5_20cr_era20c.py --datasets berkeley era5 20cr era20c
    python figures/figure5_20cr_era20c.py --datasets ushcn ushcn_bc --out Figure5_ushcn.png

20CR and ERA-20C daily TMAX is over each grid point's LOCAL midnight-to-midnight day. The 20CR files
are the ENSEMBLE MEAN of its 80 members, which smooths day-to-day weather, so its exceedances of its own
percentiles are not those of a single realization. Each dataset's thresholds come from its own record
(20CR 1900-2015, ERA-20C 1900-2010, ERA5 1940-2025).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *          # noqa: F401,F403
from master_table_io import parse_args, load_table, column, need, list_columns, out_path

check_smoother()
print()

# ══ SETTINGS ═══════════════════════════════════════════════════════════════
warnings.filterwarnings("ignore", category=RuntimeWarning)
ALL_F5 = ["berkeley", "ushcn", "ushcn_bc", "era5", "20cr", "era20c"]
# (key, column title, datasets drawn in it); drawn left to right
GROUPS = [("Berkeley", "Berkeley Earth", ["berkeley"]),
          ("USHCN", "USHCN", ["ushcn", "ushcn_bc"]),
          ("Reanalyses", "Reanalyses", ["era5", "20cr", "era20c"])]
LABEL = {"berkeley": "Berkeley Earth", "ushcn": "USHCN-Daily", "ushcn_bc": "USHCN-BC", "era5": "ERA5",
         "20cr": "20CR", "era20c": "ERA-20C"}
COLOR = {"berkeley": "#009E73", "ushcn": "#6D28D9", "ushcn_bc": "#6D28D9", "era5": "#F59E0B",
         "20cr": "#222222", "era20c": "#CC79A7"}
LS_ADJ     = "-"                                    # homogenised / bias-corrected / reanalysis
LS_UNCORR  = (0, (6.5, 1.8, 1.0, 1.8))              # long dash, dot -- uncorrected station data
STYLE = {"ushcn": LS_UNCORR}
NEW_LINES = ("20cr", "era20c")                      # may widen the y-scale, never narrow it
HAS_BAND = {"berkeley", "era5", "20cr", "era20c"}   # datasets with a global-strip column
SUPTITLE_F5 = "CONUS VS Northern Mid-latitude Band Heatwave Days"
PLOT_YRS = (1900, 2025)
ROW_LABELS = ["CONUS", "Global 24–50°N"]

PANEL_W, PANEL_H = 5.6, 3.25
YLIM_FROM        = "annual"        # scale to the annual traces ("smoothed" would scale to the curves)
SHOW_ANNUAL_F5   = (YLIM_FROM == "annual")
LW_SMOOTH, LW_ANNUAL, ALPHA_ANNUAL = 2.5, 0.55, 0.26
FS_COLTITLE, FS_YLABEL, FS_XLABEL = 19, 16, 16
FS_TICK, FS_EMPTY, FS_PANLEG = 14, 15, 13.5
FS_SUPTITLE = 26
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 15, "axes.titlesize": 18,
    "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 14, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.1, "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 5, "ytick.major.size": 5,
    "xtick.major.width": 1.1, "ytick.major.width": 1.1,
})

ARGS = parse_args(__doc__, ALL_F5, ALL_F5, "Figure5_20CR_ERA20C.png")
TABLE = load_table(ARGS.csv)
if ARGS.list:
    list_columns(TABLE, ["conus_hw_days__", "band_hw_days__"]); raise SystemExit
DATASETS = [d for d in ALL_F5 if d in ARGS.datasets]

# ══ THE SERIES THIS FIGURE DRAWS, FROM THE TABLE ═══════════════════════════
need(TABLE, [f"conus_hw_days__{d}" for d in DATASETS]
     + [f"band_hw_days__{d}" for d in DATASETS if d in HAS_BAND], "Figure 5")
R = [{}, {}]                                  # row -> {dataset: {year: value}}
for _d in DATASETS:
    R[0][_d] = column(TABLE, f"conus_hw_days__{_d}").dropna().to_dict()
    if _d in HAS_BAND:
        R[1][_d] = column(TABLE, f"band_hw_days__{_d}").dropna().to_dict()

# ══ CHECK -- Figure 5 series, before plotting ══════════════════════════════
def _pm_F5(d, y0, y1):
    _v = [d[y] for y in range(y0, y1 + 1) if y in d and not np.isnan(d[y])]
    return np.mean(_v) if _v else np.nan

def _recent_F5(d):
    """2015-24 where the line has at least five of those years, else its own last ten: 20CR ends in
    2015 and ERA-20C in 2010, so 'the last decade' has to be taken from their records."""
    v = [d[y] for y in range(2015, 2025) if y in d and not np.isnan(d[y])]
    if len(v) >= 5:
        return float(np.mean(v)), "2015-24"
    yy = sorted(y for y in d if not np.isnan(d[y]))[-10:]
    return float(np.mean([d[y] for y in yy])), f"{yy[0]}-{str(yy[-1])[2:]}"

print(f"{'series':<34}{'years':>7}{'1930s':>8}{'recent':>9}{'ratio':>7}  recent =")
for _row, _name in ((0, "CONUS"), (1, "Global")):
    for _d in DATASETS:
        _s = R[_row].get(_d)
        if not _s:
            continue
        _a = _pm_F5(_s, 1930, 1939); _b, _w = _recent_F5(_s)
        print(f"  {_name + ' ' + LABEL[_d]:<32}{len(_s):>7}{_a:>8.2f}{_b:>9.2f}{_a / _b:>7.2f}  {_w}")
print("\nratio above 1.00 means the 1930s had more heat-wave days than the last decade\n")

# ══ PLOT ═══════════════════════════════════════════════════════════════════
def _smooth(d):
    """common.lowess_smooth() on a {year: value} dict, returned as one: a 17-year tricube LOWESS, the same
    smooth as Figures 1, 2 and 4, running from the series' first year to its last. The dict is laid on a
    gap-free calendar first, so a year the heat-wave method left uncovered is a NaN; the smoother keeps such
    a hole (nothing is drawn across it, each side is smoothed as a record of its own)."""
    if not d: return {}
    s = pd.Series(d).sort_index()
    s = s.reindex(range(int(s.index.min()), int(s.index.max()) + 1))
    return lowess_smooth(s).dropna().to_dict()

def _draw(ax, hw, color, ls="-", label=None, thin=True):
    if not hw: return False
    y = np.array(sorted(hw.keys())); v = np.array([hw[k] for k in y], float)
    ok = (y >= PLOT_YRS[0]) & (y <= PLOT_YRS[1])
    y, v = y[ok], v[ok]
    if y.size == 0: return False
    if thin and SHOW_ANNUAL_F5:
        ax.plot(y, v, lw=LW_ANNUAL, color=color, ls=ls, alpha=ALPHA_ANNUAL)
    sm = _smooth({a: b for a, b in zip(y, v) if not np.isnan(b)})
    if sm:
        sy = np.array(sorted(sm.keys()))
        ax.plot(sy, [sm[k] for k in sy], lw=LW_SMOOTH, color=color, ls=ls,
                label=label, solid_capstyle="round", dash_capstyle="round")
    return True

def _ymax_of(series):
    """Scale to the annual traces (99th percentile x 1.2) or to the smoothed curves."""
    if YLIM_FROM == "smoothed":
        series = [_smooth(d) for d in series]
    vals = [v for d in series for v in d.values() if not np.isnan(v)]
    if not vals:
        return None
    return (np.percentile(vals, 99) * 1.20 if YLIM_FROM == "annual" else max(vals) * 1.18)

def _row_ymax(row):
    """The row is scaled from the pooled 99th percentile of the lines that are not 20CR / ERA-20C; those two
    can only RAISE it. Pooling them in would let their many low values drag the percentile down and clip
    Berkeley's last years."""
    orig = [R[row][d] for d in DATASETS if d in R[row] and d not in NEW_LINES]
    new = [R[row][d] for d in DATASETS if d in R[row] and d in NEW_LINES]
    top = [m for m in ([_ymax_of(orig)] + [_ymax_of([d]) for d in new]) if m is not None]
    return max(top) if top else 20.

COLS = [(k, title, [d for d in members if d in DATASETS]) for k, title, members in GROUPS
        if any(d in DATASETS for d in members)]
ncol = len(COLS)
fig, axes = plt.subplots(2, ncol, figsize=(PANEL_W * ncol, PANEL_H * 2.28), squeeze=False,
                         gridspec_kw={"hspace": 0.30, "wspace": 0.075})
_letters = "abcdefghijklmnopqrstuvwxyz"
_panel_i = 0                          # panel letters skip the empty panels, so they stay contiguous
for row, row_label in enumerate(ROW_LABELS):
    ymax = _row_ymax(row)
    for col in range(1, ncol):
        axes[row, col].sharey(axes[row, 0])
    for col, (key, title, members) in enumerate(COLS):
        ax = axes[row, col]
        lines = [(d, R[row][d]) for d in members if d in R[row]]
        if not lines:                 # e.g. USHCN has no global strip: leave the panel out
            ax.axis("off")
            continue
        for d, series in lines:
            _draw(ax, series, COLOR[d], STYLE.get(d, LS_ADJ), label=LABEL[d] if len(lines) > 1 else None)
        if len(lines) > 1:
            ax.legend(loc="upper left", frameon=False, fontsize=FS_PANLEG, handlelength=1.9,
                      borderaxespad=0.2, labelspacing=0.25, handletextpad=0.5)
        ax.set_xlim(*PLOT_YRS); ax.set_ylim(-1, ymax)
        ax.tick_params(axis="both", labelsize=FS_TICK, length=5, width=1.1)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
        ax.tick_params(axis="x", which="minor", length=2.5, width=0.9)
        ax.grid(True, axis="y", color="0.90", lw=0.7)
        ax.set_axisbelow(True)
        ax.axhline(0, color="0.6", lw=0.5, ls="--")
        ax.axvspan(1930, 1939, alpha=0.07, color="firebrick", zorder=0)
        # panel letter, lower right: clear of the legend (upper left) and of the curves
        ax.text(0.97, 0.05, f"({_letters[_panel_i]})", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=FS_COLTITLE, fontweight="bold")
        _panel_i += 1
        if row == 0:
            ax.set_title(title, color="black", fontweight="bold", pad=8, fontsize=FS_COLTITLE)
        if col == 0:
            ax.set_ylabel(f"{row_label}\nHW days/year", fontsize=FS_YLABEL, labelpad=8)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        if row == 1:
            ax.set_xlabel("Year", fontsize=FS_XLABEL, labelpad=6)
if SUPTITLE_F5:
    fig.suptitle(SUPTITLE_F5, fontsize=FS_SUPTITLE, fontweight="bold", y=0.985)
OUT_F5 = out_path(ARGS.out, FIGD)
fig.savefig(OUT_F5, dpi=200, bbox_inches="tight", facecolor="white")
plt.show(); plt.close(fig)

print(f"\nwrote {OUT_F5}")

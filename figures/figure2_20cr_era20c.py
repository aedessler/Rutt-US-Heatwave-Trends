#!/usr/bin/env python3
"""Figure 2 -- CONUS daily TMAX record frequency, May-September, drawn from the master annual table.

Plot-only: the columns conus_records__<dataset> of PAPER_FIGURES_FINAL/master_annual_table.csv,
which build_master_table.py writes. For each station or grid cell and each May-September calendar
day, which year holds the dataset's highest TMAX; ties are split; the value is records per year per
station or grid cell. One panel per selected dataset, two panels to a row. The heavy lines are LOWESS
curves (common.lowess_smooth), computed here; the trend, end-revision and sampling tables are too.

    python figures/figure2_20cr_era20c.py                                  # every dataset
    python figures/figure2_20cr_era20c.py --datasets berkeley 20cr era20c
    python figures/figure2_20cr_era20c.py --datasets ushcn_bc berkeley --out Figure2_two.png

Each series is counted over ITS OWN years (Berkeley 1900-2023, 20CR 1900-2015, ERA-20C 1900-2010,
the USHCN lines 1900-2025). With N years in a record a stationary climate gives every year 153/N
records per cell (1.21 at N = 126, 1.32 at 116, 1.38 at 111); the CHECK table prints that baseline
beside each line. 20CR and ERA-20C daily TMAX is over each grid point's LOCAL midnight-to-midnight
day; the 20CR files are the ENSEMBLE MEAN of its 80 members, which smooths day-to-day weather.
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
ALL_F2 = ["ushcn", "ushcn_bc", "berkeley", "berkeley_at_ushcn", "20cr", "era20c"]   # panel order
LABEL_F2 = {"ushcn": "USHCN-Daily", "ushcn_bc": "USHCN-BC", "berkeley": "Berkeley Earth",
            "berkeley_at_ushcn": "Berkeley Earth at USHCN Stations", "20cr": "20CR", "era20c": "ERA-20C",
            "ushcn_bc_x_sampling": "USHCN-BC × BE/BE-at-USHCN"}
COL_BE_F2, COL_UH = "#009E73", "#6D28D9"
COL_RA_F2 = {"20cr": "#222222", "era20c": "#CC79A7"}
LS_UNCORR_F2  = (0, (6.5, 1.8, 1.0, 1.8))   # uncorrected station data
LS_SAMP_UH    = (0, (2.6, 1.8))             # Berkeley re-sampled onto USHCN, the shorter dash
LS_CORR_F2    = (0, (4.2, 2.0))             # a station product carried to full coverage
C_SAMP_UH_F2  = "#3F9B78"                   # the re-sampled Berkeley line is a lighter green than the full grid
C_CORR_F2     = "#A78BFA"                   # USHCN's hue, lightened: same data, rescaled
PANEL_W_F2, PANEL_H_F2 = 5.6, 3.25
PANEL_YLIM_F2 = (0, 6)                      # the 1930s spikes clip
Y0_F2, Y1_F2 = 1900, 2025
NDAYS = 153                                 # May-September days; every series integrates to this


ARGS = parse_args(__doc__, ALL_F2, ALL_F2, "Figure2_20CR_ERA20C.png")
TABLE = load_table(ARGS.csv)
if ARGS.list:
    list_columns(TABLE, ["conus_records__"]); raise SystemExit
DATASETS = [d for d in ALL_F2 if d in ARGS.datasets]

# ══ THE SERIES THIS FIGURE DRAWS, FROM THE TABLE ═══════════════════════════
need(TABLE, [f"conus_records__{d}" for d in DATASETS], "Figure 2")
YEARS = np.arange(Y0_F2, Y1_F2 + 1)
SER = {LABEL_F2[d]: column(TABLE, f"conus_records__{d}").reindex(YEARS) for d in DATASETS}
CORR = None                                  # USHCN-BC carried to full coverage, drawn in the USHCN-BC panel
if "ushcn_bc" in DATASETS and "conus_records__ushcn_bc_x_sampling" in TABLE.columns:
    CORR = column(TABLE, "conus_records__ushcn_bc_x_sampling").reindex(YEARS)
    SER[LABEL_F2["ushcn_bc_x_sampling"]] = CORR

# ══ CHECK -- Figure 2 series. Each must integrate to 153: every location holds exactly one
# record per calendar day it observed (build_master_table.py asserts this while building). ══
def _recent_F2(d):
    """2015-24 where the series has at least five of those years, else its own last ten --
    the reanalyses end in 2015 and 2010, and 'last decade' must still mean something."""
    w = d.loc[2015:2024].dropna()
    if len(w) >= 5:
        return float(w.mean()), "2015-24"
    v = d.dropna().iloc[-10:]
    return float(v.mean()), f"{v.index[0]}-{str(v.index[-1])[2:]}"
def _ratio_F2(d):
    _a = float(d.loc[1930:1939].mean()); _b = _recent_F2(d)[0]
    return _a, _b, _a / _b
print(f"{'dataset':<34}{'1930s':>8}{'recent':>9}{'ratio':>7}{'integral':>10}{'  recent =':>11}{'  153/N':>8}")
for _lbl, _s in SER.items():
    _a, _b, _r = _ratio_F2(_s)
    print(f"  {_lbl:<32}{_a:>8.2f}{_b:>9.2f}{_r:>7.2f}{float(np.nansum(_s)):>10.1f}"
          f"  {_recent_F2(_s)[1]:>9}{NDAYS / int(_s.notna().sum()):>8.2f}")
print("\nratio above 1.00 means the Dust Bowl set more records than the last decade. 'recent' is 2015-24 where\n"
      "the line has it and its own last ten years where it does not (20CR ends 2015, ERA-20C 2010).\n"
      "153/N is what a stationary climate would give every year over that line's N years: the lines are not\n"
      "on one baseline. (The sampling-corrected line is a product, so its integral is not 153.)")
if {"conus_records__berkeley", "conus_records__berkeley_at_ushcn"} <= set(TABLE.columns):
    _num = lowess_smooth(column(TABLE, "conus_records__berkeley").reindex(YEARS))
    _den = lowess_smooth(column(TABLE, "conus_records__berkeley_at_ushcn").reindex(YEARS))
    be_samp_ratio = _num / _den
    print(f"\nthe USHCN sampling factor, Berkeley full grid / Berkeley at the USHCN sites"
          f"\n  1930s {float(be_samp_ratio.loc[1930:1939].mean()):.3f}   "
          f"2015-24 {float(np.nanmean(be_samp_ratio.loc[2015:2024])):.3f}   "
          f"range {np.nanmin(be_samp_ratio):.3f}-{np.nanmax(be_samp_ratio):.3f}"
          f"\n  above 1 = the USHCN footprint under-counts records relative to full coverage")

# ══ TRENDS -- the numbers the figure's paragraph quotes ════════════════════
# OLS trend on the ANNUAL series, never the smoothed one: a smooth manufactures serial correlation and
# would make every trend look far more significant than the data support. Record counts are strongly
# autocorrelated even unsmoothed -- a run of hot summers sets records together, and the metric itself is
# path dependent -- so the ordinary OLS t-test overstates significance. The effective sample size is the
# lag-1 correction of Santer et al. (2000), n_eff = n (1 - r1) / (1 + r1), applied to the regression
# residuals. Both p-values are printed: quote the adjusted one.
TREND_STARTS_F2 = (1900, 1920, 1940, 1955, 1960, 1970, 1980, 1990)

def trend_F2(s, y0=Y0_F2, y1=Y1_F2):
    """(trend per century, p_ols, p_ar1, n, r1) for the annual series."""
    from scipy import stats
    d = s.loc[y0:y1].dropna()
    n = len(d)
    if n < 10:
        return np.nan, np.nan, np.nan, n, np.nan
    x, y = d.index.values.astype(float), d.values.astype(float)
    xm, ym = x.mean(), y.mean()
    sxx = np.sum((x - xm) ** 2)
    slope = np.sum((x - xm) * (y - ym)) / sxx
    resid = y - (ym - slope * xm + slope * x)
    se = np.sqrt(np.sum(resid ** 2) / ((n - 2) * sxx))
    p_ols = float(2 * stats.t.sf(abs(slope / se), n - 2))
    r1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
    r1e = min(max(r1, 0.0), 0.99)        # only positive AR(1) inflates significance
    n_eff = n * (1 - r1e) / (1 + r1e)
    if n_eff > 2.5:
        p_ar1 = float(2 * stats.t.sf(abs(slope / (se * np.sqrt((n - 2) / (n_eff - 2)))), n_eff - 2))
    else:
        p_ar1 = 1.0
    return slope * 100.0, p_ols, p_ar1, n, r1

def _star(p):
    return "*" if np.isfinite(p) and p < 0.05 else " "

print(f"\ntrends in records per year per station or grid cell, PER CENTURY, {Y0_F2}-{Y1_F2}\n"
      f"  * = p < 0.05 after the lag-1 correction")
print(f"  {'dataset':<34}{'trend':>9}{'p(OLS)':>9}{'p(AR1)':>9}{'r1':>7}{'n':>5}")
for _lbl, _s in SER.items():
    _t, _po, _pa, _n, _r1 = trend_F2(_s)
    print(f"  {_lbl:<34}{_t:>+8.2f}{_star(_pa)}{_po:>9.3f}{_pa:>9.3f}{_r1:>7.2f}{_n:>5}")
print(f"\ntrend per century by start year, all ending {Y1_F2} (* = p < 0.05, lag-1 corrected)")
print(f"  {'dataset':<34}" + "".join(f"{y:>9}" for y in TREND_STARTS_F2))
for _lbl, _s in SER.items():
    _cells = []
    for _y0 in TREND_STARTS_F2:
        _t, _po, _pa, _n, _r1 = trend_F2(_s, _y0)
        _cells.append("       --" if not np.isfinite(_t) else f"{_t:>+8.2f}{_star(_pa)}")
    print(f"  {_lbl:<34}" + "".join(_cells))

# ══ THE SMOOTHED SERIES ════════════════════════════════════════════════════
SM = {l: lowess_smooth(s) for l, s in SER.items()}
print(f"\nLOWESS, {LOWESS_YEARS}-year tricube window, robustness iterations: {LOWESS_IT}")
print(f"  {'dataset':<34}{'data':>12}{'line':>12}{'frac':>8}")
for _lbl, _s in SER.items():
    _m = SM[_lbl]
    assert (_m.notna() == _s.notna()).all(), f"{_lbl}: the line does not span exactly the data"
    print(f"  {_lbl:<34}{_s.first_valid_index():>6}-{_s.last_valid_index():<5}"
          f"{_m.first_valid_index():>6}-{_m.last_valid_index():<5}{LOWESS_YEARS / _s.count():>8.3f}")

# The last years of ANY smooth are provisional: they rest on one side of the record and move as later
# years arrive. This measures how much, on these series.
REV = {l: end_revision(s) for l, s in SER.items()}
print(f"\nhow far the smooth's last years move as later years arrive -- RMSE against the value the\n"
      f"full record eventually gives there:")
print(f"  {'dataset':<34}" + "".join(
    f"{('last yr' if l == 0 else f'-{l}'):>9}" for l in range(len(next(iter(REV.values()))))))
for _lbl, _u in REV.items():
    print(f"  {_lbl:<34}" + "".join(f"{v:>9.2f}" for v in _u))

# Why LOWESS_IT is 0: the same smooth with statsmodels' default three robustness iterations, which read
# the 1930s spikes as outliers and pull the curve through them.
print(f"\nwhy LOWESS_IT = {LOWESS_IT}: the 1930s peak of the smooth, without and with robustness iterations")
print(f"  {'dataset':<34}{'annual':>9}{'it=0':>8}{'it=3':>8}{'lowered':>9}")
for _lbl, _s in SER.items():
    _p0 = float(lowess_smooth(_s, it=0).loc[1930:1939].max())
    _p3 = float(lowess_smooth(_s, it=3).loc[1930:1939].max())
    print(f"  {_lbl:<34}{_s.loc[1930:1939].max():>9.2f}{_p0:>8.2f}{_p3:>8.2f}{1 - _p3 / _p0:>9.0%}")

# ══ PLOT ═══════════════════════════════════════════════════════════════════
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 14, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "savefig.dpi": 300})

# One series per panel, two panels to a row: bold titles, shared y so the panels compare by eye, the annual
# trace under a heavy smoothed curve, the Dust Bowl in firebrick.
#   solid     homogenised / bias-corrected / reanalysis    USHCN-BC, Berkeley Earth, 20CR, ERA-20C
#   dot-dash  UNCORRECTED station data                     USHCN-Daily
#   dashed    re-sampled onto another network              Berkeley Earth at USHCN Stations
# (title, [(label, series, colour, linestyle, draw the annual trace too)])
PANEL_LINES = {
    "ushcn": [("USHCN-Daily", SER.get("USHCN-Daily"), COL_UH, LS_UNCORR_F2, True)],
    "ushcn_bc": [("USHCN-BC", SER.get("USHCN-BC"), COL_UH, "-", True)]
                + ([("USHCN-BC × BE/BE-at-USHCN", CORR, C_CORR_F2, LS_CORR_F2, False)] if CORR is not None else []),
    "berkeley": [("Berkeley Earth", SER.get("Berkeley Earth"), COL_BE_F2, "-", True)],
    "berkeley_at_ushcn": [("Berkeley Earth at USHCN Stations", SER.get("Berkeley Earth at USHCN Stations"),
                           C_SAMP_UH_F2, LS_SAMP_UH, True)],
    "20cr": [("20CR", SER.get("20CR"), COL_RA_F2["20cr"], "-", True)],
    "era20c": [("ERA-20C", SER.get("ERA-20C"), COL_RA_F2["era20c"], "-", True)],
}
PANELS = [(LABEL_F2[d], PANEL_LINES[d]) for d in DATASETS]
N_PANELS = len(PANELS)
NCOL = 1 if N_PANELS == 1 else 2
NROW = -(-N_PANELS // NCOL)
figp, axesp = plt.subplots(NROW, NCOL, figsize=(PANEL_W_F2 * NCOL, PANEL_H_F2 * NROW * 1.14),
                           sharey=True, squeeze=False, gridspec_kw={"hspace": 0.34, "wspace": 0.075})
for _i, _ax in enumerate(axesp.ravel()):
    if _i >= N_PANELS:
        _ax.set_visible(False)
        continue
    _title, _lines = PANELS[_i]
    _ax.axvspan(1930, 1939, color="firebrick", alpha=0.07, lw=0, zorder=0)
    for _l, _s, _c, _ls, _thin in _lines:
        if _thin:
            _ax.plot(_s.index, _s.values, color=_c, lw=0.7, ls="-", alpha=0.30, zorder=1)
        _ax.plot(_s.index, SM[_l].values, color=_c, lw=2.5, ls=_ls, zorder=3,
                 label=_l, solid_capstyle="round", dash_capstyle="round")
    _ax.set_xlim(Y0_F2 - 1, Y1_F2 + 1)
    _ax.set_ylim(*PANEL_YLIM_F2)
    _ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    _ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
    _ax.tick_params(axis="both", labelsize=14, length=5, width=1.1)
    _ax.tick_params(axis="x", which="minor", length=2.5, width=0.9)
    _ax.grid(True, axis="y", color="0.90", lw=0.7)
    _ax.grid(False, axis="x")
    _ax.set_axisbelow(True)
    _ax.set_title(_title, fontweight="bold", pad=8, fontsize=17)
    # panel letter, top right, in reading order
    _ax.text(0.985, 0.96, f"({'abcdef'[_i]})", transform=_ax.transAxes,
             ha="right", va="top", fontsize=16, fontweight="bold")
    if len(_lines) > 1:
        _h9, _l9 = _ax.get_legend_handles_labels()
        _ax.legend(_h9, _l9, loc="upper left", frameon=False, fontsize=11.5, handlelength=2.4,
                   borderaxespad=0.2, labelspacing=0.22, handletextpad=0.5, alignment="left")
    if _i % NCOL == 0:
        _ax.set_ylabel("Records per year\n(per station or grid cell)", fontsize=15, labelpad=8)
    if _i + NCOL >= N_PANELS:            # nothing below this panel in its column
        _ax.set_xlabel("Year", fontsize=15, labelpad=6)
OUT_P2 = out_path(ARGS.out, FIGD)
figp.savefig(OUT_P2, bbox_inches="tight", dpi=200, facecolor="white")
plt.show()
plt.close(figp)
print(f"\nwrote {OUT_P2}")

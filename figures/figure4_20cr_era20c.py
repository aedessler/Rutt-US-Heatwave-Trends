#!/usr/bin/env python3
"""Figure 4 -- the northern mid-latitude band, 24-50N, JJA, drawn from the master annual table.

Plot-only: the columns band_jja_{tmax,tmin,tavg}_anom__<dataset> of
PAPER_FIGURES_FINAL/master_annual_table.csv, which build_master_table.py writes. Each is the
cos(lat)-weighted JJA anomaly (deg C against 1951-1980) over ALL LAND in 24-50N at every
longitude, on the dataset's own grid, land being Berkeley's land footprint (see figure4.py's
docstring and reanalysis.berkeley_land_fraction). Only the LOWESS line is computed here.

    python figures/figure4_20cr_era20c.py                                   # every dataset
    python figures/figure4_20cr_era20c.py --datasets berkeley 20cr era20c
    python figures/figure4_20cr_era20c.py --tavg mean --out Figure4_daymean.png

20CR and ERA-20C daily statistics are over each grid point's LOCAL midnight-to-midnight day. TAVG
is (TMAX + TMIN) / 2 as for ERA5 and Berkeley unless `--tavg mean` takes the mean of the day's
samples. The 20CR files are the ENSEMBLE MEAN of its 80 members.
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
Y0_F4, Y1_F4 = 1900, 2025
DUSTBOWL      = (1930, 1940)
SHOW_ANNUAL_F4   = True                # thin annual JJA values under the smoothed curve ...
ANNUAL_SETS_F4   = {"berkeley"}        # ... for these datasets only (2026-09-21 request)
LW_ANN_F4, ALPHA_ANN_F4 = 0.9, 0.28    # Figure 1's thin-line weight and alpha
YPAD_FRAC     = 0.08                   # padding as a fraction of the drawn range
ALL_F4 = ["berkeley", "era5", "20cr", "era20c"]            # drawing order
COL_F4 = {"berkeley": "#009E73", "era5": "#D55E00", "20cr": "#222222", "era20c": "#CC79A7"}
LABEL_F4 = {"berkeley": "Berkeley Earth", "era5": "ERA5", "20cr": "20CR", "era20c": "ERA-20C"}
PANELS = [("(a) TMAX: JJA", "tmax"), ("(b) TMIN: JJA", "tmin"), ("(c) TAVG: JJA", "tavg")]

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 16, "axes.titlesize": 20, "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5, "savefig.dpi": 300,
})


def _extra(ap):
    ap.add_argument("--tavg", choices=("midrange", "mean"), default="midrange",
                    help="TAVG of 20CR / ERA-20C: (TMAX+TMIN)/2 as for ERA5 and Berkeley, or the day's mean")


ARGS = parse_args(__doc__, ALL_F4, ALL_F4, "Figure4_20CR_ERA20C.png", _extra)
TABLE = load_table(ARGS.csv)
if ARGS.list:
    list_columns(TABLE, ["band_jja_"]); raise SystemExit
DATASETS = [d for d in ALL_F4 if d in ARGS.datasets]

# ══ THE SERIES THIS FIGURE DRAWS, FROM THE TABLE ═══════════════════════════
YEARS_F4 = np.arange(Y0_F4, Y1_F4 + 1)
S = {}                                      # (dataset, element) -> annual Series on YEARS_F4
for _d in DATASETS:
    for _e in ("tmax", "tmin", "tavg"):
        _name = f"band_jja_{_e}_anom__{_d}"
        if _e == "tavg" and ARGS.tavg == "mean" and _d in ("20cr", "era20c"):
            _name = f"band_jja_tavgmean_anom__{_d}"
        need(TABLE, [_name], "Figure 4")
        S[(_d, _e)] = column(TABLE, _name).reindex(YEARS_F4)

# ══ CHECK -- Figure 4 series, before plotting. JJA anomaly over band land. ══
print(f"{'series':<28}{'1930s':>8}{'2010-24':>9}{'trend/century':>15}{'years':>7}   record")
for _d in DATASETS:
    for _e in ("tmax", "tmin", "tavg"):
        _s = S[(_d, _e)].dropna()
        _tr = np.polyfit(_s.index, _s.values, 1)[0] * 100
        print(f"  {LABEL_F4[_d] + ' ' + _e.upper():<26}{_s.reindex(range(1930, 1940)).mean():>+8.2f}"
              f"{_s.reindex(range(2010, 2025)).mean():>+9.2f}{_tr:>+15.2f}{len(_s):>7}   "
              f"{int(_s.index.min())}-{int(_s.index.max())}")
print("  (a dataset's 2010-24 mean uses the years it has: 20CR ends in 2015, ERA-20C in 2010)\n")

# ══ PLOT — smoothed curves (plus Berkeley's annual values), scaled to what is drawn ══
SM = {k: lowess_smooth(s) for k, s in S.items()}
_drawn_parts = [v.to_numpy(float) for v in SM.values()]
if SHOW_ANNUAL_F4:                    # the thin annual traces must fit on the axes too
    _drawn_parts += [S[(d, e)].to_numpy(float) for _t, e in PANELS for d in DATASETS if d in ANNUAL_SETS_F4]
_drawn = np.concatenate(_drawn_parts)
_drawn = _drawn[np.isfinite(_drawn)]
_pad = YPAD_FRAC * (np.nanmax(_drawn) - np.nanmin(_drawn))
_lo, _hi = np.nanmin(_drawn) - _pad, np.nanmax(_drawn) + _pad

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
for ax, (title, el) in zip(axes, PANELS):
    ax.axvspan(*DUSTBOWL, color="#caa472", alpha=0.18, lw=0, zorder=0)
    for d in DATASETS:
        if SHOW_ANNUAL_F4 and d in ANNUAL_SETS_F4:
            ax.plot(YEARS_F4, S[(d, el)].values, color=COL_F4[d], lw=LW_ANN_F4, alpha=ALPHA_ANN_F4, zorder=2)
        ax.plot(YEARS_F4, SM[(d, el)].values, color=COL_F4[d], lw=2.4, label=LABEL_F4[d], zorder=3)
    ax.axhline(0, color="0.2", lw=0.9, alpha=0.8, zorder=0)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=18)
    ax.set_ylabel("Anomaly (°C)")
    ax.grid(True, axis="y", color="0.85", lw=0.8); ax.grid(False, axis="x")
    ax.set_ylim(_lo, _hi); ax.set_xlim(1900, 2026)
    ax.tick_params(axis="both", labelsize=14)
axes[2].set_xlabel("Year", fontsize=17)
axes[2].xaxis.set_major_locator(mticker.MultipleLocator(20))
axes[2].xaxis.set_minor_locator(mticker.MultipleLocator(10))

h, l, seen = [], [], set()
for ax in axes:
    for hh, ll in zip(*ax.get_legend_handles_labels()):
        if ll not in seen:
            h.append(hh); l.append(ll); seen.add(ll)
fig.legend(h, l, loc="upper center", ncol=max(len(l), 1), frameon=False,
           bbox_to_anchor=(0.5, 1.06), handlelength=2.0, columnspacing=1.5, fontsize=16)
OUT_F4 = out_path(ARGS.out, FIGD)
fig.savefig(OUT_F4, bbox_inches="tight", dpi=300, facecolor="white")
plt.show()
plt.close(fig)

print(f"\nwrote {OUT_F4}")

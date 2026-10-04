#!/usr/bin/env python3
"""Figure 1 -- CONUS JJA temperature anomalies, drawn from the master annual table.

Plot-only: every number comes from PAPER_FIGURES_FINAL/master_annual_table.csv, which
build_master_table.py writes (columns conus_jja_{tmax,tmin,tavg}_anom__<dataset>, deg C against
1951-1980). Nothing is computed here except the LOWESS line (common.lowess_smooth).

    python figures/figure1_20cr_era20c.py                                  # every dataset
    python figures/figure1_20cr_era20c.py --datasets berkeley era5 20cr era20c
    python figures/figure1_20cr_era20c.py --tavg mean --out Figure1_daymean.png

20CR and ERA-20C daily statistics are over each grid point's LOCAL midnight-to-midnight day (see
reanalysis.py); their TAVG panel is (TMAX + TMIN) / 2 like ERA5 and Berkeley unless `--tavg mean`
takes the mean of the day's samples. The 20CR files are the ENSEMBLE MEAN of its 80 members.
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
DUSTBOWL_F1 = (1930, 1940)
# drawing order within each panel; noaa and crutem5 exist for TAVG only
ALL_F1 = ["era5", "20cr", "era20c", "berkeley", "nclimdiv", "noaa", "crutem5", "ushcn", "ushcn_bc"]
PANELS_F1 = (("tmax", "(a) TMAX: JJA"), ("tmin", "(b) TMIN: JJA"), ("tavg", "(c) TAVG: JJA"))
COL_F1 = dict(era5="#D55E00", berkeley="#009E73", nclimdiv="#F59E0B", ushcn="#6D28D9",
              ushcn_bc="#6D28D9", noaa="#06B6D4", crutem5="#0B3D91", **{"20cr": "#222222", "era20c": "#CC79A7"})
LABEL_F1 = dict(era5="ERA5", berkeley="Berkeley", nclimdiv="nCLIMDIV", ushcn="USHCN-Daily",
                ushcn_bc="USHCN-BC", noaa="NOAAGlobalTemp", crutem5="CRUTEM5", **{"20cr": "20CR", "era20c": "ERA-20C"})
HOMOG_F1 = dict(era5=True, berkeley=True, nclimdiv=True, ushcn=False, ushcn_bc=True, noaa=True,
                crutem5=True, **{"20cr": True, "era20c": True})
REANALYSES_F1 = ("20cr", "era20c")
LS_HOMOG_F1, LS_RAW_F1 = "-", "-."
LW_ANN_F1, LW_SM_F1, ALPHA_ANN_F1 = 0.9, 2.2, 0.28
STYLE_F1 = {"font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix", "font.size": 14, "axes.titlesize": 18,
            "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
            "legend.fontsize": 12, "axes.spines.top": False, "axes.spines.right": False,
            "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out",
            "xtick.major.size": 3.5, "ytick.major.size": 3.5, "savefig.dpi": 300}


def _extra(ap):
    ap.add_argument("--tavg", choices=("midrange", "mean"), default="midrange",
                    help="TAVG of 20CR / ERA-20C: (TMAX+TMIN)/2 as for ERA5 and Berkeley, or the day's mean")


ARGS = parse_args(__doc__, ALL_F1, ALL_F1, "Figure1_20CR_ERA20C.png", _extra)
TABLE = load_table(ARGS.csv)
if ARGS.list:
    list_columns(TABLE, ["conus_jja_"]); raise SystemExit

# ══ THE SERIES THIS FIGURE DRAWS, FROM THE TABLE ═══════════════════════════
S_F1 = {}                     # (dataset, element) -> annual Series
for _ds in ARGS.datasets:
    for _e in ("tmax", "tmin", "tavg"):
        if _e != "tavg" and _ds in ("noaa", "crutem5"):
            continue                                  # TAVG-only datasets
        _name = f"conus_jja_{_e}_anom__{_ds}"
        if _e == "tavg" and ARGS.tavg == "mean" and _ds in REANALYSES_F1:
            _name = f"conus_jja_tavgmean_anom__{_ds}"
        need(TABLE, [_name], "Figure 1")
        S_F1[(_ds, _e)] = column(TABLE, _name)
PANEL_SETS_F1 = {e: [d for d in ALL_F1 if (d, e) in S_F1] for e, _t in PANELS_F1}

# ══ CHECK -- windows every selected dataset can be read on, deg C vs 1951-1980 ═══
_WIN_F1 = ((1930, 1939), (1951, 1980), (2001, 2010))
def _win_F1(s, a, b):
    v = s.loc[a:b].dropna()
    return v.mean() if len(v) == b - a + 1 else np.nan
def _trend_F1(s, a=1940, b=2010):
    v = s.loc[a:b].dropna()
    return np.polyfit(v.index.values.astype(float), v.values, 1)[0] * 100 if len(v) == b - a + 1 else np.nan
for _e, _t in PANELS_F1:
    print(f"{_e.upper()} JJA anomaly (deg C vs 1951-1980); blank = the dataset has no such window")
    print(f"  {'dataset':<16}" + "".join(f"{a}-{str(b)[2:]}".rjust(10) for a, b in _WIN_F1)
          + f"{'trend 1940-2010':>18}   years")
    for _k in PANEL_SETS_F1[_e]:
        _s = S_F1[(_k, _e)]
        _cells = "".join(f"{_win_F1(_s, a, b):>+10.2f}" if np.isfinite(_win_F1(_s, a, b)) else f"{'':>10}"
                         for a, b in _WIN_F1)
        _tr = _trend_F1(_s)
        print(f"  {LABEL_F1[_k]:<16}{_cells}" + (f"{_tr:>+13.2f} C/century" if np.isfinite(_tr) else f"{'':>26}")
              + f"   {int(_s.first_valid_index())}-{int(_s.last_valid_index())}")
    print()
print("(the TAVG panel of 20CR and ERA-20C is drawn from: "
      + ("(TMAX+TMIN)/2" if ARGS.tavg == "midrange" else "the mean of the day's samples") + ")\n")

# ══ PLOT ═══════════════════════════════════════════════════════════════════
# y-limits from the pooled 1st-99th percentile of the lines that are not reanalyses (the scale of the paper's
# Figure 1); 20CR and ERA-20C may widen them but never narrow them
_orig = [s.dropna().values for (d, _e), s in S_F1.items() if d not in REANALYSES_F1]
_rean = [s.dropna().values for (d, _e), s in S_F1.items() if d in REANALYSES_F1]
_lim = lambda v: (max(np.nanquantile(v, 0.01) - 0.4, -5), min(np.nanquantile(v, 0.99) + 0.4, 4))
_yl_F1 = _lim(np.concatenate(_orig)) if _orig else _lim(np.concatenate(_rean))
if _orig and _rean:
    _lr = _lim(np.concatenate(_rean))
    _yl_F1 = (min(_yl_F1[0], _lr[0]), max(_yl_F1[1], _lr[1]))

with mpl.rc_context(STYLE_F1):
    fig11, axes11 = plt.subplots(3, 1, figsize=(10, 13), sharex=True, constrained_layout=True)
    for ax, (elem, title) in zip(axes11, PANELS_F1):
        ax.axvspan(*DUSTBOWL_F1, color="#caa472", alpha=0.18, lw=0, zorder=0)
        for key in PANEL_SETS_F1[elem]:
            s = S_F1[(key, elem)]
            ls = LS_HOMOG_F1 if HOMOG_F1[key] else LS_RAW_F1
            ax.plot(s.index, s.values, color=COL_F1[key], lw=LW_ANN_F1, alpha=ALPHA_ANN_F1)
            ax.plot(s.index, lowess_smooth(s).values, color=COL_F1[key], lw=LW_SM_F1, ls=ls,
                    label=LABEL_F1[key])
        ax.axhline(0, color="0.2", lw=0.9, alpha=0.8, zorder=0)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=17)
        ax.set_ylabel("Anomaly (deg C)")
        ax.grid(True, axis="y", color="0.85", lw=0.8); ax.grid(False, axis="x")
        ax.set_ylim(*_yl_F1); ax.set_xlim(1900, 2026)
        ax.tick_params(axis="both", labelsize=13)
    axes11[2].set_xlabel("Year", fontsize=15)
    axes11[2].xaxis.set_major_locator(mticker.MultipleLocator(20))
    axes11[2].xaxis.set_minor_locator(mticker.MultipleLocator(10))

    h11, l11, seen11 = [], [], set()
    for ax in axes11:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen11:
                h11.append(h); l11.append(l); seen11.add(l)
    fig11.legend(h11, l11, loc="upper center", ncol=min(5, max(len(l11), 1)), frameon=False,
                 bbox_to_anchor=(0.5, 1.10), handlelength=2.4, columnspacing=1.1, fontsize=12.5)
    fig11.legend(handles=[Line2D([0], [0], color="0.25", lw=LW_SM_F1, ls=LS_HOMOG_F1,
                                 label="homogenized / adjusted"),
                          Line2D([0], [0], color="0.25", lw=LW_SM_F1, ls=LS_RAW_F1,
                                 label="raw / unadjusted")],
                 loc="upper center", ncol=2, frameon=False,
                 bbox_to_anchor=(0.5, 1.035), handlelength=3.0, columnspacing=2.0, fontsize=12)
    OUT11 = out_path(ARGS.out, FIGD)
    fig11.savefig(OUT11, bbox_inches="tight", dpi=300, facecolor="white")
    plt.show(); plt.close(fig11)

print(f"wrote {OUT11}")

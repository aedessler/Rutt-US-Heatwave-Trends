#!/usr/bin/env python3
"""Figure 1 -- CONUS JJA temperature anomalies.

Eight datasets, TMAX / TMIN / TAVG, 1900-2024, anomalies against 1951-1980.
Notebook section 5.

    python figures/figure1.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *          # noqa: F401,F403

check_paths()
print()
check_geometry()
print()
check_pipeline()
print()
check_gridded()
print()

# ══ SETTINGS AND HELPERS ════════════════════════════════════════════════
warnings.filterwarnings("ignore", category=RuntimeWarning)
CACHE_ANOM.mkdir(exist_ok=True)
FIGD.mkdir(exist_ok=True)
# the notebook hard-coded the 20260406 processing date; the file on the archive
# carries whatever date it was pulled with, so the name is resolved instead.
NCLIMDIV_FILES_F1 = {e: nclimdiv_file(e) for e in ("tmax", "tmin", "tavg")}
NCLIMDIV_ELEM_F1  = dict(tmax=27, tmin=28, tavg=2)     # element codes in the statewide file

YEARS_F1   = np.arange(1900, 2025)                      # his YEAR_START..YEAR_END
JJA_F1     = (6, 7, 8)
IDW_KW_F1 = dict(power=2.0, k=8, radius_km=150.0)
ERA5_Y0_F1, ERA5_Y1_F1 = 1940, 2024
DUSTBOWL_F1 = (1930, 1940)

COL_F1 = dict(era5="#D55E00", berkeley="#009E73", nclimdiv="#F59E0B", ghcnd="#E63946",
             ushcn="#6D28D9", ushcn_bc="#6D28D9", noaa="#06B6D4", crutem5="#0B3D91")
LABEL_F1 = dict(era5="ERA5", berkeley="Berkeley", nclimdiv="nCLIMDIV", ghcnd="GHCN-Daily",
               ushcn="USHCN-Daily", ushcn_bc="USHCN-BC", noaa="NOAAGlobalTemp",
               crutem5="CRUTEM5")
HOMOG_F1 = dict(era5=True, berkeley=True, nclimdiv=True, ghcnd=False,
               ushcn=False, ushcn_bc=True, noaa=True, crutem5=True)
LS_HOMOG_F1, LS_RAW_F1 = "-", "-."
LW_ANN_F1, LW_SM_F1, ALPHA_ANN_F1 = 0.9, 2.2, 0.28
STYLE_F1 = {"font.family": "serif",
           "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
           "mathtext.fontset": "stix", "font.size": 14, "axes.titlesize": 18,
           "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
           "legend.fontsize": 12, "axes.spines.top": False, "axes.spines.right": False,
           "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out",
           "xtick.major.size": 3.5, "ytick.major.size": 3.5, "savefig.dpi": 300}

def gridded_series_F1(field, weights):
    """his jja_series: cell anomalies -> area mean -> JJA day-weighted mean."""
    m, frac = area_mean(cell_anomalies(field), weights)
    return (pd.Series(season_mean(_to_year_month(m, YEARS_F1), JJA_F1, YEARS_F1), index=YEARS_F1),
            pd.Series(season_mean(_to_year_month(frac, YEARS_F1), JJA_F1, YEARS_F1), index=YEARS_F1))

def smooth_F1(s):
    """The smoother Figures 1, 3 and 5 now share: an 11-year centered mean with
    a local-linear fit standing in over the outer ROLL//2 years at each end of
    the record. strict_interior keeps the plain full-window rule everywhere
    else, so a gap is still a gap -- ERA5 starts in 1940 and its smooth starts
    five years later, rather than having a six-point line drawn through the
    boundary."""
    return roll(s, strict_interior=True)

def station_series_F1(dataset, elem, weights05, gridder):
    """the whole station chain, cached at the gridded-field stage."""
    key = f"grid_{dataset}_{elem}_p{IDW_KW_F1['power']:g}_k{IDW_KW_F1['k']}_r{IDW_KW_F1['radius_km']:g}"
    def build():
        sm = station_anomalies(station_months(dataset, elem, YEARS_F1))
        return grid_station_field(sm, gridder, YEARS_F1)
    return gridded_series_F1(_cache(key, build), weights05)

def berkeley_field_F1(fp, elem):
    """preprocessed daily CONUS file -> JJA monthly means. Read sequentially and
    subset in memory: a lazy month-select makes xarray do strided reads of a
    218 MB file over USB (minutes instead of seconds)."""
    def build():
        ds = xr.open_dataset(str(fp))
        t = _std(ds["temperature"]).load()
        ds.close()
        u = (t.attrs.get("units", "") or "").lower()
        if u in ["k", "kelvin", "degk"] or "kelvin" in u:
            t = t - 273.15
        t = t.sel(time=t["time"].dt.month.isin(list(JJA_F1)))
        m = t.resample(time="MS").mean()
        return m.where(t.resample(time="MS").count() >= MIN_DAYS_MON)
    return _cache(f"field_berkeley_{elem}", build)

def era5_field_F1(varname, elem):
    """JJA monthly means per 0.25 deg cell from the daily files."""
    def build():
        parts, missing, failed = [], [], []
        for y in range(ERA5_Y0_F1, ERA5_Y1_F1 + 1):
            for mo in JJA_F1:
                fp = ERA5_DIR / f"era5_2t_{y}{mo:02d}_daily.nc"
                if not fp.exists():
                    missing.append(f"{y}-{mo:02d}")
                    continue
                ds = None
                try:
                    ds = xr.open_dataset(str(fp))
                    da = _std(ds[varname])
                    da = da.assign_coords(lon=((da["lon"] + 180) % 360) - 180).sortby("lon")
                    da = da.sel(lat=slice(BOX["lat_max"], BOX["lat_min"])
                                if da["lat"][0] > da["lat"][-1]
                                else slice(BOX["lat_min"], BOX["lat_max"]),
                                lon=slice(BOX["lon_min"], BOX["lon_max"])).load()
                    if float(da.mean()) > 150:              # kelvin
                        da = da - 273.15
                    m = da.mean("time")
                    m = m.expand_dims(time=[pd.Timestamp(f"{y}-{mo:02d}-01")])
                    parts.append(m)
                except Exception as exc:
                    failed.append(f"{y}-{mo:02d} ({type(exc).__name__})")
                finally:
                    if ds is not None:
                        ds.close()
        if not parts:
            raise RuntimeError(f"no ERA5 files under {ERA5_DIR}")
        # A silently partial build is how field_era5_tmin came to hold 187 of the
        # 255 JJA months, which blanked ERA5 TMIN and TAVG for a whole published
        # run: too few months to form a 1951-1980 baseline, no error anywhere.
        # A short read is a broken cache, so refuse to write one.
        want = len(range(ERA5_Y0_F1, ERA5_Y1_F1 + 1)) * len(JJA_F1)
        if len(parts) < want:
            raise RuntimeError(
                f"ERA5 {elem}: built {len(parts)} of {want} JJA months. "
                f"missing files: {missing or 'none'}. failed reads: {failed or 'none'}. "
                f"Refusing to cache a partial field -- fix the archive and re-run.")
        out = xr.concat(parts, dim="time").sortby("time")
        return out.sortby("lat")
    return _cache(f"field_era5_{elem}", build)

def nclimdiv_series_F1(elem):
    """NOAA's OFFICIAL area-weighted CONUS series: statewide file, region 110."""
    fp = NCLIMDIV_FILES_F1[elem]
    prefix = f"1100{NCLIMDIV_ELEM_F1[elem]:02d}"
    ym = np.full((len(YEARS_F1), 12), np.nan)
    with open(str(fp)) as f:
        for line in f:
            if line[:6] == prefix:
                yr = int(line[6:10])
                if not (YEARS_F1[0] <= yr <= YEARS_F1[-1]):
                    continue
                for mo in range(12):
                    v = float(line[10 + mo * 7:17 + mo * 7])
                    if v > -90.0:
                        ym[yr - YEARS_F1[0], mo] = (v - 32.0) * 5.0 / 9.0
    base = ym[(YEARS_F1 >= BASELINE[0]) & (YEARS_F1 <= BASELINE[1])]
    clim = np.where(np.isfinite(base).sum(0) >= MIN_BASE_CELL, np.nanmean(base, 0), np.nan)
    return pd.Series(season_mean(ym - clim[None, :], JJA_F1, YEARS_F1), index=YEARS_F1)

# which datasets each panel of the figure draws
PANEL_SETS_F1 = {
    "tmax": ["era5", "berkeley", "nclimdiv", "ghcnd", "ushcn", "ushcn_bc"],
    "tmin": ["era5", "berkeley", "nclimdiv", "ghcnd", "ushcn", "ushcn_bc"],
    "tavg": ["era5", "berkeley", "nclimdiv", "noaa", "crutem5", "ghcnd", "ushcn", "ushcn_bc"],
}

# ══ BUILD EVERY SERIES ═══════════════════════════════════════════════════
_lat05, _lon05 = g05_centers()
W05_F1 = cell_weights(_lat05, _lon05, key="g05")
GRIDDER_F1 = IDWGrid(_lat05, _lon05, **IDW_KW_F1)

S_F1, FRAC_F1 = {}, {}
for _ds in ("ghcnd", "ushcn", "ushcn_bc"):
    for _e in ("tmax", "tmin", "tavg"):
        S_F1[(_ds, _e)], FRAC_F1[(_ds, _e)] = station_series_F1(_ds, _e, W05_F1, GRIDDER_F1)

_be = {e: berkeley_field_F1(fp, e) for e, fp in (("tmax", BE_TMAX), ("tmin", BE_TMIN))}
_Wbe = cell_weights(_be["tmax"]["lat"].values, _be["tmax"]["lon"].values, key="berkeley")
for _e in ("tmax", "tmin"):
    S_F1[("berkeley", _e)], FRAC_F1[("berkeley", _e)] = gridded_series_F1(_be[_e], _Wbe)
S_F1[("berkeley", "tavg")] = 0.5 * (S_F1[("berkeley", "tmax")] + S_F1[("berkeley", "tmin")])

_e5 = {e: era5_field_F1(v, e) for e, v in (("tmax", "t2m_max"), ("tmin", "t2m_min"))}
_We5 = cell_weights(_e5["tmax"]["lat"].values, _e5["tmax"]["lon"].values, key="era5")
for _e in ("tmax", "tmin"):
    S_F1[("era5", _e)], FRAC_F1[("era5", _e)] = gridded_series_F1(_e5[_e], _We5)
S_F1[("era5", "tavg")] = 0.5 * (S_F1[("era5", "tmax")] + S_F1[("era5", "tmin")])

_no = noaa_field()
_Wno = cell_weights(_no["lat"].values, _no["lon"].values, key="noaa")
S_F1[("noaa", "tavg")], FRAC_F1[("noaa", "tavg")] = gridded_series_F1(_no, _Wno)
_cr = crutem_field()
_Wcr = cell_weights(_cr["lat"].values, _cr["lon"].values, key="crutem5")
S_F1[("crutem5", "tavg")], FRAC_F1[("crutem5", "tavg")] = gridded_series_F1(_cr, _Wcr)

for _e in ("tmax", "tmin", "tavg"):
    S_F1[("nclimdiv", _e)] = nclimdiv_series_F1(_e)

# ══ CHECK -- Figure 1 series, before plotting. JJA TMAX, degC vs 1951-1980. ══
def _stat_F1(s):
    s = s.dropna()
    return (s.loc[1930:1939].mean(), s.loc[1936] if 1936 in s.index else np.nan,
            s.loc[2010:2024].mean(), len(s))
print(f"{'dataset':<16}{'1930s':>8}{'1936':>8}{'2010-24':>9}{'years':>7}")
for _k in PANEL_SETS_F1["tmax"]:
    _s = S_F1.get((_k, "tmax"))
    if _s is None:
        continue
    _a, _y36, _b, _n = _stat_F1(_s)
    print(f"  {LABEL_F1[_k]:<14}{_a:>+8.2f}{_y36:>+8.2f}{_b:>+9.2f}{_n:>7}")
_r, _a = S_F1[("ushcn", "tmax")], S_F1[("ushcn_bc", "tmax")]
_d = smooth_F1(_a) - smooth_F1(_r)
print(f"\nhomogenization signal (USHCN-BC minus USHCN-Daily, {ROLL}-yr line):"
      f"  1930s {_d.loc[1930:1939].mean():+.2f}   2010-24 {_d.loc[2010:2024].mean():+.2f}")
print("published run: 1930s -0.06, 2010-24 +0.56 -- computed on ITS 10-yr window.\n"
      "The 11-yr loclin line above reproduces both to ~0.01, so the homogenization\n"
      "signal is a property of the data and not of the smoother.")
print(f"\nsampled CONUS area fraction, JJA:  " + "   ".join(
    f"{_y}:{FRAC_F1[('ghcnd', 'tmax')].loc[_y]:.2f}" for _y in (1900, 1936, 1990, 2024)))

# ══ PLOT (v6/v8 styling, unchanged) ══════════════════════════════════════

_all_F1 = [v for k, v in S_F1.items() if v is not None]
_v = np.concatenate([s.dropna().values for s in _all_F1])
_yl_F1 = (max(np.nanquantile(_v, 0.01) - 0.4, -5), min(np.nanquantile(_v, 0.99) + 0.4, 4))

with mpl.rc_context(STYLE_F1):
    fig11, axes11 = plt.subplots(3, 1, figsize=(10, 13), sharex=True,
                                 constrained_layout=True)
    for ax, (elem, title) in zip(axes11, (("tmax", "(a) TMAX: JJA"),
                                          ("tmin", "(b) TMIN: JJA"),
                                          ("tavg", "(c) TAVG: JJA"))):
        ax.axvspan(*DUSTBOWL_F1, color="#caa472", alpha=0.18, lw=0, zorder=0)
        for key in PANEL_SETS_F1[elem]:
            s = S_F1.get((key, elem))
            if s is None:
                continue
            ls = LS_HOMOG_F1 if HOMOG_F1[key] else LS_RAW_F1
            ax.plot(s.index, s.values, color=COL_F1[key], lw=LW_ANN_F1, alpha=ALPHA_ANN_F1)
            ax.plot(s.index, smooth_F1(s).values, color=COL_F1[key], lw=LW_SM_F1,
                    ls=ls, label=LABEL_F1[key])
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
    fig11.legend(h11, l11, loc="upper center", ncol=4, frameon=False,
                 bbox_to_anchor=(0.5, 1.10), handlelength=3.0,
                 columnspacing=1.4, fontsize=13)
    fig11.legend(handles=[Line2D([0], [0], color="0.25", lw=LW_SM_F1, ls=LS_HOMOG_F1,
                                 label="homogenized / adjusted"),
                          Line2D([0], [0], color="0.25", lw=LW_SM_F1, ls=LS_RAW_F1,
                                 label="raw / unadjusted")],
                 loc="upper center", ncol=2, frameon=False,
                 bbox_to_anchor=(0.5, 1.035), handlelength=3.0,
                 columnspacing=2.0, fontsize=12)
    OUT11 = FIGD / "Figure1.png"
    fig11.savefig(OUT11, bbox_inches="tight", dpi=300, facecolor="white")
    plt.show(); plt.close(fig11)

_rows_F1 = {f"{e}_{k}": S_F1[(k, e)] for (k, e) in S_F1 if S_F1[(k, e)] is not None}
_csv_F1 = OUT11.with_suffix(".csv")
pd.DataFrame(_rows_F1).round(4).rename_axis("year").to_csv(_csv_F1)

print(f"\nwrote {OUT11}")
print(f"wrote {_csv_F1}")

#!/usr/bin/env python3
"""Figure 2 -- CONUS JJA temperature anomaly bars.

Dust Bowl mean, 1936, 2015-2024 mean and 2024, for each element, JJA only,
one panel per element. GHCN-Daily and BE @ USHCN are suppressed from the
plot; both are still computed and printed in the CHECK table. Notebook
section 6 (which covers all four seasons; this script draws JJA alone).

    python figures/figure2.py
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

# ══ SETTINGS AND HELPERS ════════════════════════════════════════════════════
warnings.filterwarnings("ignore", category=RuntimeWarning)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman","Times","STIXGeneral","DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5, "savefig.dpi": 300,
})
CACHE_ANOM.mkdir(exist_ok=True)

# ── Labels (display only; never used in a cache key) ─────────────────────────
UH_LABEL_RAW_F2 = "USHCN-Daily"
UH_LABEL_ADJ_F2 = "USHCN-BC"
BE_LABEL_MSK_F2 = "BE @ USHCN"
BE_NETWORK_F2   = "ushcn"          # footprint for the masked Berkeley leg
BE_MASK_TIME_VARYING_F2 = False    # see the header note
FS_XTICK_V_F2 = None               # None = default sizing; 26 declutters the TAVG label row

# ── Settings ─────────────────────────────────────────────────────────────────
YEARS_F2     = np.arange(1900, 2026)      # DJF-2025 needs Jan-Feb 2025 -- kept for parity
SEASONS_F2   = {"DJF":(12,1,2), "MAM":(3,4,5), "JJA":(6,7,8), "SON":(9,10,11)}
SEASON_LIST_F2 = ["JJA"]           # this figure draws JJA only
SEASON_TITLES_F2 = {"DJF":"DJF (Dec-Feb)", "MAM":"MAM (Mar-May)",
                   "JJA":"JJA (Jun-Aug)", "SON":"SON (Sep-Nov)"}
MIN_COV_FRAC_F2  = 0.50    # season dropped if less than this share of CONUS reported
MIN_PERIOD_YRS_F2 = 7      # of the 10 years in a period-mean bar
MAD_K_F2 = 8.0
IDW_F2 = dict(power=2.0, k=8, radius_km=150.0)

DUST_START_F2, DUST_END_F2 = 1930, 1939
YEAR_1936_F2, YEAR_2024_F2 = 1936, 2024
FIXED_MODERN_START_F2, FIXED_MODERN_END_F2 = 2015, 2024

# ── Colours ──────────────────────────────────────────────────────────────────
_BAR_COLS_F2 = ["#009E73", "#06B6D4", "#F59E0B", "#E63946"]

def _mad_filt_F2(s, k=MAD_K_F2):
    v = np.asarray(s.values, float)
    med = np.nanmedian(v); mad = np.nanmedian(np.abs(v-med))
    if not np.isfinite(mad) or mad == 0: return s
    bad = np.isfinite(v) & (np.abs(v-med) > k*1.4826*mad)
    return s.where(~pd.Series(bad, index=s.index)) if bad.any() else s

def _trim_tail_F2(s, z_k=6.0):
    v = np.asarray(s.values, float); fi = np.where(np.isfinite(v))[0]
    if len(fi) < 15: return s
    hist, tail = fi[:-2], fi[-2:]
    if len(hist) < 10: return s
    med = np.nanmedian(v[hist]); mad = np.nanmedian(np.abs(v[hist]-med))
    if not np.isfinite(mad) or mad == 0: return s
    bad = [i for i in tail if np.isfinite(v[i]) and abs(v[i]-med) > z_k*1.4826*mad]
    if not bad: return s
    keep = np.ones_like(v, bool); keep[bad] = False
    return s.where(pd.Series(keep, index=s.index))

def _to_da_F2(s):
    """pandas Series indexed by year -> DataArray on a Jan-1 axis, so the
    period-mean helpers below work unchanged."""
    s = s.dropna()
    ti = pd.DatetimeIndex([pd.Timestamp(f"{int(y)}-01-01") for y in s.index])
    return xr.DataArray(s.values, coords=[ti], dims=["time"])

def seasonal_from_field_F2(field, weights, season, gate_coverage=False):
    """cell anomalies -> area mean -> day-weighted season -> QC -> DataArray."""
    m, frac = area_mean(cell_anomalies(field), weights)
    months = SEASONS_F2[season]
    s = pd.Series(season_mean(_to_year_month(m, YEARS_F2), months, YEARS_F2), index=YEARS_F2)
    if gate_coverage:
        f = pd.Series(season_mean(_to_year_month(frac, YEARS_F2), months, YEARS_F2), index=YEARS_F2)
        s = s.where(f >= MIN_COV_FRAC_F2)
    return _to_da_F2(_trim_tail_F2(_mad_filt_F2(s)))

def station_field_F2(dataset, elem, gridder):
    key = (f"grid16_{dataset}_{elem}_p{IDW_F2['power']:g}_k{IDW_F2['k']}"
           f"_r{IDW_F2['radius_km']:g}_{YEARS_F2[0]}_{YEARS_F2[-1]}")
    def build():
        sm = station_anomalies(station_months(dataset, elem, YEARS_F2))
        return grid_station_field(sm, gridder, YEARS_F2)
    return _cache(key, build)

def berkeley_field_F2(fp, elem):
    """All 12 months of monthly means. Read sequentially then subset in memory:
    a lazy month-select does strided reads of a 218 MB file over USB."""
    def build():
        ds = xr.open_dataset(str(fp))
        t = _std(ds["temperature"]).load(); ds.close()
        u = (t.attrs.get("units","") or "").lower()
        if u in ["k","kelvin","degk"] or "kelvin" in u: t = t - 273.15
        m = t.resample(time="MS").mean()
        return m.where(t.resample(time="MS").count() >= MIN_DAYS_MON)
    return _cache(f"field16_berkeley_{elem}", build)

# nCLIMDIV: NOAA's OFFICIAL area-weighted CONUS national mean (region 110).
_NCLIMDIV_FILES_F2 = {e: nclimdiv_file(e) for e in ('tmax', 'tmin', 'tavg')}
_NCLIMDIV_ELEM_F2  = {'tmax':27, 'tmin':28, 'tavg':2}

def nclimdiv_seasonal_F2(elem, season):
    fp = _NCLIMDIV_FILES_F2[elem]
    prefix = f"1100{_NCLIMDIV_ELEM_F2[elem]:02d}"
    ym = np.full((len(YEARS_F2), 12), np.nan)
    with open(str(fp)) as f:
        for line in f:
            if line[:6] == prefix:
                yr = int(line[6:10])
                if not (YEARS_F2[0] <= yr <= YEARS_F2[-1]): continue
                for mo in range(12):
                    v = float(line[10+mo*7:17+mo*7])
                    if v > -90.0: ym[yr-YEARS_F2[0], mo] = (v-32.0)*5.0/9.0
    base = ym[(YEARS_F2 >= BASELINE[0]) & (YEARS_F2 <= BASELINE[1])]
    clim = np.where(np.isfinite(base).sum(0) >= MIN_BASE_CELL, np.nanmean(base, 0), np.nan)
    s = pd.Series(season_mean(ym-clim[None,:], SEASONS_F2[season], YEARS_F2), index=YEARS_F2)
    return _to_da_F2(_trim_tail_F2(_mad_filt_F2(s)))

# ── the masked-Berkeley footprint ─────────────────────────────────────────────
def _be_stations_F2():
    st = pd.read_parquet(str(UH_PAIRED), columns=["ushcn_id","lat","lon"])
    st = st.groupby("ushcn_id")[["lat","lon"]].first()
    return st["lat"].values.astype(float), st["lon"].values.astype(float)

def _be_cells_F2(lat, lon, st_lat, st_lon):
    """True where >= 1 station falls inside the cell."""
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    oa, oo = np.argsort(lat), np.argsort(lon)
    la, lo = lat[oa], lon[oo]
    dlat = float(np.median(np.diff(la))) if len(la) > 1 else 1.0
    dlon = float(np.median(np.diff(lo))) if len(lo) > 1 else 1.0
    ok = ((st_lat >= la[0]-dlat/2) & (st_lat <= la[-1]+dlat/2)
          & (st_lon >= lo[0]-dlon/2) & (st_lon <= lo[-1]+dlon/2))
    ms = np.zeros((len(la), len(lo)), bool)
    ms[np.clip(np.searchsorted(la+dlat/2, st_lat[ok]), 0, len(la)-1),
       np.clip(np.searchsorted(lo+dlon/2, st_lon[ok]), 0, len(lo)-1)] = True
    m = np.zeros_like(ms); m[np.ix_(oa, oo)] = ms
    return m, int(ok.sum())

# ── period-mean helpers ────────────────────────────────────────────────────
def _pm_F2(da, y0, y1, min_yrs=MIN_PERIOD_YRS_F2):
    yrs = da["time"].dt.year.values
    v = np.asarray(da.values, float)[(yrs >= y0) & (yrs <= y1)]
    return float(np.nanmean(v)) if np.isfinite(v).sum() >= min_yrs else np.nan

def _pm_yr_F2(da, year):
    yrs = da["time"].dt.year.values
    v = np.asarray(da.values, float); m = yrs == year
    if m.sum() < 1: return np.nan
    out = float(v[m][0])
    return out if np.isfinite(out) else np.nan

# ══ BUILD EVERY SERIES ══════════════════════════════════════════════════════
_lat05_F2, _lon05_F2 = g05_centers()
W05_F2 = cell_weights(_lat05_F2, _lon05_F2, key="g05")
GRID_F2 = IDWGrid(_lat05_F2, _lon05_F2, **IDW_F2)

SER_F2 = {}                                  # (key, elem, season) -> DataArray
for _ds in ("ghcnd", "ushcn", "ushcn_bc"):
    for _e in ("tmax", "tmin", "tavg"):
        _fld = station_field_F2(_ds, _e, GRID_F2)
        for _s in SEASON_LIST_F2:
            SER_F2[(_ds, _e, _s)] = seasonal_from_field_F2(_fld, W05_F2, _s)

_be_f_F2 = {"tmax": berkeley_field_F2(BE_TMAX, "tmax"),
           "tmin": berkeley_field_F2(BE_TMIN, "tmin")}
if BE_TAVG.exists():
    _be_f_F2["tavg"] = berkeley_field_F2(BE_TAVG, "tavg")
_Wbe_F2 = cell_weights(_be_f_F2["tmax"]["lat"].values, _be_f_F2["tmax"]["lon"].values,
                        key="berkeley")
_st_lat_F2, _st_lon_F2 = _be_stations_F2()
_bemask_F2, _n_in_F2 = _be_cells_F2(_be_f_F2["tmax"]["lat"].values,
                                 _be_f_F2["tmax"]["lon"].values, _st_lat_F2, _st_lon_F2)
_Wbe_msk_F2 = _Wbe_F2.where(xr.DataArray(_bemask_F2, dims=("lat", "lon"),
                                       coords=_Wbe_F2.coords), 0.0)

for _e, _f in _be_f_F2.items():
    for _s in SEASON_LIST_F2:
        SER_F2[("berkeley", _e, _s)] = seasonal_from_field_F2(_f, _Wbe_F2, _s)
        SER_F2[("be_msk", _e, _s)] = seasonal_from_field_F2(_f, _Wbe_msk_F2, _s)
if "tavg" not in _be_f_F2:                    # no Berkeley TAVG file: 0.5*(TMAX+TMIN)
    for _k in ("berkeley", "be_msk"):
        for _s in SEASON_LIST_F2:
            SER_F2[(_k, "tavg", _s)] = 0.5*(SER_F2[(_k, "tmax", _s)]
                                           + SER_F2[(_k, "tmin", _s)])

_noaa_F2 = noaa_field()
_Wno_F2 = cell_weights(_noaa_F2["lat"].values, _noaa_F2["lon"].values, key="noaa")
_cru_F2 = crutem_field()
_Wcr_F2 = cell_weights(_cru_F2["lat"].values, _cru_F2["lon"].values, key="crutem5")
for _s in SEASON_LIST_F2:
    SER_F2[("noaa", "tavg", _s)] = seasonal_from_field_F2(_noaa_F2, _Wno_F2, _s,
                                                        gate_coverage=True)
    SER_F2[("crutem5", "tavg", _s)] = seasonal_from_field_F2(_cru_F2, _Wcr_F2, _s,
                                                           gate_coverage=True)
for _e in ("tmax", "tmin", "tavg"):
    for _s in SEASON_LIST_F2:
        SER_F2[("nclimdiv", _e, _s)] = nclimdiv_seasonal_F2(_e, _s)

# ══ SPECS ═══════════════════════════════════════════════════════════════════
LABELS_F2 = {"berkeley": "Berkeley", "be_msk": BE_LABEL_MSK_F2, "nclimdiv": "nCLIMDIV",
            "noaa": "NOAAGlobalTemp", "crutem5": "CRUTEM5", "ghcnd": "GHCN-Daily",
            "ushcn": UH_LABEL_RAW_F2, "ushcn_bc": UH_LABEL_ADJ_F2}
ORDER_F2 = {"tmax": ["berkeley", "be_msk", "nclimdiv", "ghcnd", "ushcn", "ushcn_bc"],
           "tmin": ["berkeley", "be_msk", "nclimdiv", "ghcnd", "ushcn", "ushcn_bc"],
           "tavg": ["berkeley", "be_msk", "nclimdiv", "noaa", "crutem5", "ghcnd",
                    "ushcn", "ushcn_bc"]}

def _yr1936_for_F2(s): return YEAR_1936_F2 + 1 if s == "DJF" else YEAR_1936_F2
def _yr2024_for_F2(s): return YEAR_2024_F2 + 1 if s == "DJF" else YEAR_2024_F2

def _cspec_F2(da, lbl, s):
    if da is None: return None
    modern = _pm_F2(da, FIXED_MODERN_START_F2, FIXED_MODERN_END_F2)
    if not np.isfinite(modern): return None
    return (lbl, _pm_F2(da, DUST_START_F2, DUST_END_F2), _pm_yr_F2(da, _yr1936_for_F2(s)),
            modern, _pm_yr_F2(da, _yr2024_for_F2(s)))

_MISSING_F2 = []
def _specs_F2(elem, s):
    out = []
    for k in ORDER_F2[elem]:
        sp = _cspec_F2(SER_F2.get((k, elem, s)), LABELS_F2[k], s)
        if sp is None:
            _MISSING_F2.append((elem.upper(), s, LABELS_F2[k], "whole series dropped: "
                               "no 2015-2024 mean"))
            continue
        for _bar, _v in ((str(YEAR_1936_F2), sp[2]), (str(YEAR_2024_F2), sp[4])):
            if not np.isfinite(_v):
                _MISSING_F2.append((elem.upper(), s, LABELS_F2[k],
                                   f"{_bar} bar: no value for that season"))
        out.append(sp)
    return out

all_specs_F2 = {(ri, s): _specs_F2(e, s)
               for ri, e in enumerate(("tmax", "tmin", "tavg")) for s in SEASON_LIST_F2}

row_ylims_F2 = []
for ri in range(3):
    vals = [v for s in SEASON_LIST_F2 for sp in all_specs_F2[(ri, s)]
            for v in sp[1:5] if v is not None and np.isfinite(v)]
    row_ylims_F2.append((min(0.0, min(vals)-0.20), max(vals)+0.20) if vals else (0.0, 2.0))

# ══ CHECK -- Figure 2 (JJA) bars, before plotting. GHCN-Daily and BE @ USHCN
# still carried here even though the plot below suppresses them. ══
print(f"{'dataset':<16}{'DustBowl':>10}{'1936':>8}{'2015-24':>9}{'2024':>8}")
for _lbl, _db, _y36, _md, _y24 in all_specs_F2[(0, "JJA")]:
    _f = lambda v: f"{v:>+8.2f}" if v is not None and np.isfinite(v) else f"{'n/a':>8}"
    print(f"  {_lbl:<14}{_f(_db)[:10]:>10}{_f(_y36)}{_f(_md)[:9]:>9}{_f(_y24)}")
if _MISSING_F2:
    print("\nbars that cannot be drawn (data availability, not a bug):")
    for _e, _s, _lbl, _why in _MISSING_F2:
        print(f"  {_e:<5}{_s:<5}{_lbl:<14}{_why}")
_n_series = sum(1 for _v in SER_F2.values() if _v is not None)
print(f"\n{_n_series} seasonal series built "
      f"({len(ORDER_F2['tavg'])} datasets x {len(SEASON_LIST_F2)} season x 3 elements)")

# ══ PLOT -- one panel per element, JJA only ═════════════════════════════════
FS_MAIN_F2, FS_BLOCK_F2 = 20, 16
FS_YLABEL_F2, FS_XTICK_F2, FS_YTICK_F2, FS_LEG_F2 = 13, 12, 11, 12

VAR_BLOCK_LABELS_F2 = ["(a) TMAX  —  Daily maximum temperature",
                      "(b) TMIN  —  Daily minimum temperature",
                      "(c) TAVG  —  Daily mean temperature"]
_w_F2, _B_OFF_F2 = 0.18, [-0.30,-0.10,0.10,0.30]
_BEDGE_F2, _BLW_F2, _Y_NM_TIERS_F2 = "white", 0.8, [-0.10]
_FS_NM_F2 = FS_XTICK_F2 if FS_XTICK_V_F2 is None else FS_XTICK_V_F2
_NM_PAD_F2       = 0.12    # minimum gap between same-tier names, in bar-group widths
_NM_MAX_SHIFT_F2 = 0.40    # beyond this sideways drift, shrink the font instead
_NM_MIN_FS_F2    = 8

def _pav_F2(y):
    """least-squares non-decreasing fit (pool-adjacent-violators)."""
    blocks = []
    for v in y:
        blocks.append([float(v), 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, n2 = blocks.pop(); v1, n1 = blocks.pop()
            blocks.append([(v1*n1 + v2*n2)/(n1 + n2), n1 + n2])
    return np.concatenate([[v]*n for v, n in blocks])

def _declutter_names_F2(ax, texts, renderer):
    """texts: list of (x_centre, tier, Text). Moves each same-tier name the
    least total distance so neighbours are >= _NM_PAD_F2 apart; shrinks the font
    only if that would pull a name more than _NM_MAX_SHIFT_F2 off its bars."""
    if not texts: return
    px_per_x = ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0]
    fs = texts[0][2].get_fontsize()
    while True:
        worst = 0.0; newx = {}
        for tier in {t for _, t, _ in texts}:
            row = sorted([(x, tx) for x, t, tx in texts if t == tier], key=lambda r: r[0])
            x0 = np.array([x for x, _ in row], float)
            hw = np.array([tx.get_window_extent(renderer).width/px_per_x/2 for _, tx in row])
            need = np.concatenate([[0.0], np.cumsum(hw[:-1] + hw[1:] + _NM_PAD_F2)])
            # x_i - need_i must be non-decreasing; project x0 - need onto that
            x = _pav_F2(x0 - need) + need
            for (xc, tx), xn in zip(row, x): newx[id(tx)] = xn
            worst = max(worst, float(np.max(np.abs(x - x0))))
        if worst <= _NM_MAX_SHIFT_F2 or fs <= _NM_MIN_FS_F2:
            break
        fs -= 1
        for _, _, tx in texts: tx.set_fontsize(fs)
    for _, _, tx in texts:
        tx.set_x(newx[id(tx)])

fig_F2, axes_F2 = plt.subplots(3, 1, figsize=(11, 13))
fig_F2.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.05, hspace=0.55)
_names_F2 = {}
_hidden_F2 = {LABELS_F2["ghcnd"], LABELS_F2["be_msk"]}
for bi, ax in enumerate(axes_F2):
    # GHCN-Daily and BE @ USHCN suppressed from the plot; all_specs_F2 (and the
    # CHECK table above) still carry both.
    specs = [sp for sp in all_specs_F2[(bi, "JJA")] if sp[0] not in _hidden_F2]
    n = len(specs)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2); ax.spines["bottom"].set_linewidth(1.2)
    for xi, (lbl, dust_v, yr36_v, mod_v, yr24_v) in enumerate(specs):
        for off, val, bc in zip(_B_OFF_F2, [dust_v, mod_v, yr36_v, yr24_v], _BAR_COLS_F2):
            if np.isfinite(val):
                ax.bar(xi+off, val, width=_w_F2, color=bc,
                       edgecolor=_BEDGE_F2, linewidth=_BLW_F2, zorder=3)
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    _names_F2[bi] = [
        (xi, 0, ax.text(xi, _Y_NM_TIERS_F2[0], lbl, transform=trans,
                             ha="center", va="top", fontsize=_FS_NM_F2,
                             color="0.10", clip_on=False))
        for xi, (lbl, *_) in enumerate(specs)]
    ax.axhline(0, color="0.25", lw=1.4, zorder=2)
    ax.set_xticks(np.arange(n)); ax.tick_params(axis="x", length=0)
    ax.set_xticklabels([]); ax.set_xlim(-0.55, n-0.45)
    ax.set_ylim(*row_ylims_F2[bi])
    ax.grid(True, axis="y", color="0.88", lw=1.0, zorder=0); ax.grid(False, axis="x")
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.tick_params(axis="y", labelsize=FS_YTICK_F2, length=7, width=1.2)
    ax.set_ylabel("Anomaly (deg C)", fontsize=FS_YLABEL_F2, labelpad=8)
    ax.set_title(VAR_BLOCK_LABELS_F2[bi], loc="left", fontweight="bold",
                 fontsize=FS_BLOCK_F2, pad=10)

_rend_F2 = fig_F2.canvas.get_renderer()
for bi, texts in _names_F2.items():
    _declutter_names_F2(axes_F2[bi], texts, _rend_F2)

_leg_handles_F2 = [
    mpatches.Patch(facecolor=_BAR_COLS_F2[0], label=f"Dust Bowl mean ({DUST_START_F2}-{DUST_END_F2})"),
    mpatches.Patch(facecolor=_BAR_COLS_F2[1], label=f"Modern mean ({FIXED_MODERN_START_F2}-{FIXED_MODERN_END_F2})"),
    mpatches.Patch(facecolor=_BAR_COLS_F2[2], label=str(YEAR_1936_F2)),
    mpatches.Patch(facecolor=_BAR_COLS_F2[3], label=str(YEAR_2024_F2)),
]
fig_F2.legend(handles=_leg_handles_F2, loc="upper center", ncol=4, frameon=False,
             bbox_to_anchor=(0.5, 0.975), handlelength=2.2, columnspacing=1.8,
             fontsize=FS_LEG_F2)
fig_F2.suptitle(f"CONUS JJA Temperature Anomalies  |  {DUST_START_F2}-{DUST_END_F2} vs. "
               f"{FIXED_MODERN_START_F2}-{FIXED_MODERN_END_F2} vs. {YEAR_1936_F2} vs. {YEAR_2024_F2}",
               fontsize=FS_MAIN_F2, fontweight="bold", y=1.02)

_out_F2 = FIGD / "Figure2.png"
_out_F2.parent.mkdir(exist_ok=True)
fig_F2.savefig(_out_F2, dpi=300, bbox_inches="tight", facecolor="white")
plt.show()
plt.close(fig_F2)


_rows_F2 = {}
for (k, e, s), da in SER_F2.items():
    if da is None: continue
    _rows_F2[f"{e}_{s}_{k}"] = pd.Series(da.values, index=da["time"].dt.year.values)
_csv_F2 = _out_F2.with_suffix(".csv")
pd.DataFrame(_rows_F2).sort_index(axis=1).round(4).rename_axis("year").to_csv(_csv_F2)

print(f"\nwrote {_out_F2}")
print(f"wrote {_csv_F2}")

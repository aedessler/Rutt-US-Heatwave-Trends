#!/usr/bin/env python3
"""Figure 6 -- heat-wave days, CONUS against the global 24-50N strip.

Runs of six or more days above a cell's own day-of-season 90th percentile,
Christy (2026) method. Notebook section 10.

The GHCN-Daily CONUS line reads the May-September station cube; in the notebook
that cube had to be built by section 7 first, here it is built on demand by
common.ghcnd_daily_cube() and shares Figure 3's cache.

    python figures/figure6.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *          # noqa: F401,F403

check_paths()
print()
check_geometry()
print()

# ══ SETTINGS AND HELPERS ═══════════════════════════════════════════════════
warnings.filterwarnings("ignore", category=RuntimeWarning)
CACHE_HW.mkdir(exist_ok=True)
F3_TAG      = TAG_F3            # same cube, same cache, as notebook section 7
F3_DAILY_FP = DAILY_FP

UH_YR0, UH_YR1 = 1900, 2024
UH_LAT_BNDS, UH_LON_BNDS = (24.0, 50.0), (-130.0, -60.0)
USHCN_LEGS = ("raw", "adj")

GHCND_CONUS_SOURCE = "stations"   # "stations" (fixed) | "christy_grid" (old)
COMPARE_BOTH_GHCND = False
BLANK_BE_PARTIAL   = True
MIN_THRESH_N       = 30
SMOOTH_W           = 5
USREG_MODE         = "polygon"

# The paper's copy of this figure carries a title above the panel grid; the
# notebook's plot block did not draw one. Set to None to get the bare grid.
SUPTITLE_F6 = "CONUS VS Northern Mid-latitude Band Heatwave Days"

# (A) which rows draw GHCN-Daily as uncorrected (dot-dashed).
#     (0, 1) = both rows, one style per dataset;  (0,) = top row only, literally
#     what the advisor asked for.
GHCND_UNCORR_ROWS = (0, 1)

# (W1)(W5) panel geometry and how the y-axis is scaled.
#   PANEL_W x PANEL_H is the box each subplot gets, in inches. 5.6 x 3.25 is
#   about 1.7:1; v G was effectively 3.9 x 3.4, i.e. square.
#   YLIM_FROM "annual"   -- scale to the annual traces, as v G did
#             "smoothed" -- scale to the smoothed curves, ~35% taller, and hide
#                           the annual traces so that nothing is clipped
PANEL_W, PANEL_H = 5.6, 3.25
YLIM_FROM        = "smoothed"
SHOW_ANNUAL_F6      = (YLIM_FROM == "annual")
LW_SMOOTH, LW_ANNUAL, ALPHA_ANNUAL = 2.5, 0.55, 0.26      # (W2)

# (B) Berkeley re-sampled onto a station network's footprint, Figure 3 style.
#     Each entry is a network whose station positions define the units.
BE_AT_NETWORKS   = ("ushcn",)
DRAW_BE_AT_USHCN = True         # the dashed line in the Berkeley CONUS panel
PLOT_BE_NETWORKS = False         # the extra GHCN-vs-USHCN comparison figure

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
FS_COLTITLE, FS_YLABEL, FS_XLABEL = 19, 16, 16
FS_TICK, FS_EMPTY, FS_PANLEG = 14, 15, 13.5
FS_SUPTITLE = 26
C = {"Berkeley": "#009E73", "GHCND": "#E63946", "ERA5": "#F59E0B",
     "USHCN": "#6D28D9"}
# (W3) the re-sampled Berkeley line is a DARKER GREEN, not the same green. Two
# 2.5 pt lines in one colour merge into a single blob wherever they cross, which
# is most of the record; same hue keeps it reading as Berkeley, the darker value
# keeps it readable as a separate line.
C_SAMPLED = "#00513C"
# (A) one style vocabulary for the whole figure:
#   solid     homogenised / bias-corrected / reanalysis
#   dot-dash  UNCORRECTED station data (GHCN-Daily, USHCN-Daily raw)
#   dashed    a corrected product re-sampled onto another network's footprint
# (W3) spelled-out dash patterns: at this panel width a default "-." and "--"
# are easy to confuse, and telling them apart is the whole point of the figure.
LS_ADJ     = "-"
LS_UNCORR  = (0, (6.5, 1.8, 1.0, 1.8))     # long dash, dot -- uncorrected
LS_SAMPLED = (0, (7.0, 2.4))               # long open dash -- re-sampled
LS_RAW = LS_UNCORR                      # v F name kept so nothing downstream breaks

_PCTILE, _WINDOW, _MIN_RUN, _MIN_FRAC = 90, 3, 6, 0.70
_HW_MON, _N_DOY = {5, 6, 7, 8, 9}, 153

def mask_conus(lat, lon, key):
    fp = CACHE_HW / f"maskconus_{key}.npy"
    if fp.exists(): return np.load(fp)
    m = points_in_grid(conus_polygon(), lat, lon); np.save(fp, m); return m

def mask_land(lat, lon, key):
    fp = CACHE_HW / f"maskland_{key}.npy"
    if fp.exists(): return np.load(fp)
    m = points_in_grid(land_polygon(), lat, lon); np.save(fp, m); return m

def _jaccard(a, b):
    u = int((a | b).sum())
    return float((a & b).sum() / u) if u else 0.0

# ══ CORE ALGORITHM ═════════════════════════════════════════════════════════
def _season_doy(times):
    """Day of the May-September season, 1..153."""
    ts = pd.DatetimeIndex(times)
    return (ts.dayofyear.to_numpy() - np.where(ts.is_leap_year, 121, 120)).astype(np.int16)

def _thresholds(tx_s, doys):
    T, nlat, nlon = tx_s.shape
    thr = np.full((_N_DOY, nlat, nlon), np.nan, dtype=np.float32)
    doys = np.asarray(doys, np.int16)
    for doy in range(1, _N_DOY + 1):
        m = np.abs(doys - doy) <= _WINDOW
        vals = tx_s[m]
        if vals.shape[0] == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = np.nanpercentile(vals, _PCTILE, axis=0)
        n = np.isfinite(vals).sum(axis=0)          # (7)
        thr[doy - 1] = np.where(n >= MIN_THRESH_N, p, np.nan)
    return thr

def _count_hw(exceed):
    T = exceed.shape[0]
    exc = exceed.astype(np.int16)
    rl = np.zeros_like(exc); rl[0] = exc[0]
    for t in range(1, T):
        rl[t] = np.where(exc[t], rl[t - 1] + 1, 0)
    rt = rl.copy()
    for t in range(T - 2, -1, -1):
        same = (exc[t] == 1) & (rl[t + 1] == rl[t] + 1)
        rt[t] = np.where(same, rt[t + 1], rt[t])
    return np.sum(exceed & (rt >= _MIN_RUN), axis=0).astype(np.float32)

def _assert_contiguous(dy, yr, label):
    """(8) runs are only meaningful if a year's days are ordered and unbroken."""
    for y in np.unique(yr):
        d = dy[yr == y]
        if d.size < 2:
            continue
        if not np.all(np.diff(d) == 1):
            gaps = int((np.diff(d) != 1).sum())
            raise ValueError(f"{label}: year {y} has {gaps} break(s) in its "
                             f"day-of-season axis. Runs would be fused across the "
                             f"gap. Reindex the time axis to the full MJJAS calendar.")

def christy_fields(tx_s, doys, yrs, label=""):
    """(C) {year: HW-days grid}. Thresholds and run counting only -- no spatial
    reduction, so one pass over a dataset can serve several domains/footprints."""
    doys = np.asarray(doys, np.int16); yrs = np.asarray(yrs, int)
    _assert_contiguous(doys, yrs, label)
    thr = _thresholds(tx_s, doys)
    fields = {}
    for yr in np.unique(yrs):
        m = yrs == yr; tx = tx_s[m]; dy = doys[m]
        vf = np.sum(~np.isnan(tx), axis=0) / _N_DOY
        exc = (~np.isnan(tx)) & np.isfinite(thr[dy - 1]) & (tx > thr[dy - 1])
        hw = _count_hw(exc)
        hw[vf < _MIN_FRAC] = np.nan
        fields[int(yr)] = hw
    return fields

def _reduce_fields(fields, lat, lon, mask=None,
                   lat_bnds=(24., 50.), lon_bnds=(-180., 180.), label=""):
    """(C) cos(lat)-weighted spatial mean of the per-cell HW fields."""
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    lat_ok = (lat >= lat_bnds[0]) & (lat <= lat_bnds[1])
    lo360, hi360 = lon_bnds[0] % 360, lon_bnds[1] % 360
    lon360 = lon % 360
    lon_ok = ((lon360 >= lo360) & (lon360 <= hi360)
              if hi360 > lo360 else (lon360 >= lo360) | (lon360 <= hi360))
    dom = lat_ok[:, None] & lon_ok[None, :]
    if mask is not None:
        mask = np.asarray(mask, bool)
        assert mask.shape == dom.shape, \
            f"{label}: mask {mask.shape} does not match grid {dom.shape}"
        dom = dom & mask
    cosw = np.cos(np.radians(lat))[:, None] * np.ones(len(lon))
    series = {}
    for yr in sorted(fields):
        valid = dom & ~np.isnan(fields[yr])
        if not np.any(valid): continue
        w = cosw[valid]
        series[int(yr)] = float(np.nansum(fields[yr][valid] * w) / np.nansum(w))
    return series

def christy_series(tx_s, doys, yrs, lat, lon, mask=None,
                   lat_bnds=(24., 50.), lon_bnds=(-180., 180.), label=""):
    """v F signature, now a thin wrapper over christy_fields + _reduce_fields."""
    return _reduce_fields(christy_fields(tx_s, doys, yrs, label=label),
                          lat, lon, mask=mask,
                          lat_bnds=lat_bnds, lon_bnds=lon_bnds, label=label)

def christy_series_stations(tx_s, doys, yrs, stn_lats, label=""):
    doys = np.asarray(doys, np.int16); yrs = np.asarray(yrs, int)
    _assert_contiguous(doys, yrs, label)
    tx3 = tx_s[:, np.newaxis, :]
    thr = _thresholds(tx3, doys)
    cosw = np.cos(np.radians(stn_lats))
    series = {}
    for yr in np.unique(yrs):
        m = yrs == yr; tx = tx3[m]; dy = doys[m]
        vf = np.sum(~np.isnan(tx), axis=0)[0] / _N_DOY
        exc = (~np.isnan(tx)) & np.isfinite(thr[dy - 1]) & (tx > thr[dy - 1])
        hw = _count_hw(exc)[0]
        hw[vf < _MIN_FRAC] = np.nan
        valid = ~np.isnan(hw)
        if not np.any(valid): continue
        w = cosw[valid]
        series[int(yr)] = float(np.nansum(hw[valid] * w) / np.nansum(w))
    return series

def _reindex_full(vals, times, label):
    """Force the array onto the complete MJJAS calendar, so no missing row can
    fuse two runs. Returns (vals, doys, yrs)."""
    times = pd.DatetimeIndex(times)
    y0, y1 = int(times.year.min()), int(times.year.max())
    full = pd.date_range(f"{y0}-01-01", f"{y1}-12-31", freq="D")
    full = full[np.isin(full.month, list(_HW_MON))]
    if len(times) != len(full) or not (times == full).all():
        pos = pd.Index(full).get_indexer(times)
        out = np.full((len(full),) + vals.shape[1:], np.nan, np.float32)
        g = pos >= 0
        out[pos[g]] = vals[g]
        vals = out
    return vals, _season_doy(full), full.year.values.astype(int)

# ══ CONUS LOADERS ══════════════════════════════════════════════════════════
def ld_be_conus():
    fp = BE_TMAX
    ds = xr.open_dataset(fp)
    lat, lon = ds.latitude.values, ds.longitude.values
    lm = ds["land_mask"]
    lm = (lm.isel(time=0) if "time" in lm.dims else lm).values >= 0.5
    times = pd.DatetimeIndex(ds.time.values)
    sm = np.isin(times.month, list(_HW_MON))
    tx = ds["temperature"].values[sm].astype(np.float32)
    ds.close()
    tx, dy, yr = _reindex_full(tx, times[sm], "BE US")
    return tx, dy, yr, lat, lon, lm & mask_conus(lat, lon, "be_us")

def _resolve_conus_grid_mask(lat, lon):
    """(5) usreg_half.txt is checked for orientation, not trusted."""
    poly = mask_conus(lat, lon, "ghcnd_idw")
    if not USREG_FP.exists():
        return poly

    U = np.loadtxt(USREG_FP, dtype=int) > 0
    if U.shape != (len(lat), len(lon)):
        return poly

    opts = {"as-is": U, "lat flipped": U[::-1, :],
            "lon flipped": U[:, ::-1], "both flipped": U[::-1, ::-1]}
    scores = {k: _jaccard(v, poly) for k, v in opts.items()}
    best = max(scores, key=scores.get)

    m = {"polygon": poly, "auto": opts[best], "usreg": U}[USREG_MODE]
    return m

def ld_ghcnd_conus_grid():
    dirp = GHCND_DIR / "Global_Gridded_Christy_IDW"
    files = sorted(dirp.glob("ghcn_grid_????.nc"))
    ds = xr.open_mfdataset(files, combine="by_coords")
    lat, lon = ds.lat.values, ds.lon.values
    times = pd.DatetimeIndex(ds.time.values)
    sm = np.isin(times.month, list(_HW_MON))
    tx = ds["tmax_c"].values[sm].astype(np.float32)
    ds.close()
    m = _resolve_conus_grid_mask(lat, lon)
    tx, dy, yr = _reindex_full(tx, times[sm], "GHCND IDW")
    return tx, dy, yr, lat, lon, m

def ld_ghcnd_conus_stations():
    """(1) GHCN-Daily at stations, from the Figure 3 daily cube.
    common.ghcnd_daily_cube() builds it if the cache is not there yet."""
    ids, V, pos = ghcnd_daily_cube()
    V = V.astype(np.float32)
    pos = pos.reindex(pd.Index(ids))
    assert pos["lat"].notna().all(), "some station ids are missing from the inventory"
    full = pd.date_range(f"{UH_YR0}-01-01", f"{UH_YR1}-12-31", freq="D")
    full = full[np.isin(full.month, list(_HW_MON))]
    assert V.shape[0] == len(full), \
        f"cube has {V.shape[0]} rows, the MJJAS calendar has {len(full)}"
    return V, _season_doy(full), full.year.values.astype(int), pos["lat"].values.astype(np.float32)

def ld_era5_conus():
    files = [ERA5_DIR / f"era5_2t_{y:04d}{m:02d}_daily.nc"
             for y in range(1940, 2025) for m in sorted(_HW_MON)
             if (ERA5_DIR / f"era5_2t_{y:04d}{m:02d}_daily.nc").exists()]
    ds = xr.open_mfdataset(sorted(files), combine="by_coords")
    lat_all = ds.latitude.values; lon_all = ds.longitude.values
    lat_ok = (lat_all >= 23) & (lat_all <= 51)             # (8) subset first
    lon180 = np.where(lon_all > 180, lon_all - 360, lon_all)
    lon_ok = (lon180 >= -126) & (lon180 <= -65)
    assert lat_ok.any() and lon_ok.any(), \
        f"ERA5 CONUS box selected nothing; lat {lat_all.min():.1f}..{lat_all.max():.1f}, " \
        f"lon {lon_all.min():.1f}..{lon_all.max():.1f}"
    sub = ds.isel(latitude=np.where(lat_ok)[0], longitude=np.where(lon_ok)[0])
    lat, lon = sub.latitude.values, sub.longitude.values
    tx = (sub["t2m_max"].values - 273.15).astype(np.float32)
    times = pd.DatetimeIndex(sub.time.values)
    ds.close()
    tx, dy, yr = _reindex_full(tx, times, "ERA5 CONUS")
    m = mask_conus(lat, lon, "era5_conus")                 # (2)
    return tx, dy, yr, lat, lon, m

# ══ GLOBAL LOADERS ═════════════════════════════════════════════════════════
def ld_be_global():
    files = sorted(BE_PROC.glob(
        "processed_nh_TMAX_Complete_TMAX_Daily_LatLong1_????.nc"))
    ds = xr.open_mfdataset(files, combine="by_coords")
    lat_all = ds.latitude.values
    keep = (lat_all >= 23) & (lat_all <= 51)                # (8) subset first
    sub = ds.isel(latitude=np.where(keep)[0])
    lat, lon = sub.latitude.values, sub.longitude.values
    times = pd.DatetimeIndex(sub.time.values)
    _, uniq = np.unique(times.values, return_index=True)    # dedup decades
    sm = np.isin(times.month, list(_HW_MON))
    sm = sm & np.isin(np.arange(len(times)), uniq)
    tx = sub["temperature"].values[sm].astype(np.float32)
    lm = sub["land_mask"]
    lm = (lm.isel(time=0) if "time" in lm.dims else lm).values > 0   # (3)
    ds.close()
    tx, dy, yr = _reindex_full(tx, times[sm], "BE NH")
    return tx, dy, yr, lat, lon, lm

def ld_ghcnd_global():
    dirp = GHCND_GLOBAL_GRID
    files = sorted(dirp.glob("ghcn_grid_????.nc"))
    ds = xr.open_mfdataset(files, combine="by_coords")
    lat_all = ds.lat.values
    keep = (lat_all >= 23) & (lat_all <= 51)
    sub = ds.isel(lat=np.where(keep)[0])
    lat, lon = sub.lat.values, sub.lon.values
    times = pd.DatetimeIndex(sub.time.values)
    sm = np.isin(times.month, list(_HW_MON))
    tx = sub["tmax_c"].values[sm].astype(np.float32)
    ds.close()
    tx, dy, yr = _reindex_full(tx, times[sm], "GHCND 2deg")
    return tx, dy, yr, lat, lon, None            # station-derived: already land

def ld_era5_global():
    files = [ERA5_DIR / f"era5_2t_{y:04d}{m:02d}_daily.nc"
             for y in range(1940, 2025) for m in sorted(_HW_MON)
             if (ERA5_DIR / f"era5_2t_{y:04d}{m:02d}_daily.nc").exists()]
    ds = xr.open_mfdataset(sorted(files), combine="by_coords")
    lat_all = ds.latitude.values
    keep = np.where((lat_all >= 23) & (lat_all <= 51))[0][::4]   # (8)
    sub = ds.isel(latitude=keep, longitude=slice(None, None, 4))
    lat, lon = sub.latitude.values, sub.longitude.values
    tx = (sub["t2m_max"].values - 273.15).astype(np.float32)
    times = pd.DatetimeIndex(sub.time.values)
    ds.close()
    tx, dy, yr = _reindex_full(tx, times, "ERA5 global")
    m = mask_land(lat, lon, "era5_global")       # (2)
    return tx, dy, yr, lat, lon, m


# ══ USHCN PAIR (unchanged: the matched design is right) ════════════════════
def _uh_piv_fp(leg):
    _mtag = "".join(str(m) for m in sorted(_HW_MON))
    return UH_PIV_DIR / f"ushcn_pair_tmax_pivot_m{_mtag}_{leg}.parquet"

def uh_available():
    return (UH_STN_FP.exists()
            and ((_uh_piv_fp("raw").exists() and _uh_piv_fp("adj").exists())
                 or (UH_DAILY_DIR.is_dir() and any(UH_DAILY_DIR.glob("*.csv.gz")))))

def _uh_build_pivots():
    import glob, os
    fs = sorted(glob.glob(str(UH_DAILY_DIR / "*.csv.gz")))
    if not fs:
        raise RuntimeError(f"no station files under {UH_DAILY_DIR}")
    UH_PIV_DIR.mkdir(exist_ok=True)
    raw, adj = {}, {}
    for i, fp in enumerate(fs, 1):
        sid = os.path.basename(fp)[:11]
        d = pd.read_csv(fp, dtype={"date": str},
                        usecols=["date", "tmax_raw_c", "tmax_adj_c"])
        dt = pd.to_datetime(d["date"], format="%Y%m%d", errors="coerce")
        vr = pd.to_numeric(d["tmax_raw_c"], errors="coerce")
        va = pd.to_numeric(d["tmax_adj_c"], errors="coerce")
        both = dt.notna() & dt.dt.month.isin(list(_HW_MON)) & vr.notna() & va.notna()
        if not both.any(): continue
        idx = pd.DatetimeIndex(dt[both])
        for store, vv in ((raw, vr), (adj, va)):
            s = pd.Series(vv[both].to_numpy("float32"), index=idx)
            store[sid] = s[~s.index.duplicated()]
    for leg, store in (("raw", raw), ("adj", adj)):
        p = pd.concat(store, axis=1).sort_index(); p.index.name = "date"
        p.to_parquet(str(_uh_piv_fp(leg)))

def _uh_stations():
    """The station table the purple lines are built from: inside the CONUS box
    and present in the matched pivots."""
    stns = pd.read_csv(str(UH_STN_FP))[["ushcn_id", "lat", "lon"]]
    stns = stns[stns["lat"].between(*UH_LAT_BNDS)
                & stns["lon"].between(*UH_LON_BNDS)].reset_index(drop=True)
    return stns

def _uh_calendar():
    full = pd.date_range(f"{UH_YR0}-01-01", f"{UH_YR1}-12-31", freq="D")
    return full[np.isin(full.month, list(_HW_MON))]

def ld_ushcn_pair():
    if not (_uh_piv_fp("raw").exists() and _uh_piv_fp("adj").exists()):
        _uh_build_pivots()
    stns = _uh_stations()
    piv = {}
    for leg in ("raw", "adj"):
        p = pd.read_parquet(str(_uh_piv_fp(leg))); p.index = pd.DatetimeIndex(p.index)
        piv[leg] = p
    stns = stns[stns["ushcn_id"].isin(set(piv["raw"].columns))].reset_index(drop=True)
    full = _uh_calendar()
    legs = {}
    for leg in ("raw", "adj"):
        v = (piv[leg].reindex(columns=stns["ushcn_id"].values)
                     .reindex(index=full).values.astype(np.float32))
        legs[leg] = np.where((v > -60) & (v < 60), v, np.nan)
    assert np.array_equal(np.isfinite(legs["raw"]), np.isfinite(legs["adj"])), \
        "the two legs are not on the same days -- delete the pivots and rebuild"
    yrs = full.year.values.astype(int)
    return legs, _season_doy(full), yrs, stns

# ══ (B) BERKELEY RE-SAMPLED ONTO A STATION NETWORK, FIGURE 3 STYLE ════════
def _network_positions(net):
    """(lat, lon) of the stations that actually build this figure's line for
    that network -- not the full inventory."""
    if net == "ushcn":
        if not (_uh_piv_fp("raw").exists() and _uh_piv_fp("adj").exists()):
            _uh_build_pivots()
        stns = _uh_stations()
        cols = set(pd.read_parquet(str(_uh_piv_fp("raw"))).columns)
        stns = stns[stns["ushcn_id"].isin(cols)]
        return stns["lat"].to_numpy(float), stns["lon"].to_numpy(float)
    if net == "ghcnd":
        ids, _V, pos = ghcnd_daily_cube()
        pos = pos.reindex(pd.Index(ids))
        assert pos["lat"].notna().all(), "station ids missing from the inventory"
        return pos["lat"].to_numpy(float), pos["lon"].to_numpy(float)
    raise ValueError(net)

NET_TITLE = {"ushcn": "USHCN", "ghcnd": "GHCN-Daily"}

def _be_at_network(fields, lat, lon, be_mask, net, label=""):
    """FIGURE 3's convention, applied to the heat-wave field.

    Fig 3 does   BE_AT_V = BV[:, _flat[_good]]   -- one column per STATION, so a
    Berkeley cell holding n stations is duplicated n times and carries n times
    the weight. That is what makes the line inherit the network's DENSITY and
    not merely its footprint. Here the same indexing is applied to the per-cell
    heat-wave counts (identical, since every duplicate column of a cell has the
    same values and therefore the same thresholds), and the units are then
    reduced by cos(lat) exactly as christy_series_stations reduces the real
    station lines."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    lonw = np.where(lon > 180.0, lon - 360.0, lon)
    slat, slon = _network_positions(net)
    il = np.rint((slat - lat[0]) / (lat[1] - lat[0])).astype(int)
    io = np.rint((slon - lonw[0]) / (lonw[1] - lonw[0])).astype(int)
    ok = (il >= 0) & (il < lat.size) & (io >= 0) & (io < lonw.size)
    flat = np.ravel_multi_index((il[ok], io[ok]), (lat.size, lonw.size))
    good = np.asarray(be_mask, bool).ravel()[flat]      # Fig 3's _good
    cols = flat[good]
    ulat = slat[ok][good]
    cosw = np.cos(np.radians(ulat))
    series = {}
    for yr in sorted(fields):
        v = fields[yr].ravel()[cols]
        valid = ~np.isnan(v)
        if not np.any(valid): continue
        w = cosw[valid]
        series[int(yr)] = float(np.nansum(v[valid] * w) / np.nansum(w))
    return series

def _run_be_conus_set():
    """(B)(C) one Berkeley read, three lines: the full CONUS average (identical
    to v F, same cache) and the network-sampled averages."""
    pkl_full = CACHE_HW / "berkeley_landmasked.pkl"
    pkls = {n: CACHE_HW / f"berkeley_at_{n}_stations.pkl" for n in BE_AT_NETWORKS}
    nets = [n for n in BE_AT_NETWORKS
            if n != "ushcn" or uh_available()]
    out = {}
    full = None
    if pkl_full.exists():
        with open(pkl_full, "rb") as f: full = pickle.load(f)
    for n in nets:
        if pkls[n].exists():
            with open(pkls[n], "rb") as f: out[n] = pickle.load(f)
    todo = [n for n in nets if n not in out]
    if full is not None and not todo:
        return full, out

    tx, doys, yrs, lat, lon, be_mask = ld_be_conus()
    fields = christy_fields(tx, doys, yrs, label="Berkeley CONUS")
    del tx
    if full is None:
        full = _reduce_fields(fields, lat, lon, mask=be_mask,
                              lat_bnds=(24., 50.), lon_bnds=(-125., -65.),
                              label="Berkeley CONUS")
        with open(pkl_full, "wb") as f: pickle.dump(full, f)
    for n in todo:
        out[n] = _be_at_network(fields, lat, lon, be_mask, n,
                                label=f"Berkeley @ {NET_TITLE[n]}")
        with open(pkls[n], "wb") as f: pickle.dump(out[n], f)
    for lbl, d in [("full CONUS", full)] + [(f"at {NET_TITLE[n]}", out[n])
                                            for n in nets]:
        if d:
            v = [x for x in d.values() if not np.isnan(x)]
    return full, out

# ══ CHECK -- the heat-wave kernel on synthetic numbers. Needs no data files. ══
# 1. day-of-season numbering: 1 May is day 1, 30 September is day 153
_d = _season_doy(pd.DatetimeIndex(["2001-05-01", "2001-09-30", "2004-05-01", "2004-09-30"]))
assert list(_d) == [1, 153, 1, 153], list(_d)      # leap years too
print(f"_season_doy: 1 May -> {_d[0]}, 30 Sep -> {_d[1]} (leap year: {_d[2]}, {_d[3]})")

# 2. a run must reach MIN_RUN days to count, and then every day of it counts
def _run_days(n):
    _e = np.zeros((20, 1, 1), bool); _e[3:3 + n] = True
    return float(_count_hw(_e)[0, 0])
assert _run_days(_MIN_RUN) == _MIN_RUN, _run_days(_MIN_RUN)
assert _run_days(_MIN_RUN - 1) == 0.0
assert _run_days(9) == 9.0
print(f"_count_hw: {_MIN_RUN - 1} hot days in a row count 0, "
      f"{_MIN_RUN} count {_run_days(_MIN_RUN):.0f}, 9 count {_run_days(9):.0f}")

# 3. a missing day breaks a run rather than joining it
_e = np.zeros((20, 1, 1), bool); _e[3:9] = True; _e[6] = False
assert float(_count_hw(_e)[0, 0]) == 0.0
print("_count_hw: a gap in the middle of a 6-day spell breaks it, as it should")

# 4. thresholds need a minimum sample, otherwise the cell gets NaN for that day
_tx = np.full((_N_DOY, 1, 1), np.nan, np.float32)
_doys = np.arange(1, _N_DOY + 1, dtype=np.int16)
_tx[:, 0, 0] = np.linspace(20, 40, _N_DOY)
_thr = _thresholds(_tx, _doys)
_n_ok = int(np.isfinite(_thr).sum())
print(f"_thresholds: one year of data gives {_n_ok} usable day-of-season thresholds "
      f"(a +/-{_WINDOW}-day window holds {2 * _WINDOW + 1} values, "
      f"below the {MIN_THRESH_N} required)")
assert _n_ok == 0

# 5. a year whose days are not contiguous would fuse two spells: refuse it
try:
    _assert_contiguous(np.array([1, 2, 4], np.int16), np.array([2000, 2000, 2000]), "test")
    raise AssertionError("a broken day axis should have been rejected")
except ValueError as _e:
    print(f"_assert_contiguous: broken day axis rejected -- {str(_e)[:60]}...")
print("\nheat-wave kernel checks passed\n")

# ══ RUN ════════════════════════════════════════════════════════════════════
def _run(label, loader, pkl_name, lon_bnds=(-125., -65.)):
    pkl = CACHE_HW / pkl_name
    if pkl.exists():
        with open(pkl, "rb") as f: return pickle.load(f)
    tx_s, doys, yrs, lat, lon, mask = loader()      # (8) no blanket except
    hw = christy_series(tx_s, doys, yrs, lat, lon, mask=mask,
                        lat_bnds=(24., 50.), lon_bnds=lon_bnds, label=label)
    del tx_s
    with open(pkl, "wb") as f: pickle.dump(hw, f)
    v = [x for x in hw.values() if not np.isnan(x)]
    return hw

def _run_stations(label, loader, pkl_name):
    pkl = CACHE_HW / pkl_name
    if pkl.exists():
        with open(pkl, "rb") as f: return pickle.load(f)
    tx, doys, yrs, lats = loader()
    hw = christy_series_stations(tx, doys, yrs, lats, label=label)
    with open(pkl, "wb") as f: pickle.dump(hw, f)
    v = [x for x in hw.values() if not np.isnan(x)]
    return hw

def _run_ushcn_pair(legs_wanted=USHCN_LEGS):
    pkls = {leg: CACHE_HW / f"ushcn_pair_{leg}_stations.pkl" for leg in legs_wanted}
    out = {}
    todo = [leg for leg in legs_wanted if not pkls[leg].exists()]
    for leg in legs_wanted:
        if pkls[leg].exists():
            with open(pkls[leg], "rb") as f: out[leg] = pickle.load(f)
    if not todo: return out
    legs, doys, yrs, stns = ld_ushcn_pair()
    lats = stns["lat"].values.astype(np.float32)
    for leg in todo:
        hw = christy_series_stations(legs[leg], doys, yrs, lats, label=f"USHCN {leg}")
        with open(pkls[leg], "wb") as f: pickle.dump(hw, f)
        v = [x for x in hw.values() if not np.isnan(x)]
        out[leg] = hw
    return out

_be_c, _be_net = _run_be_conus_set()                      # (B)
if BLANK_BE_PARTIAL:                                      # (4)
    _be_c = {k: v for k, v in _be_c.items() if k != 2024}
    _be_net = {n: {k: v for k, v in d.items() if k != 2024}
               for n, d in _be_net.items()}
_be_u = _be_net.get("ushcn", {}) if DRAW_BE_AT_USHCN else {}

_gh_stn = _run_stations("GHCN-Daily CONUS (stations)", ld_ghcnd_conus_stations,
                        "ghcnd_stations.pkl")
_gh_grd = (_run("GHCND-Daily CONUS (Christy grid)", ld_ghcnd_conus_grid,
                f"ghcnd_idw_mask-{USREG_MODE}.pkl", (-125., -65.))
           if (COMPARE_BOTH_GHCND or GHCND_CONUS_SOURCE == "christy_grid") else {})
_gh_c = _gh_stn if GHCND_CONUS_SOURCE == "stations" else _gh_grd

R_conus = {"Berkeley": _be_c, "GHCND": _gh_c,
           "ERA5": _run("ERA5 CONUS (land-masked)", ld_era5_conus,
                        "era5_landmasked.pkl", (-125., -65.))}

_be_g = _run("Berkeley NH global (land-masked)", ld_be_global,
             "berkeley_nh_landmasked.pkl", (-180., 180.))
if BLANK_BE_PARTIAL and 2024 in _be_g:
    _be_g = {k: v for k, v in _be_g.items() if k != 2024}
R_global = {"Berkeley": _be_g,
            "GHCND": _run("GHCND 2deg global", ld_ghcnd_global,
                          "ghcnd_2deg_glb.pkl", (-180., 180.)),
            "ERA5": _run("ERA5 global strip (land-masked)", ld_era5_global,
                         "era5_global_landmasked.pkl", (-180., 180.))}

_uh = _run_ushcn_pair() if uh_available() else {}
R_ushcn_raw, R_ushcn_adj = _uh.get("raw", {}), _uh.get("adj", {})

# ══ CHECK -- Figure 6 series, before plotting. ═════════════════════════════
def _pm_F6(d, y0, y1):
    _v = [d[y] for y in range(y0, y1 + 1) if y in d and not np.isnan(d[y])]
    return np.mean(_v) if _v else np.nan
print(f"{'series':<34}{'years':>7}{'1930s':>8}{'2015-24':>9}{'ratio':>7}")
_rows = [("CONUS Berkeley", R_conus["Berkeley"]), ("CONUS GHCN-Daily", R_conus["GHCND"]),
         ("CONUS ERA5", R_conus["ERA5"]), ("USHCN-Daily (raw)", R_ushcn_raw),
         ("USHCN-BC (adjusted)", R_ushcn_adj), ("Berkeley sampled as USHCN", _be_u),
         ("Global Berkeley", R_global["Berkeley"]), ("Global GHCN 2deg", R_global["GHCND"]),
         ("Global ERA5", R_global["ERA5"])]
for _lbl, _d in _rows:
    if not _d:
        print(f"  {_lbl:<32}{'not built':>7}"); continue
    _a, _b = _pm_F6(_d, 1930, 1939), _pm_F6(_d, 2015, 2024)
    print(f"  {_lbl:<32}{len(_d):>7}{_a:>8.2f}{_b:>9.2f}{_a / _b:>7.2f}")
print("\nthe point of the figure: with USHCN sampling the 1930s still exceed the present"
      "\n(ratio above 1.00), while the full Berkeley grid does not")

# ══ PLOT ═══════════════════════════════════════════════════════════════════
def _smooth(d, w=SMOOTH_W):
    """(6) year-indexed centred mean, full window required."""
    if not d: return {}
    s = pd.Series(d).sort_index()
    s = s.reindex(range(int(s.index.min()), int(s.index.max()) + 1))
    sm = s.rolling(w, center=True, min_periods=w).mean().dropna()
    return sm.to_dict()

PLOT_YRS = (1900, 2024)
ROW_LABELS = ["CONUS", "Global 24–50°N"]
ROW_DATA = [R_conus, R_global]
COL_TITLES = {"Berkeley": "Berkeley Earth", "GHCND": "GHCN/USHCN",
              "USHCN": "USHCN", "ERA5": "ERA5"}
EMPTY_NOTE = {("USHCN", 1): "CONUS-only network\n(no global coverage)"}
PRIMARY_LABEL = {("GHCND", 0): "GHCN", ("GHCND", 1): "GHCN",
                 ("USHCN", 0): "USHCN-BC", ("Berkeley", 0): "Berkeley Earth"}
# (A) uncorrected datasets are dot-dashed wherever they appear
PRIMARY_LS = {("GHCND", r): LS_UNCORR for r in GHCND_UNCORR_ROWS}

def _draw(ax, hw, color, ls="-", label=None, thin=True):
    """thin=False draws the smoothed curve only. Used for the Berkeley
    USHCN-sampling line: its annual values sit almost on top of the full-CONUS
    ones, so a second thin trace in the same colour only hides both curves."""
    if not hw: return False
    y = np.array(sorted(hw.keys())); v = np.array([hw[k] for k in y], float)
    ok = (y >= PLOT_YRS[0]) & (y <= PLOT_YRS[1])
    y, v = y[ok], v[ok]
    if y.size == 0: return False
    if thin and SHOW_ANNUAL_F6:                                  # (W2)(W5)
        ax.plot(y, v, lw=LW_ANNUAL, color=color, ls=ls, alpha=ALPHA_ANNUAL)
    sm = _smooth({a: b for a, b in zip(y, v) if not np.isnan(b)})
    if sm:
        sy = np.array(sorted(sm.keys()))
        ax.plot(sy, [sm[k] for k in sy], lw=LW_SMOOTH, color=color, ls=ls,
                label=label, solid_capstyle="round", dash_capstyle="round")
    return True

def build_figure(datasets, overlays, out_name, extra_for_scale=None):
    extra_for_scale = extra_for_scale or {}
    def _row_ymax(R, row):
        """(W5) scale to the annual traces (v G) or to the smoothed curves."""
        series = [R.get(dn, {}) for dn in datasets] + extra_for_scale.get(row, [])
        if YLIM_FROM == "smoothed":
            series = [_smooth(d) for d in series]
        vals = [v for d in series for v in d.values() if not np.isnan(v)]
        if not vals:
            return 20.
        return (np.percentile(vals, 99) * 1.20 if YLIM_FROM == "annual"
                else max(vals) * 1.18)
    ncol = len(datasets)
    fig, axes = plt.subplots(2, ncol, figsize=(PANEL_W * ncol, PANEL_H * 2.28),
                             gridspec_kw={"hspace": 0.30, "wspace": 0.075})
    for row, (row_label, R) in enumerate(zip(ROW_LABELS, ROW_DATA)):
        ymax = _row_ymax(R, row)
        for col in range(1, ncol):
            axes[row, col].sharey(axes[row, 0])
        for col, dn in enumerate(datasets):
            ax = axes[row, col]; hw = R.get(dn, {}); ovl = overlays.get((dn, row), [])
            drew = _draw(ax, hw, C[dn], PRIMARY_LS.get((dn, row), LS_ADJ),
                         label=(PRIMARY_LABEL.get((dn, row), COL_TITLES[dn])
                                if ovl else None))
            for _o in ovl:                      # (label, series, colour, ls[, thin])
                _l, _s, _c, _ls = _o[:4]
                drew |= _draw(ax, _s, _c, _ls, label=_l,
                              thin=(_o[4] if len(_o) > 4 else True))
            if not drew:
                ax.text(0.5, 0.5, EMPTY_NOTE.get((dn, row), "No data"),
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=FS_EMPTY, color="0.55", style="italic")
                ax.set_facecolor("0.97")
            elif ovl:
                ax.legend(loc="upper left", frameon=False, fontsize=FS_PANLEG,
                          handlelength=1.9, borderaxespad=0.2,
                          labelspacing=0.25, handletextpad=0.5)
            ax.set_xlim(*PLOT_YRS); ax.set_ylim(-1, ymax)
            ax.tick_params(axis="both", labelsize=FS_TICK, length=5, width=1.1)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(20))      # (W4)
            ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
            ax.tick_params(axis="x", which="minor", length=2.5, width=0.9)
            ax.grid(True, axis="y", color="0.90", lw=0.7)                # (W4)
            ax.set_axisbelow(True)
            ax.axhline(0, color="0.6", lw=0.5, ls="--")
            ax.axvspan(1930, 1939, alpha=0.07, color="firebrick", zorder=0)
            if row == 0:
                ax.set_title(COL_TITLES[dn], color="black", fontweight="bold",
                             pad=8, fontsize=FS_COLTITLE)
            if col == 0:
                ax.set_ylabel(f"{row_label}\nHW days/year", fontsize=FS_YLABEL,
                              labelpad=8)
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
            if row == 1:
                ax.set_xlabel("Year", fontsize=FS_XLABEL, labelpad=6)
    if SUPTITLE_F6:
        fig.suptitle(SUPTITLE_F6, fontsize=FS_SUPTITLE, fontweight="bold", y=0.985)
    # the notebook wrote this one to the work directory while every other
    # figure went to FIGD; keep them together instead
    out = FIGD / out_name
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.show(); plt.close(fig)
    return out

_USHCN_LINES = [(lab, s, C["USHCN"], ls) for lab, s, ls in
                [("USHCN-Daily", R_ushcn_raw, LS_UNCORR),
                 ("USHCN-BC", R_ushcn_adj, LS_ADJ)] if s]
# (B) the dashed green line lives in the Berkeley panel of the CONUS row only
BE_AT_USHCN_THIN = False        # True to also draw its noisy annual trace
_BE_LINES = ([("Berkeley, sampled as USHCN", _be_u, C_SAMPLED, LS_SAMPLED,
               BE_AT_USHCN_THIN)] if _be_u else [])

_SCALE0 = [o[1] for o in _USHCN_LINES] + ([_be_u] if _be_u else [])

# figure 1: USHCN rides in the GHCN panel
_OVL_OVERLAY = {}
if _USHCN_LINES: _OVL_OVERLAY[("GHCND", 0)] = _USHCN_LINES
if _BE_LINES:    _OVL_OVERLAY[("Berkeley", 0)] = _BE_LINES

_out_F6 = build_figure(["Berkeley", "GHCND", "ERA5"], _OVL_OVERLAY,
                       "FigHW_conus_vs_global_ushcn_overlay_wide.png", {0: _SCALE0})

print(f"\nwrote {_out_F6}")

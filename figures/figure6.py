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
USREG_MODE         = "polygon"

# ══ (I) CHRISTY (2026) COMPLIANCE ══════════════════════════════════════════
# Four places where this script used to differ from the reference implementation,
# /Users/adessler/Desktop/Christy 2026 analysis/us_dly_waves.py (a Python port of
# J.R. Christy's us_dly_waves.f, UAH/NSSTC). Each is a switch, so the old
# behaviour is one edit away and the CHECK block can price the difference.
#
# (I1) the percentile estimator. Christy sorts the pooled window and takes
#      sorted[int(n*p/100)] -- a nearest-rank rule, so the threshold is an
#      OBSERVED value. numpy's default interpolates between two of them.
PCTILE_METHOD = "nearest_rank"    # "nearest_rank" (Christy) | "linear" (v F)
#
# (I2) the exceedance test. Coupled to (I1): against an interpolated threshold
#      no day lands exactly on it and > vs >= is academic, but a nearest-rank
#      threshold IS a data value and GHCN-Daily is quantised to 0.1 degC, so the
#      tied days are a real population -- a few tenths of a percent of all days,
#      about half a day per season against totals in the 5-15 day range.
EXCEED_INCLUSIVE = True           # tx >= thr (Christy) | tx > thr (v F)
#
# (I3) how many values a day-of-season threshold needs. Christy: icnt > 200.
#      On gridded data this never binds (Berkeley offers ~875 per cell-day);
#      on stations it IS the record-length filter, ~29 near-complete seasons.
MIN_THRESH_N = 201                # 30 was v F
#
# (I4) the spatial reduction, and the one that matters. Christy counts wave days
#      at each station, IDW-interpolates THE COUNTS onto a grid, and then takes a
#      cos(lat) area average of the accepted cells. v F averaged over stations
#      directly, which weights by where the observers are rather than by land.
STATION_REDUCTION = "idw_grid"    # "idw_grid" (Christy) | "station_mean" (v F)

# The IDW itself (us_dly_waves.py:358-425). The radius is a property of the
# SEARCH, not of the cell, so Christy's 115 km carries over unchanged to the 1
# deg Berkeley grid this script interpolates onto. The global band needs a wider
# one: outside the US the network is sparse enough that 115 km would leave most
# of the band empty. Set it from the coverage diagnostic the CHECK block prints.
IDW_RADIUS_KM_CONUS  = 115.0
IDW_RADIUS_KM_GLOBAL = 300.0
IDW_MIN_STATIONS     = 2          # cells with fewer in-range reporters fail ...
IDW_SOLO_WSUM        = 1.5        # ... unless the one they have is this close:
                                  # sqrt(sum (R/d)^2) > 1.5 means d < R/1.5.
# (I5) the bottom-row GHCN panel. "stations" puts it in Christy's order --
#      count at stations, then grid -- instead of reading the 2 deg grid of
#      daily TEMPERATURES that prepare_data.py builds, which counts second.
GHCND_GLOBAL_SOURCE = "stations"  # "stations" (Christy order) | "grid2deg" (v F)

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
# (I9) the third Berkeley value: light enough to separate from both greens above
# at 2.5 pt, still unmistakably the Berkeley hue.
C_COSAMP = "#7FD3B8"
# (A) one style vocabulary for the whole figure:
#   solid     homogenised / bias-corrected / reanalysis
#   dot-dash  UNCORRECTED station data (GHCN-Daily, USHCN-Daily raw)
#   dashed    a corrected product re-sampled onto another network's footprint
# (W3) spelled-out dash patterns: at this panel width a default "-." and "--"
# are easy to confuse, and telling them apart is the whole point of the figure.
LS_ADJ     = "-"
LS_UNCORR  = (0, (6.5, 1.8, 1.0, 1.8))     # long dash, dot -- uncorrected
LS_SAMPLED = (0, (7.0, 2.4))               # long open dash -- re-sampled
LS_COSAMP  = (0, (1.6, 1.9))               # dot -- (I9) a SUBSET of the cells
LS_RAW = LS_UNCORR                      # v F name kept so nothing downstream breaks

_PCTILE, _WINDOW, _MIN_RUN, _MIN_FRAC = 90, 3, 6, 0.70
_HW_MON, _N_DOY = {5, 6, 7, 8, 9}, 153

# (I6) every cached series is tagged with the method that produced it. The one
# failure mode here that would look entirely plausible on screen is a figure
# drawn half from v F pickles and half from Christy's -- (I1)-(I3) change
# christy_fields, so the GRIDDED caches go stale too, not just the station ones.
# A run under one set of switches must not be able to read a cache written under
# another, so the switches are in the filename. Same idea as common.TAG_F3.
_TAG_THR = (f"p{_PCTILE}r{_MIN_RUN}w{_WINDOW}n{MIN_THRESH_N}"
            f"_{'nr' if PCTILE_METHOD == 'nearest_rank' else 'lin'}"
            f"{'ge' if EXCEED_INCLUSIVE else 'gt'}")
_TAG_STN = f"{_TAG_THR}_{'idw' if STATION_REDUCTION == 'idw_grid' else 'stnmean'}"

def _tag_idw(radius_km):
    return f"{_TAG_STN}{radius_km:g}"

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

def _nearest_rank(vals, pctile):
    """(I1) Christy's percentile, us_dly_waves.py:197-201: sort the finite values
    of each column and take sorted[int(n * pctile / 100)], clipped to n-1.

    np.sort sends NaN to the end of each column, so a column's n finite values
    occupy its first n rows and the rank index can differ from column to column.

    Not np.percentile(method="inverted_cdf"): that agrees with Christy whenever
    n*p/100 falls between two integers and sits one rank low when it lands on
    one. With a 7-day window at the 90th percentile the product is 6.3*n_years,
    an integer for every record whose length is a multiple of 5 -- common enough
    that the two estimators would visibly disagree.

    The arithmetic is written in Christy's order, n * pctile * 0.01 rather than
    n * (pctile * 0.01), because int() of the two differs in the last bit."""
    s = np.sort(vals, axis=0)
    n = np.isfinite(vals).sum(axis=0)
    idx = np.minimum((n * pctile * 0.01).astype(np.int64), np.maximum(n - 1, 0))
    out = np.take_along_axis(s, idx[np.newaxis], axis=0)[0]
    return np.where(n > 0, out, np.nan)

def _exceeds(tx, thr):
    """(I2) Christy counts a day that sits exactly ON its threshold."""
    return tx >= thr if EXCEED_INCLUSIVE else tx > thr

def _thresholds(tx_s, doys):
    T, nlat, nlon = tx_s.shape
    thr = np.full((_N_DOY, nlat, nlon), np.nan, dtype=np.float32)
    doys = np.asarray(doys, np.int16)
    for doy in range(1, _N_DOY + 1):
        m = np.abs(doys - doy) <= _WINDOW
        vals = tx_s[m]
        if vals.shape[0] == 0:
            continue
        if PCTILE_METHOD == "nearest_rank":                        # (I1)
            p = _nearest_rank(vals, _PCTILE)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = np.nanpercentile(vals, _PCTILE, axis=0)
        n = np.isfinite(vals).sum(axis=0)          # (7)
        thr[doy - 1] = np.where(n >= MIN_THRESH_N, p, np.nan)      # (I3)
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
        exc = (~np.isnan(tx)) & np.isfinite(thr[dy - 1]) & _exceeds(tx, thr[dy - 1])
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

def christy_station_counts(tx_s, doys, yrs, label=""):
    """(I4) Christy's nval_all, us_dly_waves.py:511: annual heat-wave-day counts
    per station, NaN where the station-year missed the _MIN_FRAC rule.

    Stops before any spatial reduction, so one pass over a dataset feeds both the
    station mean and the IDW grid and the two can be priced against each other.
    Returns (counts, years) with counts shaped (n_years, n_stn)."""
    doys = np.asarray(doys, np.int16); yrs = np.asarray(yrs, int)
    _assert_contiguous(doys, yrs, label)
    tx3 = tx_s[:, np.newaxis, :]
    thr = _thresholds(tx3, doys)
    # (I7) A DEPARTURE FROM CHRISTY, and a necessary one. His loop sets
    # nval[yi] = total for any year that clears the 70% data gate, even when the
    # station has NO thresholds at all -- so a short record reports a hard ZERO
    # heat-wave days rather than "unknown". Among his 1,218 uniformly long USHCN
    # records that can never happen. Among the 24,874 stations of the global band
    # more than half are in exactly that state, and a hard zero is not a missing
    # value: it would be interpolated, averaged, and would drag the field down.
    # So a unit also has to HAVE thresholds on _MIN_FRAC of the season.
    tfrac = np.isfinite(thr).mean(axis=0)[0]
    years = np.unique(yrs)
    counts = np.full((years.size, tx_s.shape[1]), np.nan, np.float32)
    for k, yr in enumerate(years):
        m = yrs == yr; tx = tx3[m]; dy = doys[m]
        vf = np.sum(~np.isnan(tx), axis=0)[0] / _N_DOY
        exc = (~np.isnan(tx)) & np.isfinite(thr[dy - 1]) & _exceeds(tx, thr[dy - 1])
        hw = _count_hw(exc)[0]
        hw[(vf < _MIN_FRAC) | (tfrac < _MIN_FRAC)] = np.nan
        counts[k] = hw
    return counts, years

def station_mean_series(counts, years, stn_lats):
    """v F's reduction: one cos(lat) mean over STATIONS. Kept for
    STATION_REDUCTION = "station_mean" and for the old-vs-new CHECK column."""
    cosw = np.cos(np.radians(np.asarray(stn_lats, float)))
    series = {}
    for k, yr in enumerate(years):
        hw = counts[k]; valid = ~np.isnan(hw)
        if not np.any(valid): continue
        w = cosw[valid]
        series[int(yr)] = float(np.nansum(hw[valid] * w) / np.nansum(w))
    return series

def christy_series_stations(tx_s, doys, yrs, stn_lats, label=""):
    """v F's signature, now a thin wrapper, exactly as christy_series is."""
    counts, years = christy_station_counts(tx_s, doys, yrs, label=label)
    return station_mean_series(counts, years, stn_lats)

# ══ (I4) CHRISTY'S IDW: grid the COUNTS, then area-average ═════════════════
EARTH_R_CHRISTY = 6335.44        # us_dly_waves.py:26 -- not the mean radius
_PI180 = np.pi / 180.0

def _christy_dist_km(cell_lat, cell_lon, stn_lat, stn_lon):
    """Great-circle distance, transcribed from us_dly_waves.py:334, itself a
    transcription of us_dly_waves.f:104-131. Subscript 1 = grid cell, 2 = station.

    Two Fortran quirks are kept deliberately, because the object is to reproduce
    Christy's station SELECTION rather than to improve on it:
      * r = 6335.44 km, not the mean Earth radius 6371, so his kilometre runs
        about 0.6% short and his 115 km reaches 115.6 real ones;
      * the x1 term uses cos(cell_lat) where the textbook Vincenty form uses
        cos(stn_lat).

    Beyond a quarter of the globe the denominator turns negative and arctan
    returns a NEGATIVE distance. Christy never meets that case -- his cells and
    his stations are all inside the CONUS -- but the global row does, so callers
    must reject d < 0 rather than let it pass the radius test."""
    clat1 = np.cos(cell_lat * _PI180); slat1 = np.sin(cell_lat * _PI180)
    clat2 = np.cos(stn_lat * _PI180);  slat2 = np.sin(stn_lat * _PI180)
    slon21 = np.sin((stn_lon - cell_lon) * _PI180)
    clon21 = np.cos((stn_lon - cell_lon) * _PI180)
    x1 = (clat1 * slon21) ** 2
    x2 = (clat2 * slat1 - slat2 * clat1 * clon21) ** 2
    x5 = np.sqrt(x1 + x2) / (slat2 * slat1 + clat2 * clat1 * clon21)
    return np.arctan(x5) * EARTH_R_CHRISTY

def _christy_weights(cell_lat, cell_lon, stn_lat, stn_lon, radius_km):
    """Sparse (n_cells, n_stn) matrix of (radius/d)^2 for every station STRICTLY
    within radius_km of a cell centre (us_dly_waves.py:358-376).

    A KD-tree on the unit sphere pre-selects candidate pairs and _christy_dist_km
    then decides membership, so the answer is Christy's and only the search is
    ours. The pre-filter is generous -- a true great-circle radius 5% wider than
    the target, which more than covers both the 6335.44 km radius and the cosine
    quirk -- so it cannot drop a pair Christy would have kept.

    It is not an optimisation either. Dense, the global row would be 9,360 cells
    by ~10,000 stations, and every far pair would have to be screened for the
    negative-distance case above."""
    from scipy.sparse import csr_matrix
    cell_lat = np.asarray(cell_lat, float); cell_lon = np.asarray(cell_lon, float)
    stn_lat = np.asarray(stn_lat, float);   stn_lon = np.asarray(stn_lon, float)
    slack = 1.05 * EARTH_R / EARTH_R_CHRISTY
    chord = 2.0 * np.sin(0.5 * (radius_km * slack) / EARTH_R)
    tree = cKDTree(_xyz(stn_lat, stn_lon))
    cand = tree.query_ball_point(_xyz(cell_lat, cell_lon), chord)
    rows, cols, vals = [], [], []
    for ic, js in enumerate(cand):
        if not js: continue
        js = np.asarray(js)
        d = _christy_dist_km(cell_lat[ic], cell_lon[ic], stn_lat[js], stn_lon[js])
        keep = (d >= 0) & (d < radius_km)              # strict, and sign-guarded
        if not keep.any(): continue
        js = js[keep]
        d = np.maximum(d[keep], 1e-3)                  # us_dly_waves.py:371-373
        rows.append(np.full(js.size, ic)); cols.append(js)
        vals.append((radius_km / d) ** 2)
    shape = (cell_lat.size, stn_lat.size)
    if not rows:
        return csr_matrix(shape)
    return csr_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))), shape=shape)

def christy_idw_fields(counts, years, stn_lat, stn_lon, lat, lon, mask,
                       radius_km, label=""):
    """(I4) {year: (nlat, nlon) field} and {year: coverage}, us_dly_waves.py:379.

    `counts` is christy_station_counts' (n_years, n_stn), NaN standing in for the
    Fortran's -99: a station that did not report in a given year is dropped from
    that year's interpolation entirely, so the grid follows the network as it
    opens and closes. A cell is accepted only if IDW_MIN_STATIONS of them were in
    range, or exactly one whose weights sum to more than IDW_SOLO_WSUM squared;
    everything else stays NaN and _reduce_fields renormalises over what survived.

    Coverage is Christy's ddd(0)/ddtot diagnostic (us_dly_waves.py:423): the
    cos(lat) share of the masked domain that carried a value that year. Read it
    before trusting a series -- it is what says whether the domain is holding
    still or quietly shrinking into the well-observed corners."""
    lat = np.asarray(lat, float)
    lon = np.where(np.asarray(lon, float) > 180.0,
                   np.asarray(lon, float) - 360.0, np.asarray(lon, float))
    LON, LAT = np.meshgrid(lon, lat)
    m = np.ones(LAT.shape, bool) if mask is None else np.asarray(mask, bool)
    assert m.shape == LAT.shape, \
        f"{label}: mask {m.shape} does not match grid {LAT.shape}"
    ci = np.flatnonzero(m.ravel())
    assert ci.size, f"{label}: the mask selects no cells"
    clat, clon = LAT.ravel()[ci], LON.ravel()[ci]

    W = _christy_weights(clat, clon, stn_lat, stn_lon, radius_km)
    Wb = W.copy(); Wb.data = np.ones_like(Wb.data)     # in-range INDICATOR
    cosw = np.cos(np.radians(clat)); cos_tot = cosw.sum()

    fields, coverage = {}, {}
    for k, yr in enumerate(years):
        v = counts[k]
        rep = np.isfinite(v).astype(np.float64)        # reported this year
        if not rep.any(): continue
        dd  = W  @ rep                                 # sum of weights, in range
        num = W  @ np.where(rep > 0, v, 0.0)
        nst = Wb @ rep                                 # how many, in range
        # us_dly_waves.py:408. The (radius/d)^2 constant cancels in num/dd but
        # NOT in this test, which is why the weights are carried in Christy's
        # units rather than as a bare d^-2.
        acc = ((nst >= IDW_MIN_STATIONS)
               | ((nst == 1) & (np.sqrt(dd) > IDW_SOLO_WSUM))) & (dd > 0)
        if not acc.any(): continue
        flat = np.full(m.size, np.nan)
        flat[ci[acc]] = num[acc] / dd[acc]
        fields[int(yr)] = flat.reshape(m.shape)
        coverage[int(yr)] = float(cosw[acc].sum() / cos_tot)
    return fields, coverage

FIXED_CELLS = {}       # (I8) label -> (series on the never-missing cells, n, n_tot)
IDW_FIELDS = {}        # (I9) label -> {year: gridded field}, for co-sampling

def reduce_on_footprint(fields, lat, lon, mask, footprint, lon_bnds=(-180., 180.),
                        label=""):
    """(I9) The same field, averaged only over the cells another line had a
    value in THAT year.

    This is the honest way to show what a moving footprint is worth. The
    fixed-cell diagnostic (I8) answers the question by throwing away every cell
    that was ever missing -- on the global band that is three quarters of the
    land. Here nothing is thrown away and nothing about the data changes; only
    which cells are allowed to speak, year by year. The gap between this line
    and the full-grid one is then attributable to coverage alone, because the
    two are the same dataset on the same grid in the same year."""
    base = np.asarray(mask, bool)
    out = {}
    for y, f in fields.items():
        fp = footprint.get(y)
        if fp is None:
            continue
        m = base & fp
        if not m.any():
            continue
        s = _reduce_fields({y: f}, lat, lon, mask=m, lat_bnds=(24., 50.),
                           lon_bnds=lon_bnds, label=label)
        if y in s:
            out[y] = s[y]
    return out

def footprint_of(label):
    """(I9) which cells an IDW line actually carried a value in, per year."""
    return {y: ~np.isnan(f) for y, f in IDW_FIELDS.get(label, {}).items()}

def christy_idw_series(counts, years, stn_lat, stn_lon, lat, lon, mask,
                       radius_km, lon_bnds=(-180., 180.), label=""):
    """(I4) the whole of Christy's reduction: grid the counts, then hand them to
    _reduce_fields -- the SAME cos(lat) area mean the gridded datasets go
    through, on the same cells, so a station line and the Berkeley line beside it
    differ in their data and in nothing else."""
    fields, coverage = christy_idw_fields(counts, years, stn_lat, stn_lon,
                                          lat, lon, mask, radius_km, label=label)
    series = _reduce_fields(fields, lat, lon, mask=mask,
                            lat_bnds=(24., 50.), lon_bnds=lon_bnds, label=label)
    # (I8) The same series over the cells accepted in EVERY year. Christy lets
    # the footprint drift -- his denominator is the accepted cells of that year,
    # which is what `series` above reproduces -- and on his dense, stable USHCN
    # network the drift is small. It is not small here, and a drifting footprint
    # means an early decade is being compared against a different piece of
    # ground, not only a different climate. Reported, never substituted: which
    # of the two belongs in the paper is a judgement about the science.
    common = np.ones_like(np.asarray(mask, bool))
    for f in fields.values():
        common &= ~np.isnan(f)
    fixed = (_reduce_fields({y: f for y, f in fields.items()}, lat, lon,
                            mask=np.asarray(mask, bool) & common,
                            lat_bnds=(24., 50.), lon_bnds=lon_bnds, label=label)
             if common.any() else {})
    FIXED_CELLS[label] = (fixed, int(common.sum()), int(np.asarray(mask, bool).sum()))
    IDW_FIELDS[label] = fields                          # (I9)
    return series, coverage

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

# ══ (I4) THE TARGET GRIDS ══════════════════════════════════════════════════
# Christy grids to his own half-degree CONUS mask (usreg_half.txt, 116x50). This
# script grids to the BERKELEY grid instead, in both rows, so that every line in
# a panel sits on the same cells and the panels can be differenced cell by cell.
# The IDW's radius and acceptance rule are properties of the search and not of
# the cell, so they carry over to the coarser grid untouched.
#
# If a half-degree sensitivity test is ever wanted, common.g05_centers() already
# reproduces Christy's grid exactly -- 25.25..49.75 N by -124.25..-66.75 E.
def be_conus_grid():
    """Berkeley's CONUS grid and mask without reading its 4 GB temperature cube.
    Must agree with ld_be_conus() cell for cell; the CHECK block asserts it."""
    fp = CACHE_HW / "be_conus_grid.npz"
    if fp.exists():
        z = np.load(fp); return z["lat"], z["lon"], z["mask"]
    ds = xr.open_dataset(BE_TMAX)
    lat, lon = ds.latitude.values, ds.longitude.values
    lm = ds["land_mask"]
    lm = (lm.isel(time=0) if "time" in lm.dims else lm).values >= 0.5
    ds.close()
    m = lm & mask_conus(lat, lon, "be_us")
    np.savez(fp, lat=lat, lon=lon, mask=m)
    return lat, lon, m

def be_global_grid():
    """Berkeley's northern-band grid and land mask, ld_be_global()'s cells."""
    fp = CACHE_HW / "be_global_grid.npz"
    if fp.exists():
        z = np.load(fp); return z["lat"], z["lon"], z["mask"]
    files = sorted(BE_PROC.glob(
        "processed_nh_TMAX_Complete_TMAX_Daily_LatLong1_????.nc"))
    assert files, f"no Berkeley NH decade files under {BE_PROC}"
    ds = xr.open_dataset(files[0])
    lat_all = ds.latitude.values
    keep = (lat_all >= 23) & (lat_all <= 51)           # ld_be_global's subset
    sub = ds.isel(latitude=np.where(keep)[0])
    lat, lon = sub.latitude.values, sub.longitude.values
    lm = sub["land_mask"]
    m = (lm.isel(time=0) if "time" in lm.dims else lm).values > 0
    ds.close()
    np.savez(fp, lat=lat, lon=lon, mask=m)
    return lat, lon, m

# ══ CONUS LOADERS ══════════════════════════════════════════════════════════
def ld_be_conus():
    ds = xr.open_dataset(BE_TMAX)
    times = pd.DatetimeIndex(ds.time.values)
    sm = np.isin(times.month, list(_HW_MON))
    tx = ds["temperature"].values[sm].astype(np.float32)
    ds.close()
    tx, dy, yr = _reindex_full(tx, times[sm], "BE US")
    lat, lon, m = be_conus_grid()       # (I4) one definition of grid and mask
    return tx, dy, yr, lat, lon, m

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
    return (V, _season_doy(full), full.year.values.astype(int),
            pos["lat"].values.astype(np.float64),
            pos["lon"].values.astype(np.float64))      # (I4) lon, for the IDW

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
    times = pd.DatetimeIndex(sub.time.values)
    _, uniq = np.unique(times.values, return_index=True)    # dedup decades
    sm = np.isin(times.month, list(_HW_MON))
    sm = sm & np.isin(np.arange(len(times)), uniq)
    tx = sub["temperature"].values[sm].astype(np.float32)
    ds.close()
    tx, dy, yr = _reindex_full(tx, times[sm], "BE NH")
    lat, lon, lm = be_global_grid()     # (I4)(3) one definition of grid and mask
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

def ld_ghcnd_global_stations():
    """(I5) the band as STATIONS, so the bottom row can count first and grid
    second like the top row does. Same archive prepare_data.py builds the 2 deg
    field from, read through common.ghcnd_band_cube().

    The screen handed to that function is chosen to be LOSSLESS -- it may only
    drop stations that provably cannot affect a single output number:

      * a year needs ceil(_MIN_FRAC * _N_DOY) days, or (I7) blanks it anyway;
      * a station needs at least one such year, or every year is NaN;
      * and it needs enough observations for (I7)'s threshold coverage to be
        reachable at all. A threshold at day d pools a window of at most
        2*_WINDOW+1 days, so summing the window counts over the season counts
        each observation at most that many times. Needing MIN_THRESH_N in each
        of ceil(_MIN_FRAC * _N_DOY) windows therefore needs at least
        ceil(_MIN_FRAC * _N_DOY) * MIN_THRESH_N / (2*_WINDOW+1) observations
        in total -- about 3,100 at the current settings. Below that tfrac cannot
        reach _MIN_FRAC, so every year of that station is NaN regardless.

    Nothing here is a judgement about which stations "count"; the dropped ones
    contribute NaN either way, and the screen exists only so the cube fits in
    memory."""
    need_days = int(np.ceil(_MIN_FRAC * _N_DOY))
    need_tot = int(np.ceil(need_days * MIN_THRESH_N / (2 * _WINDOW + 1)))
    ids, V, pos = ghcnd_band_cube(min_total_days=need_tot, min_good_years=1,
                                  min_days_in_year=need_days)
    V = V.astype(np.float32, copy=False)
    assert pos["lat"].notna().all(), "band stations missing from the position table"
    full = pd.date_range(f"{UH_YR0}-01-01", f"{UH_YR1}-12-31", freq="D")
    full = full[np.isin(full.month, list(_HW_MON))]
    assert V.shape[0] == len(full), \
        f"band cube has {V.shape[0]} rows, the MJJAS calendar has {len(full)}"
    print(f"  GHCN-Daily band: {len(ids):,} stations clear the lossless screen "
          f"({need_tot:,} MJJAS days, one year of {need_days}+)")
    return (V, _season_doy(full), full.year.values.astype(int),
            pos["lat"].values.astype(np.float64),
            pos["lon"].values.astype(np.float64))

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

def _be_at_network_counts(fields, lat, lon, be_mask, net):
    """FIGURE 3's convention, applied to the heat-wave field.

    Fig 3 does   BE_AT_V = BV[:, _flat[_good]]   -- one column per STATION, so a
    Berkeley cell holding n stations is duplicated n times and carries n times
    the weight. That is what makes the line inherit the network's DENSITY and
    not merely its footprint. Here the same indexing is applied to the per-cell
    heat-wave counts (identical, since every duplicate column of a cell has the
    same values and therefore the same thresholds).

    Returns (counts, years, slat, slon) -- the same shape christy_station_counts
    returns, so _reduce_stations can take it through whichever reduction the real
    station lines are using."""
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
    # (I4) The comparator has to travel the same road as the lines it is
    # compared with. If the USHCN lines are counted at stations and then gridded
    # while this one stays a station mean, their ratio stops isolating SAMPLING
    # and starts carrying the reduction as well -- which is the one thing this
    # dashed line exists to hold fixed. So this function now stops at the
    # pseudo-station counts and hands them to the same _reduce_stations.
    years = np.array(sorted(fields))
    counts = np.stack([fields[y].ravel()[cols] for y in years]).astype(np.float32)
    return counts, years, slat[ok][good], slon[ok][good]

def be_fields(which):
    """(I9) Berkeley's per-CELL heat-wave counts, cached as a cube.

    Needed twice over: once for the Berkeley line itself, and once for the
    co-sampled line, which has to be re-reduced on a different set of cells
    every year and so cannot be formed from an already-reduced series. Small
    enough to keep -- 0.8 MB for CONUS, 4.7 MB for the band -- and it means a
    warm run never has to reopen the 4 GB Berkeley file to add the second line.
    Returns ({year: field}, lat, lon, mask)."""
    fp = CACHE_HW / f"berkeley_{which}_fields_{_TAG_THR}.npz"
    lat, lon, mask = be_conus_grid() if which == "conus" else be_global_grid()
    if fp.exists():
        z = np.load(fp)
        return ({int(y): f for y, f in zip(z["years"], z["fields"])},
                lat, lon, mask)
    tx, doys, yrs, lat, lon, mask = (ld_be_conus() if which == "conus"
                                     else ld_be_global())
    fields = christy_fields(tx, doys, yrs, label=f"Berkeley {which}")
    del tx
    yy = np.array(sorted(fields))
    np.savez(fp, years=yy, fields=np.stack([fields[y] for y in yy]))
    return fields, lat, lon, mask

def _run_be_conus_set():
    """(B)(C) one Berkeley read, three lines: the full CONUS average (identical
    to v F, same cache) and the network-sampled averages."""
    pkl_full = CACHE_HW / f"berkeley_landmasked_{_TAG_THR}.pkl"
    # (I4) cache the per-station COUNTS, not the finished series -- the same
    # rule _run_stations follows. Caching the series would skip _reduce_stations
    # on a warm run, and this line would then vanish from the coverage, the
    # old-versus-new and the fixed-cell tables: the reduction is what fills them.
    npzs = {n: CACHE_HW / f"berkeley_at_{n}_counts_{_TAG_THR}.npz"
            for n in BE_AT_NETWORKS}
    nets = [n for n in BE_AT_NETWORKS
            if n != "ushcn" or uh_available()]
    held = {}
    full = None
    if pkl_full.exists():
        with open(pkl_full, "rb") as f: full = pickle.load(f)
    for n in nets:
        if npzs[n].exists():
            z = np.load(npzs[n])
            held[n] = (z["counts"], z["years"], z["slat"], z["slon"])
    todo = [n for n in nets if n not in held]

    if full is None or todo:
        fields, lat, lon, be_mask = be_fields("conus")        # (I9) cached cube
        if full is None:
            full = _reduce_fields(fields, lat, lon, mask=be_mask,
                                  lat_bnds=(24., 50.), lon_bnds=(-125., -65.),
                                  label="Berkeley CONUS")
            with open(pkl_full, "wb") as f: pickle.dump(full, f)
        for n in todo:
            held[n] = _be_at_network_counts(fields, lat, lon, be_mask, n)
            np.savez(npzs[n], counts=held[n][0], years=held[n][1],
                     slat=held[n][2], slon=held[n][3])

    grid = be_conus_grid()
    out = {n: _reduce_stations(*held[n], grid, IDW_RADIUS_KM_CONUS,
                               (-125., -65.), f"Berkeley @ {NET_TITLE[n]}")
           for n in nets}
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

# ── (I) the Christy-compliance checks ──────────────────────────────────────
# 6. (I1) the percentile is a RANK into the sorted sample, not an interpolation.
#    n = 10, p = 90 puts n*p/100 exactly on 9, and the three estimators split.
_v = np.arange(1.0, 11.0, dtype=np.float32).reshape(10, 1, 1)
assert _nearest_rank(_v, 90)[0, 0] == 10.0
assert np.percentile(_v.ravel(), 90, method="inverted_cdf") == 9.0
assert np.isclose(np.percentile(_v.ravel(), 90), 9.1)
print("_nearest_rank: n=10 p=90 -> 10.0 "
      "(numpy's inverted_cdf gives 9.0, its default 9.1)")

# 7. NaN sorts to the end, so each column's rank is read off its finite head
_x = np.array([[[5.0]], [[np.nan]], [[1.0]], [[3.0]], [[9.0]]], np.float32)
assert _nearest_rank(_x, 50)[0, 0] == 5.0          # finite [1,3,5,9], idx 2
assert _nearest_rank(_x, 25)[0, 0] == 3.0          # finite [1,3,5,9], idx 1
print("_nearest_rank: NaN sort out of the way; the rank indexes the finite values")

# 8. (I2) a day sitting exactly ON its threshold is a hot day
_tx = np.array([[[10.0]]], np.float32); _th = np.array([[10.0]], np.float32)
assert bool(_exceeds(_tx[0], _th)[0, 0]) is EXCEED_INCLUSIVE
print(f"_exceeds: a day equal to its threshold counts = {EXCEED_INCLUSIVE} "
      "(Christy's >=; v F used a strict >)")

# 9. (I4) Christy's distance, with its two deliberate quirks intact
_d1 = _christy_dist_km(40.0, -100.0, 41.0, -100.0)
assert 110.0 < _d1 < 111.0, _d1
print(f"_christy_dist_km: 1 deg of latitude at 40N = {_d1:.2f} km; a 6371 km "
      f"sphere gives 111.19, and the 0.6% shortfall is Christy's 6335.44")
#    beyond a quarter of the globe arctan flips sign. Christy never meets the
#    case, the global row would meet it constantly, and a negative distance
#    would sail through a "< radius" test and then weight as if colocated.
assert _christy_dist_km(40.0, -100.0, -40.0, 80.0) < 0
print("_christy_dist_km: an antipodal pair returns a NEGATIVE distance -- "
      "_christy_weights rejects d < 0 rather than letting it pass the radius")

# 10. (I4) the IDW reproduces a station sitting on the cell centre
_cl, _co = np.array([40.0]), np.array([-100.0])
_sl, _so = np.array([40.0, 40.5]), np.array([-100.0, -100.0])
_f, _cov = christy_idw_fields(np.array([[7.0, 3.0]], np.float32), np.array([2000]),
                              _sl, _so, _cl, _co, np.ones((1, 1), bool), 115.0)
assert abs(_f[2000][0, 0] - 7.0) < 1e-6, _f[2000]
print(f"christy_idw_fields: a station on the cell centre is reproduced exactly "
      f"({_f[2000][0, 0]:.4f}), the 1 m floor giving it all the weight")

# 11. (I4) the acceptance rule, and why the 115 km constant has to stay. It
#     cancels out of the weighted mean, so it looks removable -- but the solo
#     test reads it, and without it every one-station cell would be thrown away.
for _n_in, _dist, _want in ((0, 200.0, False), (1, 90.0, False),
                            (1, 70.0, True), (2, 110.0, True)):
    _dd = _n_in * (115.0 / _dist) ** 2
    _got = (_n_in >= IDW_MIN_STATIONS
            or (_n_in == 1 and np.sqrt(_dd) > IDW_SOLO_WSUM))
    assert _got == _want, (_n_in, _dist, _got)
assert not np.sqrt(((1.0 / np.array([50.0])) ** 2).sum()) > IDW_SOLO_WSUM
print(f"IDW acceptance: {IDW_MIN_STATIONS}+ stations in range, or one inside "
      f"{115.0 / IDW_SOLO_WSUM:.1f} km; drop the constant and the solo rule "
      f"would silently reject every cell")

# 12. (I7) a unit with data but no thresholds must report NaN, not a hard zero.
#     One year of data can never clear MIN_THRESH_N, so nothing is countable.
_tx1 = np.linspace(20, 40, _N_DOY, dtype=np.float32)[:, None]
_c1, _y1 = christy_station_counts(_tx1, np.arange(1, _N_DOY + 1, dtype=np.int16),
                                  np.full(_N_DOY, 2000), label="check")
assert np.isnan(_c1[0, 0]), _c1
print("christy_station_counts: a station with no usable thresholds reports NaN, "
      "not 0 heat-wave days (Christy's loop would have said 0)")
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

COVERAGE = {}          # (I4) label -> {year: accepted cos(lat) share}
OLD_METHOD = {}        # (I4) label -> the station-mean series, for the CHECK table

def _reduce_stations(counts, years, slat, slon, grid, radius_km, lon_bnds, label):
    """(I4) one place where a station network becomes a series, so the GHCN,
    USHCN and Berkeley-at-network lines cannot drift apart. Both reductions are
    computed -- the IDW is cheap once the counts exist, and the CHECK table
    prints what the choice costs."""
    old = station_mean_series(counts, years, slat)
    OLD_METHOD[label] = old
    if STATION_REDUCTION != "idw_grid":
        return old
    lat, lon, mask = grid
    hw, cov = christy_idw_series(counts, years, slat, slon, lat, lon, mask,
                                 radius_km, lon_bnds=lon_bnds, label=label)
    COVERAGE[label] = cov
    return hw

def _run_stations(label, loader, pkl_name, grid=None,
                  radius_km=IDW_RADIUS_KM_CONUS, lon_bnds=(-125., -65.)):
    """The counts are cached, not the series: they are the expensive half and
    they do not depend on the reduction, so switching STATION_REDUCTION re-reads
    them in seconds instead of re-thresholding the whole cube."""
    npz = CACHE_HW / f"{pkl_name}_counts_{_TAG_THR}.npz"
    if npz.exists():
        z = np.load(npz)
        counts, years, slat, slon = z["counts"], z["years"], z["slat"], z["slon"]
    else:
        tx, doys, yrs, slat, slon = loader()
        counts, years = christy_station_counts(tx, doys, yrs, label=label)
        del tx
        np.savez(npz, counts=counts, years=years, slat=slat, slon=slon)
    if grid is None:
        grid = be_conus_grid()
    return _reduce_stations(counts, years, slat, slon, grid, radius_km,
                            lon_bnds, label)

def _run_ushcn_pair(legs_wanted=USHCN_LEGS):
    npzs = {leg: CACHE_HW / f"ushcn_pair_{leg}_counts_{_TAG_THR}.npz"
            for leg in legs_wanted}
    held, todo = {}, [leg for leg in legs_wanted if not npzs[leg].exists()]
    for leg in legs_wanted:
        if npzs[leg].exists():
            z = np.load(npzs[leg])
            held[leg] = (z["counts"], z["years"], z["slat"], z["slon"])
    if todo:
        legs, doys, yrs, stns = ld_ushcn_pair()
        slat = stns["lat"].values.astype(np.float64)
        slon = stns["lon"].values.astype(np.float64)
        for leg in todo:
            counts, years = christy_station_counts(legs[leg], doys, yrs,
                                                   label=f"USHCN {leg}")
            np.savez(npzs[leg], counts=counts, years=years, slat=slat, slon=slon)
            held[leg] = (counts, years, slat, slon)
    grid = be_conus_grid()
    return {leg: _reduce_stations(*held[leg], grid, IDW_RADIUS_KM_CONUS,
                                  (-125., -65.), f"USHCN {leg}")
            for leg in legs_wanted}

_be_c, _be_net = _run_be_conus_set()                      # (B)
if BLANK_BE_PARTIAL:                                      # (4)
    _be_c = {k: v for k, v in _be_c.items() if k != 2024}
    _be_net = {n: {k: v for k, v in d.items() if k != 2024}
               for n, d in _be_net.items()}
_be_u = _be_net.get("ushcn", {}) if DRAW_BE_AT_USHCN else {}

_gh_stn = _run_stations("GHCN-Daily CONUS (stations)", ld_ghcnd_conus_stations,
                        "ghcnd_stations")
_gh_grd = (_run("GHCND-Daily CONUS (Christy grid)", ld_ghcnd_conus_grid,
                f"ghcnd_idw_mask-{USREG_MODE}_{_TAG_THR}.pkl", (-125., -65.))
           if (COMPARE_BOTH_GHCND or GHCND_CONUS_SOURCE == "christy_grid") else {})
_gh_c = _gh_stn if GHCND_CONUS_SOURCE == "stations" else _gh_grd

R_conus = {"Berkeley": _be_c, "GHCND": _gh_c,
           "ERA5": _run("ERA5 CONUS (land-masked)", ld_era5_conus,
                        f"era5_landmasked_{_TAG_THR}.pkl", (-125., -65.))}

_be_g = _run("Berkeley NH global (land-masked)", ld_be_global,
             f"berkeley_nh_landmasked_{_TAG_THR}.pkl", (-180., 180.))
if BLANK_BE_PARTIAL and 2024 in _be_g:
    _be_g = {k: v for k, v in _be_g.items() if k != 2024}
# (I5) the bottom-row GHCN line, in Christy's order. "grid2deg" is v F: the 2 deg
# field of daily TEMPERATURES that prepare_data.py builds, which counts second.
_gh_g = (_run_stations("GHCN-Daily band (stations)", ld_ghcnd_global_stations,
                       "ghcnd_band_stations", grid=be_global_grid(),
                       radius_km=IDW_RADIUS_KM_GLOBAL, lon_bnds=(-180., 180.))
         if GHCND_GLOBAL_SOURCE == "stations" else
         _run("GHCND 2deg global", ld_ghcnd_global,
              f"ghcnd_2deg_glb_{_TAG_THR}.pkl", (-180., 180.)))
R_global = {"Berkeley": _be_g, "GHCND": _gh_g,
            "ERA5": _run("ERA5 global strip (land-masked)", ld_era5_global,
                         f"era5_global_landmasked_{_TAG_THR}.pkl", (-180., 180.))}

_uh = _run_ushcn_pair() if uh_available() else {}
R_ushcn_raw, R_ushcn_adj = _uh.get("raw", {}), _uh.get("adj", {})

# ══ (I9) BERKELEY ON GHCN'S OWN FOOTPRINT ══════════════════════════════════
# Christy lets the interpolated footprint drift with the network, and that is
# kept -- but on the global band it drifts from 47% of the land in the 1930s to
# 84% today, so the era comparison there is partly a comparison of coverage.
# Rather than pin the domain and lose three quarters of it, each row gains a
# SECOND Berkeley line: the same Berkeley field, reduced each year over exactly
# the cells GHCN carried a value in that year. Berkeley is the only dataset here
# that exists everywhere, so it is the only one that can hold the climate fixed
# and vary the footprint alone. The gap between the two green lines is what the
# drift is worth; where they lie on top of each other, coverage is not the story.
_GH_LBL_CONUS = "GHCN-Daily CONUS (stations)"
_GH_LBL_BAND  = "GHCN-Daily band (stations)"

def _be_on_ghcn(which, gh_label, lon_bnds):
    fpr = footprint_of(gh_label)
    if not fpr:
        return {}
    flds, lat, lon, mask = be_fields(which)
    s = reduce_on_footprint(flds, lat, lon, mask, fpr, lon_bnds=lon_bnds,
                            label=f"Berkeley on {gh_label}")
    return {k: v for k, v in s.items() if not (BLANK_BE_PARTIAL and k == 2024)}

_be_c_gh = _be_on_ghcn("conus", _GH_LBL_CONUS, (-125., -65.))
_be_g_gh = _be_on_ghcn("global", _GH_LBL_BAND, (-180., 180.))

# ══ CHECK -- Figure 6 series, before plotting. ═════════════════════════════
def _pm_F6(d, y0, y1):
    _v = [d[y] for y in range(y0, y1 + 1) if y in d and not np.isnan(d[y])]
    return np.mean(_v) if _v else np.nan

# (I4) Christy's ddd(0)/ddtot. A series is only as trustworthy as the share of
# the domain its accepted cells cover, and that share MOVES: the IDW follows the
# network as it opens and closes, so an era with sparse coverage is being
# compared against a different footprint, not just a different climate.
if COVERAGE:
    print("\nIDW coverage -- cos(lat) share of the masked domain carrying a value")
    _eras = [(1900, 1929), (1930, 1939), (1940, 1969), (1970, 1999), (2000, 2024)]
    print(f"  {'line':<30}" + "".join(f"{a}-{str(b)[2:]:>3}" for a, b in _eras))
    for _lbl, _cov in COVERAGE.items():
        _cells = [f"{_pm_F6(_cov, a, b):>8.2f}" for a, b in _eras]
        print(f"  {_lbl:<30}" + "".join(_cells))

print(f"\n{'series':<34}{'years':>7}{'1930s':>8}{'2015-24':>9}{'ratio':>7}")
_rows = [("CONUS Berkeley", R_conus["Berkeley"]), ("CONUS GHCN-Daily", R_conus["GHCND"]),
         ("CONUS ERA5", R_conus["ERA5"]), ("USHCN-Daily (raw)", R_ushcn_raw),
         ("USHCN-BC (adjusted)", R_ushcn_adj), ("Berkeley sampled as USHCN", _be_u),
         ("Berkeley on GHCN's CONUS cells", _be_c_gh),
         ("Global Berkeley", R_global["Berkeley"]),
         ("Berkeley on GHCN's band cells", _be_g_gh),
         (f"Global GHCN ({'stations' if GHCND_GLOBAL_SOURCE == 'stations' else '2deg'})",
          R_global["GHCND"]),
         ("Global ERA5", R_global["ERA5"])]
for _lbl, _d in _rows:
    if not _d:
        print(f"  {_lbl:<32}{'not built':>7}"); continue
    _a, _b = _pm_F6(_d, 1930, 1939), _pm_F6(_d, 2015, 2024)
    print(f"  {_lbl:<32}{len(_d):>7}{_a:>8.2f}{_b:>9.2f}{_a / _b:>7.2f}")
print("\nthe point of the figure: with USHCN sampling the 1930s still exceed the present"
      "\n(ratio above 1.00), while the full Berkeley grid does not")

# (I4) what the reduction alone is worth. Both are computed from one pass over
# the counts, so this costs nothing and it is the only way to tell a change in
# the SCIENCE from a change in the METHOD when reading this table against an
# older run. Every number above is Christy's; every number here is v F's.
if OLD_METHOD and STATION_REDUCTION == "idw_grid":
    print(f"\nwhat the reduction is worth -- same counts, cos(lat) mean over "
          f"STATIONS instead of over the grid ({_TAG_THR})")
    print(f"  {'line':<32}{'1930s':>8}{'2015-24':>9}{'ratio':>7}")
    for _lbl, _d in OLD_METHOD.items():
        _a, _b = _pm_F6(_d, 1930, 1939), _pm_F6(_d, 2015, 2024)
        print(f"  {_lbl:<32}{_a:>8.2f}{_b:>9.2f}{_a / _b:>7.2f}")

# (I8) and what the DRIFTING footprint is worth. Read this next to the coverage
# table above: where coverage moves a lot between eras, so does this.
if FIXED_CELLS:
    print("\nthe same lines over the cells accepted in EVERY year -- Christy lets "
          "the\nfootprint drift, and this is the size of that choice")
    print(f"  {'line':<32}{'cells':>12}{'1930s':>8}{'2015-24':>9}{'ratio':>7}")
    for _lbl, (_d, _n, _nt) in FIXED_CELLS.items():
        if not _d:
            print(f"  {_lbl:<32}{'none':>12}"); continue
        _a, _b = _pm_F6(_d, 1930, 1939), _pm_F6(_d, 2015, 2024)
        print(f"  {_lbl:<32}{f'{_n}/{_nt}':>12}{_a:>8.2f}{_b:>9.2f}{_a / _b:>7.2f}")

# ══ PLOT ═══════════════════════════════════════════════════════════════════
def _smooth(d):
    """common.roll(): an 11-year centred mean (ROLL=11), matching Figure 3,
    with its outer 5 years on each end filled by a local linear fit read at the
    endpoint (END_METHOD="loclin") rather than left blank -- the least-squares
    line through the same shrinking window, which inside the record IS the
    window mean and only diverges from it at the ends, where it tracks a
    trending tail instead of lagging it the way a plain mean would.

    strict_interior=True keeps the plain full-window rule everywhere except
    those outer positions: this script's IDW acceptance rule can in principle
    leave a year uncovered in the interior of a record (figure6.py's `_reindex_
    full` already guarantees the calendar has no holes, but a sparse year can
    still fail every cell's accept test), and such a gap should stay a gap
    rather than have a six-point line drawn through it."""
    if not d: return {}
    s = pd.Series(d).sort_index()
    s = s.reindex(range(int(s.index.min()), int(s.index.max()) + 1))
    return roll(s, strict_interior=True).dropna().to_dict()

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
# (I9) the co-sampled line is a third Berkeley trace, so it stays in the
# Berkeley hue and takes a third VALUE of it -- light against the mid-green of
# the full grid and the dark green of the USHCN-sampled line -- plus a dotted
# pattern, which reads as "a subset of" next to solid and long-dash.
_BE_LINES += ([("Berkeley, on GHCN's cells", _be_c_gh, C_COSAMP, LS_COSAMP,
                False)] if _be_c_gh else [])
_BE_LINES_G = ([("Berkeley, on GHCN's cells", _be_g_gh, C_COSAMP, LS_COSAMP,
                 False)] if _be_g_gh else [])

_SCALE0 = ([o[1] for o in _USHCN_LINES] + ([_be_u] if _be_u else [])
           + ([_be_c_gh] if _be_c_gh else []))
_SCALE1 = [_be_g_gh] if _be_g_gh else []

# figure 1: USHCN rides in the GHCN panel
_OVL_OVERLAY = {}
if _USHCN_LINES: _OVL_OVERLAY[("GHCND", 0)] = _USHCN_LINES
if _BE_LINES:    _OVL_OVERLAY[("Berkeley", 0)] = _BE_LINES
if _BE_LINES_G:  _OVL_OVERLAY[("Berkeley", 1)] = _BE_LINES_G

_out_F6 = build_figure(["Berkeley", "GHCND", "ERA5"], _OVL_OVERLAY,
                       "Figure6.png", {0: _SCALE0, 1: _SCALE1})

print(f"\nwrote {_out_F6}")

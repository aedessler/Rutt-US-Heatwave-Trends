#!/usr/bin/env python3
"""Figure 3 -- CONUS daily TMAX record frequency, May-September.

For each station or grid cell and each May-September calendar day, which year
holds the highest TMAX of 1900-2024. Ties are split. Notebook section 7.

This also builds the daily station cube that Figure 6 reads; the cube lives in
common.py, so either figure can be run first.

    python figures/figure3.py
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
for _d in (FIGD, CACHE_REC): _d.mkdir(parents=True, exist_ok=True)

# Y0_F3, Y1_F3, MONTHS, NDAYS, YEARS_F3, MIN_YEARS, MIN_FRAC, TMAX_BOUNDS_F3,
# TAG_F3, COV_FP, DAILY_FP and the KEYS/KPOS/YR/MO/DY calendar come from
# common.py, because Figure 6 needs the same cube and the same tag.
MIN_CELL_FRAC = 0.50             # a Berkeley cell must be >=50% inside CONUS
BE_LAND_MIN   = 0.50             # ... and >50% land in Berkeley's own mask
ROLL = 11
END_METHOD = "loclin"            # how the smooth handles the last/first ROLL//2
                                 # years: loclin | mean | savgol | minrough | none
END_SHADE  = True                # shade those reduced-window years
END_BAND   = True                # ... and draw their out-of-sample uncertainty
END_BAND_K = 1.0                 # envelope half-width, in units of that RMSE
BLANK_PARTIAL = True             # Berkeley's daily release stops 2024-08-31
SERIES_FP = CACHE_REC / f"fig3_all_series_{TAG_F3}_cf{MIN_CELL_FRAC:g}.csv"

# ══ THE RECORD KERNEL — one implementation for every line ═══════════════════
def credit(sub):
    """Fractional credit for holding the all-time max of one calendar day.
    N years sharing the maximum get 1/N each -- unbiased and deterministic."""
    fin = np.isfinite(sub); m = np.where(fin, sub, -np.inf)
    match = fin & (m == m.max(axis=0)[None, :])
    n = match.sum(axis=0)
    return np.where(match, 1.0 / np.where(n > 0, n, 1)[None, :], 0.0)

def record_counts(vals, yr, mo, dy):
    """(n_year, n_unit) all-time record counts. `vals` is (n_time, n_unit)."""
    ymap = {y: i for i, y in enumerate(YEARS_F3)}
    out = np.zeros((len(YEARS_F3), vals.shape[1]))
    for m in MONTHS:
        for d in np.unique(dy[mo == m]):
            k = (mo == m) & (dy == d)
            if k.sum() < 2:
                continue
            sub = vals[k]
            if not np.isfinite(sub).any():
                continue
            rows = np.array([ymap[y] for y in yr[k]])
            assert np.unique(rows).size == rows.size, "repeated years on one date"
            np.add.at(out, rows, credit(sub))
    return out

def to_series(counts_arr, keep, w=None, label="", check=True):
    c = counts_arr[:, keep]
    ww = np.ones(c.shape[1]) if w is None else np.asarray(w)[keep]
    s = pd.Series((c * ww[None, :]).sum(axis=1) / ww.sum(), index=YEARS_F3)
    integ = float(s.sum())
    if check:
        assert abs(integ - NDAYS) < 0.5, (
            f"{label}: integral {integ:.2f} != {NDAYS}. Every unit must contribute "
            f"exactly one record per day-of-year it observed -- a masking or "
            f"kernel bug, not a result.")
    return s

def ratio(s):
    a = float(s.loc[1930:1939].mean()); b = float(np.nanmean(s.loc[2015:2024]))
    return a, b, a / b

def conus_cell_fraction(lat_c, lon_c, key):
    """Fraction of each grid cell inside CONUS -- the same polygon the stations
    are tested against, so the two domains cannot disagree."""
    fp = CACHE_REC / f"conusfrac_poly_{key}.npy"
    if fp.exists():
        return np.load(fp)
    import shapely
    from shapely.geometry import box as sbox
    poly = conus_polygon()
    la, lo = _edges(lat_c), _edges(lon_c)
    frac = np.zeros((len(lat_c), len(lon_c)))
    cells, idx = [], []
    for i in range(len(lat_c)):
        for j in range(len(lon_c)):
            cells.append(sbox(min(lo[j], lo[j + 1]), min(la[i], la[i + 1]),
                              max(lo[j], lo[j + 1]), max(la[i], la[i + 1])))
            idx.append((i, j))
    tree = shapely.STRtree(np.array(cells, dtype=object))
    for h in np.atleast_1d(tree.query(poly, predicate="intersects")):
        i, j = idx[h]; c = cells[h]
        frac[i, j] = poly.intersection(c).area / c.area if c.area > 0 else 0.0
    np.save(fp, frac)
    return frac

# ══ CHECK -- the record kernel on synthetic numbers. Needs no data files. ══
# 1. ties are split, not given to the earliest year
_sub = np.array([[1.0], [3.0], [3.0], [np.nan]])          # 4 years, 1 unit, 2-way tie
_c = credit(_sub)
assert np.allclose(_c.ravel(), [0.0, 0.5, 0.5, 0.0]), _c.ravel()
print(f"credit: a two-way tie gives {_c[1, 0]:.2f} / {_c[2, 0]:.2f}, not 1 / 0")

# 2. a clear winner takes the whole day
_sub2 = np.array([[1.0], [5.0], [3.0]])
assert np.allclose(credit(_sub2).ravel(), [0.0, 1.0, 0.0])

# 3. conservation: one unit with a full record holds exactly 153 records
_rng = np.random.default_rng(0)
_v = _rng.normal(30.0, 5.0, size=(len(KEYS), 1)).astype(np.float32)
_cnt = record_counts(_v, YR, MO, DY)
assert abs(_cnt.sum() - NDAYS) < 1e-6, _cnt.sum()
print(f"record_counts: a complete synthetic station holds {_cnt.sum():.1f} records "
      f"(= {NDAYS} May-September days), spread over {(_cnt > 0).sum()} years")

# 4. the completeness rule: years present and fraction of possible days
_v2 = _v.copy(); _v2[: len(KEYS) // 2] = np.nan        # drop the first half of the record
_ny, _fr = coverage(_v2, YR)
assert not complete(_ny, _fr)[0], "a station missing half its record must fail"
print(f"coverage/complete: half a record -> {int(_ny[0])} years, {_fr[0]:.2f} of days, "
      f"rejected (needs >= {MIN_YEARS} years and {MIN_FRAC:.0%} of days)")
print("\nrecord kernel checks passed\n")

# ══ (1) GHCN-DAILY, FROM THE STATION ARCHIVE ═══════════════════════════════
IDS, V, POS = ghcnd_daily_cube()

_all_gh = np.ones(len(IDS), bool)
ghcnd = to_series(record_counts(V, YR, MO, DY), _all_gh, label="GHCN-Daily")

# ══ (2) THE MATCHED USHCN PAIR ═════════════════════════════════════════════
_mt = "".join(map(str, MONTHS))
piv = {}
for w in ("raw", "adj"):
    p = pd.read_parquet(str(UH_PIV_DIR / f"ushcn_pair_tmax_pivot_m{_mt}_{w}.parquet"))
    p.index = pd.DatetimeIndex(p.index); piv[w] = p
_cols = piv["raw"].columns.intersection(piv["adj"].columns)
piv = {w: p.reindex(columns=_cols) for w, p in piv.items()}
assert np.array_equal(np.isfinite(piv["raw"].values), np.isfinite(piv["adj"].values)), \
    ("raw and adjusted USHCN pivots are not on the same days -- rebuild them. The "
     "whole point of this pair is that the adjustment is the ONLY difference.")
_ut = pd.DatetimeIndex(piv["raw"].index)
_us = np.isin(_ut.month, MONTHS) & (_ut.year >= Y0_F3) & (_ut.year <= Y1_F3)
UYR, UMO, UDY = _ut.year.values[_us], _ut.month.values[_us], _ut.day.values[_us]
RAW, ADJ = piv["raw"].values[_us], piv["adj"].values[_us]
_uny, _ufr = coverage(RAW, UYR)            # identical for ADJ by construction
KEEP_UH = complete(_uny, _ufr)
ushcn    = to_series(record_counts(RAW, UYR, UMO, UDY), KEEP_UH, label="USHCN-Daily")
ushcn_bc = to_series(record_counts(ADJ, UYR, UMO, UDY), KEEP_UH, label="USHCN-BC")

# ══ (3) BERKELEY, AND BERKELEY AT THE GHCN-DAILY STATIONS ══════════════════
_ds = xr.open_dataset(str(BE_TMAX))
_da = _ds["temperature"]
_bt = pd.DatetimeIndex(_da["time"].values)
BLAT = _ds["latitude"].values.astype(float)
BLON = _ds["longitude"].values.astype(float)
if BLON.max() > 180.0:
    BLON = np.where(BLON > 180.0, BLON - 360.0, BLON)
_bland = _ds["land_mask"]
_bland = (_bland.isel(time=0) if "time" in _bland.dims else _bland).values
_bs = np.isin(_bt.month, MONTHS) & (_bt.year >= Y0_F3) & (_bt.year <= Y1_F3)
BV = _da.values[_bs].reshape(int(_bs.sum()), -1)
BYR, BMO, BDY = _bt.year.values[_bs], _bt.month.values[_bs], _bt.day.values[_bs]
_blast = _bt[-1]
_ds.close()
PARTIAL = [] if (_blast.month == 12 and _blast.day == 31) else [int(_blast.year)]

_bfrac = conus_cell_fraction(BLAT, BLON, "berkeley").ravel()
_bny, _bfr = coverage(BV, BYR)
KEEP_BE = (_bfrac >= MIN_CELL_FRAC) & (_bland.ravel() > BE_LAND_MIN) & complete(_bny, _bfr)
_bcounts = record_counts(BV, BYR, BMO, BDY)
berkeley = to_series(_bcounts, KEEP_BE, w=_bfrac, label="Berkeley Earth")

# each kept GHCN-Daily station carries its own Berkeley cell. ONE UNIT PER
# STATION, counted at the point. Duplicates are kept on purpose: two stations in
# one cell means that cell counts twice, which is exactly the density weighting
# the station network imposes and the effect this line exists to expose.
_kpos = POS.loc[POS.index.intersection(pd.Index(IDS))]
_il = np.rint((_kpos["lat"].to_numpy() - BLAT[0]) / (BLAT[1] - BLAT[0])).astype(int)
_io = np.rint((_kpos["lon"].to_numpy() - BLON[0]) / (BLON[1] - BLON[0])).astype(int)
_ok = (_il >= 0) & (_il < len(BLAT)) & (_io >= 0) & (_io < len(BLON))
_flat = np.ravel_multi_index((_il[_ok], _io[_ok]), (len(BLAT), len(BLON)))
_flat = _flat[KEEP_BE[_flat]]
be_at = to_series(record_counts(BV[:, _flat], BYR, BMO, BDY),
                  np.ones(len(_flat), bool),
                  label="Berkeley Earth at GHCN Stations")

# sensitivities on that construction, printed only
_cells = KEEP_BE & np.isin(np.arange(KEEP_BE.size), np.unique(_flat))
be_at_cells = to_series(_bcounts, _cells, w=_bfrac,
                        label="  [sens] at 1-deg cells instead")
_bkeys = (BYR.astype(np.int64) * 10000 + BMO * 100 + BDY)
_rowmap = KPOS.get_indexer(_bkeys)
_avail = np.zeros((len(_bkeys), len(_flat)), bool)
_g = _rowmap >= 0
_stncol = np.nonzero(_ok)[0][KEEP_BE[np.ravel_multi_index(
    (_il[_ok], _io[_ok]), (len(BLAT), len(BLON)))]]
_avail[_g] = np.isfinite(V[_rowmap[_g]][:, _stncol])
be_at_masked = to_series(record_counts(np.where(_avail, BV[:, _flat], np.nan),
                                       BYR, BMO, BDY),
                         np.ones(len(_flat), bool),
                         label="  [sens] + masked to reporting days", check=False)

# Berkeley at the USHCN sites. Same construction as the GHCN version above --
# one unit per STATION, counted at its Berkeley cell, duplicates kept so the
# line inherits the network's DENSITY and not merely its footprint -- but on the
# stations that actually build the purple lines, i.e. after the completeness cut.
_ustn = pd.read_csv(str(UH_STN_FP)).set_index("ushcn_id")
_ukept = _ustn.reindex(pd.Index(_cols)[KEEP_UH])
assert _ukept[["lat", "lon"]].notna().all().all(), (
    "a USHCN station in the matched pivots is missing from stations.csv -- the "
    "footprint of the purple lines and of this one would then differ, which is "
    "the one thing this comparison may not do.")
_uil = np.rint((_ukept["lat"].to_numpy() - BLAT[0]) / (BLAT[1] - BLAT[0])).astype(int)
_uio = np.rint((_ukept["lon"].to_numpy() - BLON[0]) / (BLON[1] - BLON[0])).astype(int)
_uok = (_uil >= 0) & (_uil < len(BLAT)) & (_uio >= 0) & (_uio < len(BLON))
_uflat = np.ravel_multi_index((_uil[_uok], _uio[_uok]), (len(BLAT), len(BLON)))
_uflat = _uflat[KEEP_BE[_uflat]]
be_at_uh = to_series(record_counts(BV[:, _uflat], BYR, BMO, BDY),
                     np.ones(len(_uflat), bool),
                     label="Berkeley Earth at USHCN Stations")

if BLANK_PARTIAL and PARTIAL:
    for _s in (berkeley, be_at, be_at_uh, be_at_cells, be_at_masked):
        _s.loc[[y for y in PARTIAL if y in _s.index]] = np.nan

# ══ THE SMOOTHER ══════════════════════════════════════════════════════════════
# A centered ROLL-year mean. The window runs off the record in the last and
# first ROLL//2 years, and END_METHOD decides what happens there. Every option
# below leaves 1905-2019 bit-for-bit identical to the plain centered mean -- only
# the ends differ -- and each is a different answer to the same bias/variance
# question, because a one-sided window must either lag a trending series or pay
# variance to extrapolate it.
#
#   none      stop ROLL//2 years short (the original behaviour)
#   mean      average whatever the window catches. Lowest variance, but it
#             reports the mean of the last six years at the position of the
#             last one, so it sits ~2.5 yr behind and DAMPS a rising tail.
#   loclin    least-squares line through the same six-plus years, read at the
#             endpoint. Unbiased under a local trend, ~2.4x the interior's
#             standard error. Inside the record it IS the window mean, because
#             the line through a symmetric window read at its own centre is
#             that window's average.
#   savgol    as loclin, but the line is fit to the full ROLL-year window rather
#             than the shrinking one: less variance, more lag.
#   minrough  Mann (2004) -- pad the series past the end with the reflection
#             (flat, mirrored, or mirrored-through-the-endpoint) that leaves the
#             smoothed tail smoothest, then take the ordinary centered mean.
#
# Out-of-sample on these five series -- truncate at year T, compare each rule's
# estimate at T against the centered mean the boxcar eventually reports there --
# `mean` wins overall (RMSE 0.35 vs 0.69 for loclin) because most of the record
# is not trending, but that reverses exactly where it matters: over the 22
# episodes rising faster than +0.15 records/yr, which is the regime every one of
# these series is in after 2015, loclin scores 0.39 against 0.57 and carries a
# bias of -0.11 where `mean` under-reports the rise by -0.54.
HALF, MIN_WIN = ROLL // 2, ROLL // 2 + 1

def roll(s):
    """No smoothed value where the series itself has none: `savgol` and
    `minrough` would otherwise happily draw Berkeley's blanked 2024 out of the
    surrounding years, and a deliberately blanked year must stay blank."""
    return _roll(s).where(np.isfinite(s.values))

def _roll(s):
    x, y = s.index.values.astype(float), s.values.astype(float)
    box = s.rolling(ROLL, center=True, min_periods=ROLL).mean()
    if END_METHOD == "none":
        return box
    if END_METHOD in ("loclin", "savgol"):
        out = np.full(y.size, np.nan)
        for i in range(y.size):
            k = (slice(max(0, i - HALF), i + HALF + 1) if END_METHOD == "loclin" else
                 slice(min(max(0, i - HALF), max(0, y.size - ROLL)),
                       max(i + HALF + 1, min(ROLL, y.size))))
            xx, yy = x[k] - x[i], y[k]
            g = np.isfinite(yy)
            if g.sum() >= MIN_WIN:
                out[i] = np.polyfit(xx[g], yy[g], 1)[1]     # the line, read at x[i]
        return pd.Series(out, index=s.index)
    if END_METHOD == "mean":
        return s.rolling(ROLL, center=True, min_periods=MIN_WIN).mean()
    if END_METHOD == "minrough":
        def _pad(v, side):
            t = v[-(HALF + 1):] if side > 0 else v[:HALF + 1][::-1]
            t = t[np.isfinite(t)]
            if t.size < 2:
                return {k: np.full(HALF, np.nan) for k in ("flat", "even", "odd")}
            return {"flat": np.repeat(t[-1], HALF),
                    "even": t[-2::-1][:HALF],
                    "odd":  2 * t[-1] - t[-2::-1][:HALF]}
        best, bs = None, np.inf
        for kind in ("flat", "even", "odd"):
            lo, hi = _pad(y, -1)[kind][::-1], _pad(y, +1)[kind]
            z = pd.Series(np.concatenate([lo, y, hi])).rolling(
                ROLL, center=True, min_periods=MIN_WIN).mean().values[HALF:HALF + y.size]
            r = np.nansum(np.diff(z[:ROLL], 2) ** 2) + np.nansum(np.diff(z[-ROLL:], 2) ** 2)
            if r < bs:
                best, bs = z, r
        return pd.Series(best, index=s.index)
    raise ValueError(f"END_METHOD={END_METHOD!r}")

def end_uncertainty(s, n_min=3 * ROLL):
    """How wrong the reduced-window years are, measured on this series rather
    than asserted. Truncate at each year T, smooth the truncated record, and
    compare its estimate at T, T-1 ... against the centered mean the FULL record
    eventually reports there. Returns RMSE by distance from the endpoint -- the
    envelope the figure draws, and a number a caption can quote."""
    y = s.dropna().values.astype(float)
    truth = pd.Series(y).rolling(ROLL, center=True, min_periods=ROLL).mean().values
    err = {lag: [] for lag in range(HALF + 1)}
    for T in range(n_min, len(y) - HALF):
        est = _roll(pd.Series(y[:T + 1], index=np.arange(T + 1, dtype=float))).values
        for lag in range(HALF + 1):
            i = T - lag
            if np.isfinite(truth[i]) and np.isfinite(est[i]):
                err[lag].append(est[i] - truth[i])
    return np.array([np.sqrt(np.mean(np.square(err[l]))) if err[l] else np.nan
                     for l in range(HALF + 1)])

# ══ CHECK -- the smoother. Inside the record it must reproduce the plain
# centered mean to machine precision; only the ends are new. ══
_c9 = pd.Series(np.random.default_rng(1).normal(size=len(YEARS_F3)), index=YEARS_F3)
_r9 = _c9.rolling(ROLL, center=True, min_periods=ROLL).mean()
assert np.nanmax(np.abs(roll(_c9) - _r9)) < 1e-12, (
    f"END_METHOD={END_METHOD!r} moved the interior. Every option is allowed to "
    f"invent the outer {HALF} years and nothing else.")
_new9 = int(np.isfinite(roll(_c9)).sum() - np.isfinite(_r9).sum())
print(f"roll(): END_METHOD={END_METHOD!r} -- interior identical to the centered "
      f"{ROLL}-year mean, {_new9} reduced-window years added\n")

def end_band(ax, sm, unc, color, alpha=0.13):
    """Shade +/- END_BAND_K * the out-of-sample RMSE over the reduced-window
    years at each end of a smoothed curve."""
    yy = sm.dropna()
    if yy.empty:
        return
    for sgn in (+1, -1):                            # the tail, then the head
        y0 = yy.index[-1] if sgn > 0 else yy.index[0]
        ix = [y0 - sgn * l for l in range(HALF + 1)][::sgn]
        w = np.array([unc[abs(y0 - i)] for i in ix]) * END_BAND_K
        v = sm.reindex(ix).values
        ax.fill_between(ix, v - w, v + w, color=color, alpha=alpha, lw=0, zorder=2)

# ══ THE SAMPLING CORRECTION ════════════════════════════════════════════════
# Berkeley is the only dataset here that exists BOTH ways: on the full CONUS
# grid and on the USHCN footprint, from identical underlying fields. Their ratio
# is therefore what USHCN's sampling alone does to a record count, with the
# climate held fixed -- above 1 in years when the station network under-counts
# relative to full coverage, below 1 when it over-counts. Multiplying USHCN-BC
# through by it asks what USHCN-BC would have reported had it seen the whole
# country. Note both series integrate to the same 153 by construction, so the
# ratio redistributes records in TIME and invents none.
#
# The factor is taken from the SMOOTHED Berkeley pair, not year by year. Network
# sampling is a slowly varying property of where the stations are; an annual
# ratio of two noisy series is dominated by the noise (it ranges 0.35 to 2.4 here,
# and its denominator passes close to zero), and multiplying USHCN-BC by that
# would inject Berkeley's year-to-year scatter into a USHCN line: annually the
# ratio runs 0.58 to 1.99 against 0.76 to 1.36 for the smoothed pair, and its
# denominator falls to 0.057 records/yr. RATIO_SMOOTH False uses the raw one.
RATIO_SMOOTH = True
_num, _den = (roll(berkeley), roll(be_at_uh)) if RATIO_SMOOTH else (berkeley, be_at_uh)
be_samp_ratio = _num / _den
ushcn_bc_full = ushcn_bc * be_samp_ratio

NU  = {"GHCN-Daily": len(IDS), "USHCN-Daily": int(KEEP_UH.sum()),
       "Berkeley Earth": int(KEEP_BE.sum()), "USHCN-BC": int(KEEP_UH.sum()),
       "Berkeley Earth at GHCN Stations": len(_flat),
       "Berkeley Earth at USHCN Stations": len(_uflat),
       "USHCN-BC × BE/BE-at-USHCN": int(KEEP_UH.sum())}
SER = {"GHCN-Daily": ghcnd, "USHCN-Daily": ushcn, "Berkeley Earth": berkeley,
       "USHCN-BC": ushcn_bc, "Berkeley Earth at GHCN Stations": be_at,
       "Berkeley Earth at USHCN Stations": be_at_uh,
       "USHCN-BC × BE/BE-at-USHCN": ushcn_bc_full}

pd.DataFrame(SER).rename_axis("year").to_csv(SERIES_FP)

# ══ CHECK -- Figure 3 series. Each series must integrate to 153: every
# location holds exactly one record per calendar day it observed. to_series
# asserts this while building, so reaching this point already proves it. ══
def _ratio_F3(d):
    _a = float(d.loc[1930:1939].mean()); _b = float(np.nanmean(d.loc[2015:2024]))
    return _a, _b, _a / _b
print(f"{'dataset':<34}{'units':>8}{'1930s':>8}{'2015-24':>9}{'ratio':>7}{'integral':>10}")
for _lbl, _s in SER.items():
    _a, _b, _r = _ratio_F3(_s)
    print(f"  {_lbl:<32}{NU[_lbl]:>8,}{_a:>8.2f}{_b:>9.2f}{_r:>7.2f}{float(np.nansum(_s)):>10.1f}")
print("\nratio above 1.00 means the Dust Bowl set more records than the last decade")
_r30 = float(be_samp_ratio.loc[1930:1939].mean())
_r15 = float(np.nanmean(be_samp_ratio.loc[2015:2024]))
print(f"\nthe USHCN sampling factor, Berkeley full grid / Berkeley at the USHCN sites"
      f"\n  1930s {_r30:.3f}   2015-24 {_r15:.3f}   range {np.nanmin(be_samp_ratio):.3f}"
      f"-{np.nanmax(be_samp_ratio):.3f}"
      f"\n  above 1 = the USHCN footprint under-counts records relative to full coverage")

# ══ PLOT ═══════════════════════════════════════════════════════════════════
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 14, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "savefig.dpi": 300})
UNC = {l: end_uncertainty(s) for l, s in SER.items()} if END_METHOD != "none" else {}
if UNC:
    print(f"\nout-of-sample error of the END_METHOD={END_METHOD!r} years, RMSE against the "
          f"centered mean\nthe full record eventually reports there:")
    print(f"  {'dataset':<34}" + "".join(
        f"{('last yr' if l == 0 else f'-{l}'):>10}" for l in range(HALF + 1)))
    for _lbl, _u in UNC.items():
        print(f"  {_lbl:<34}" + "".join(f"{v:>10.2f}" for v in _u))
COL_BE_F3, COL_GH_F3, COL_UH = "#009E73", "#E63946", "#6D28D9"
LINES = [("GHCN-Daily",                      ghcnd,    COL_GH_F3, "-."),
         ("USHCN-Daily",                     ushcn,    COL_UH, "-."),
         ("Berkeley Earth",                  berkeley, COL_BE_F3, "-"),
         ("USHCN-BC",                        ushcn_bc, COL_UH, "-"),
         ("Berkeley Earth at GHCN Stations",  be_at,   COL_BE_F3, (0, (5, 1.6)))]

# no constrained_layout: it fights the figure-level legend and clips the title
fig = plt.figure(figsize=(11.5, 6.6))
fig.subplots_adjust(left=0.095, right=0.98, top=0.755, bottom=0.105)
ax = fig.add_subplot(111)
ax.axvspan(1930, 1939, color="tan", alpha=0.22, lw=0, zorder=0)
if END_METHOD != "none" and END_SHADE:
    for _x0, _x1 in ((1899, Y0_F3 + HALF - 0.5), (Y1_F3 - HALF + 0.5, 2025)):
        ax.axvspan(_x0, _x1, color="0.5", alpha=0.07, lw=0, zorder=0)
for _l, _s, _c, _ls in LINES:
    ax.plot(_s.index, _s.values, color=_c, lw=0.9, ls=_ls, alpha=0.26, zorder=1)
if END_METHOD != "none" and END_BAND:
    for _l, _s, _c, _ls in LINES:
        end_band(ax, roll(_s), UNC[_l], _c)
for _l, _s, _c, _ls in LINES:
    ax.plot(_s.index, roll(_s).values, color=_c, lw=2.6, ls=_ls, label=_l, zorder=3)
ax.set_xlim(1899, 2025)
ax.set_ylim(0, 6)
ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
ax.set_xlabel("Year")
ax.set_ylabel("Per grid-cell / per-station average\nnumber of TMAX records per year")
ax.grid(True, axis="y", color="0.88", lw=0.8)
ax.grid(False, axis="x")
h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.915), ncol=3,
           frameon=False, handlelength=2.8, columnspacing=1.6, fontsize=12)
fig.suptitle("CONUS Daily TMAX Record Frequency",
             fontsize=20, fontweight="bold", y=0.975)
OUT_F3 = FIGD / (f"Fig3_final_{TAG_F3}_roll{ROLL}"
                 f"{'' if END_METHOD == 'none' else '_end-' + END_METHOD}_ylim0-6.png")
fig.savefig(OUT_F3, bbox_inches="tight", dpi=300, facecolor="white")
fig.savefig(OUT_F3.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
plt.show()
plt.close(fig)

print(f"\nwrote {OUT_F3}")
print(f"wrote {OUT_F3.with_suffix('.pdf')}")
print(f"wrote {SERIES_FP}")

# ══ PANEL VERSION ══════════════════════════════════════════════════════════
# The same five series, one per panel, in the layout and style vocabulary of
# Figure 6's FigHW_conus_vs_global_ushcn_overlay_wide: a grid of wide panels,
# bold column titles, shared y so the panels are comparable by eye, annual
# trace under a heavy smoothed curve, Dust Bowl in firebrick.
#
#   solid     homogenised / bias-corrected     USHCN-BC, Berkeley Earth
#   dot-dash  UNCORRECTED station data         GHCN-Daily, USHCN-Daily
#   dashed    re-sampled onto another network  Berkeley Earth at GHCN Stations
#
# Note the colour change against the overlay figure: the re-sampled Berkeley
# line is the DARKER green here, not the same green as the full grid, which is
# Figure 6's fix for two heavy lines of one colour merging wherever they cross.
# It only matters in the sixth panel, where all five are drawn together.
PANEL_W_F3, PANEL_H_F3 = 5.6, 3.25
PANEL_YLIM_F3 = (0, 6)           # as the overlay figure: the 1930s spikes clip
PANEL_OVERLAY = False            # True spends the spare slot on all of them together
PANEL_STAT    = False            # True adds each line's 1930s / 2015-24 ratio
LS_UNCORR_F3  = (0, (6.5, 1.8, 1.0, 1.8))   # uncorrected station data
LS_SAMP_GH    = (0, (7.0, 2.4))             # Berkeley re-sampled onto GHCN-Daily
LS_SAMP_UH    = (0, (2.6, 1.8))             # ... and onto USHCN, the shorter dash
LS_CORR_F3    = (0, (4.2, 2.0))             # a station product carried to full coverage
C_SAMPLED_F3  = "#00513C"                   # Figure 6's colour for a re-sampled line
C_SAMP_UH_F3  = "#3F9B78"                   # ... and a second value of it, for USHCN
C_CORR_F3     = "#A78BFA"                   # USHCN's hue, lightened: same data, rescaled
# (title, [(label, series, colour, linestyle, draw the annual trace too)])
PANELS = [
    ("GHCN-Daily",  [("GHCN-Daily", ghcnd, COL_GH_F3, LS_UNCORR_F3, True)]),
    ("USHCN-Daily", [("USHCN-Daily", ushcn, COL_UH, LS_UNCORR_F3, True)]),
    ("USHCN-BC",    [("USHCN-BC", ushcn_bc, COL_UH, "-", True),
                     ("USHCN-BC × BE/BE-at-USHCN", ushcn_bc_full,
                      C_CORR_F3, LS_CORR_F3, False)]),
    ("Berkeley Earth", [("Berkeley Earth", berkeley, COL_BE_F3, "-", True)]),
    # both re-samplings share a panel, so the reader sees what changes when the
    # SAME Berkeley field is read at one network's sites rather than the other's
    ("Berkeley Earth at Station Sites",
                    [("Berkeley Earth at GHCN Stations", be_at,
                      C_SAMPLED_F3, LS_SAMP_GH, True),
                     ("Berkeley Earth at USHCN Stations", be_at_uh,
                      C_SAMP_UH_F3, LS_SAMP_UH, False)]),
]
OVERLAY = [ln for _t, lns in PANELS for ln in lns]
N_PANELS = len(PANELS) + (1 if PANEL_OVERLAY else 0)

figp, axesp = plt.subplots(2, 3, figsize=(PANEL_W_F3 * 3, PANEL_H_F3 * 2.28),
                           sharey=True, gridspec_kw={"hspace": 0.34, "wspace": 0.075})
for _i, _ax in enumerate(axesp.ravel()):
    _last = _i == len(PANELS)
    if _last and PANEL_OVERLAY:
        _title, _lines = "All series, overlaid", OVERLAY
    elif _i < len(PANELS):
        _title, _lines = PANELS[_i]
    else:
        _ax.set_visible(False)
        continue
    _ax.axvspan(1930, 1939, color="firebrick", alpha=0.07, lw=0, zorder=0)
    if END_METHOD != "none" and END_SHADE:
        # the shaded years follow each panel's own record: Berkeley's blanked
        # 2024 means its reduced-window years are 2019-2023, not 2020-2024
        _e0 = min(s.dropna().index[0] for _, s, _, _, _ in _lines)
        _e1 = max(s.dropna().index[-1] for _, s, _, _, _ in _lines)
        for _x0, _x1 in ((Y0_F3 - 1, _e0 + HALF - 0.5), (_e1 - HALF + 0.5, Y1_F3 + 1)):
            _ax.axvspan(_x0, _x1, color="0.5", alpha=0.07, lw=0, zorder=0)
    for _l, _s, _c, _ls, _thin in _lines:
        if not _last:
            if _thin:
                _ax.plot(_s.index, _s.values, color=_c, lw=0.7, ls=_ls, alpha=0.30, zorder=1)
            if END_METHOD != "none" and END_BAND:
                end_band(_ax, roll(_s), UNC[_l], _c)
        _ax.plot(_s.index, roll(_s).values, color=_c, lw=2.5, ls=_ls, zorder=3,
                 label=_l, solid_capstyle="round", dash_capstyle="round")
    _ax.set_xlim(Y0_F3 - 1, Y1_F3 + 1)
    _ax.set_ylim(*PANEL_YLIM_F3)
    _ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    _ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
    _ax.tick_params(axis="both", labelsize=14, length=5, width=1.1)
    _ax.tick_params(axis="x", which="minor", length=2.5, width=0.9)
    _ax.grid(True, axis="y", color="0.90", lw=0.7)
    _ax.grid(False, axis="x")
    _ax.set_axisbelow(True)
    _ax.set_title(_title, fontweight="bold", pad=8, fontsize=17)
    # the 1930s / 2015-24 ratio rides in the legend where there is one -- a
    # second text block in the same corner collides with it -- and sits top
    # right in the single-line panels, which have no legend to carry it
    if len(_lines) > 1:
        _h9, _l9 = _ax.get_legend_handles_labels()
        _stat9 = PANEL_STAT and not _last
        if _stat9:
            _l9 = [f"{_t9}  ({_ratio_F3(_s9)[2]:.2f})"
                   for _t9, (_, _s9, _, _, _) in zip(_l9, _lines)]
        _ax.legend(_h9, _l9, loc="upper left", frameon=False,
                   fontsize=10.5 if _last else 11.5, handlelength=2.4,
                   borderaxespad=0.2, labelspacing=0.22, handletextpad=0.5,
                   title="1930s / 2015–24 in ( )" if _stat9 else None,
                   title_fontsize=10.5, alignment="left")
    elif PANEL_STAT and not _last:
        _ax.text(0.97, 0.95, f"1930s / 2015–24 = {_ratio_F3(_lines[0][1])[2]:.2f}",
                 transform=_ax.transAxes, ha="right", va="top", fontsize=12.5,
                 color="0.30")
    if _i % 3 == 0:
        _ax.set_ylabel("Records per year\n(per station or grid cell)",
                       fontsize=15, labelpad=8)
    if _i + 3 >= N_PANELS:            # nothing below this panel in its column
        _ax.set_xlabel("Year", fontsize=15, labelpad=6)
figp.suptitle("CONUS Daily TMAX Record Frequency", fontsize=25, fontweight="bold",
              y=0.985)
OUT_P3 = FIGD / (f"Fig3_panels_{TAG_F3}_roll{ROLL}"
                 f"{'' if END_METHOD == 'none' else '_end-' + END_METHOD}_wide.png")
figp.savefig(OUT_P3, bbox_inches="tight", dpi=200, facecolor="white")
figp.savefig(OUT_P3.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
plt.show()
plt.close(figp)
print(f"wrote {OUT_P3}")
print(f"wrote {OUT_P3.with_suffix('.pdf')}")

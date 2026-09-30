#!/usr/bin/env python3
"""Figure 2 -- CONUS daily TMAX record frequency, May-September.

For each station or grid cell and each May-September calendar day, which year
holds the highest TMAX of 1900-2025. Ties are split. Notebook section 7, from
which this script has since diverged; see figures/README.md.

The heavy lines are LOWESS curves (common.lowess_smooth), which run the whole
length of each record, first year of data to last.

This also builds the daily station cube that Figure 5 reads; the cube lives in
common.py, so either figure can be run first.

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
check_smoother()
print()

# ══ SETTINGS AND HELPERS ═══════════════════════════════════════════════════
warnings.filterwarnings("ignore", category=RuntimeWarning)
for _d in (FIGD, CACHE_REC): _d.mkdir(parents=True, exist_ok=True)

# Y0_F2, Y1_F2, MONTHS, NDAYS, YEARS_F2, MIN_YEARS, MIN_FRAC, TMAX_BOUNDS_F2,
# TAG_F2, COV_FP, DAILY_FP and the KEYS/KPOS/YR/MO/DY calendar come from
# common.py, because Figure 5 needs the same cube and the same tag.
MIN_CELL_FRAC = 0.50             # a Berkeley cell must be >=50% inside CONUS
BE_LAND_MIN   = 0.50             # ... and >50% land in Berkeley's own mask
BLANK_PARTIAL = True             # Berkeley's daily release stops 2024-08-31
SERIES_FP = CACHE_REC / f"fig2_all_series_{TAG_F2}_cf{MIN_CELL_FRAC:g}.csv"
SMOOTH_FP = CACHE_REC / f"fig2_smooth_{TAG_F2}_cf{MIN_CELL_FRAC:g}.csv"    # the plotted lines

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
    ymap = {y: i for i, y in enumerate(YEARS_F2)}
    out = np.zeros((len(YEARS_F2), vals.shape[1]))
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
    s = pd.Series((c * ww[None, :]).sum(axis=1) / ww.sum(), index=YEARS_F2)
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
_us = np.isin(_ut.month, MONTHS) & (_ut.year >= Y0_F2) & (_ut.year <= Y1_F2)
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
_bs = np.isin(_bt.month, MONTHS) & (_bt.year >= Y0_F2) & (_bt.year <= Y1_F2)
BV = _da.values[_bs].reshape(int(_bs.sum()), -1)
BYR, BMO, BDY = _bt.year.values[_bs], _bt.month.values[_bs], _bt.day.values[_bs]
_blast = _bt[-1]
_ds.close()
# Every year the Berkeley release does not cover in full. Its last year is
# partial whenever the file does not end on 31 December, and once the record
# runs past that year the remaining ones are absent from the file entirely --
# which the record kernel would otherwise score as a hard ZERO rather than as
# missing, dragging the line to the floor. Both cases are blanked below.
_blast_full = int(_blast.year) - (0 if (_blast.month == 12 and _blast.day == 31) else 1)
PARTIAL = [int(y) for y in YEARS_F2 if y > _blast_full]

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
# LOWESS, from common.py: lowess_smooth() and end_revision(), with LOWESS_YEARS /
# LOWESS_IT, and the synthetic check_smoother() that ran at the top. Figures 1,
# 4 and 5 draw the same smooth. This figure's series are gap-free apart from the
# years Berkeley lacks at the end, which the smooth simply stops short of.

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
# The factor is taken from the SMOOTHED Berkeley pair, not year by year, and here
# the smooth is the same LOWESS the figure draws. Network sampling is a slowly
# varying property of where the stations are; an annual ratio of two noisy series
# is dominated by the noise, and multiplying USHCN-BC by that would inject
# Berkeley's year-to-year scatter into a USHCN line: annually the ratio runs about
# 0.6 to 2.0 and its denominator falls to about 0.06 records/yr, against 0.76 to
# 1.38 for the LOWESS pair. RATIO_SMOOTH False uses the raw one.
RATIO_SMOOTH = True
_num, _den = ((lowess_smooth(berkeley), lowess_smooth(be_at_uh)) if RATIO_SMOOTH
              else (berkeley, be_at_uh))
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

# ══ CHECK -- Figure 2 series. Each series must integrate to 153: every
# location holds exactly one record per calendar day it observed. to_series
# asserts this while building, so reaching this point already proves it. ══
def _ratio_F2(d):
    _a = float(d.loc[1930:1939].mean()); _b = float(np.nanmean(d.loc[2015:2024]))
    return _a, _b, _a / _b
print(f"{'dataset':<34}{'units':>8}{'1930s':>8}{'2015-24':>9}{'ratio':>7}{'integral':>10}")
for _lbl, _s in SER.items():
    _a, _b, _r = _ratio_F2(_s)
    print(f"  {_lbl:<32}{NU[_lbl]:>8,}{_a:>8.2f}{_b:>9.2f}{_r:>7.2f}{float(np.nansum(_s)):>10.1f}")
print("\nratio above 1.00 means the Dust Bowl set more records than the last decade")
_r30 = float(be_samp_ratio.loc[1930:1939].mean())
_r15 = float(np.nanmean(be_samp_ratio.loc[2015:2024]))
print(f"\nthe USHCN sampling factor, Berkeley full grid / Berkeley at the USHCN sites"
      f"\n  1930s {_r30:.3f}   2015-24 {_r15:.3f}   range {np.nanmin(be_samp_ratio):.3f}"
      f"-{np.nanmax(be_samp_ratio):.3f}"
      f"\n  above 1 = the USHCN footprint under-counts records relative to full coverage")

# ══ TRENDS -- the numbers the figure's paragraph quotes ════════════════════
# OLS trend on the ANNUAL series, never the smoothed one: an 11-year mean
# manufactures serial correlation and would make every trend look far more
# significant than the data support.
#
# Record counts are strongly autocorrelated even unsmoothed -- a run of hot
# summers sets records together, and the metric itself is path dependent, since
# a record once set cannot be reset by a later year unless that year is hotter.
# The ordinary OLS t-test assumes independent residuals and so overstates
# significance here. The effective sample size below is the standard lag-1
# correction of Santer et al. (2000), n_eff = n (1 - r1) / (1 + r1), applied to
# the regression residuals. Both p-values are printed: quote the adjusted one.
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
        p_ar1 = float(2 * stats.t.sf(abs(slope / (se * np.sqrt((n - 2) / (n_eff - 2)))),
                                     n_eff - 2))
    else:
        p_ar1 = 1.0
    return slope * 100.0, p_ols, p_ar1, n, r1

def _star(p):
    return "*" if np.isfinite(p) and p < 0.05 else " "

print(f"\ntrends in records per year per station or grid cell, PER CENTURY, "
      f"{Y0_F2}-{Y1_F2}\n  * = p < 0.05 after the lag-1 correction")
print(f"  {'dataset':<34}{'trend':>9}{'p(OLS)':>9}{'p(AR1)':>9}{'r1':>7}{'n':>5}")
for _lbl, _s in SER.items():
    _t, _po, _pa, _n, _r1 = trend_F2(_s)
    print(f"  {_lbl:<34}{_t:>+8.2f}{_star(_pa)}{_po:>9.3f}{_pa:>9.3f}{_r1:>7.2f}{_n:>5}")

print(f"\ntrend per century by start year, all ending {Y1_F2} "
      f"(* = p < 0.05, lag-1 corrected)")
print(f"  {'dataset':<34}" + "".join(f"{y:>9}" for y in TREND_STARTS_F2))
for _lbl, _s in SER.items():
    _cells = []
    for _y0 in TREND_STARTS_F2:
        _t, _po, _pa, _n, _r1 = trend_F2(_s, _y0)
        _cells.append("       --" if not np.isfinite(_t)
                      else f"{_t:>+8.2f}{_star(_pa)}")
    print(f"  {_lbl:<34}" + "".join(_cells))

# ══ THE SMOOTHED SERIES ════════════════════════════════════════════════════
SM = {l: lowess_smooth(s) for l, s in SER.items()}

# ══ CHECK -- the line must run the whole length of every record, first year of
# data to last, and stop where the data stop. lowess_smooth() does this by
# construction; asserted here so that no later change can quietly shorten a line. ══
print(f"\nLOWESS, {LOWESS_YEARS}-year tricube window, robustness iterations: {LOWESS_IT}")
print(f"  {'dataset':<34}{'data':>12}{'line':>12}{'frac':>8}")
for _lbl, _s in SER.items():
    _m = SM[_lbl]
    assert (_m.notna() == _s.notna()).all(), f"{_lbl}: the line does not span exactly the data"
    print(f"  {_lbl:<34}{_s.first_valid_index():>6}-{_s.last_valid_index():<5}"
          f"{_m.first_valid_index():>6}-{_m.last_valid_index():<5}{LOWESS_YEARS / _s.count():>8.3f}")
pd.DataFrame(SM).rename_axis("year").to_csv(SMOOTH_FP)

# The last years of ANY smooth are provisional: they rest on one side of the
# record and move as later years arrive. This measures how much, on these series.
REV = {l: end_revision(s) for l, s in SER.items()}
print(f"\nhow far the smooth's last years move as later years arrive -- RMSE against the "
      f"value the\nfull record eventually gives there:")
print(f"  {'dataset':<34}" + "".join(
    f"{('last yr' if l == 0 else f'-{l}'):>9}" for l in range(len(next(iter(REV.values()))))))
for _lbl, _u in REV.items():
    print(f"  {_lbl:<34}" + "".join(f"{v:>9.2f}" for v in _u))

# Why LOWESS_IT is 0: the same smooth with statsmodels' default three robustness
# iterations, which read the 1930s spikes as outliers and pull the curve through them.
print(f"\nwhy LOWESS_IT = {LOWESS_IT}: the 1930s peak of the smooth, without and with "
      f"robustness iterations")
print(f"  {'dataset':<34}{'annual':>9}{'it=0':>8}{'it=3':>8}{'lowered':>9}")
for _lbl, _s in SER.items():
    _p0 = float(lowess_smooth(_s, it=0).loc[1930:1939].max())
    _p3 = float(lowess_smooth(_s, it=3).loc[1930:1939].max())
    print(f"  {_lbl:<34}{_s.loc[1930:1939].max():>9.2f}{_p0:>8.2f}{_p3:>8.2f}"
          f"{1 - _p3 / _p0:>9.0%}")

# ══ PLOT ═══════════════════════════════════════════════════════════════════
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 14, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "savefig.dpi": 300})
COL_BE_F2, COL_GH_F2, COL_UH = "#009E73", "#E63946", "#6D28D9"
print(f"\nwrote {SERIES_FP}\nwrote {SMOOTH_FP}")

# ══ PANEL VERSION ══════════════════════════════════════════════════════════
# Four series (GHCN-Daily suppressed, see PANELS below), one per panel in a
# 2x2 grid, in the layout and style vocabulary of Figure 5's
# FigHW_conus_vs_global_ushcn_overlay_wide: bold column titles, shared y so
# the panels are comparable by eye, annual trace under a heavy smoothed
# curve, Dust Bowl in firebrick.
#
#   solid     homogenised / bias-corrected     USHCN-BC, Berkeley Earth
#   dot-dash  UNCORRECTED station data         USHCN-Daily
#   dashed    re-sampled onto another network  Berkeley Earth at USHCN Stations
#
# Note the colour change against the overlay figure: the re-sampled Berkeley
# line is a DARKER green here, not the same green as the full grid, which is
# Figure 5's fix for two heavy lines of one colour merging wherever they cross.
PANEL_W_F2, PANEL_H_F2 = 5.6, 3.25
PANEL_YLIM_F2 = (0, 6)           # as the overlay figure: the 1930s spikes clip
PANEL_OVERLAY = False            # True spends the spare slot on all of them together
PANEL_STAT    = False            # True adds each line's 1930s / 2015-24 ratio
LS_UNCORR_F2  = (0, (6.5, 1.8, 1.0, 1.8))   # uncorrected station data
LS_SAMP_GH    = (0, (7.0, 2.4))             # Berkeley re-sampled onto GHCN-Daily
LS_SAMP_UH    = (0, (2.6, 1.8))             # ... and onto USHCN, the shorter dash
LS_CORR_F2    = (0, (4.2, 2.0))             # a station product carried to full coverage
C_SAMPLED_F2  = "#00513C"                   # Figure 5's colour for a re-sampled line
C_SAMP_UH_F2  = "#3F9B78"                   # ... and a second value of it, for USHCN
C_CORR_F2     = "#A78BFA"                   # USHCN's hue, lightened: same data, rescaled
# (title, [(label, series, colour, linestyle, draw the annual trace too)])
PANELS = [
    # ("GHCN-Daily", [("GHCN-Daily", ghcnd, COL_GH_F2, LS_UNCORR_F2, True)]),
    # GHCN-Daily suppressed from the plot; ghcnd is still computed above and
    # carried in SER/NU for the console CHECK and TRENDS tables and the CSV.
    ("USHCN-Daily", [("USHCN-Daily", ushcn, COL_UH, LS_UNCORR_F2, True)]),
    ("USHCN-BC",    [("USHCN-BC", ushcn_bc, COL_UH, "-", True),
                     ("USHCN-BC × BE/BE-at-USHCN", ushcn_bc_full,
                      C_CORR_F2, LS_CORR_F2, False)]),
    ("Berkeley Earth", [("Berkeley Earth", berkeley, COL_BE_F2, "-", True)]),
    # ("Berkeley Earth at GHCN Stations", be_at, C_SAMPLED_F2, LS_SAMP_GH, True)
    # removed from this panel; be_at is still computed above and carried in
    # SER/NU for the console CHECK and TRENDS tables and the CSV.
    ("Berkeley Earth at USHCN Stations",
                    [("Berkeley Earth at USHCN Stations", be_at_uh,
                      C_SAMP_UH_F2, LS_SAMP_UH, True)]),
]
OVERLAY = [ln for _t, lns in PANELS for ln in lns]
N_PANELS = len(PANELS) + (1 if PANEL_OVERLAY else 0)

figp, axesp = plt.subplots(2, 2, figsize=(PANEL_W_F2 * 2, PANEL_H_F2 * 2.28),
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
    # no end shading and no end band: LOWESS has no reduced-window years to mark,
    # and the smooth is drawn over exactly the years its series has data for
    for _l, _s, _c, _ls, _thin in _lines:
        if _thin and not _last:
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
    # panel letter, top right. axesp.ravel() is row-major, so this is (a)-(b)
    # across the top row and (c)-(d) on the bottom, matching the reading
    # order of the titles.
    _ax.text(0.985, 0.96, f"({'abcdef'[_i]})", transform=_ax.transAxes,
             ha="right", va="top", fontsize=16, fontweight="bold")
    # the 1930s / 2015-24 ratio rides in the legend where there is one -- a
    # second text block in the same corner collides with it -- and sits top
    # right in the single-line panels, which have no legend to carry it
    if len(_lines) > 1:
        _h9, _l9 = _ax.get_legend_handles_labels()
        _stat9 = PANEL_STAT and not _last
        if _stat9:
            _l9 = [f"{_t9}  ({_ratio_F2(_s9)[2]:.2f})"
                   for _t9, (_, _s9, _, _, _) in zip(_l9, _lines)]
        _ax.legend(_h9, _l9, loc="upper left", frameon=False,
                   fontsize=10.5 if _last else 11.5, handlelength=2.4,
                   borderaxespad=0.2, labelspacing=0.22, handletextpad=0.5,
                   title="1930s / 2015–24 in ( )" if _stat9 else None,
                   title_fontsize=10.5, alignment="left")
    elif PANEL_STAT and not _last:
        _ax.text(0.97, 0.86, f"1930s / 2015–24 = {_ratio_F2(_lines[0][1])[2]:.2f}",
                 transform=_ax.transAxes, ha="right", va="top", fontsize=12.5,
                 color="0.30")
    if _i % 2 == 0:
        _ax.set_ylabel("Records per year\n(per station or grid cell)",
                       fontsize=15, labelpad=8)
    if _i + 2 >= N_PANELS:            # nothing below this panel in its column
        _ax.set_xlabel("Year", fontsize=15, labelpad=6)
_SUPTITLE_P2 = None              # the panel titles carry the figure; no banner
if _SUPTITLE_P2:
    figp.suptitle(_SUPTITLE_P2, fontsize=25, fontweight="bold", y=0.985)
OUT_P2 = FIGD / "Figure2.png"
figp.savefig(OUT_P2, bbox_inches="tight", dpi=200, facecolor="white")
plt.show()
plt.close(figp)
print(f"wrote {OUT_P2}")

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
ROLL = 5
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

if BLANK_PARTIAL and PARTIAL:
    for _s in (berkeley, be_at, be_at_cells, be_at_masked):
        _s.loc[[y for y in PARTIAL if y in _s.index]] = np.nan

SER = {"GHCN-Daily": ghcnd, "USHCN-Daily": ushcn, "Berkeley Earth": berkeley,
       "USHCN-BC": ushcn_bc, "Berkeley Earth at GHCN Stations": be_at}
NU  = {"GHCN-Daily": len(IDS), "USHCN-Daily": int(KEEP_UH.sum()),
       "Berkeley Earth": int(KEEP_BE.sum()), "USHCN-BC": int(KEEP_UH.sum()),
       "Berkeley Earth at GHCN Stations": len(_flat)}

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

# ══ PLOT ═══════════════════════════════════════════════════════════════════
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 14, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "savefig.dpi": 300})
roll = lambda s: s.rolling(ROLL, center=True, min_periods=ROLL).mean()
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
for _l, _s, _c, _ls in LINES:
    ax.plot(_s.index, _s.values, color=_c, lw=0.9, ls=_ls, alpha=0.26, zorder=1)
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
OUT_F3 = FIGD / f"Fig3_final_{TAG_F3}_roll{ROLL}_ylim0-6.png"
fig.savefig(OUT_F3, bbox_inches="tight", dpi=300, facecolor="white")
fig.savefig(OUT_F3.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
plt.show()
plt.close(fig)

print(f"\nwrote {OUT_F3}")
print(f"wrote {OUT_F3.with_suffix('.pdf')}")
print(f"wrote {SERIES_FP}")

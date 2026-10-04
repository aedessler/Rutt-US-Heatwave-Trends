#!/usr/bin/env python3
"""Build the master annual table: one CSV with every annual series the figures plot.

Rows are years (1900-2026); columns are named  <domain>_<quantity>__<dataset>  and hold, for the
paper's figures 1, 2, 4 and 5:

    conus_jja_{tmax,tmin,tavg}_anom__<ds>   Figure 1  CONUS JJA anomaly vs 1951-1980 (deg C)
    conus_jja_tavgmean_anom__<ds>           the same with TAVG from the mean of the day's samples (reanalyses)
    conus_records__<ds>                     Figure 2  all-time daily TMAX records per cell per year, May-Sep
    band_jja_{tmax,tmin,tavg,tavgmean}_anom__<ds>   Figure 4  24-50N land JJA anomaly (deg C)
    conus_hw_days__<ds>, band_hw_days__<ds> Figure 5  heat-wave days per year (CONUS; 24-50N land)

with <ds> in  era5 berkeley nclimdiv ushcn ushcn_bc noaa crutem5 20cr era20c  (and, for the records and
heat waves, berkeley_at_ushcn, ushcn_bc_x_sampling). No GHCN data is used. `--list` prints every
column; master_annual_table_columns.csv, written beside the CSV, describes each one. The figure
scripts figure{1,2,4,5}_20cr_era20c.py only read this CSV (LOWESS smoothing is theirs, not stored).

The computation is the figures' own, moved here: each section's functions are the originals'
(figure1.py, figure2.py, figure4.py, figure5.py), with the GHCN branches removed and the SAME cache
names, so datasets already cached load from cache and give identical numbers. 20CR and ERA-20C come
from the NH_mid_latitude local-time files on the lab archive through reanalysis.py.

    python figures/build_master_table.py                       # every missing column
    python figures/build_master_table.py --sections 1 4        # only those figures' columns
    python figures/build_master_table.py --datasets 20cr era20c
    python figures/build_master_table.py --rebuild --datasets era5   # recompute columns already present
    python figures/build_master_table.py --list

Existing columns are kept unless asked to rebuild, so adding a dataset later is incremental.

COPIES ARE DELETED once the CSV and its columns file are written: everything this run copied from
/Volumes/adessler_lab -- the staged archive files under _stage/ (including those left by earlier
runs) and the per-year 20CR / ERA-20C extracts -- is removed. The derived caches (_cache_*) are
computed products, not copies, and stay. `--keep-copies` skips the deletion.
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *          # noqa: F401,F403
from reanalysis import (RA, jja_monthly_field, jja_annual, mjjas_cube, berkeley_land_fraction,
                        check_local_day, check_localday_files, remove_copies)

warnings.filterwarnings("ignore", category=RuntimeWarning)
for _d in (FIGD, CACHE_ANOM, CACHE_REC, CACHE_BAND, CACHE_HW):
    _d.mkdir(parents=True, exist_ok=True)

CSV_DEFAULT = FIGD / "master_annual_table.csv"
YEARS_MASTER = np.arange(1900, 2027)

# ── what the table can hold ───────────────────────────────────────────────────
DATASET_LABEL = {
    "era5": "ERA5", "berkeley": "Berkeley Earth", "nclimdiv": "nCLIMDIV", "ushcn": "USHCN-Daily",
    "ushcn_bc": "USHCN-BC", "noaa": "NOAAGlobalTemp", "crutem5": "CRUTEM5", "20cr": "20CR",
    "era20c": "ERA-20C", "berkeley_at_ushcn": "Berkeley Earth at USHCN Stations",
    "ushcn_bc_x_sampling": "USHCN-BC x BE/BE-at-USHCN"}
SECTION_DATASETS = {
    1: ["era5", "berkeley", "nclimdiv", "ushcn", "ushcn_bc", "noaa", "crutem5", "20cr", "era20c"],
    2: ["ushcn", "ushcn_bc", "berkeley", "berkeley_at_ushcn", "ushcn_bc_x_sampling", "20cr", "era20c"],
    4: ["berkeley", "era5", "20cr", "era20c"],
    5: ["berkeley", "era5", "20cr", "era20c", "ushcn", "ushcn_bc", "berkeley_at_ushcn"],
}
_BAND_HW = {"berkeley", "era5", "20cr", "era20c"}


def section_columns(sec, ds):
    """[(column, units, note)] that section `sec` writes for dataset `ds`."""
    anom = "JJA mean anomaly against the 1951-1980 monthly climatology, anomaly-first"
    if sec == 1:
        elems = ("tavg",) if ds in ("noaa", "crutem5") else ("tmax", "tmin", "tavg")
        cols = [(f"conus_jja_{e}_anom__{ds}", "degC", f"CONUS {anom}, area-weighted") for e in elems]
        if ds in RA:
            cols.append((f"conus_jja_tavgmean_anom__{ds}", "degC",
                         f"CONUS {anom}; TAVG = mean of the day's 8 samples, not (TMAX+TMIN)/2"))
        return cols
    if sec == 2:
        note = ("number of May-Sep days whose all-time TMAX record fell in that year, per station or grid "
                "cell (ties split); each series is counted over its own years")
        if ds == "ushcn_bc_x_sampling":
            note = "USHCN-BC records multiplied by the LOWESS ratio Berkeley / Berkeley-at-USHCN (sampling correction)"
        return [(f"conus_records__{ds}", "records per year per cell", note)]
    if sec == 4:
        elems = ("tmax", "tmin", "tavg") + (("tavgmean",) if ds in RA else ())
        return [(f"band_jja_{e}_anom__{ds}", "degC", f"24-50N land-mean {anom}, native grid, MAD guard") for e in elems]
    if sec == 5:
        hw = "days in runs of >= 6 above the cell's day-of-season 90th percentile (Christy 2026), May-Sep"
        cols = [(f"conus_hw_days__{ds}", "heat-wave days per year", f"CONUS mean, {hw}")]
        if ds in _BAND_HW:
            cols.append((f"band_hw_days__{ds}", "heat-wave days per year", f"24-50N land mean, {hw}"))
        return cols
    raise KeyError(sec)



# ==============================================================================
# SECTION 1 -- CONUS JJA temperature anomalies (figure1.py's computation)
# ==============================================================================

NCLIMDIV_FILES_F1 = {e: nclimdiv_file(e) for e in ("tmax", "tmin", "tavg")}
NCLIMDIV_ELEM_F1  = dict(tmax=27, tmin=28, tavg=2)     # element codes in the statewide file
YEARS_F1   = np.arange(1900, 2027)                      # his YEAR_START..YEAR_END
JJA_F1     = (6, 7, 8)
IDW_KW_F1 = dict(power=2.0, k=8, radius_km=150.0)
ERA5_Y0_F1, ERA5_Y1_F1 = 1940, 2025

def gridded_series_F1(field, weights):
    """his jja_series: cell anomalies -> area mean -> JJA day-weighted mean."""
    m, frac = area_mean(cell_anomalies(field), weights)
    return (pd.Series(season_mean(_to_year_month(m, YEARS_F1), JJA_F1, YEARS_F1), index=YEARS_F1),
            pd.Series(season_mean(_to_year_month(frac, YEARS_F1), JJA_F1, YEARS_F1), index=YEARS_F1))

def station_series_F1(dataset, elem, weights05, gridder):
    """the whole station chain, cached at the gridded-field stage."""
    key = (f"grid_{dataset}_{elem}_p{IDW_KW_F1['power']:g}_k{IDW_KW_F1['k']}"
           f"_r{IDW_KW_F1['radius_km']:g}_{YEARS_F1[0]}_{YEARS_F1[-1]}")
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
                fp = era5_file(y, mo)
                if not fp.exists():
                    missing.append(f"{y}-{mo:02d}")
                    continue
                ds = None
                try:
                    ds = xr.open_dataset(str(stage(fp)))
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
    return _cache(f"field_era5_{elem}_{ERA5_Y0_F1}_{ERA5_Y1_F1}", build)

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


def section1(want):
    """{column: Series} for the Figure 1 datasets in `want`."""
    S = {}
    if want & {"ushcn", "ushcn_bc"}:
        lat05, lon05 = g05_centers()
        W05 = cell_weights(lat05, lon05, key="g05")
        gridder = IDWGrid(lat05, lon05, **IDW_KW_F1)
        for ds in ("ushcn", "ushcn_bc"):
            if ds in want:
                for e in ("tmax", "tmin", "tavg"):
                    S[(ds, e)] = station_series_F1(ds, e, W05, gridder)[0]
    if "berkeley" in want:
        be = {e: berkeley_field_F1(fp, e) for e, fp in (("tmax", BE_TMAX), ("tmin", BE_TMIN))}
        W = cell_weights(be["tmax"]["lat"].values, be["tmax"]["lon"].values, key="berkeley")
        for e in ("tmax", "tmin"):
            S[("berkeley", e)] = gridded_series_F1(be[e], W)[0]
        S[("berkeley", "tavg")] = 0.5 * (S[("berkeley", "tmax")] + S[("berkeley", "tmin")])
    if "era5" in want:
        e5 = {e: era5_field_F1(v, e) for e, v in (("tmax", "t2m_max"), ("tmin", "t2m_min"))}
        W = cell_weights(e5["tmax"]["lat"].values, e5["tmax"]["lon"].values, key="era5")
        for e in ("tmax", "tmin"):
            S[("era5", e)] = gridded_series_F1(e5[e], W)[0]
        S[("era5", "tavg")] = 0.5 * (S[("era5", "tmax")] + S[("era5", "tmin")])
    if "noaa" in want:
        no = noaa_field()
        W = cell_weights(no["lat"].values, no["lon"].values, key="noaa")
        S[("noaa", "tavg")] = gridded_series_F1(no, W)[0]
    if "crutem5" in want:
        cr = crutem_field()
        W = cell_weights(cr["lat"].values, cr["lon"].values, key="crutem5")
        S[("crutem5", "tavg")] = gridded_series_F1(cr, W)[0]
    if "nclimdiv" in want:
        for e in ("tmax", "tmin", "tavg"):
            S[("nclimdiv", e)] = nclimdiv_series_F1(e)
    # 20CR and ERA-20C: JJA monthly means of the LOCAL-day statistic per native cell, then the same
    # anomaly -> CONUS-polygon area mean -> JJA mean as every gridded line above
    for k in RA:
        if k in want:
            f = {e: jja_monthly_field(k, e) for e in ("tmax", "tmin", "tmean")}
            W = cell_weights(f["tmax"]["lat"].values, f["tmax"]["lon"].values, key=f"{k}_nh")
            for e in ("tmax", "tmin"):
                S[(k, e)] = gridded_series_F1(f[e], W)[0]
            S[(k, "tavg")] = 0.5 * (S[(k, "tmax")] + S[(k, "tmin")])
            S[(k, "tavgmean")] = gridded_series_F1(f["tmean"], W)[0]
    return {f"conus_jja_{e}_anom__{ds}": s for (ds, e), s in S.items()}



# ==============================================================================
# SECTION 2 -- CONUS daily TMAX record counts (figure2.py's computation)
# ==============================================================================

MIN_CELL_FRAC = 0.50             # a Berkeley cell must be >=50% inside CONUS
BE_LAND_MIN_F2 = 0.50            # ... and >50% land in Berkeley's own mask
BLANK_PARTIAL = True             # Berkeley's daily release stops 2024-08-31

# ══ THE RECORD KERNEL — one implementation for every line ═══════════════════
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


def section2(want):
    """{column: Series} of record counts for the Figure 2 datasets in `want`."""
    need = set(want)
    if "ushcn_bc_x_sampling" in need:              # the correction is a ratio of two Berkeley lines
        need |= {"ushcn_bc", "berkeley", "berkeley_at_ushcn"}
    ser = {}
    if need & {"ushcn", "ushcn_bc", "berkeley_at_ushcn"}:
        # ══ THE MATCHED USHCN PAIR ═════════════════════════════════════════════
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
        if "ushcn" in need:
            ushcn = ser["ushcn"] = to_series(record_counts(RAW, UYR, UMO, UDY), KEEP_UH, label="USHCN-Daily")
        if "ushcn_bc" in need:
            ushcn_bc = ser["ushcn_bc"] = to_series(record_counts(ADJ, UYR, UMO, UDY), KEEP_UH, label="USHCN-BC")
    if need & {"berkeley", "berkeley_at_ushcn"}:
        # ══ BERKELEY, AND BERKELEY AT THE USHCN STATIONS ═══════════════════════
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
        KEEP_BE = (_bfrac >= MIN_CELL_FRAC) & (_bland.ravel() > BE_LAND_MIN_F2) & complete(_bny, _bfr)
        _bcounts = record_counts(BV, BYR, BMO, BDY)
        berkeley = to_series(_bcounts, KEEP_BE, w=_bfrac, label="Berkeley Earth")
        ser["berkeley"] = berkeley
        if "berkeley_at_ushcn" in need:
            # Berkeley at the USHCN sites. Same construction as the full-grid line above --
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
            ser["berkeley_at_ushcn"] = be_at_uh
        if BLANK_PARTIAL and PARTIAL:
            for _s in (berkeley, ser.get("berkeley_at_ushcn")):
                if _s is not None:
                    _s.loc[[y for y in PARTIAL if y in _s.index]] = np.nan
    if "ushcn_bc_x_sampling" in need:
        # ══ THE SAMPLING CORRECTION ════════════════════════════════════════════
        RATIO_SMOOTH = True
        _num, _den = ((lowess_smooth(berkeley), lowess_smooth(be_at_uh)) if RATIO_SMOOTH
                      else (berkeley, be_at_uh))
        be_samp_ratio = _num / _den
        ushcn_bc_full = ushcn_bc * be_samp_ratio
        ser["ushcn_bc_x_sampling"] = ushcn_bc_full
    # ══ 20CR AND ERA-20C, ON LOCAL-TIME DAYS ═══════════════════════════════
    # One unit per native grid cell with at least MIN_CELL_FRAC of it inside the CONUS polygon (the polygon
    # the stations are tested against), weighted by that fraction -- Berkeley's rule, without its land mask
    # because neither reanalysis ships one here. A year after the record ends is not a year of zero
    # records, so it is blanked, exactly as Berkeley's missing years are.
    for k, meta in RA.items():
        if k not in need:
            continue
        rx, rlat, rlon, rtimes = mjjas_cube(
            k, "tmax", lat_bnds=(BOX["lat_min"] - 3, BOX["lat_max"] + 3),
            lon_bnds=(BOX["lon_min"] - 3, BOX["lon_max"] + 3))
        rfrac = conus_cell_fraction(rlat, rlon, f"{k}_nh").ravel()
        rV = rx.reshape(rx.shape[0], -1)
        rYR, rMO, rDY = rtimes.year.values, rtimes.month.values, rtimes.day.values
        rny, rfr = coverage(rV, rYR)
        rkeep = (rfrac >= MIN_CELL_FRAC) & complete(rny, rfr)
        rs = to_series(record_counts(rV, rYR, rMO, rDY), rkeep, w=rfrac, label=meta["label"])
        rs.loc[[y for y in YEARS_F2 if y not in set(rYR)]] = np.nan
        ser[k] = rs
    return {f"conus_records__{ds}": s for ds, s in ser.items() if ds in want}



# ==============================================================================
# SECTION 4 -- northern mid-latitude band JJA anomalies (figure4.py's computation)
# ==============================================================================

Y0_F4, Y1_F4  = 1900, 2025
JJA           = [6, 7, 8]
BAND_LAT      = (24.0, 50.0)
MIN_JJA_DAYS  = 75                  # of 92; a cell needs this many in a year
MIN_BASE_YRS  = 15                  # of 30, per cell
BE_LAND_MIN_F4 = 0.0                # a Berkeley 1-deg cell is land if land_mask > this (its daily product
                                    # has no data off its own mask, so 0.0 = every cell it reports land for)
MAD_K         = 8.0                 # light guard on a single absurd year
FORCE_F4      = False
ERA5_Y0_F4, ERA5_Y1_F4 = 1940, 2025
RA_LAND_MIN   = 0.5                 # a reanalysis cell is land if this share of it is, in Berkeley's mask
TAG_F4 = f"land{BE_LAND_MIN_F4:g}_d{MIN_JJA_DAYS}_b{MIN_BASE_YRS}_{Y0_F4}-{Y1_F4}"
YEARS_F4 = np.arange(Y0_F4, Y1_F4 + 1)
BASE_OK  = (YEARS_F4 >= BASELINE[0]) & (YEARS_F4 <= BASELINE[1])

def band_mean_native(anom, lat, land):
    """cos(lat)-weighted mean of a (year, lat, lon) anomaly field over the land
    cells that have data, plus the fraction of the band's land area those cells
    cover. Averaging on the dataset's own grid, rather than on a coarse common
    grid, keeps a 2-degree box that is one quarter land from carrying the same
    weight as one that is all land."""
    w1 = np.repeat(np.cos(np.deg2rad(np.asarray(lat, float)))[:, None],
                   anom.shape[2], axis=1)
    w1 = np.where(land, w1, 0.0)
    tot = w1.sum()
    out = np.full(anom.shape[0], np.nan)
    cov = np.zeros(anom.shape[0])
    for yi in range(anom.shape[0]):
        ok = np.isfinite(anom[yi]) & land
        if not ok.any():
            continue
        w = np.where(ok, w1, 0.0)
        # NaN * 0 is NaN, so the field is zeroed off `ok` before it is weighted
        out[yi] = float(np.sum(np.where(ok, anom[yi], 0.0) * w) / w.sum())
        cov[yi] = float(w.sum() / tot)
    return pd.Series(out, index=YEARS_F4), pd.Series(cov, index=YEARS_F4)

def anomalise(ann):
    """Per-cell JJA anomaly against that cell's own 1951-1980 mean, computed
    BEFORE any spatial averaging, so a changing set of reporting cells cannot
    inject climatology into the band mean."""
    nb = np.isfinite(ann[BASE_OK]).sum(axis=0)
    clim = np.where(nb >= MIN_BASE_YRS, np.nanmean(ann[BASE_OK], axis=0), np.nan)
    return ann - clim[None, :, :]

# ══ (1) BERKELEY'S OWN 1-DEGREE GRID AND LAND MASK ═════════════════════════
@lru_cache(maxsize=1)
def berkeley_grid():
    """(lat, lon, land) for Berkeley's 1-degree band grid. The land mask is a
    land FRACTION, and is NaN over open ocean, where the daily product carries
    no temperature at all -- so `> BE_LAND_MIN_F4` with BE_LAND_MIN_F4 = 0 selects
    exactly the cells Berkeley treats as land."""
    for dec in range(1900, 2030, 10):
        f = BE_PROC / f"processed_nh_TMAX_Complete_TMAX_Daily_LatLong1_{dec}.nc"
        if not f.exists():
            continue
        ds = xr.open_dataset(stage(f))
        la = ds["latitude"].values.astype(float)
        lo = ds["longitude"].values.astype(float)
        if lo.max() > 180.0:
            lo = np.where(lo > 180.0, lo - 360.0, lo)
        keep = (la >= BAND_LAT[0]) & (la <= BAND_LAT[1])
        lm = ds["land_mask"]
        lm = (lm.isel(time=0) if "time" in lm.dims else lm).values[keep, :]
        ds.close()
        return la[keep], lo, np.isfinite(lm) & (lm > BE_LAND_MIN_F4)
    raise FileNotFoundError(f"no Berkeley NH decade files under {BE_PROC}")

# ══ (2) BERKELEY: JJA ANOMALY ON ITS OWN GRID, THEN THE LAND BAND MEAN ═════
def berkeley_band():
    """Per-year JJA band mean anomaly over all Berkeley land in 24-50N."""
    fp = CACHE_BAND / f"berkeley_land_band_{TAG_F4}.npz"
    if fp.exists() and not FORCE_F4:
        z = np.load(fp)
        return ({v: pd.Series(z[v], index=YEARS_F4) for v in ("tmax", "tmin")},
                {v: pd.Series(z[f"cov_{v}"], index=YEARS_F4) for v in ("tmax", "tmin")})
    blat, blon, land = berkeley_grid()
    ser, cov = {}, {}
    for var in ("TMAX", "TMIN"):
        ann = np.full((len(YEARS_F4), blat.size, blon.size), np.nan, np.float32)
        for dec in range(1900, 2030, 10):
            f = BE_PROC / f"processed_nh_{var}_Complete_{var}_Daily_LatLong1_{dec}.nc"
            if not f.exists():
                continue
            ds = xr.open_dataset(stage(f))
            t = ds["temperature"]
            la = ds["latitude"].values.astype(float)
            keep = (la >= BAND_LAT[0]) & (la <= BAND_LAT[1])   # direction-safe
            tt = pd.DatetimeIndex(t["time"].values)
            sel = np.isin(tt.month, JJA)
            v = t.values[sel][:, keep, :].astype(np.float32)
            v = np.where(np.isfinite(v) & (np.abs(v) < 1e4), v, np.nan)
            yrs = tt.year.values[sel]
            ds.close()
            for y in np.unique(yrs):
                if not (Y0_F4 <= y <= Y1_F4):
                    continue
                k = yrs == y
                cnt = np.isfinite(v[k]).sum(axis=0)
                ann[y - Y0_F4] = np.where(cnt >= MIN_JJA_DAYS,
                                          np.nanmean(v[k], axis=0), np.nan)
        anom = np.where(land[None, :, :], anomalise(ann), np.nan)
        ser[var.lower()], cov[var.lower()] = band_mean_native(anom, blat, land)
    np.savez_compressed(fp, tmax=ser["tmax"].to_numpy(), tmin=ser["tmin"].to_numpy(),
                        cov_tmax=cov["tmax"].to_numpy(), cov_tmin=cov["tmin"].to_numpy())
    return ser, cov

# ══ (3) ERA5, ON THE SAME LAND FOOTPRINT ═══════════════════════════════════
def era5_band():
    """Per-year JJA band mean anomaly over the same land, from ERA5's native
    0.25-degree grid. Berkeley's 1-degree land mask is carried onto the ERA5
    grid by nearest cell centre, so both curves average the same geography.
    The archived files already span 24-50N at all longitudes; ERA5 starts in
    1940, so the series is blank before then."""
    fp = CACHE_BAND / f"era5_land_band_{TAG_F4}.npz"
    if fp.exists() and not FORCE_F4:
        z = np.load(fp)
        return ({v: pd.Series(z[v], index=YEARS_F4) for v in ("tmax", "tmin")},
                {v: pd.Series(z[f"cov_{v}"], index=YEARS_F4) for v in ("tmax", "tmin")})
    blat, blon, bland = berkeley_grid()
    elat = elon = order = land = None
    sum_tx = sum_tn = cnt = None
    missing = []
    for y in range(ERA5_Y0_F4, ERA5_Y1_F4 + 1):
        for mo in JJA:
            f = era5_file(y, mo)
            if not f.exists():
                missing.append(f"{y}-{mo:02d}")
                continue
            ds = xr.open_dataset(stage(f))
            if elat is None:
                elat = ds["latitude"].values.astype(float)
                lon0 = ds["longitude"].values.astype(float)
                elon = ((lon0 + 180.0) % 360.0) - 180.0
                order = np.argsort(elon)
                elon = elon[order]
                # Berkeley's land mask, nearest cell centre -> the ERA5 grid
                ia = np.abs(elat[:, None] - blat[None, :]).argmin(axis=1)
                io = np.abs(elon[:, None] - blon[None, :]).argmin(axis=1)
                land = bland[np.ix_(ia, io)]
                shape = (len(YEARS_F4), elat.size, elon.size)
                sum_tx = np.zeros(shape); sum_tn = np.zeros(shape)
                cnt = np.zeros(shape, np.int32)
            yi = y - Y0_F4
            tx = ds["t2m_max"].values[:, :, order].astype(np.float64)
            tn = ds["t2m_min"].values[:, :, order].astype(np.float64)
            ds.close()
            sum_tx[yi] += np.nansum(tx, axis=0)
            sum_tn[yi] += np.nansum(tn, axis=0)
            cnt[yi] += np.isfinite(tx).sum(axis=0)
    if missing:
        print(f"ERA5 band field: missing {len(missing)} of "
              f"{(ERA5_Y1_F4 - ERA5_Y0_F4 + 1) * len(JJA)} JJA months "
              f"(e.g. {missing[:3]})")
    ser, cov = {}, {}
    for s, key in ((sum_tx, "tmax"), (sum_tn, "tmin")):
        ann = np.where(cnt >= MIN_JJA_DAYS, s / np.maximum(cnt, 1), np.nan)
        anom = np.where(land[None, :, :], anomalise(ann), np.nan)
        ser[key], cov[key] = band_mean_native(anom, elat, land)
    np.savez_compressed(fp, tmax=ser["tmax"].to_numpy(), tmin=ser["tmin"].to_numpy(),
                        cov_tmax=cov["tmax"].to_numpy(), cov_tmin=cov["tmin"].to_numpy())
    return ser, cov

# ══ (4) 20CR AND ERA-20C, ON LOCAL-TIME DAYS AND THE SAME LAND FOOTPRINT ═══
def reanalysis_band(name):
    """Per-year JJA band-mean anomaly over Berkeley's land footprint, from the reanalysis'
    own grid, for the local-day TMAX, TMIN and mean. Same steps as era5_band(): JJA mean
    per cell (>= MIN_JJA_DAYS days), anomaly against the cell's own 1951-1980 mean BEFORE
    any spatial average, cos(lat) mean over land cells. Returns ({stat: series},
    {stat: land coverage}, land-mask area share of the band)."""
    fp = CACHE_BAND / f"{name}_nh_land_band_{TAG_F4}_f{RA_LAND_MIN:g}.npz"
    stats = ("tmax", "tmin", "tmean")
    if fp.exists() and not FORCE_F4:
        z = np.load(fp)
        return ({v: pd.Series(z[v], index=YEARS_F4) for v in stats},
                {v: pd.Series(z[f"cov_{v}"], index=YEARS_F4) for v in stats},
                float(z["area_share"]))
    ser, cov, land, share = {}, {}, None, np.nan
    for v in stats:
        ann_y, lat, lon, yrs = jja_annual(name, v, lat_bnds=BAND_LAT, min_days=MIN_JJA_DAYS)
        if land is None:
            land = berkeley_land_fraction(lat, lon) >= RA_LAND_MIN
            w = np.cos(np.deg2rad(lat))[:, None] * np.ones(lon.size)
            share = float((w * land).sum() / w.sum())
        ann = np.full((len(YEARS_F4), lat.size, lon.size), np.nan, np.float32)
        ann[yrs - Y0_F4] = ann_y
        anom = np.where(land[None, :, :], anomalise(ann), np.nan)
        ser[v], cov[v] = band_mean_native(anom, lat, land)
    np.savez_compressed(fp, area_share=share,
                        **{v: ser[v].to_numpy() for v in stats},
                        **{f"cov_{v}": cov[v].to_numpy() for v in stats})
    return ser, cov, share

# ══ SERIES AND A LIGHT OUTLIER GUARD ═══════════════════════════════════════
def mad_flag(v, k=MAD_K):
    v = np.asarray(v, float)
    med = np.nanmedian(v); mad = np.nanmedian(np.abs(v - med))
    if not np.isfinite(mad) or mad == 0:
        return np.zeros(v.shape, bool)
    return np.isfinite(v) & (np.abs(v - med) > k * 1.4826 * mad)

def finish(s, label):
    """One MAD pass, only to stop a single absurd year from bending the smooth.
    Anything it removes is reported rather than removed silently."""
    m = s.to_numpy(float).copy()
    bad = mad_flag(m)
    if bad.any():
        print(f"  MAD guard blanked {int(bad.sum())} year(s) of {label}: "
              f"{list(YEARS_F4[bad])}")
        m[bad] = np.nan
    return pd.Series(m, index=YEARS_F4)


def section4(want):
    """{column: Series} of band JJA anomalies for the Figure 4 datasets in `want`."""
    S = {}
    if "berkeley" in want:
        BE, _ = berkeley_band()
        for v in ("tmax", "tmin"):
            S[("berkeley", v)] = finish(BE[v], f"Berkeley {v.upper()}")
        S[("berkeley", "tavg")] = 0.5 * (S[("berkeley", "tmax")] + S[("berkeley", "tmin")])
    if "era5" in want:
        ER, _ = era5_band()
        for v in ("tmax", "tmin"):
            S[("era5", v)] = finish(ER[v], f"ERA5 {v.upper()}")
        S[("era5", "tavg")] = 0.5 * (S[("era5", "tmax")] + S[("era5", "tmin")])
    for k in RA:
        if k in want and RA[k]["band"]:
            ser, _cov, share = reanalysis_band(k)
            print(f"  {RA[k]['label']}: land mask covers {share:.3f} of the band's cos(lat) area")
            for v in ("tmax", "tmin"):
                S[(k, v)] = finish(ser[v], f"{RA[k]['label']} {v.upper()}")
            S[(k, "tavgmean")] = finish(ser["tmean"], f"{RA[k]['label']} day-mean TAVG")
            S[(k, "tavg")] = 0.5 * (S[(k, "tmax")] + S[(k, "tmin")])
    return {f"band_jja_{e}_anom__{ds}": s for (ds, e), s in S.items()}



# ==============================================================================
# SECTION 5 -- heat-wave days, Christy (2026) method (figure5.py's computation)
# ==============================================================================

UH_YR0, UH_YR1 = 1900, 2025
UH_LAT_BNDS, UH_LON_BNDS = (24.0, 50.0), (-130.0, -60.0)
USHCN_LEGS = ("raw", "adj")
BLANK_BE_PARTIAL   = True
BE_AT_NETWORKS     = ("ushcn",)

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

_PCTILE, _WINDOW, _MIN_RUN, _MIN_FRAC = 90, 3, 6, 0.70
_HW_MON, _N_DOY = {5, 6, 7, 8, 9}, 153

# (I6) every cached series is tagged with the method that produced it. The one
# failure mode here that would look entirely plausible on screen is a figure
# drawn half from v F pickles and half from Christy's -- (I1)-(I3) change
# christy_fields, so the GRIDDED caches go stale too, not just the station ones.
# A run under one set of switches must not be able to read a cache written under
# another, so the switches are in the filename. Same idea as common.TAG_F2.
_TAG_THR = (f"p{_PCTILE}r{_MIN_RUN}w{_WINDOW}n{MIN_THRESH_N}"
            f"_{'nr' if PCTILE_METHOD == 'nearest_rank' else 'lin'}"
            f"{'ge' if EXCEED_INCLUSIVE else 'gt'}_{UH_YR0}-{UH_YR1}")
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

def ld_era5_conus():
    files = [stage(era5_file(y, m))
             for y in range(1940, UH_YR1 + 1) for m in sorted(_HW_MON)
             if era5_file(y, m).exists()]
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

def ld_ra_conus(name):
    """20CR / ERA-20C local-day TMAX on the CONUS box, ld_era5_conus()'s way: the box of
    lat 23-51, lon -126..-65 is read, and the CONUS polygon picks the cells whose CENTRE
    is inside it (the mask ERA5 gets, so the reanalyses are judged on the same footing)."""
    tx, lat, lon, times = mjjas_cube(name, "tmax", lat_bnds=(23, 51), lon_bnds=(-126, -65))
    tx, dy, yr = _reindex_full(tx, times, f"{RA[name]['label']} CONUS")
    m = mask_conus(lat, lon, f"{name}_nh_conus")
    return tx, dy, yr, lat, lon, m

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

# ══ GLOBAL LOADERS ═════════════════════════════════════════════════════════
def ld_ra_global(name):
    """20CR / ERA-20C local-day TMAX over 23-51N at every longitude, land-masked by the
    Natural Earth polygon at the cell centres, as ld_era5_global() does for ERA5. No
    thinning: unlike ERA5 at 0.25 degrees these grids are already 1 to 2 degrees."""
    tx, lat, lon, times = mjjas_cube(name, "tmax", lat_bnds=(23, 51))
    tx, dy, yr = _reindex_full(tx, times, f"{RA[name]['label']} global")
    m = mask_land(lat, lon, f"{name}_nh_global")
    return tx, dy, yr, lat, lon, m

def ld_era5_global():
    files = [stage(era5_file(y, m))
             for y in range(1940, UH_YR1 + 1) for m in sorted(_HW_MON)
             if era5_file(y, m).exists()]
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

# ══ (B) BERKELEY RE-SAMPLED ONTO THE USHCN NETWORK ═════════════════════════
def _network_positions(net):
    """(lat, lon) of the stations that actually build this figure's line for that network."""
    if net == "ushcn":
        if not (_uh_piv_fp("raw").exists() and _uh_piv_fp("adj").exists()):
            _uh_build_pivots()
        stns = _uh_stations()
        cols = set(pd.read_parquet(str(_uh_piv_fp("raw"))).columns)
        stns = stns[stns["ushcn_id"].isin(cols)]
        return stns["lat"].to_numpy(float), stns["lon"].to_numpy(float)
    raise ValueError(net)

NET_TITLE = {"ushcn": "USHCN"}

def _be_at_network_counts(fields, lat, lon, be_mask, net):
    """FIGURE 3's convention, applied to the heat-wave field.

    Fig 2 does   BE_AT_V = BV[:, _flat[_good]]   -- one column per STATION, so a
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
    good = np.asarray(be_mask, bool).ravel()[flat]      # Fig 2's _good
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
    """(I4) one place where a station network becomes a series, so the USHCN
    and Berkeley-at-network lines cannot drift apart. Both reductions are
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



def section5(want):
    """{column: Series} of heat-wave days per year for the Figure 5 datasets in `want`."""
    R = {}                                     # column -> {year: value}
    if want & {"berkeley", "berkeley_at_ushcn"}:
        be_c, be_net = _run_be_conus_set()
        if BLANK_BE_PARTIAL:                   # Berkeley's daily release stops 2024-08-31
            be_c = {k: v for k, v in be_c.items() if k != 2024}
            be_net = {n: {k: v for k, v in d.items() if k != 2024} for n, d in be_net.items()}
        R["conus_hw_days__berkeley"] = be_c
        R["conus_hw_days__berkeley_at_ushcn"] = be_net.get("ushcn", {})
    if "era5" in want:
        R["conus_hw_days__era5"] = _run("ERA5 CONUS (land-masked)", ld_era5_conus,
                                        f"era5_landmasked_{_TAG_THR}.pkl", (-125., -65.))
        R["band_hw_days__era5"] = _run("ERA5 global strip (land-masked)", ld_era5_global,
                                       f"era5_global_landmasked_{_TAG_THR}.pkl", (-180., 180.))
    if "berkeley" in want:
        be_g = _run("Berkeley NH global (land-masked)", ld_be_global,
                    f"berkeley_nh_landmasked_{_TAG_THR}.pkl", (-180., 180.))
        R["band_hw_days__berkeley"] = {k: v for k, v in be_g.items() if not (BLANK_BE_PARTIAL and k == 2024)}
    # 20CR and ERA-20C: the cache names carry their own span, because _TAG_THR's 1900-2025 is not theirs
    for k, meta in RA.items():
        if k in want:
            tag = f"{k}_nh_{meta['years'][0]}-{meta['years'][1]}"
            R[f"conus_hw_days__{k}"] = _run(f"{meta['label']} CONUS (land-masked)",
                                            lambda k=k: ld_ra_conus(k),
                                            f"{tag}_conus_landmasked_{_TAG_THR}.pkl", (-125., -65.))
            if meta["band"]:
                R[f"band_hw_days__{k}"] = _run(f"{meta['label']} global strip (land-masked)",
                                               lambda k=k: ld_ra_global(k),
                                               f"{tag}_global_landmasked_{_TAG_THR}.pkl", (-180., 180.))
    if want & {"ushcn", "ushcn_bc"}:
        uh = _run_ushcn_pair() if uh_available() else {}
        R["conus_hw_days__ushcn"] = uh.get("raw", {})
        R["conus_hw_days__ushcn_bc"] = uh.get("adj", {})
    keep = {c for ds in want for c, _u, _n in section_columns(5, ds)}
    return {c: pd.Series(d, dtype=float).sort_index() for c, d in R.items() if c in keep and d}



# ==============================================================================
# SELF-CHECKS -- the record and heat-wave kernels on synthetic numbers (no data needed)
# ==============================================================================

def selfchecks():
    """The synthetic checks figure2.py and figure5.py run at the top, in one place."""
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



# ══ THE TABLE AND ITS CATALOGUE ════════════════════════════════════════════════
_SECTION_FIGURE = {1: 1, 2: 2, 4: 4, 5: 5}
_SECTIONS = {1: section1, 2: section2, 4: section4, 5: section5}


def catalogue():
    """One row per column the table can hold: where it comes from and what it is."""
    rows = []
    for sec, dss in SECTION_DATASETS.items():
        for ds in dss:
            for col, units, note in section_columns(sec, ds):
                head, _ = col.split("__")
                domain, quantity = head.split("_", 1)
                rows.append(dict(column=col, figure=_SECTION_FIGURE[sec], domain=domain, quantity=quantity,
                                 dataset=ds, label=DATASET_LABEL[ds], units=units, note=note))
    return pd.DataFrame(rows)


def wanted(sec, datasets, have, rebuild):
    """Datasets of section `sec` to compute: those asked for whose columns are not all in the table yet."""
    out = set()
    for ds in SECTION_DATASETS[sec]:
        if datasets and ds not in datasets:
            continue
        cols = [c for c, _u, _n in section_columns(sec, ds)]
        if rebuild or any(c not in have for c in cols):
            out.add(ds)
    return out


def write_table(df, csv):
    cat = catalogue()
    cat = cat[cat["column"].isin(df.columns)].copy()
    df = df[list(cat["column"]) + [c for c in df.columns if c not in set(cat["column"])]]
    first, last, n = [], [], []
    for c in cat["column"]:
        v = df[c].dropna()
        first.append(int(v.index.min()) if len(v) else ""); last.append(int(v.index.max()) if len(v) else "")
        n.append(len(v))
    cat["first_year"], cat["last_year"], cat["n_years"] = first, last, n
    df.index.name = "year"
    tmp = csv.with_name(csv.name + ".tmp")
    df.to_csv(tmp)                                  # full precision: the figures must see the series exactly
    meta = csv.with_name(csv.stem + "_columns.csv")
    cat.to_csv(meta.with_name(meta.name + ".tmp"), index=False)
    tmp.replace(csv); meta.with_name(meta.name + ".tmp").replace(meta)
    return df, meta


def summary(df, cols):
    print(f"\n{'column':<46}{'years':>6}{'first':>7}{'last':>6}{'1930s':>9}{'last 10 yr':>12}")
    for c in cols:
        v = df[c].dropna()
        if v.empty:
            print(f"  {c:<44}{'(empty)':>6}"); continue
        d30 = v.loc[1930:1939].mean() if v.index.min() <= 1930 else np.nan
        print(f"  {c:<44}{len(v):>6}{int(v.index.min()):>7}{int(v.index.max()):>6}"
              f"{d30:>9.2f}{v.iloc[-10:].mean():>12.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", nargs="*", type=int, choices=sorted(SECTION_DATASETS),
                    help="figures to build columns for (default: 1 2 4 5)")
    ap.add_argument("--datasets", nargs="*", help="only these datasets (names as in --list)")
    ap.add_argument("--csv", default=str(CSV_DEFAULT))
    ap.add_argument("--rebuild", action="store_true", help="recompute columns that are already in the table")
    ap.add_argument("--list", action="store_true", help="print the columns and whether the table has them")
    ap.add_argument("--keep-copies", action="store_true",
                    help="do not delete the staged archive files and the 20CR/ERA-20C extracts afterwards")
    a = ap.parse_args()
    csv = Path(a.csv)
    have = set(pd.read_csv(csv, index_col=0, nrows=0).columns) if csv.exists() else set()
    if a.list:
        cat = catalogue()
        for sec in SECTION_DATASETS:
            print(f"\nFigure {sec}")
            for _, r in cat[cat["figure"] == sec].iterrows():
                print(f"  {'*' if r['column'] in have else ' '} {r['column']:<46} {r['units']}")
        print(f"\n(* = already in {csv})")
        return
    unknown = set(a.datasets or []) - set(DATASET_LABEL)
    if unknown:
        raise SystemExit(f"unknown dataset(s) {sorted(unknown)}; choose from {sorted(DATASET_LABEL)}")

    check_pipeline(); print()
    check_geometry(); print()
    check_smoother(); print()
    check_local_day(); print()
    selfchecks(); print()
    check_localday_files(); print()

    df = (pd.read_csv(csv, index_col=0) if csv.exists() else pd.DataFrame(index=YEARS_MASTER))
    df = df.reindex(YEARS_MASTER)
    datasets = set(a.datasets or [])
    new = {}
    for sec in (a.sections or sorted(SECTION_DATASETS)):
        want = wanted(sec, datasets, have, a.rebuild)
        if not want:
            print(f"Figure {sec}: nothing to build"); continue
        t0 = time.time()
        print(f"\n{'=' * 70}\nFigure {sec}: building {sorted(want)}\n{'=' * 70}", flush=True)
        out = _SECTIONS[sec](want)
        new.update(out)
        print(f"Figure {sec}: {len(out)} columns in {time.time() - t0:.0f}s", flush=True)
    if new:
        for c, s in new.items():
            df[c] = pd.to_numeric(s, errors="coerce").reindex(YEARS_MASTER).astype(float)
        df, meta = write_table(df, csv)
        print(f"\nwrote {csv}  ({df.shape[0]} years x {df.shape[1]} columns)\nwrote {meta}")
        summary(df, list(new))
    else:
        print("\nthe table already has everything asked for")

    if a.keep_copies:
        print("\n--keep-copies: staged archive files and extracts left in place")
    else:
        # the table is on disk (just written, or already there): remove everything copied from the archive
        n = remove_copies()
        from prepare_data import prune_stage
        prune_stage(True)
        print(f"removed {n} extract/temporary files; the archive copies under _stage/ are gone too")


if __name__ == "__main__":
    main()

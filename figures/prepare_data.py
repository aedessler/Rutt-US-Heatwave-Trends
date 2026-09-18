#!/usr/bin/env python3
"""Build the inputs the figure scripts need but the archive does not hold.

`/Volumes/adessler_lab` has every dataset the paper uses, but three of the
things the notebook read are not on it in that form:

  (a) the GHCN-Daily yearly archive as `ghcn_global_YYYY.parquet`. The drive has
      the raw by-year CSVs (global) and a QC'd CONUS-only daily cube. Figure 5
      and Figure 6's global row need stations outside the United States, so the
      archive is rebuilt from the raw CSVs, restricted to 24-50N (every
      longitude) -- the widest domain any figure asks for.

  (b) the USHCN daily raw / bias-corrected pair. The drive has USHCN v2.5
      MONTHLY raw and FLs.52j, and their monthly difference as
      `ushcn_offsets_1900_2024.nc`. The daily pair is reconstructed by adding
      each station-month's offset to that station's daily GHCN-Daily values:
      the adjustment is then the only difference between the two legs, which is
      exactly the invariant Figures 3 and 6 assert.

  (c) the 2-degree global gridded GHCN-Daily TMAX that Figure 6's bottom-row
      GHCN panel reads. Built from (a) by binning stations into 2-degree cells.

Everything is written under HEATWAVE_WORK, is cached, and is skipped if it is
already there. Run it once:

    python figures/prepare_data.py            # all three
    python figures/prepare_data.py band       # just one step
    python figures/prepare_data.py ushcn grid

`--force` rebuilds, `--workers N` sets the by-year parallelism (the archive is
on a network share, so several concurrent readers help).
"""
import argparse, gzip, os, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ROOT, WORK, DATA, GHCND_BY_YEAR, GHCND_STATIONS_TXT,
                    GHCND_STATION_DAILY, USHCN_OFFSETS_FP, USHCN_CROSSWALK_FP,
                    USHCN_STATIONS_TXT, GHCND_GLOBAL, UH_STN_FP, UH_DAILY_DIR,
                    UH_PAIRED, UH_PIV_DIR, GHCND_GLOBAL_GRID,
                    BAND_LAT_PREP, Y0_PREP, Y1_PREP)

# ══════════════════════════════════════════════════════════════════════════════
# station metadata
# ══════════════════════════════════════════════════════════════════════════════
def ghcnd_stations():
    """id -> lat, lon, from ghcnd-stations.txt (the authoritative NOAA file)."""
    df = pd.read_fwf(str(GHCND_STATIONS_TXT),
                     colspecs=[(0, 11), (12, 20), (21, 30)],
                     names=["id", "lat", "lon"])
    return df.set_index("id")[["lat", "lon"]].astype("float64")


# ══════════════════════════════════════════════════════════════════════════════
# (a) the GHCN-Daily 24-50N band archive
# ══════════════════════════════════════════════════════════════════════════════
_CSV_KW = dict(header=None, compression="gzip", usecols=[0, 1, 2, 3, 5],
               names=["id", "date", "elem", "val", "qflag"],
               dtype={"id": "string", "date": "int32", "elem": "string",
                      "val": "float32", "qflag": "string"})

def _build_band_year(args):
    """One year of by_year/YYYY.csv.gz -> ghcn_global_YYYY.parquet.

    The QC is the archive's own documented rule, the same one the CONUS daily
    cube on the drive was built with: a non-blank quality flag means the value
    failed, and the first observation of a station-day-element wins.
    """
    y, keep_ids, lat, lon, out_fp = args
    src = GHCND_BY_YEAR / f"{y}.csv.gz"
    if not src.exists():
        return y, None, "no source file"
    t0 = time.time()
    d = pd.read_csv(src, **_CSV_KW)
    d = d[d["elem"].isin(("TMAX", "TMIN"))]
    d = d[d["qflag"].isna() | (d["qflag"].astype("string").str.strip() == "")]
    d = d[d["id"].isin(keep_ids)]
    if d.empty:
        return y, 0, f"{time.time() - t0:.0f}s (empty)"
    d = d.drop_duplicates(subset=["id", "date", "elem"], keep="first")
    wide = (d.pivot(index=["id", "date"], columns="elem", values="val")
              .reset_index())
    for c in ("TMAX", "TMIN"):
        if c not in wide.columns:
            wide[c] = np.nan
    out = pd.DataFrame({
        "station_id": wide["id"].astype(str),
        "date": pd.to_datetime(wide["date"].astype("int64").astype(str),
                               format="%Y%m%d"),
        "tmax_c": (wide["TMAX"] / 10.0).astype("float32"),   # archive is 0.1 degC
        "tmin_c": (wide["TMIN"] / 10.0).astype("float32"),
    })
    out["lat"] = out["station_id"].map(lat).astype("float32")
    out["lon"] = out["station_id"].map(lon).astype("float32")
    out = out.dropna(subset=["lat", "lon"])
    out = out[out["tmax_c"].notna() | out["tmin_c"].notna()]
    tmp = out_fp.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, out_fp)
    return y, len(out), f"{time.time() - t0:.0f}s"


def build_band(force=False, workers=6):
    GHCND_GLOBAL.mkdir(parents=True, exist_ok=True)
    st = ghcnd_stations()
    band = st[(st["lat"] >= BAND_LAT_PREP[0]) & (st["lat"] <= BAND_LAT_PREP[1])]
    keep_ids = set(band.index)
    lat = band["lat"].to_dict(); lon = band["lon"].to_dict()
    print(f"[band] {len(keep_ids):,} GHCN-Daily stations in "
          f"{BAND_LAT_PREP[0]:g}-{BAND_LAT_PREP[1]:g}N (all longitudes)")

    jobs = []
    for y in range(Y0_PREP, Y1_PREP + 1):
        fp = GHCND_GLOBAL / f"ghcn_global_{y}.parquet"
        if fp.exists() and not force:
            continue
        jobs.append((y, keep_ids, lat, lon, fp))
    if not jobs:
        print("[band] already built"); return
    print(f"[band] building {len(jobs)} years with {workers} workers "
          f"(the raw CSVs are ~13 GB on a network share; expect ~30-60 min)")
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_build_band_year, j): j[0] for j in jobs}
        for f in as_completed(futs):
            y, n, msg = f.result()
            done += 1
            print(f"[band] {done:3d}/{len(jobs)}  {y}  "
                  f"{'-' if n is None else format(n, ',')+' rows':>14}  {msg}",
                  flush=True)
    print("[band] done")


# ══════════════════════════════════════════════════════════════════════════════
# (b) the USHCN daily raw / bias-corrected pair, and the paired monthly table
# ══════════════════════════════════════════════════════════════════════════════
def _ushcn_tables():
    """(stations, offsets) for the USHCN network, keyed by USHCN id.

    The crosswalk maps each USHCN id to the GHCN-Daily id that carries its
    observations; the offsets file is keyed by that GHCN-Daily id.
    """
    xw = pd.read_csv(USHCN_CROSSWALK_FP).dropna(subset=["ghcnd_id"])
    xw = xw[xw["ghcnd_id"].astype(str).str.len() == 11]

    ds = xr.open_dataset(USHCN_OFFSETS_FP)
    off_ids = ds["station_id"].values.astype(str)
    off = {v: ds[f"{v}_offset"].values for v in ("tmax", "tmin", "tavg")}
    years = ds["year"].values.astype(int)
    ds.close()
    off_pos = {s: i for i, s in enumerate(off_ids)}

    xw = xw[xw["ghcnd_id"].isin(off_pos)]

    st = pd.read_fwf(str(USHCN_STATIONS_TXT),
                     colspecs=[(0, 11), (12, 20), (21, 30)],
                     names=["ushcn_id", "lat", "lon"]).set_index("ushcn_id")
    xw = xw[xw["ushcn_id"].isin(st.index)]
    xw = xw.drop_duplicates(subset="ushcn_id").set_index("ushcn_id")
    stations = st.loc[xw.index].assign(ghcnd_id=xw["ghcnd_id"])
    return stations, off, years, off_pos


def _daily_cube_for(ghcnd_ids):
    """(dates, {id: (tmax, tmin)}) daily series for the given GHCN-Daily ids,
    from the drive's QC'd CONUS daily cube (same by-year source and same QC
    rule as the band archive above)."""
    frames_tx, frames_tn, dates = [], [], None
    for y in range(Y0_PREP, Y1_PREP + 1):
        fp = GHCND_STATION_DAILY / f"{y}.nc"
        if not fp.exists():
            continue
        ds = xr.open_dataset(fp)
        ids = ds["station_id"].values.astype(str)
        sel = np.nonzero(np.isin(ids, list(ghcnd_ids)))[0]
        tx = ds["tmax"].values[sel] / 10.0          # 0.1 degC -> degC
        tn = ds["tmin"].values[sel] / 10.0
        cols = ids[sel]
        ds.close()
        # the file's calendar is 366 leap-aligned slots with slot 59 = 29 Feb,
        # so a non-leap year is the same axis with that slot removed
        keep = np.ones(366, bool)
        if not pd.Timestamp(year=y, month=1, day=1).is_leap_year:
            keep[59] = False
        real = pd.date_range(f"{y}-01-01", f"{y}-12-31", freq="D")
        assert keep.sum() == len(real), (y, keep.sum(), len(real))
        frames_tx.append(pd.DataFrame(tx[:, keep].T, index=real, columns=cols))
        frames_tn.append(pd.DataFrame(tn[:, keep].T, index=real, columns=cols))
    TX = pd.concat(frames_tx).sort_index()
    TN = pd.concat(frames_tn).sort_index()
    return TX, TN


def build_ushcn(force=False):
    if (UH_STN_FP.exists() and UH_PAIRED.exists()
            and UH_DAILY_DIR.is_dir() and any(UH_DAILY_DIR.glob("*.csv.gz"))
            and not force):
        print("[ushcn] already built"); return

    stations, off, off_years, off_pos = _ushcn_tables()
    print(f"[ushcn] {len(stations):,} USHCN stations with a GHCN-Daily id "
          f"and monthly offsets")

    TX, TN = _daily_cube_for(set(stations["ghcnd_id"]))
    print(f"[ushcn] daily cube {TX.shape[0]:,} days x {TX.shape[1]:,} stations")

    UH_STN_FP.parent.mkdir(parents=True, exist_ok=True)
    UH_DAILY_DIR.mkdir(parents=True, exist_ok=True)
    UH_PAIRED.parent.mkdir(parents=True, exist_ok=True)

    yr = TX.index.year.to_numpy()
    mo = TX.index.month.to_numpy()
    yi = pd.Index(off_years).get_indexer(yr)
    monthly_rows = []
    kept = 0

    for ushcn_id, row in stations.iterrows():
        g = row["ghcnd_id"]
        if g not in TX.columns:
            continue
        oi = off_pos[g]
        tx_raw = TX[g].to_numpy(float)
        tn_raw = TN[g].to_numpy(float)

        # the station-month offset, broadcast onto that month's days
        def spread(var):
            o = off[var][oi]                        # (year, month)
            v = np.full(len(yr), np.nan)
            ok = yi >= 0
            v[ok] = o[yi[ok], mo[ok] - 1]
            return v
        ox, on_, oa = spread("tmax"), spread("tmin"), spread("tavg")

        # the two legs must share one valid-day mask: a day counts only if the
        # raw value AND that month's adjustment both exist.
        mx = np.isfinite(tx_raw) & np.isfinite(ox)
        mn = np.isfinite(tn_raw) & np.isfinite(on_)
        if not mx.any():
            continue
        tx_r = np.where(mx, tx_raw, np.nan); tx_a = np.where(mx, tx_raw + ox, np.nan)
        tn_r = np.where(mn, tn_raw, np.nan); tn_a = np.where(mn, tn_raw + on_, np.nan)
        # TAVG needs its own mask: both raw elements AND the tavg offset, so
        # that the raw and adjusted legs again share one valid-day mask.
        ma = np.isfinite(tx_raw) & np.isfinite(tn_raw) & np.isfinite(oa)
        ta_r = np.where(ma, (tx_raw + tn_raw) / 2.0, np.nan)
        ta_a = np.where(ma, (tx_raw + tn_raw) / 2.0 + oa, np.nan)

        # -- daily file, May-September, what Figures 3 and 6 read --
        mj = np.isin(mo, (5, 6, 7, 8, 9)) & mx
        if mj.any():
            pd.DataFrame({
                "date": TX.index[mj].strftime("%Y%m%d"),
                "tmax_raw_c": np.round(tx_r[mj], 2),
                "tmax_adj_c": np.round(tx_a[mj], 2),
            }).to_csv(UH_DAILY_DIR / f"{ushcn_id}.csv.gz", index=False,
                      compression="gzip")
            kept += 1

        # -- monthly sums and counts, what Figures 1 and 2 read --
        g_df = pd.DataFrame({"year": yr, "month": mo,
                             "tmax_raw": tx_r, "tmax_adj": tx_a,
                             "tmin_raw": tn_r, "tmin_adj": tn_a,
                             "tavg_raw": ta_r, "tavg_adj": ta_a})
        agg = g_df.groupby(["year", "month"]).agg(
            tmax_raw_sum=("tmax_raw", "sum"), tmax_raw_n=("tmax_raw", "count"),
            tmax_adj_sum=("tmax_adj", "sum"), tmax_adj_n=("tmax_adj", "count"),
            tmin_raw_sum=("tmin_raw", "sum"), tmin_raw_n=("tmin_raw", "count"),
            tmin_adj_sum=("tmin_adj", "sum"), tmin_adj_n=("tmin_adj", "count"),
            tavg_raw_sum=("tavg_raw", "sum"), tavg_raw_n=("tavg_raw", "count"),
            tavg_adj_sum=("tavg_adj", "sum"), tavg_adj_n=("tavg_adj", "count"),
        ).reset_index()
        agg = agg[agg["tmax_raw_n"] + agg["tmin_raw_n"] > 0]
        agg["ushcn_id"] = ushcn_id
        agg["lat"] = row["lat"]; agg["lon"] = row["lon"]
        monthly_rows.append(agg)

    stations.reset_index()[["ushcn_id", "lat", "lon"]].to_csv(UH_STN_FP, index=False)
    paired = pd.concat(monthly_rows, ignore_index=True)
    paired.to_parquet(UH_PAIRED, index=False)
    print(f"[ushcn] wrote {kept:,} daily files, {len(paired):,} station-months, "
          f"stations.csv and the paired monthly parquet")
    build_ushcn_pivots(force=True)


def build_ushcn_pivots(force=False):
    """The May-September raw/adjusted pivots. Figure 6 can build these itself,
    but Figure 3 only reads them, so they are made here and neither figure has
    to be run before the other."""
    mtag = "56789"
    fps = {leg: UH_PIV_DIR / f"ushcn_pair_tmax_pivot_m{mtag}_{leg}.parquet"
           for leg in ("raw", "adj")}
    if all(f.exists() for f in fps.values()) and not force:
        print("[ushcn] pivots already built"); return
    import glob
    files = sorted(glob.glob(str(UH_DAILY_DIR / "*.csv.gz")))
    if not files:
        print("[ushcn] no daily files to pivot"); return
    raw, adj = {}, {}
    for fp in files:
        sid = os.path.basename(fp)[:11]
        d = pd.read_csv(fp, dtype={"date": str},
                        usecols=["date", "tmax_raw_c", "tmax_adj_c"])
        dt = pd.to_datetime(d["date"], format="%Y%m%d", errors="coerce")
        vr = pd.to_numeric(d["tmax_raw_c"], errors="coerce")
        va = pd.to_numeric(d["tmax_adj_c"], errors="coerce")
        both = dt.notna() & vr.notna() & va.notna()
        if not both.any():
            continue
        idx = pd.DatetimeIndex(dt[both])
        for store, vv in ((raw, vr), (adj, va)):
            ss = pd.Series(vv[both].to_numpy("float32"), index=idx)
            store[sid] = ss[~ss.index.duplicated()]
    for leg, store in (("raw", raw), ("adj", adj)):
        pv = pd.concat(store, axis=1).sort_index()
        pv.index.name = "date"
        pv.to_parquet(str(fps[leg]))
    print(f"[ushcn] wrote MJJAS pivots for {len(raw):,} stations")


# ══════════════════════════════════════════════════════════════════════════════
# (c) the 2-degree global gridded GHCN-Daily TMAX (Figure 6, bottom row)
# ══════════════════════════════════════════════════════════════════════════════
GRID_DEG_PREP = 2.0

def build_grid(force=False):
    GHCND_GLOBAL_GRID.mkdir(parents=True, exist_ok=True)
    lat_e = np.arange(BAND_LAT_PREP[0], BAND_LAT_PREP[1] + 1e-9, GRID_DEG_PREP)
    lon_e = np.arange(-180.0, 180.0 + 1e-9, GRID_DEG_PREP)
    latc = 0.5 * (lat_e[:-1] + lat_e[1:])
    lonc = 0.5 * (lon_e[:-1] + lon_e[1:])

    todo = [y for y in range(Y0_PREP, Y1_PREP + 1)
            if force or not (GHCND_GLOBAL_GRID / f"ghcn_grid_{y}.nc").exists()]
    if not todo:
        print("[grid] already built"); return
    print(f"[grid] binning {len(todo)} years onto {len(latc)}x{len(lonc)} "
          f"{GRID_DEG_PREP:g}-degree cells")

    for y in todo:
        src = GHCND_GLOBAL / f"ghcn_global_{y}.parquet"
        if not src.exists():
            print(f"[grid] {y}: no band archive -- run the band step first")
            continue
        d = pd.read_parquet(src, columns=["date", "tmax_c", "lat", "lon"])
        d = d[d["tmax_c"].notna()]
        days = pd.date_range(f"{y}-01-01", f"{y}-12-31", freq="D")
        ti = pd.DatetimeIndex(d["date"])
        it = pd.Index(days).get_indexer(ti)
        il = np.clip(((d["lat"].to_numpy() - lat_e[0]) / GRID_DEG_PREP).astype(int),
                     0, len(latc) - 1)
        io = np.clip(((d["lon"].to_numpy() + 180.0) / GRID_DEG_PREP).astype(int),
                     0, len(lonc) - 1)
        ok = it >= 0
        num = np.zeros((len(days), len(latc), len(lonc)))
        den = np.zeros_like(num)
        np.add.at(num, (it[ok], il[ok], io[ok]), d["tmax_c"].to_numpy(float)[ok])
        np.add.at(den, (it[ok], il[ok], io[ok]), 1.0)
        grid = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
        xr.Dataset(
            {"tmax_c": (("time", "lat", "lon"), grid.astype(np.float32))},
            coords={"time": days, "lat": latc, "lon": lonc},
            attrs={"description": f"GHCN-Daily TMAX, {GRID_DEG_PREP:g}-degree "
                                  f"cell means, built by prepare_data.py",
                   "source": str(src)},
        ).to_netcdf(GHCND_GLOBAL_GRID / f"ghcn_grid_{y}.nc",
                    encoding={"tmax_c": {"zlib": True, "complevel": 4}})
        print(f"[grid] {y}  {int((den > 0).any(axis=0).sum()):,} cells reported",
              flush=True)
    print("[grid] done")


# ══════════════════════════════════════════════════════════════════════════════
# prune the local staging copies
# ══════════════════════════════════════════════════════════════════════════════
# Two archive files are reopened on every run, because the figure code does not
# cache what it reads from them: Figure 3 recomputes the Berkeley record counts
# from the CONUS daily file, and Figure 4 reads a RAW decade file for its land
# mask and record span. Those are small, so they are kept by default.
_STAGE_KEEP = (
    "Berkeley_Earth/Processed/preprocessed_us_TMAX_data.nc",   # Figure 3
    "Berkeley_Earth/RAW/Complete_TMAX_Daily_LatLong1_1900.nc",  # Figure 4
    "Berkeley_Earth/RAW/Complete_TMAX_Daily_LatLong1_2020.nc",  # Figure 4
)


def prune_stage(keep_all=False, dry_run=False):
    """Delete local copies of archive files under _stage/.

    Always safe: staging is automatic, so anything deleted is copied in again
    the next time something opens it. It only costs time, and only for a figure
    whose own cache has also been cleared -- with the caches in place the big
    inputs (all of ERA5, the Berkeley decade files) are never reopened.
    """
    from common import STAGE
    if not STAGE.is_dir():
        print("[prune] nothing staged"); return
    keep = set() if keep_all else {STAGE / k for k in _STAGE_KEEP}
    freed = 0
    removed = 0
    for fp in sorted(STAGE.rglob("*")):
        if not fp.is_file() or fp in keep:
            continue
        n = fp.stat().st_size
        freed += n
        removed += 1
        if not dry_run:
            fp.unlink()
    if not dry_run:
        for d in sorted((p for p in STAGE.rglob("*") if p.is_dir()), reverse=True):
            try: d.rmdir()
            except OSError: pass
    kept = sum(f.stat().st_size for f in keep if f.exists())
    print(f"[prune] {'would free' if dry_run else 'freed'} "
          f"{freed / 1e9:.1f} GB across {removed:,} files; "
          f"kept {kept / 1e6:.0f} MB that Figures 3 and 4 reread every run")


# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="*", default=None,
                    choices=["band", "ushcn", "grid", "prune"],
                    help="which steps to run (default: band, ushcn, grid). "
                         "`prune` deletes the local staging copies instead.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--all", action="store_true",
                    help="prune: delete every staged file, including the few "
                         "Figures 3 and 4 reread on every run")
    ap.add_argument("--dry-run", action="store_true",
                    help="prune: report what would be deleted, delete nothing")
    a = ap.parse_args()
    steps = a.steps or ["band", "ushcn", "grid"]
    print(f"ROOT = {ROOT}\nWORK = {WORK}\nDATA = {DATA}\n")
    if "band" in steps:  build_band(a.force, a.workers)
    if "ushcn" in steps:
        build_ushcn(a.force)
        build_ushcn_pivots(a.force)
    if "grid" in steps:  build_grid(a.force)
    if "prune" in steps: prune_stage(a.all, a.dry_run)


if __name__ == "__main__":
    main()

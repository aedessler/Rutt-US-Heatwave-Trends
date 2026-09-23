#!/usr/bin/env python3
"""Rebuild the two archive-level derived products the figures read.

`prepare_data.py` consumes these but never built them; the code that made the
copies on the archive is not in this repository, so extending the record past
2024 had no reproducible path. Both rules below were recovered from the files'
own metadata and then verified by rebuilding a year (or the whole range) and
diffing against the archived copy -- `--verify` does exactly that, and both
products reproduce their archived versions exactly.

    # check the recovered rules still reproduce what is on the archive
    python figures/build_derived.py cube    --verify 2024
    python figures/build_derived.py offsets --verify

    # extend
    python figures/build_derived.py cube    2025 --out DIR
    python figures/build_derived.py offsets --y1 2026 --out DIR

**The QC'd daily cube** (`GHCND/station_daily/<year>.nc`) is the raw global
`by_year/<year>.csv.gz` reduced to the CONUS station set, under the rules the
existing files record in their own attributes: rows with a non-blank quality
flag are dropped, the first observation of a station-day-element wins, values
stay in GHCN's 0.1 degC integers, and the year is laid out on 366 leap-aligned
slots with slot 59 always 29 February. The station axis is NOT re-derived --
every archived year shares one 18,128-station axis, so it is read from an
existing file and reused, which also keeps a new year aligned with the old ones.

**The USHCN offsets** (`USHCN_v2.5/derived/ushcn_offsets_<y0>_<y1>.nc`) are the
FLs.52j monthly value minus the raw monthly value, per station/year/month, keyed
by GHCN-Daily id through `ushcn_crosswalk.csv`. One rule is not in the file's
description and was recovered from the data: a month whose FLs.52j value carries
measurement flag `E` is dropped. `E` marks a value infilled from neighbours
rather than an adjustment of the station's own observation, so the difference
there is not a bias correction for that station's daily data. Keeping those
months adds ~87k spurious offsets to TMAX alone and does not reproduce the
archive.
"""
import argparse
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (GHCND_BY_YEAR, GHCND_STATION_DAILY, USHCN_DIR,   # noqa: E402
                    USHCN_CROSSWALK_FP)

BY_YEAR_COLS = ["id", "date", "elem", "value", "mflag", "qflag", "sflag", "obstime"]
OFFSET_SCALE = 0.01           # USHCN monthly source units -> degC
MONTH_WIDTH = 9               # 6-char value + 3 flag chars
MONTH_START = 16              # id(11) + element code(1) + year(4)


# ══ THE QC'D DAILY CUBE ═════════════════════════════════════════════════════
def station_axis(template=None):
    """The 18,128-station axis every archived year shares."""
    if template is None:
        cands = sorted(Path(GHCND_STATION_DAILY).glob("[12][0-9][0-9][0-9].nc"))
        if not cands:
            raise SystemExit(f"no year files under {GHCND_STATION_DAILY} to take "
                             f"the station axis from")
        template = cands[-1]
    with xr.open_dataset(str(template)) as ds:
        return ds["station_id"].values.astype(str).copy()


def _slots(year, dates):
    """Leap-aligned slot index: slot 59 is 29 February in every year."""
    doy = dates.dayofyear.to_numpy()
    if pd.Timestamp(year=year, month=1, day=1).is_leap_year:
        return doy - 1
    return np.where(doy <= 59, doy - 1, doy)


def build_cube(year, ids=None, chunksize=4_000_000, quiet=False):
    ids = station_axis() if ids is None else ids
    src = Path(GHCND_BY_YEAR) / f"{year}.csv.gz"
    if not src.exists():
        raise SystemExit(f"{src} is not on the archive -- nothing to build from")
    pos = pd.Index(ids)
    out = {e: np.full((len(ids), 366), np.nan, np.float32) for e in ("tmax", "tmin")}
    seen = {e: np.zeros((len(ids), 366), bool) for e in ("tmax", "tmin")}
    nrows = 0
    for chunk in pd.read_csv(src, header=None, names=BY_YEAR_COLS,
                             usecols=["id", "date", "elem", "value", "qflag"],
                             dtype={"id": str, "date": str, "elem": str,
                                    "value": "float64", "qflag": str},
                             chunksize=chunksize, na_filter=False,
                             keep_default_na=False):
        nrows += len(chunk)
        c = chunk[chunk["elem"].isin(("TMAX", "TMIN"))]
        c = c[c["qflag"].str.strip() == ""]        # non-blank quality flag -> dropped
        if c.empty:
            continue
        si = pos.get_indexer(c["id"].to_numpy())
        keep = si >= 0
        c, si = c[keep], si[keep]
        if len(c) == 0:
            continue
        v = pd.to_numeric(c["value"], errors="coerce").to_numpy(float)
        dt = pd.DatetimeIndex(pd.to_datetime(c["date"].to_numpy(),
                                             format="%Y%m%d", errors="coerce"))
        good = np.isfinite(v) & (v != -9999) & dt.notna()
        si, v, dt, el = si[good], v[good], dt[good], c["elem"].to_numpy()[good]
        sl = _slots(year, dt)
        for elem, tag in (("TMAX", "tmax"), ("TMIN", "tmin")):
            m = el == elem
            if not m.any():
                continue
            r, s, val = si[m], sl[m], v[m]
            fresh = ~seen[tag][r, s]               # first observation wins
            r, s, val = r[fresh], s[fresh], val[fresh]
            if len(r) == 0:
                continue
            _, first = np.unique(r.astype(np.int64) * 366 + s, return_index=True)
            r, s, val = r[first], s[first], val[first]
            out[tag][r, s] = val.astype(np.float32)
            seen[tag][r, s] = True
    if not quiet:
        print(f"[cube {year}] {nrows:,} rows read; "
              f"tmax {np.isfinite(out['tmax']).sum():,} / "
              f"tmin {np.isfinite(out['tmin']).sum():,} valid station-days")
    return xr.Dataset(
        {"tmax": (("station", "slot"), out["tmax"]),
         "tmin": (("station", "slot"), out["tmin"])},
        coords={"station": np.arange(len(ids)),
                "station_id": ("station", ids.astype("<U11")),
                "slot": np.arange(366)},
        attrs={"year": year, "units": "0.1 degC", "missing_value": -32768,
               "calendar": "366 leap-aligned slots; slot 59 is 29 February",
               "qflag_rule": "rows with a non-blank quality flag are dropped",
               "dedup": "first observation of a station-day-element wins",
               "source": str(src)})


# ══ THE USHCN OFFSETS ═══════════════════════════════════════════════════════
def _read_monthly(fp, drop_estimated=False):
    """{year: (12,) array} in source units, NaN where missing.

    drop_estimated also blanks a month whose measurement flag is 'E' -- in the
    FLs.52j leg that marks a value infilled from neighbours, not an adjustment
    of the station's own observation."""
    out = {}
    with open(fp) as fh:
        for line in fh:
            if len(line) < MONTH_START:
                continue
            vals = np.array([float(line[MONTH_START + m * MONTH_WIDTH:
                                        MONTH_START + m * MONTH_WIDTH + 6])
                             for m in range(12)])
            vals = np.where(vals == -9999, np.nan, vals)
            if drop_estimated:
                est = np.array([line[MONTH_START + 6 + m * MONTH_WIDTH] == "E"
                                for m in range(12)])
                vals = np.where(est, np.nan, vals)
            out[int(line[12:16])] = vals
    return out


def _ushcn_src():
    """The extracted USHCN monthly station files, unpacked from the archive's
    tarballs into a temp directory that the caller cleans up."""
    tmp = tempfile.mkdtemp(prefix="ushcn_monthly_")
    for var in ("tmax", "tmin", "tavg"):
        for leg in ("raw", "FLs.52j"):
            tar = Path(USHCN_DIR) / f"ushcn.{var}.latest.{leg}.tar.gz"
            if not tar.exists():
                raise SystemExit(f"{tar} is not on the archive")
            with tarfile.open(tar) as tf:
                tf.extractall(tmp)
    inner = [p for p in Path(tmp).iterdir() if p.is_dir()]
    return (inner[0] if inner else Path(tmp)), tmp


def build_offsets(y0, y1, src_dir=None, quiet=False):
    src_dir, tmp = (src_dir, None) if src_dir else _ushcn_src()
    try:
        xw = pd.read_csv(USHCN_CROSSWALK_FP).sort_values("ghcnd_id")
        xw = xw.reset_index(drop=True)
        years = np.arange(y0, y1 + 1)
        yi = {y: i for i, y in enumerate(years)}
        data = {}
        for var in ("tmax", "tmin", "tavg"):
            off = np.full((len(xw), len(years), 12), np.nan, np.float32)
            absent = 0
            for si, uid in enumerate(xw["ushcn_id"]):
                try:
                    raw = _read_monthly(f"{src_dir}/{uid}.raw.{var}")
                    adj = _read_monthly(f"{src_dir}/{uid}.FLs.52j.{var}",
                                        drop_estimated=True)
                except FileNotFoundError:
                    absent += 1
                    continue
                for y, a in adj.items():
                    if y in yi and y in raw:
                        off[si, yi[y]] = ((a - raw[y]) * OFFSET_SCALE).astype(np.float32)
            data[f"{var}_offset"] = (("station", "year", "month"), off)
            if not quiet:
                print(f"[offsets {var}] {np.isfinite(off).sum():,} finite "
                      f"station-months" + (f" | {absent} stations absent" if absent else ""))
        return xr.Dataset(
            data,
            coords={"station": np.arange(len(xw)),
                    "station_id": ("station", xw["ghcnd_id"].to_numpy().astype("<U11")),
                    "year": years, "month": np.arange(1, 13)},
            attrs={"description": "USHCN v2.5 FLs.52j minus raw monthly offsets, "
                                  "keyed by GHCN-Daily id"})
    finally:
        if tmp:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


# ══ VERIFICATION -- rebuild something that exists and diff it ═══════════════
def _diff(new, old, names):
    ok = True
    for v in names:
        A, B = old[v].values, new[v].values
        fa, fb = np.isfinite(A), np.isfinite(B)
        both = fa & fb
        d = np.abs(A[both] - B[both])
        worst = float(d.max()) if d.size else 0.0
        same = bool((fa == fb).all()) and worst < 1e-4
        ok &= same
        print(f"  {v:<14} archived {fa.sum():>10,} | rebuilt {fb.sum():>10,} | "
              f"only-archived {int((fa & ~fb).sum()):>7,} | only-new "
              f"{int((fb & ~fa).sum()):>7,} | max|diff| {worst:.3g}  "
              f"{'MATCH' if same else 'DIFFERS'}")
    return ok


def verify_cube(year):
    fp = Path(GHCND_STATION_DAILY) / f"{year}.nc"
    if not fp.exists():
        raise SystemExit(f"{fp} does not exist, so there is nothing to verify against")
    print(f"verifying the cube rules against {fp}")
    with xr.open_dataset(str(fp)) as old:
        new = build_cube(year, ids=old["station_id"].values.astype(str))
        same_ids = bool((old["station_id"].values.astype(str)
                         == new["station_id"].values.astype(str)).all())
        print(f"  station axis identical: {same_ids}")
        return _diff(new, old, ("tmax", "tmin")) and same_ids


def verify_offsets():
    fps = sorted(Path(USHCN_DIR).glob("derived/ushcn_offsets_*.nc"))
    if not fps:
        raise SystemExit("no archived offsets file to verify against")
    fp = fps[0]
    print(f"verifying the offset rules against {fp}")
    with xr.open_dataset(str(fp)) as old:
        yrs = old["year"].values
        new = build_offsets(int(yrs[0]), int(yrs[-1]))
        same_ids = bool((old["station_id"].values.astype(str)
                         == new["station_id"].values.astype(str)).all())
        print(f"  station axis identical: {same_ids}")
        return _diff(new, old, ("tmax_offset", "tmin_offset",
                                "tavg_offset")) and same_ids


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)

    c = sub.add_parser("cube", help="the QC'd GHCN-Daily CONUS daily cube")
    c.add_argument("year", nargs="?", type=int)
    c.add_argument("--verify", type=int, metavar="YEAR",
                   help="rebuild an archived year and diff instead of writing")
    c.add_argument("--out", type=Path, default=None,
                   help="directory to write <year>.nc into")

    o = sub.add_parser("offsets", help="the USHCN FLs.52j-minus-raw offsets")
    o.add_argument("--y0", type=int, default=1900)
    o.add_argument("--y1", type=int, required=False)
    o.add_argument("--verify", action="store_true",
                   help="rebuild the archived range and diff instead of writing")
    o.add_argument("--out", type=Path, default=None)

    a = ap.parse_args()
    if a.what == "cube":
        if a.verify:
            raise SystemExit(0 if verify_cube(a.verify) else 1)
        if a.year is None or a.out is None:
            raise SystemExit("cube needs a YEAR and --out DIR (or --verify YEAR)")
        a.out.mkdir(parents=True, exist_ok=True)
        dest = a.out / f"{a.year}.nc"
        build_cube(a.year).to_netcdf(dest)
        print(f"wrote {dest}")
    else:
        if a.verify:
            raise SystemExit(0 if verify_offsets() else 1)
        if a.y1 is None or a.out is None:
            raise SystemExit("offsets needs --y1 YEAR and --out DIR (or --verify)")
        a.out.mkdir(parents=True, exist_ok=True)
        dest = a.out / f"ushcn_offsets_{a.y0}_{a.y1}.nc"
        build_offsets(a.y0, a.y1).to_netcdf(dest)
        print(f"wrote {dest}")

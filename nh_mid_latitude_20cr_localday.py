#!/usr/bin/env python3
"""20CR daily Tmax, Tmin, Tavg on LOCAL-TIME days for the NH mid-latitude band.

Input   /Volumes/adessler_lab/20CR/Daily/raw_full/air.2m.<year>.nc
        20CR 2 m air temperature, global 1 degree, 3-hourly (UTC), 1900-2015.
Output  /Volumes/adessler_lab/20CR/Daily/NH_mid_latitude/20cr_2m_LT_<year>_daily.nc
        one file per year, every calendar day of the year, 27 lats x 360 lons.

Band    LAT_BOX = (23.9, 50.1), i.e. the 1 degree rows 24..50 N, at all longitudes.

Local time: local time = UTC + round(lon / 15) hours (longitude first wrapped to
[-180, 180); exact halves go to the even hour). This is the rule used for the ERA5
local-time files and in figures/reanalysis.py. For every grid point each local
calendar day is the 8 three-hourly samples between its own local midnight and the
next one, and

    tmax         the largest of the 8 samples
    tmin         the smallest of the 8 samples
    tavg         the mean of the 8 samples
    tavg_minmax  (tmax + tmin) / 2

A local day that lacks any of its 8 samples is NaN, never a statistic of fewer
samples. A local day can reach into the neighbouring UTC year, so the adjacent years' files
are read as well. Only the first and last year lack a neighbour on one side: in 1900
local Jan 1 is NaN east of 37.5 E (offset >= +3 h) and in 2015 local Dec 31 is NaN
west of 7.5 W (offset <= -1 h, which includes the whole United States).

Temperatures are written in degrees C (float32). `utc_offset_hours(lon)` gives the
offset applied at each longitude. Longitudes keep the source convention, 0..359 E.

Each archive file is copied to local disk, read, and the copy deleted at once (the
lab share is an SMB mount that HDF5 cannot read reliably in place). Output is built
locally and then copied to OUT_DIR. Re-running skips years whose output exists.

    python nh_mid_latitude_20cr_localday.py                  # all of 1900-2015
    python nh_mid_latitude_20cr_localday.py --years 1950 1952
    python nh_mid_latitude_20cr_localday.py --out /some/dir --force
"""
import argparse
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ARCHIVE = Path("/Volumes/adessler_lab/20CR/Daily")
SRC_FMT = str(ARCHIVE / "raw_full" / "air.2m.{y}.nc")
OUT_DIR = ARCHIVE / "NH_mid_latitude"
OUT_FMT = "20cr_2m_LT_{y}_daily.nc"
LAT_BOX = (23.9, 50.1)
YEARS = (1900, 2015)
STEP_H = 3                                   # 20CR samples are 3-hourly
_EPOCH = np.datetime64("1970-01-01T00", "h")


def local_offset_hours(lon):
    """Whole-hour offset from UTC: round(lon / 15), longitude wrapped to [-180, 180)."""
    lon = ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0
    return np.rint(lon / 15.0).astype(int)


def hours_since_epoch(times):
    t = np.asarray(pd.DatetimeIndex(times).values, dtype="datetime64[h]")
    return (t - _EPOCH).astype("int64")


def days_since_epoch(date):
    return int((np.datetime64(pd.Timestamp(date).date(), "D")
                - np.datetime64("1970-01-01", "D")).astype("int64"))


def local_day_stats(x, hours_utc, lon, day0, ndays):
    """Local-day max, min, mean of regularly spaced UTC samples.

    x          (T, nlat, nlon) samples at UTC times `hours_utc`
    hours_utc  (T,) integer hours since the epoch, evenly spaced by a step dividing 24
    lon        (nlon,) degrees east
    day0       first local calendar day wanted, days since 1970-01-01
    ndays      number of consecutive local days

    Returns three (ndays, nlat, nlon) float32 arrays. For a longitude with offset k,
    local day D spans UTC hours [24 D - k, 24 D - k + 24), which holds exactly
    24 / step samples, so each offset needs one gather. A day not fully covered by
    the samples (or containing a NaN) is NaN."""
    hours_utc = np.asarray(hours_utc, dtype=np.int64)
    dh = np.unique(np.diff(hours_utc))
    if dh.size != 1 or dh[0] <= 0 or 24 % int(dh[0]):
        raise ValueError(f"samples must be evenly spaced by a step dividing 24 h; saw {dh[:6].tolist()}")
    step = int(dh[0]); n = 24 // step; h0 = int(hours_utc[0])
    off = local_offset_hours(lon)
    nlat = x.shape[1]
    out = {s: np.full((ndays, nlat, x.shape[2]), np.nan, np.float32) for s in ("max", "min", "mean")}
    for k in np.unique(off):
        cols = np.flatnonzero(off == k)
        i0 = -((h0 + int(k) - 24 * int(day0)) // step)    # first sample at/after local midnight
        first = i0 + n * np.arange(ndays)
        ok = np.flatnonzero((first >= 0) & (first + n <= x.shape[0]))
        if ok.size == 0:
            continue
        blk = x[:, :, cols][first[ok][:, None] + np.arange(n)[None, :]]   # (days, n, nlat, cols)
        ix = np.ix_(ok, np.arange(nlat), cols)
        out["max"][ix] = blk.max(axis=1)
        out["min"][ix] = blk.min(axis=1)
        out["mean"][ix] = blk.mean(axis=1, dtype=np.float32)
    return out["max"], out["min"], out["mean"]


def _cp(src, dst, tries=3):
    """/bin/cp with a size check (Python's buffered open() is unreliable on this share)."""
    want, last = Path(src).stat().st_size, None
    for _ in range(tries):
        r = subprocess.run(["cp", "-f", str(src), str(dst)], capture_output=True)
        if r.returncode == 0 and Path(dst).stat().st_size == want:
            return
        last = r.stderr.decode().strip() or f"short copy of {src}"
    Path(dst).unlink(missing_ok=True)
    raise OSError(f"could not copy {src}: {last}")


def load_year(year, tmp):
    """One archive year, cropped to the band and in degC: (hours_utc, lat, lon, x)."""
    src = Path(SRC_FMT.format(y=year))
    local = Path(tmp) / f"{year}_{src.name}"
    _cp(src, local)
    try:
        with xr.open_dataset(local) as ds:
            da = ds["air"].transpose("time", "lat", "lon").sortby("lat").sortby("lon")
            da = da.sel(lat=slice(*LAT_BOX))
            if str(da.attrs.get("units", "")).strip().lower() in ("k", "degk", "kelvin"):
                da = da - 273.15
            x = da.values.astype(np.float32)
            hours = hours_since_epoch(da["time"].values)
            lat, lon = da["lat"].values.astype(float), da["lon"].values.astype(float)
        if not -60 < float(x[:8].mean()) < 60:
            raise ValueError(f"{src.name}: mean {x[:8].mean():.1f} degC; units?")
        return hours, lat, lon, x
    finally:
        local.unlink(missing_ok=True)


def build_year(year, loaded):
    """Local-day dataset of `year`; `loaded` maps year -> load_year() result."""
    hours, lat, lon, x = loaded[year]
    pieces_h, pieces_x = [hours], [x]
    # one UTC day of margin on each side covers every offset (within +-12 h)
    if year - 1 in loaded:
        h, _, _, xp = loaded[year - 1]; keep = h >= hours[0] - 24
        pieces_h.insert(0, h[keep]); pieces_x.insert(0, xp[keep])
    if year + 1 in loaded:
        h, _, _, xn = loaded[year + 1]; keep = h < hours[-1] + STEP_H + 24
        pieces_h.append(h[keep]); pieces_x.append(xn[keep])
    hh, xx = np.concatenate(pieces_h), np.concatenate(pieces_x)
    ndays = 366 if pd.Timestamp(year=year, month=12, day=31).dayofyear == 366 else 365
    tmax, tmin, tavg = local_day_stats(xx, hh, lon, days_since_epoch(f"{year}-01-01"), ndays)
    dates = pd.date_range(f"{year}-01-01", periods=ndays, freq="D")
    dims = ("time", "lat", "lon")
    ds = xr.Dataset(
        {"tmax": (dims, tmax, dict(long_name="daily maximum 2 m air temperature, local-time day", units="degC")),
         "tmin": (dims, tmin, dict(long_name="daily minimum 2 m air temperature, local-time day", units="degC")),
         "tavg": (dims, tavg, dict(long_name="daily mean of the 8 three-hourly 2 m air temperatures, local-time day", units="degC")),
         "tavg_minmax": (dims, ((tmax + tmin) / 2).astype(np.float32), dict(long_name="(tmax + tmin) / 2", units="degC")),
         "utc_offset_hours": (("lon",), local_offset_hours(lon).astype(np.int8),
                              dict(long_name="local time minus UTC applied at this longitude", units="hours"))},
        coords={"time": dates, "lat": ("lat", lat, dict(units="degrees_north")),
                "lon": ("lon", lon, dict(units="degrees_east"))},
        attrs=dict(title=f"20CR daily Tmax/Tmin/Tavg on local-time days, NH mid-latitude band, {year}",
                   source=SRC_FMT.format(y=year),
                   convention="local time = UTC + round(lon/15) h; days are local midnight-to-midnight, "
                              "8 three-hourly samples; incomplete days are NaN",
                   lat_box=f"{LAT_BOX[0]} to {LAT_BOX[1]}",
                   note="20CR 2 m temperature is the ensemble mean, which damps daily extremes",
                   history=f"built {time.strftime('%Y-%m-%d')} by nh_mid_latitude_20cr_localday.py"))
    return ds, int(np.isnan(tmax).sum())


def write_year(ds, dest, tmp):
    """Write locally with compression, then copy to `dest` (via a .part name)."""
    local = Path(tmp) / dest.name
    enc = {v: dict(zlib=True, complevel=4, dtype="float32", _FillValue=np.float32(np.nan))
           for v in ("tmax", "tmin", "tavg", "tavg_minmax")}
    ds.to_netcdf(local, engine="netcdf4", encoding=enc)
    part = dest.with_name(dest.name + ".part")
    try:
        _cp(local, part)
        part.replace(dest)
    finally:
        local.unlink(missing_ok=True)
        part.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", nargs=2, type=int, default=YEARS, metavar=("FIRST", "LAST"))
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--force", action="store_true", help="rebuild years whose output exists")
    a = ap.parse_args()
    y0, y1 = a.years
    a.out.mkdir(parents=True, exist_ok=True)
    todo = [y for y in range(y0, y1 + 1) if a.force or not (a.out / OUT_FMT.format(y=y)).exists()]
    if not todo:
        print("nothing to do"); return
    avail = range(YEARS[0], YEARS[1] + 1)          # years the archive holds, for neighbours

    with tempfile.TemporaryDirectory(prefix="20cr_lt_") as tmp, ThreadPoolExecutor(3) as pool:
        futs, loaded = {}, {}

        def fetch(y):
            if y in avail and y not in futs:
                futs[y] = pool.submit(load_year, y, tmp)

        for i, y in enumerate(todo):
            t0 = time.time()
            for k in (y - 1, y, y + 1, y + 2):     # read ahead so copying overlaps computing
                fetch(k)
            for k in (y - 1, y, y + 1):
                if k in futs and k not in loaded:
                    loaded[k] = futs[k].result()
            ds, nnan = build_year(y, loaded)
            write_year(ds, a.out / OUT_FMT.format(y=y), tmp)
            print(f"{y}: {len(ds.time)} days written, {nnan} NaN tmax values "
                  f"({time.time() - t0:.0f} s)", flush=True)
            for k in [k for k in loaded if k < y]:   # drop what no later year needs
                del loaded[k]; futs.pop(k, None)


if __name__ == "__main__":
    sys.exit(main())

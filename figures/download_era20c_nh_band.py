#!/usr/bin/env python3
"""ERA-20C 2 m temperature over the northern mid-latitude band, as LOCAL-time daily max / min / mean.

Downloads ERA-20C's 3-hourly surface FORECAST 2 m temperature (NCAR GDEX d626000, the product the
CONUS-only files in ERA20C/3hr_forecasts came from) for every longitude and for the same 24
latitudes as those files (24.11 to 49.91 N, the N80 Gaussian rows inside 24-50 N), 1900-2010.
For every grid point the UTC time is moved to local time

    local time = UTC + round(lon / 15) hours      (lon wrapped to [-180, 180); halves to the even hour)

-- the rule of ERA5/ERA5_LT and of reanalysis.py -- and the daily maximum, minimum and mean are
taken over local midnight-to-midnight days, from the 8 three-hourly samples of each. A local day
that is not fully covered by the data is NaN. That is only the two ends of the record: 1 Jan 1900
(the record starts at 09 UTC, so every column with a UTC offset of -6 h or more lacks samples) and
the last day of 2010 where the record ends before local midnight.

One file per year, written to ERA20C/NH_mid_latitude on the lab archive:

    era20c_2t_LT_<YYYY>_daily.nc     time (local dates, every day of the year), latitude (descending,
                                     native), longitude (0..358.875, native)
        t2m_max, t2m_min, t2m_mean   degC, float32
        utc_offset_hours             the whole-hour offset used at each longitude

(`t2m_mean` is the "Tavg" of the request: the mean of the day's eight samples.) Edge days need a day
of padding from the neighbouring years, so each year is fetched with 31 Dec of the year before and
the first hours of 1 Jan of the year after.

Each file is built on local disk, copied to the share with cp (HDF5 straight onto SMB is unreliable),
size- and byte-checked there, and the local copy deleted. A re-run skips years already on the share.

    python figures/download_era20c_nh_band.py                  # all years, 3 at a time
    python figures/download_era20c_nh_band.py --years 1936 1937 --workers 1
"""
import argparse, os, subprocess, sys, tempfile, time, urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reanalysis import local_day_stats, local_offset_hours, hours_since_epoch, days_since_epoch

THREDDS = ("https://thredds.rda.ucar.edu/thredds/ncss/grid/aggregations/g/d626000/15/"
           "ERA_20th_Century_3_hourly_atmospheric_surface_forecast-{year}")
VAR = "2_metre_temperature_surface"
LAT_BOX = (23.9, 50.1)        # picks the 24 rows 24.11 ... 49.91 N, as the CONUS files have
LON_BOX = (0.0, 358.9)        # every longitude of the 320-point grid
N_LAT, N_LON = 24, 320
Y0, Y1 = 1900, 2010
DEST = Path("/Volumes/adessler_lab/ERA20C/NH_mid_latitude")
FILE = "era20c_2t_LT_{year}_daily.nc"


def _get(url, dst, tries=6):
    """Download one NCSS request to `dst` with curl (Python's own HTTP client is many times slower
    against this server), retrying with a growing pause."""
    last = None
    for i in range(tries):
        r = subprocess.run(["curl", "-sS", "-f", "-L", "-m", "600", "-o", str(dst), url],
                           capture_output=True, text=True)
        if r.returncode == 0 and Path(dst).exists() and Path(dst).stat().st_size > 0:
            return
        last = (r.stderr or f"curl exit {r.returncode}").strip()
        time.sleep(5 * (i + 1))
    raise RuntimeError(f"giving up on {url}: {last}")


def fetch(year, t0, t1, tmpdir):
    """2 m temperature (K) of aggregation `year` between two UTC times, as (time, lat, lon)."""
    q = dict(var=VAR, north=LAT_BOX[1], south=LAT_BOX[0], west=LON_BOX[0], east=LON_BOX[1],
             time_start=pd.Timestamp(t0).strftime("%Y-%m-%dT%H:%M:%SZ"),
             time_end=pd.Timestamp(t1).strftime("%Y-%m-%dT%H:%M:%SZ"), accept="netcdf")
    url = THREDDS.format(year=year) + "?" + urllib.parse.urlencode(q)
    fp = Path(tmpdir) / f"chunk_{year}_{pd.Timestamp(t0):%Y%m%d%H}.nc"
    for attempt in range(3):
        _get(url, fp)
        try:
            ds = xr.open_dataset(fp)
            da = ds[VAR]
            tdim = [d for d in da.dims if d not in ("latitude", "longitude")]
            assert len(tdim) == 1, da.dims
            da = da.rename({tdim[0]: "time"}).transpose("time", "latitude", "longitude").load()
            ds.close(); fp.unlink()
            assert da.shape[1:] == (N_LAT, N_LON), f"grid {da.shape[1:]}, expected {(N_LAT, N_LON)}"
            return da.reset_coords(drop=True)
        except Exception as e:
            last = e; fp.unlink(missing_ok=True); time.sleep(5)
    raise RuntimeError(f"bad chunk {url}: {last}")


def samples_for_year(year, tmpdir):
    """UTC samples covering every local day of `year`. The aggregation named Y holds the valid times
    from 09 UTC on 1 Jan Y to 06 UTC on 1 Jan Y+1, so the first hours of Y and the 1 Jan hours that
    close it come from the neighbouring aggregations; a request past an aggregation's end returns
    what exists. Each valid time is in exactly one aggregation. Returns (T, lat, lon) float32 K,
    the latitudes, longitudes and the UTC times."""
    parts = []
    if year > Y0:                                      # 31 Dec before, and 00-06 UTC on 1 Jan
        parts.append(fetch(year - 1, f"{year - 1}-12-31T00:00", f"{year}-01-01T12:00", tmpdir))
    for m in pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="MS"):
        end = (m + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d") + "T21:00"
        if m.month == 12:
            end = f"{year + 1}-01-01T12:00"            # to the aggregation's last step, 06 UTC
        parts.append(fetch(year, m.strftime("%Y-%m-%dT00:00"), end, tmpdir))
    if year < Y1:                                      # 09-21 UTC on 1 Jan after
        parts.append(fetch(year + 1, f"{year + 1}-01-01T00:00", f"{year + 1}-01-01T21:00", tmpdir))
    da = xr.concat(parts, dim="time").sortby("time")
    t = pd.DatetimeIndex(da["time"].values)
    assert t.is_unique, f"{year}: a valid time came from two aggregations"
    return (da.values.astype(np.float32), da["latitude"].values.astype(np.float64),
            da["longitude"].values.astype(np.float64), t)


def build_year_file(year, out):
    """Download and process one year into the local file `out`."""
    with tempfile.TemporaryDirectory(prefix=f"era20c_{year}_") as tmp:
        x, lat, lon, times = samples_for_year(year, tmp)
    hours = hours_since_epoch(times)
    steps = np.unique(np.diff(hours))
    assert steps.tolist() == [3], f"{year}: the 3-hourly record has steps {steps.tolist()[:6]}"
    x = x - 273.15
    assert -60 < float(x[:8].mean()) < 60
    ndays = 366 if pd.Timestamp(f"{year}-12-31").is_leap_year else 365
    tmax, tmin, tmean = local_day_stats(x, hours, lon, days_since_epoch(f"{year}-01-01"), ndays)
    nan = int(np.isnan(tmax).sum())
    # only the record's two ends may be incomplete: 1 Jan 1900 east of Greenwich, 31 Dec 2010 west of it
    if 1900 < year < 2010 and nan:
        raise ValueError(f"{year}: {nan} local-day values are NaN inside the record")
    dates = pd.date_range(f"{year}-01-01", periods=ndays, freq="D")
    ds = xr.Dataset(
        {"t2m_max": (("time", "latitude", "longitude"), tmax),
         "t2m_min": (("time", "latitude", "longitude"), tmin),
         "t2m_mean": (("time", "latitude", "longitude"), tmean),
         "utc_offset_hours": (("longitude",), local_offset_hours(lon).astype(np.int8))},
        coords={"time": dates, "latitude": lat.astype(np.float32), "longitude": lon.astype(np.float32)},
        attrs=dict(
            title="ERA-20C 2 m temperature, daily max/min/mean over LOCAL midnight-to-midnight days",
            source="NCAR GDEX d626000, ERA 20th Century 3 hourly atmospheric surface forecast, "
                   "2_metre_temperature_surface, 3-hourly steps, N80 Gaussian grid 320 x 160 subset",
            local_time="UTC + round(lon/15) hours; lon wrapped to [-180,180), halves to the even hour",
            statistics="max, min and mean of the 8 three-hourly samples inside each local day; "
                       "NaN where a local day is not fully covered by the record",
            units_note="degC", created_by="figures/download_era20c_nh_band.py"))
    for v in ("t2m_max", "t2m_min", "t2m_mean"):
        ds[v].attrs.update(units="degC", long_name=f"daily {v[4:]} of 2 m temperature, local-time day")
    ds["utc_offset_hours"].attrs.update(units="h", long_name="whole-hour offset of local time from UTC")
    enc = {v: dict(zlib=True, complevel=3, shuffle=True, dtype="float32", _FillValue=np.nan)
           for v in ("t2m_max", "t2m_min", "t2m_mean")}
    ds.to_netcdf(out, encoding=enc)
    return nan


def _same_bytes(a, b):
    return subprocess.run(["cmp", "-s", str(a), str(b)]).returncode == 0


def job(args):
    year, dest, force = args
    final = Path(dest) / FILE.format(year=year)
    try:
        if final.exists() and final.stat().st_size > 0 and not force:
            return year, "exists", None
        with tempfile.TemporaryDirectory(prefix=f"era20c_out_{year}_") as tmp:
            local = Path(tmp) / FILE.format(year=year)
            nan = build_year_file(year, local)
            Path(dest).mkdir(parents=True, exist_ok=True)
            part = Path(dest) / (FILE.format(year=year) + ".part")
            for attempt in range(3):
                subprocess.run(["cp", "-f", str(local), str(part)], check=True)
                if part.stat().st_size == local.stat().st_size and _same_bytes(local, part):
                    break
            else:
                part.unlink(missing_ok=True)
                raise OSError("copy to the share did not verify")
            os.replace(part, final)
        return year, "built", nan
    except Exception as e:
        return year, "FAILED", f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--years", nargs="*", type=int)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dest", default=str(DEST))
    a = ap.parse_args()
    years = a.years or list(range(Y0, Y1 + 1))
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0, bad = time.time(), []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(job, (y, a.dest, a.force)) for y in years]
        for i, f in enumerate(as_completed(futs), 1):
            y, status, info = f.result()
            if status == "FAILED":
                bad.append(y)
            print(f"[{i}/{len(years)} {time.time() - t0:5.0f}s] {y} {status}"
                  + (f"  (NaN local-day values: {info})" if status == "built" and info else "")
                  + (f"  {info}" if status == "FAILED" else ""), flush=True)
    print(f"done: {len(years) - len(bad)} ok, {len(bad)} failed {sorted(bad)}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()

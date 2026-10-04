"""20CR and ERA-20C on LOCAL-TIME days -- the reanalysis inputs of build_master_table.py.

Not a figure. Both reanalyses are archived sub-daily, and a UTC day is the wrong unit for a daily
maximum or minimum: it splits the afternoon of the western United States across two dates. The
daily statistics are therefore taken over each grid point's own LOCAL midnight-to-midnight day,

    local time = UTC + round(lon / 15) hours      (lon wrapped to [-180, 180); halves to the even hour)

the rule of the ERA5 files in ERA5/ERA5_LT. The archive holds the finished result, one file per
year, for the northern mid-latitude band (every longitude, 24-50 N):

    20CR/Daily/NH_mid_latitude/20cr_2m_LT_<y>_daily.nc          tmax, tmin, tavg, tavg_minmax   1900-2015
    ERA20C/NH_mid_latitude/era20c_2t_LT_<y>_daily.nc            t2m_max, t2m_min, t2m_mean      1900-2010

(20CR from 20CR/Daily/raw_full by the lab's own script; ERA-20C by figures/download_era20c_nh_band.py,
which uses local_day_stats() below.) This module only EXTRACTS what the figures need -- local May 1 to
Sep 30 of each year, tmax / tmin / tmean on the common grid layout -- into small per-year files, and
reads those. Figures 1 and 4 use JJA, Figures 2 and 5 use May-September.

``local_day_stats()`` and ``check_local_day()`` (a synthetic test) stay because the ERA-20C files are
made with them.
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from common import ROOT, WORK, BOX, MIN_DAYS_MON, BE_PROC, _cache, _edges

MJJAS_MONTHS = (5, 6, 7, 8, 9)
_EPOCH = np.datetime64("1970-01-01T00", "h")

# the local-day files this module reads and writes, one per dataset and year. A
# test can point HEATWAVE_RA_CACHE somewhere else without touching the real ones.
RA_CACHE = Path(os.environ.get("HEATWAVE_RA_CACHE", str(WORK / "_cache_reanalysis")))
RA_CACHE.mkdir(parents=True, exist_ok=True)

# what each reanalysis offers: the label drawn on the figures and the years it covers. `band`: the
# source files cover the whole 24-50N strip at every longitude (both do, now).
RA = {"20cr":   dict(label="20CR",    years=(1900, 2015), band=True),
      "era20c": dict(label="ERA-20C", years=(1900, 2010), band=True)}


def ra_meta(name):
    return RA[name]


def local_offset_hours(lon):
    """Whole-hour offset from UTC of each longitude: round(lon / 15).

    Longitudes are first wrapped to [-180, 180) so that the western hemisphere
    gets a negative offset (-4 h for 60 W, not +20 h). Both give the same
    24-hour windows, but a +20 h offset labels each window with the NEXT
    calendar date, which would put the western United States a day off.
    Exact halves (7.5, 22.5, ... degrees) round to the even hour, as Python's
    round() does."""
    lon = ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0
    return np.rint(lon / 15.0).astype(int)


def hours_since_epoch(times):
    """Integer hours since 1970-01-01 00:00 UTC of a datetime-like index."""
    t = np.asarray(pd.DatetimeIndex(times).values, dtype="datetime64[h]")
    return (t - _EPOCH).astype("int64")


def days_since_epoch(date):
    """Integer days since 1970-01-01 of a date-like."""
    return int((np.datetime64(pd.Timestamp(date).date(), "D")
                - np.datetime64("1970-01-01", "D")).astype("int64"))


def local_day_stats(x, hours_utc, lon, day0, ndays):
    """Local-day max, min and mean of regularly spaced UTC samples.

    x          (T, nlat, nlon)  samples at UTC times `hours_utc`
    hours_utc  (T,) integer hours since the epoch; must be evenly spaced by a
               step that divides 24 (3 h and 6 h are what the archives hold)
    lon        (nlon,) degrees east, any convention
    day0       first LOCAL calendar day wanted, as days since 1970-01-01
    ndays      how many consecutive local days

    Returns tmax, tmin, tmean, each (ndays, nlat, nlon) float32; a local day
    that is not covered by the samples, or has a NaN sample, is NaN.

    For a longitude with offset k, local day D covers the UTC hours
    [24 D - k, 24 D - k + 24). The samples are evenly spaced by `step`, so every
    such window holds exactly 24 / step of them and consecutive days are
    consecutive blocks of that size -- one gather per distinct offset, with no
    per-day or per-point loop."""
    x = np.asarray(x)
    hours_utc = np.asarray(hours_utc, dtype=np.int64)
    if x.ndim != 3 or x.shape[0] != hours_utc.size:
        raise ValueError(f"x is {x.shape} but there are {hours_utc.size} sample times")
    dh = np.unique(np.diff(hours_utc))
    if dh.size != 1 or dh[0] <= 0 or 24 % int(dh[0]):
        raise ValueError(f"samples must be evenly spaced by a step dividing 24 h; "
                         f"saw steps {dh[:6].tolist()}")
    step = int(dh[0]); n = 24 // step; h0 = int(hours_utc[0])
    off = local_offset_hours(lon)
    if off.size != x.shape[2]:
        raise ValueError(f"{off.size} longitudes for {x.shape[2]} columns")
    nlat = x.shape[1]
    out = {s: np.full((ndays, nlat, x.shape[2]), np.nan, np.float32)
           for s in ("max", "min", "mean")}
    for k in np.unique(off):
        cols = np.flatnonzero(off == k)
        # first sample at or after local midnight of day0: ceil((24*day0 - k - h0) / step)
        i0 = -((h0 + int(k) - 24 * int(day0)) // step)
        first = i0 + n * np.arange(ndays)
        # whole days only: a day the samples do not fully cover stays NaN
        ok = np.flatnonzero((first >= 0) & (first + n <= x.shape[0]))
        if ok.size == 0:
            continue
        xs = x[:, :, cols].astype(np.float32)
        blk = xs[first[ok][:, None] + np.arange(n)[None, :]]    # (days, n, nlat, cols)
        ix = np.ix_(ok, np.arange(nlat), cols)
        out["max"][ix] = blk.max(axis=1)
        out["min"][ix] = blk.min(axis=1)
        out["mean"][ix] = blk.mean(axis=1)
    return out["max"], out["min"], out["mean"]


# ══════════════════════════════════════════════════════════════════════════════
# THE LOCAL-DAY FILES -- one per dataset and year, May 1 to Sep 30 local dates.
#
#   <name>_nh_MJJAS_<year>.nc          tmax, tmin, tmean (time, lat, lon) degC float32
#
# lat ascending, lon in [-180, 180) ascending, time = the 153 local calendar
# dates. Extracted once from the NH_mid_latitude files by build_localday_year()
# (below) and read by everything else here.
# ══════════════════════════════════════════════════════════════════════════════
def localday_fp(name, year):
    return RA_CACHE / f"{name}_nh_MJJAS_{int(year)}.nc"


def write_localday(name, year, lat, lon, tmax, tmin, tmean, attrs=None):
    """Write one year of local-day statistics; each array is (153, nlat, nlon)."""
    times = pd.date_range(f"{int(year)}-05-01", f"{int(year)}-09-30", freq="D")
    for a in (tmax, tmin, tmean):
        assert a.shape == (len(times), len(lat), len(lon)), (a.shape, len(times), len(lat), len(lon))
    ds = xr.Dataset({k: (("time", "lat", "lon"), np.asarray(v, np.float32), {"units": "degC"})
                     for k, v in (("tmax", tmax), ("tmin", tmin), ("tmean", tmean))},
                    coords=dict(time=times, lat=np.asarray(lat, float), lon=np.asarray(lon, float)),
                    attrs=dict(attrs or {}))
    fp = localday_fp(name, year)
    tmp = fp.with_name(fp.name + ".tmp")
    ds.to_netcdf(tmp)
    os.replace(tmp, fp)
    return fp


def read_localday(name, year, elem, lat_bnds=None, lon_bnds=None):
    """One year of one statistic ('tmax', 'tmin' or 'tmean') as a (153, nlat, nlon)
    DataArray in degC, cropped to the (min, max) bounds given."""
    fp = localday_fp(name, year)
    if not fp.exists():
        build_localday_year(name, year)
    ds = xr.open_dataset(fp)
    da = ds[elem]
    if lat_bnds is not None:
        da = da.sel(lat=slice(lat_bnds[0], lat_bnds[1]))
    if lon_bnds is not None:
        da = da.sel(lon=slice(lon_bnds[0], lon_bnds[1]))
    da = da.load()
    ds.close()
    return da


def ra_years(name, y0=None, y1=None):
    a, b = ra_meta(name)["years"]
    return list(range(max(a, y0 or a), min(b, y1 or b) + 1))


def jja_monthly_field(name, elem, pad=3.0):
    """JJA monthly means of a local-day statistic, (time, lat, lon): the shape of
    the ERA5 field Figure 1 reads, so it goes through the same anomaly pipeline.
    Cropped to Figure 1's CONUS box plus `pad` degrees, which cell_weights then
    trims to the cells that touch CONUS. A month needs MIN_DAYS_MON days."""
    def build():
        parts = []
        for y in ra_years(name):
            da = read_localday(name, y, elem,
                               lat_bnds=(BOX["lat_min"] - pad, BOX["lat_max"] + pad),
                               lon_bnds=(BOX["lon_min"] - pad, BOX["lon_max"] + pad))
            for mo in (6, 7, 8):
                d = da.sel(time=da["time"].dt.month == mo)
                m = d.mean("time", skipna=True).where(d.notnull().sum("time") >= MIN_DAYS_MON)
                parts.append(m.expand_dims(time=[pd.Timestamp(f"{y}-{mo:02d}-01")]))
        out = xr.concat(parts, dim="time").sortby("time").sortby("lat")
        return out.astype(np.float32)
    y0, y1 = ra_meta(name)["years"]
    return _cache(f"field_{name}_{elem}_{y0}_{y1}_nh", build)


def jja_annual(name, elem, lat_bnds=(24.0, 50.0), min_days=75):
    """JJA mean of a local-day statistic per cell and year, at the dataset's own
    resolution, all longitudes: (n_years, nlat, nlon), NaN where a cell has fewer
    than min_days of the 92 days. Returns (ann, lat, lon, years)."""
    ann, years, lat, lon = [], [], None, None
    for y in ra_years(name):
        da = read_localday(name, y, elem, lat_bnds=lat_bnds)
        sel = da.sel(time=da["time"].dt.month.isin([6, 7, 8]))
        v = sel.values
        n = np.isfinite(v).sum(axis=0)
        with np.errstate(invalid="ignore"):
            m = np.where(n >= min_days, np.nanmean(v, axis=0), np.nan)
        ann.append(m.astype(np.float32)); years.append(y)
        lat, lon = da["lat"].values, da["lon"].values
    return np.stack(ann), lat, lon, np.array(years)


def mjjas_cube(name, elem="tmax", lat_bnds=None, lon_bnds=None):
    """May-September daily values on the canonical calendar, years x 153 days:
    (n_years * 153, nlat, nlon) float32 degC, plus lat, lon and the date of every
    row. This is the layout Figures 2 and 5 count records and heat waves on."""
    cube, times, lat, lon = [], [], None, None
    for y in ra_years(name):
        da = read_localday(name, y, elem, lat_bnds=lat_bnds, lon_bnds=lon_bnds)
        cube.append(da.values); times.append(pd.DatetimeIndex(da["time"].values))
        lat, lon = da["lat"].values, da["lon"].values
    return (np.concatenate(cube, axis=0).astype(np.float32, copy=False), lat, lon,
            times[0].append(times[1:]) if len(times) > 1 else times[0])


# ══════════════════════════════════════════════════════════════════════════════
# NH_mid_latitude FILES -> PER-YEAR MAY-SEPTEMBER EXTRACTS
#
# Each archive file is copied to local disk, read, and DELETED before the next one is touched: the
# archive is on a network share that HDF5 cannot read reliably in place. The small extracts that
# remain (RA_CACHE) are intermediates too: build_master_table.py removes them once its CSV is
# written, because nothing copied from the archive is to be kept.
# ══════════════════════════════════════════════════════════════════════════════
_SRC = {
    "20cr":   dict(rel="20CR/Daily/NH_mid_latitude/20cr_2m_LT_{y}_daily.nc",
                   vars=dict(tmax="tmax", tmin="tmin", tmean="tavg"), lat="lat", lon="lon"),
    "era20c": dict(rel="ERA20C/NH_mid_latitude/era20c_2t_LT_{y}_daily.nc",
                   vars=dict(tmax="t2m_max", tmin="t2m_min", tmean="t2m_mean"),
                   lat="latitude", lon="longitude"),
}
RAW_COPY_DIR = RA_CACHE / "_raw_copy"          # emptied again after every file


def _copy_from_archive(src, tag):
    """/bin/cp one archive file into RAW_COPY_DIR (Python's buffered open() fails on this
    share where cp does not), checking the size. The caller deletes it."""
    import subprocess
    src = Path(src)
    dst = RAW_COPY_DIR / f"{tag}_{os.getpid()}_{src.name}"
    want, last = src.stat().st_size, None
    for _ in range(3):
        try:
            RAW_COPY_DIR.mkdir(parents=True, exist_ok=True)
            subprocess.run(["cp", "-f", str(src), str(dst)], check=True, capture_output=True)
            if dst.stat().st_size == want:
                return dst
            last = f"short copy {dst.stat().st_size} of {want} bytes"
        except subprocess.CalledProcessError as e:
            last = (e.stderr or b"").decode().strip() or str(e)
        except OSError as e:
            last = str(e)
    dst.unlink(missing_ok=True)
    raise OSError(f"could not copy {src} after 3 attempts: {last}")


def build_localday_year(name, year):
    """The NH_mid_latitude file of one year -> its May 1-Sep 30 extract (tmax, tmin, tmean).
    Copies the archive file, uses it, deletes the copy."""
    spec = _SRC[name]
    src = ROOT / spec["rel"].format(y=year)
    tmp = _copy_from_archive(src, name)
    try:
        ds = xr.open_dataset(tmp)
        try:
            out = {}
            for k, v in spec["vars"].items():
                da = ds[v].rename({spec["lat"]: "lat", spec["lon"]: "lon"}).transpose("time", "lat", "lon")
                if str(da.attrs.get("units", "")).strip() != "degC":
                    raise ValueError(f"{src.name}: {v} is in {da.attrs.get('units')!r}, expected degC")
                # lat ascending, lon wrapped to [-180, 180) and ascending: what every reader here assumes
                da = da.assign_coords(lon=((da["lon"] + 180.0) % 360.0) - 180.0).sortby("lon").sortby("lat")
                da = da.sel(time=slice(f"{year}-05-01", f"{year}-09-30")).load()
                out[k] = da
            meta_src = dict(ds.attrs)
        finally:
            ds.close()
        t = pd.DatetimeIndex(out["tmax"]["time"].values)
        want = pd.date_range(f"{year}-05-01", f"{year}-09-30")
        if not t.equals(want):
            raise ValueError(f"{src.name}: expected {len(want)} days May-Sep, found {len(t)}")
        bad = sum(int(np.isnan(a.values).sum()) for a in out.values())
        if bad:
            raise ValueError(f"{name} {year}: {bad} NaN values in May-September")
        lat = out["tmax"]["lat"].values.astype(float); lon = out["tmax"]["lon"].values.astype(float)
        write_localday(name, year, lat, lon, out["tmax"].values, out["tmin"].values, out["tmean"].values,
                       attrs=dict(source=str(src), dataset=ra_meta(name)["label"],
                                  convention=str(meta_src.get("convention", meta_src.get("local_time", ""))),
                                  note="May-Sep extract made by figures/reanalysis.py"))
    finally:
        tmp.unlink(missing_ok=True)               # the empty temp dir is removed by remove_copies()
    return year


def remove_copies():
    """Delete the temporary archive copies and the per-year extracts: everything this module
    copied from the archive. Returns the number of files removed."""
    n = 0
    for fp in list(RA_CACHE.glob("*_nh_MJJAS_*.nc")) + (list(RAW_COPY_DIR.glob("*")) if RAW_COPY_DIR.exists() else []):
        fp.unlink(); n += 1
    if RAW_COPY_DIR.exists():
        RAW_COPY_DIR.rmdir()
    return n


def _build_job(job):
    name, year, force = job
    try:
        if force or not localday_fp(name, year).exists():     # a re-run keeps what is built
            build_localday_year(name, year)
        return name, year, None
    except Exception as e:                        # report and carry on; a re-run fills the gaps
        return name, year, f"{type(e).__name__}: {e}"


def build_localday(names=("20cr", "era20c"), years=None, workers=4, force=False):
    """Build every missing local-day file for `names` (years default to each dataset's span),
    `workers` at a time. Safe to interrupt and re-run."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    jobs = []
    for n in names:
        for y in (years or ra_years(n)):
            if force or not localday_fp(n, y).exists():
                jobs.append((n, y, force))
    print(f"building {len(jobs)} local-day files with {workers} workers -> {RA_CACHE}")
    failed, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_build_job, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            n, y, err = f.result()
            if err:
                failed.append((n, y, err)); print(f"  FAILED {n} {y}: {err}", flush=True)
            elif i % 5 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done ({time.time() - t0:.0f}s)", flush=True)
    print(f"{len(jobs) - len(failed)} built, {len(failed)} failed")
    return failed


def _overlap_weights(target_c, source_c, period=None):
    """(n_target, n_source): the share of each target cell covered by each source cell, in one
    dimension, with cell edges halfway between centres. Both centre lists ascending; `period`
    (360 for longitude) lets a cell at the seam overlap the cells on the other side of it."""
    te, se = _edges(target_c), _edges(source_c)
    tlo, thi, slo, shi = te[:-1, None], te[1:, None], se[None, :-1], se[None, 1:]
    w = np.zeros((len(target_c), len(source_c)))
    for sh in ((0.0,) if period is None else (-period, 0.0, period)):
        w += np.clip(np.minimum(thi, shi + sh) - np.maximum(tlo, slo + sh), 0.0, None)
    return w / (thi - tlo)


def berkeley_land_fraction(lat, lon):
    """Share of every (lat, lon) cell that Berkeley Earth's 1-degree land mask calls land,
    in [0, 1]. Figure 4 masks ERA5 to Berkeley's land footprint by nearest cell centre, which
    is exact for a 0.25-degree grid. A 1- to 2-degree reanalysis cell holds parts of several
    Berkeley cells, and a 1-degree grid sits half a cell off Berkeley's, so here the mask is
    carried over by AREA: each cell takes the overlap-weighted share of the Berkeley cells
    under it that are land, and the caller thresholds that. `lat` must be ascending."""
    lat = np.asarray(lat, float)
    lon = ((np.asarray(lon, float) + 180.0) % 360.0) - 180.0
    ds = None
    for dec in range(1900, 2030, 10):
        f = BE_PROC / f"processed_nh_TMAX_Complete_TMAX_Daily_LatLong1_{dec}.nc"
        if f.exists():
            ds = xr.open_dataset(f)
            break
    if ds is None:
        raise FileNotFoundError(f"no Berkeley NH decade files under {BE_PROC}")
    bl = ds["latitude"].values.astype(float)
    bo = ((ds["longitude"].values.astype(float) + 180.0) % 360.0) - 180.0
    lm = ds["land_mask"]
    lm = (lm.isel(time=0) if "time" in lm.dims else lm).values
    ds.close()
    land = (np.isfinite(lm) & (lm > 0)).astype(float)
    ol, oo = np.argsort(bl), np.argsort(bo)
    land, bl, bo = land[np.ix_(ol, oo)], bl[ol], bo[oo]
    order = np.argsort(lon)
    wy = _overlap_weights(lat, bl)
    wx = _overlap_weights(lon[order], bo, period=360.0)
    cover = wy.sum(1)[:, None] * wx.sum(1)[None, :]            # < 1 only outside Berkeley's grid
    frac = (wy @ land @ wx.T) / np.where(cover > 0, cover, 1.0)
    out = np.empty_like(frac)
    out[:, order] = frac
    return out



def check_localday_files():
    """CHECK -- which local-day files are on disk. A missing year is built from the raw
    sub-daily archive the first time something asks for it (slow: the archive is on a
    network share); a cached one is read in a fraction of a second."""
    print(f"{'dataset':<10}{'years':<12}{'cached':>9}   {RA_CACHE}")
    for name, meta in RA.items():
        yrs = ra_years(name)
        have = sum(localday_fp(name, y).exists() for y in yrs)
        print(f"  {meta['label']:<8}{yrs[0]}-{yrs[-1]:<7}{have:>5}/{len(yrs):<4}"
              + ("" if have == len(yrs) else "  (the rest are built from the lab archive)"))


def check_local_day():
    """CHECK -- local_day_stats against a longhand reference, on synthetic numbers.
    Needs no files."""
    rng = np.random.default_rng(0)
    lon = np.array([-122.5, -97.0, -75.0, -7.5, 0.0, 22.5, 100.0, 179.0, -179.0, 359.0])
    off = local_offset_hours(lon)
    # offsets: wrapped to the western hemisphere, halves to the even hour
    assert off.tolist() == [-8, -6, -5, 0, 0, 2, 7, 12, -12, 0], off.tolist()   # 359 E is 1 W
    day0, nd = days_since_epoch("2001-05-01"), 30

    for step in (3, 6):
        T = (24 // step) * 60
        hrs = hours_since_epoch(pd.date_range("2001-04-20", periods=T, freq=f"{step}h"))
        x = rng.normal(size=(T, 2, lon.size)).astype(np.float32)
        mx, mn, me = local_day_stats(x, hrs, lon, day0, nd)
        # longhand: pick each local day's samples by their local calendar date
        for j, k in enumerate(off):
            for d in range(nd):
                sel = (hrs + k) // 24 == day0 + d
                assert sel.sum() == 24 // step, "a local day must hold 24/step samples"
                ref = x[sel][:, :, j]
                assert np.array_equal(mx[d, :, j], ref.max(0))
                assert np.array_equal(mn[d, :, j], ref.min(0))
                assert np.allclose(me[d, :, j], ref.mean(0), atol=1e-6)

    # window placement, exactly: a ramp equal to the hour itself. For 100 W (-7 h)
    # local May 1 is UTC May 1 07:00 to May 2 07:00, so the first three-hourly
    # sample inside it is 09:00 and the last 06:00 the next morning.
    hrs = hours_since_epoch(pd.date_range("2001-04-20", periods=8 * 60, freq="3h"))
    ramp = np.broadcast_to(hrs[:, None, None].astype(np.float64), (hrs.size, 1, 1))
    mx, mn, me = local_day_stats(ramp, hrs, np.array([-100.0]), day0, 3)
    h_may1_0900 = hours_since_epoch(["2001-05-01 09:00"])[0]
    assert mn[0, 0, 0] == h_may1_0900 and mx[0, 0, 0] == h_may1_0900 + 21
    assert me[0, 0, 0] == h_may1_0900 + 10.5 and mn[1, 0, 0] - mn[0, 0, 0] == 24

    # a day the samples do not fully cover, or that has a missing sample, is NaN
    early = hours_since_epoch(pd.date_range("2001-05-01", periods=8 * 5, freq="3h"))
    xe = np.zeros((early.size, 1, 2), np.float32)
    mx, mn, me = local_day_stats(xe, early, np.array([-100.0, 100.0]), day0, 8)
    assert np.isnan(mx[0, 0, 1])          # 100 E is +7 h: its May 1 starts before the first sample
    assert np.isfinite(mx[1, 0, 1]) and np.isnan(mx[5:, 0, :]).all()
    xg = xe.copy(); xg[12, 0, 0] = np.nan
    mx, mn, me = local_day_stats(xg, early, np.array([-100.0, 100.0]), day0, 5)
    assert np.isnan(mx[1, 0, 0]) and np.isnan(me[1, 0, 0]) and np.isfinite(mx[1, 0, 1])
    # irregular spacing is refused, not averaged over
    try:
        local_day_stats(xe[:-1], np.delete(early, 3), np.array([-100.0, 100.0]), day0, 2)
        raise AssertionError("an irregular time axis should have been rejected")
    except ValueError:
        pass
    print("local_day_stats: matches the longhand local-date selection at 3 h and 6 h, "
          "places the 100 W window at\n  07:00 UTC, leaves uncovered or incomplete "
          "days NaN, and refuses an irregular time axis")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Extract the May-Sep 20CR / ERA-20C files the figures need from the "
                                             "NH_mid_latitude files on the lab archive.")
    ap.add_argument("names", nargs="*", default=list(RA), help="20cr and/or era20c")
    ap.add_argument("--years", nargs="*", type=int, help="only these years")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="rebuild files that already exist")
    a = ap.parse_args()
    check_local_day()
    raise SystemExit(1 if build_localday(a.names, a.years, a.workers, a.force) else 0)

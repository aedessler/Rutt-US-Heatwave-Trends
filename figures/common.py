"""Shared machinery for every figure script in this directory.

This is sections 1-4 of ``US-Heatwave-Trends-Analysis-Code.ipynb`` -- paths,
the CONUS polygon, the anomaly-first pipeline and the gridded loaders -- plus
the May-September GHCN-Daily station cube that Figures 2 and 5 both read
(notebook section 7 builds it; section 10 reads it back off disk).

Nothing here draws anything. Each ``figureN.py`` does ``from common import *``
and then reproduces exactly one figure of the paper.

Paths can be redirected without editing this file:

    HEATWAVE_ROOT=/path/to/archives   raw data archives  (default: the external drive)
    HEATWAVE_WORK=/path/to/workdir    caches and figure output
                                      (default: the original work directory if it
                                      exists, otherwise the repository root)
    HEATWAVE_SHOW=1                   use an interactive matplotlib backend
                                      instead of writing files headless
"""

# ══════════════════════════════════════════════════════════════════════════════
# SETUP -- imports, data locations, and the settings every figure shares.
# ══════════════════════════════════════════════════════════════════════════════
import os, re, calendar, warnings, pickle
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl

# headless by default: a script must be able to write its PNG with no display
if os.environ.get("HEATWAVE_SHOW", "") != "1":
    mpl.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import blended_transform_factory
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import cKDTree
from statsmodels.nonparametric.smoothers_lowess import lowess
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings("ignore", category=RuntimeWarning)
# the scripts keep the notebook's plt.show() calls; under Agg that is a no-op
# and matplotlib says so once per figure, which is only noise here.
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")

# ── where the data live ────────────────────────────────────────────────────
# The archive layout is the one on /Volumes/adessler_lab, with a fallback to
# the layout the notebook used, so either drive works. `prepare_data.py`
# builds the three inputs no archive carries in the form the figures want;
# those live under DATA, inside the work directory.
_ROOT_CANDIDATES = [Path("/Volumes/adessler_lab"),
                    Path("/Volumes/Expansion/homogenized_dataset_analysis")]
if "HEATWAVE_ROOT" in os.environ:
    ROOT = Path(os.environ["HEATWAVE_ROOT"])
else:
    ROOT = next((p for p in _ROOT_CANDIDATES if p.exists()), _ROOT_CANDIDATES[0])

_WORK_DEFAULT = Path("/Users/jessesmac/Myprojects/PHD_work/homogenized_dataset_work")
WORK = (Path(os.environ["HEATWAVE_WORK"]) if "HEATWAVE_WORK" in os.environ else
        (_WORK_DEFAULT if _WORK_DEFAULT.exists()
         else Path(__file__).resolve().parent.parent))
FIGD = WORK / "PAPER_FIGURES_FINAL"
DATA = WORK / "_data"          # what prepare_data.py builds


def _pick(*candidates):
    """First path that exists, else the first one, so a missing input is
    reported under its preferred name."""
    cands = [Path(c) for c in candidates]
    return next((c for c in cands if c.exists()), cands[0])


def _pick_glob(directory, pattern, fallback):
    """Newest file matching `pattern`. nCLIMDIV and NOAAGlobalTemp carry a
    processing date in the filename, so the name is resolved, not hard-coded."""
    hits = sorted(Path(directory).glob(pattern)) if Path(directory).is_dir() else []
    return hits[-1] if hits else Path(fallback)


# -- the raw archive, as it sits on the drive ---------------------------------
GHCND_DIR           = _pick(ROOT / "GHCND")
GHCND_BY_YEAR       = _pick(GHCND_DIR / "by_year")           # raw global CSVs
GHCND_STATION_DAILY = _pick(GHCND_DIR / "station_daily")     # QC'd CONUS cube
GHCND_INV           = _pick(GHCND_DIR / "meta" / "ghcnd-inventory.txt",
                            GHCND_DIR / "ghcnd-inventory.txt")
GHCND_STATIONS_TXT  = _pick(GHCND_DIR / "meta" / "ghcnd-stations.txt",
                            GHCND_DIR / "ghcnd-stations.txt")

BE_DIR       = _pick(ROOT / "Berkeley_Earth")
BE_PROC      = BE_DIR / "Processed"
BE_RAW       = BE_DIR / "RAW"
BE_TMAX      = BE_PROC / "preprocessed_us_TMAX_data.nc"
BE_TMIN      = BE_PROC / "preprocessed_us_TMIN_data.nc"
BE_TAVG      = BE_PROC / "preprocessed_us_TAVG_data.nc"   # optional; read by table2.py

ERA5_DIR     = _pick(ROOT / "ERA5")
NOAA_DIR     = _pick(ROOT / "NOAAGlobalTemp", ROOT / "noaaglobaltemp")
NOAA_FP      = _pick_glob(NOAA_DIR, "NOAAGlobalTemp_v6.0.0_gridded_*.nc",
                          NOAA_DIR / "NOAAGlobalTemp_v6.0.0_gridded.nc")
CRUTEM_DIR   = _pick(ROOT / "CRUTEM5", ROOT / "crutem5")
CRUTEM_FP    = CRUTEM_DIR / "CRUTEM.5.1.0.0.anomalies.nc"
NCLIMDIV_DIR = _pick(ROOT / "nClimDiv", ROOT / "nclimdiv")

def nclimdiv_file(elem):
    """The statewide file for one element, whatever its processing date."""
    stem = {"tmax": "tmaxst", "tmin": "tminst", "tavg": "tmpcst"}[elem]
    return _pick_glob(NCLIMDIV_DIR, f"climdiv-{stem}-v1.0.0-*",
                      NCLIMDIV_DIR / f"climdiv-{stem}-v1.0.0")

USHCN_DIR          = _pick(ROOT / "USHCN_v2.5")
# the offsets carry their year span in the name, so the newest one wins rather
# than the span being pinned here. build_derived.py writes them.
USHCN_OFFSETS_FP   = _pick_glob(USHCN_DIR / "derived", "ushcn_offsets_*.nc",
                                USHCN_DIR / "derived" / "ushcn_offsets_1900_2024.nc")
USHCN_CROSSWALK_FP = USHCN_DIR / "derived" / "ushcn_crosswalk.csv"
USHCN_STATIONS_TXT = USHCN_DIR / "ushcn-v2.5-stations.txt"

# Only used by Figure 5's non-default GHCND_CONUS_SOURCE = "christy_grid".
USREG_FP = Path(os.environ.get("HEATWAVE_USREG",
                               "/Volumes/Expansion/christy2026/usreg_half.txt"))

# -- built by prepare_data.py --------------------------------------------------
BAND_LAT_PREP    = (24.0, 50.0)   # the widest station domain any figure needs
Y0_PREP, Y1_PREP = 1900, 2026

GHCND_GLOBAL      = _pick(DATA / "ghcnd_band", GHCND_DIR / "Global")
GHCND_GLOBAL_GRID = _pick(DATA / "ghcnd_global_grid", GHCND_DIR / "Global_Gridded")
UH_WORKDIR   = _pick(DATA / "ushcn_daily_homog", ROOT / "ushcn_daily_homog")
UH_STN_FP    = UH_WORKDIR / "output" / "stations.csv"
UH_DAILY_DIR = UH_WORKDIR / "output" / "daily"
UH_PIV_DIR   = WORK / "_cache_ushcn_homog"
UH_PAIRED    = UH_PIV_DIR / "ushcn_homog_station_months_paired.parquet"

# ── local staging of archive files ───────────────────────────────────────────
# HDF5 does many small random-access reads, and doing them against the SMB
# share this archive lives on is unreliable: it returns "Bad file descriptor",
# or segfaults the HDF5 library outright, on files that a plain sequential
# copy reads perfectly. Which file it hits varies between runs. So any netCDF
# under ROOT is copied to local disk once, and opened from there. Copies are
# kept, so this is also why a second run is much faster than the first.
#
# Set HEATWAVE_STAGE=0 to read the archive in place (fine on a local disk).
STAGE = WORK / "_stage"
STAGE_ENABLED = os.environ.get("HEATWAVE_STAGE", "1") != "0"


def stage(fp):
    """Local path for an archive file, copying it in on first use."""
    fp = Path(fp)
    if not STAGE_ENABLED:
        return fp
    try:
        rel = fp.resolve().relative_to(ROOT.resolve())
    except (ValueError, OSError):
        return fp                      # not under the archive: already local
    dst = STAGE / rel
    try:
        if dst.exists() and dst.stat().st_size == fp.stat().st_size:
            return dst
    except OSError:
        return fp
    # Use /bin/cp, not shutil: on this share Python's buffered open() can fail
    # with "Bad file descriptor" on a file that cp (fcopyfile) copies cleanly.
    import subprocess
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    want = fp.stat().st_size
    last = None
    for attempt in range(3):
        try:
            subprocess.run(["cp", "-f", str(fp), str(tmp)], check=True,
                           capture_output=True)
            if tmp.stat().st_size == want:
                os.replace(tmp, dst)
                return dst
            last = f"short copy: {tmp.stat().st_size} of {want} bytes"
        except subprocess.CalledProcessError as e:
            last = (e.stderr or b"").decode().strip() or str(e)
        except OSError as e:
            last = str(e)
    tmp.unlink(missing_ok=True)
    raise OSError(f"could not stage {fp} to {dst} after 3 attempts: {last}")


def _staging_open(orig, many=False):
    def wrapped(path, *a, **kw):
        if many:
            if isinstance(path, (str, os.PathLike)):
                path = stage(path)
            elif isinstance(path, (list, tuple)):
                path = [stage(p) for p in path]
        elif isinstance(path, (str, os.PathLike)):
            path = stage(path)
        return orig(path, *a, **kw)
    return wrapped

# the figure scripts call xr.open_dataset / xr.open_mfdataset exactly as the
# notebook did; staging is applied underneath so none of that code changes.
if STAGE_ENABLED:
    xr.open_dataset = _staging_open(xr.open_dataset)
    xr.open_mfdataset = _staging_open(xr.open_mfdataset, many=True)

# ── caches (built on first run, reused afterwards) ───────────────────────────
CACHE_ANOM  = WORK / "_cache_fig1_adessler"     # Figure 1 and Table 2
CACHE_REC   = FIGD / "cache_fixed_metrics"      # Figure 2, and the cube Figure 5 reads
CACHE_MAPS  = CACHE_REC / "global"              # Figure 3
CACHE_BAND  = WORK / "_cache_band_jja"          # Figure 4
CACHE_HW    = WORK / "_cache_christy_hw"        # Figure 5
for _d in (FIGD, CACHE_ANOM, CACHE_REC, CACHE_MAPS, CACHE_BAND, CACHE_HW,
           UH_PIV_DIR, STAGE):
    _d.mkdir(parents=True, exist_ok=True)

# ── settings Figure 1 and Table 2 share ────────────────────────────────────────
BASELINE  = (1951, 1980)          # anomaly baseline, both figures
BOX       = dict(lat_min=24.0, lat_max=50.0, lon_min=-125.0, lon_max=-66.0)
MIN_BASE_STN  = 15        # of 30 baseline years, per station per calendar month
MIN_BASE_CELL = 20        # of 30, per grid cell per calendar month
MIN_DAYS_FRAC = 0.80      # station-month completeness
MIN_DAYS_MON  = 20        # days needed for a monthly mean of a gridded daily field
EARTH_R = 6371.0088
G05_LAT_EDGES = np.round(np.arange(25.0, 50.0 + 1e-9, 0.5), 3)   # 0.5 deg CONUS grid
G05_LON_EDGES = np.round(np.arange(-124.5, -66.5 + 1e-9, 0.5), 3)


def check_paths():
    """CHECK -- can this machine see the data? The raw archive must be mounted;
    the `built` rows come from prepare_data.py, the `optional` ones are only
    needed by settings the figures do not use by default."""
    _req = {"GHCN-Daily by-year CSVs": GHCND_BY_YEAR,
            "GHCN-Daily inventory": GHCND_INV,
            "GHCN-Daily station list": GHCND_STATIONS_TXT,
            "Berkeley (CONUS daily)": BE_TMAX,
            "Berkeley (NH decades)": BE_PROC,
            "Berkeley RAW": BE_RAW,
            "ERA5": ERA5_DIR,
            "NOAAGlobalTemp": NOAA_FP,
            "CRUTEM5": CRUTEM_FP,
            "nCLIMDIV (TMAX)": nclimdiv_file("tmax"),
            "USHCN offsets": USHCN_OFFSETS_FP,
            "USHCN crosswalk": USHCN_CROSSWALK_FP}
    _built = {"GHCN-Daily band archive": GHCND_GLOBAL,
              "GHCN-Daily 2deg global grid": GHCND_GLOBAL_GRID,
              "USHCN daily pair": UH_DAILY_DIR,
              "USHCN paired months": UH_PAIRED}
    _opt = {"Berkeley TAVG": BE_TAVG, "Christy usreg mask": USREG_FP}

    print(f"{'dataset':<32}{'status':<10}path")
    for _label, _group in (("raw", _req), ("built", _built), ("optional", _opt)):
        for _name, _p in _group.items():
            _ok = "OK" if _p.exists() else ("MISSING" if _label != "optional"
                                            else "absent")
            print(f"  {_name:<30}{_ok:<10}{_p}")
    _missing = [n for n, p in _req.items() if not p.exists()]
    _unbuilt = [n for n, p in _built.items() if not p.exists()]
    if _missing:
        print(f"\nmissing from the archive: {', '.join(_missing)}")
    if _unbuilt:
        print(f"\nnot built yet: {', '.join(_unbuilt)}"
              f"  --  run  python figures/prepare_data.py")
    if not _missing and not _unbuilt:
        print("\nall present")


# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY -- one CONUS polygon for every figure.
#
# Natural Earth 50m United States, clipped to 24-50N / 125-66W, minus lakes.
# Two ways of using it are provided: a FRACTIONAL AREA WEIGHT per cell (what the
# anomaly figures average with) and a POINT-IN-POLYGON test (what the station
# and heat-wave figures mask with).
# ══════════════════════════════════════════════════════════════════════════════
def conus_polygon():
    import geopandas as gpd
    from cartopy.io import shapereader
    from shapely.geometry import box as sbox
    from shapely.ops import unary_union
    b = sbox(BOX["lon_min"], BOX["lat_min"], BOX["lon_max"], BOX["lat_max"])
    c = gpd.read_file(shapereader.natural_earth("50m","cultural","admin_0_countries"))
    col = "ADMIN" if "ADMIN" in c.columns else "NAME"
    usa = c[c[col] == "United States of America"]
    parts = [g for g in usa.geometry.explode(index_parts=False) if g.intersects(b)]
    poly = unary_union(parts).intersection(b)
    lakes = gpd.read_file(shapereader.natural_earth("50m","physical","lakes"))
    return poly.difference(unary_union([g for g in lakes.geometry if g.intersects(b)]))

@lru_cache(maxsize=1)
def land_polygon():
    """Natural Earth 50m land. Figure 5's global strip is masked with this, so
    that ocean persistence is not counted as heat waves."""
    import geopandas as gpd
    from cartopy.io import shapereader
    from shapely.ops import unary_union
    land = gpd.read_file(shapereader.natural_earth("50m", "physical", "land"))
    return unary_union(list(land.geometry))


def _edges(c):
    c = np.asarray(c, float); d = np.diff(c)
    return np.concatenate([[c[0]-d[0]/2], c[:-1]+d/2, [c[-1]+d[-1]/2]])

def cell_weights(lat_c, lon_c, key=None):
    if key is not None and (CACHE_ANOM / f"weights_{key}.nc").exists():
        return xr.load_dataarray(CACHE_ANOM / f"weights_{key}.nc")
    import shapely
    from shapely.geometry import box as sbox
    poly = conus_polygon()
    la, lo = _edges(lat_c), _edges(lon_c)
    frac = np.zeros((len(lat_c), len(lon_c)))
    cells, idx = [], []
    for i in range(len(lat_c)):
        for j in range(len(lon_c)):
            cells.append(sbox(lo[j], la[i], lo[j+1], la[i+1])); idx.append((i,j))
    tree = shapely.STRtree(np.array(cells, dtype=object))
    for h in np.atleast_1d(tree.query(poly, predicate="intersects")):
        i, j = idx[h]; c = cells[h]
        frac[i, j] = poly.intersection(c).area / c.area if c.area > 0 else 0.0
    band = np.abs(np.sin(np.radians(la[1:])) - np.sin(np.radians(la[:-1])))[:, None]
    w = xr.DataArray(frac*band, dims=("lat","lon"), coords={"lat":lat_c,"lon":lon_c})
    if key is not None:
        w.name = f"weights_{key}"; w.to_netcdf(CACHE_ANOM / f"weights_{key}.nc")
    return w

def g05_centers():
    return (0.5*(G05_LAT_EDGES[:-1]+G05_LAT_EDGES[1:]),
            0.5*(G05_LON_EDGES[:-1]+G05_LON_EDGES[1:]))

def points_inside(lat, lon):
    poly = conus_polygon()
    try:
        import shapely
        return np.asarray(shapely.contains_xy(poly, np.asarray(lon), np.asarray(lat)))
    except Exception:                                   # shapely 1.x
        from shapely.geometry import Point
        from shapely.prepared import prep
        pp = prep(poly)
        return np.array([pp.contains(Point(o, a)) for a, o in zip(lat, lon)])

def points_in_grid(geom, lat, lon):
    lon = np.where(np.asarray(lon) > 180.0, np.asarray(lon) - 360.0, np.asarray(lon))
    LON, LAT = np.meshgrid(lon, np.asarray(lat))
    try:
        import shapely
        m = np.asarray(shapely.contains_xy(geom, LON.ravel(), LAT.ravel()))
    except Exception:
        from shapely.geometry import Point
        from shapely.prepared import prep
        pp = prep(geom)
        m = np.array([pp.contains(Point(o, a))
                      for a, o in zip(LAT.ravel(), LON.ravel())])
    return m.reshape(LAT.shape)


def check_geometry():
    """CHECK -- the polygon and the weights, without touching the data drive."""
    _poly = conus_polygon()
    _inside  = [(-98.5, 39.0, "Kansas"), (-84.4, 33.7, "Georgia"), (-119.0, 36.5, "California")]
    _outside = [(-103.0, 25.0, "northern Mexico"), (-113.5, 51.0, "Alberta"),
                (-87.5, 44.0, "Lake Michigan"), (-70.0, 41.0, "Atlantic")]
    for _lon, _lat, _nm in _inside:
        assert points_inside([_lat], [_lon])[0], f"{_nm} should be inside CONUS"
    for _lon, _lat, _nm in _outside:
        assert not points_inside([_lat], [_lon])[0], f"{_nm} should be outside CONUS"
    print(f"point-in-polygon: {len(_inside)} inside / {len(_outside)} outside all correct")

    _lat05, _lon05 = g05_centers()
    _W05 = cell_weights(_lat05, _lon05, key="g05")
    assert float(_W05.min()) >= 0 and float(_W05.sum()) > 0
    print(f"0.5 deg grid: {_W05.shape[0]}x{_W05.shape[1]} cells, "
          f"{int((_W05 > 0).sum()):,} with weight, "
          f"weights sum to {float(_W05.sum()):.4f} of a hemisphere-band unit")
    print(f"widest-weighted cell: lat {float(_W05.idxmax(dim='lat').max()):.2f} "
          f"(band area falls off with latitude, as it should)")


# ══════════════════════════════════════════════════════════════════════════════
# THE ANOMALY-FIRST PIPELINE (Figure 1 and Table 2)
#
# Every station is converted to an anomaly against ITS OWN 1951-1980 monthly
# climatology BEFORE gridding, and every grid cell against its own before the
# cells are averaged. Averaging absolute temperatures first lets a change in
# which stations report move the series, because stations differ in elevation
# and exposure.
#
# The year axis is an argument (`years`), because the callers have run different
# spans: Figure 1 and Table 2 now both run 1900-2026, where the retired bar chart
# stopped at 2025.
# ══════════════════════════════════════════════════════════════════════════════

def _cache(key, build):
    fp = CACHE_ANOM / f"{key}.nc"
    if fp.exists():
        return xr.load_dataarray(fp)
    da = build(); da.name = key; da.to_netcdf(fp); return da

# ══ THE COMMON TAIL ═════════════════════════════════════════════════════════
def cell_anomalies(da, min_base=MIN_BASE_CELL):
    w = da.sel(time=slice(f"{BASELINE[0]}-01-01", f"{BASELINE[1]}-12-31"))
    clim = w.groupby("time.month").mean("time", skipna=True)
    clim = clim.where(w.groupby("time.month").count("time") >= min_base)
    return (da.groupby("time.month") - clim).drop_vars("month", errors="ignore")

def area_mean(da, weights):
    w = weights.where(da.notnull(), 0.0)
    den = w.sum(("lat","lon")); num = (da.fillna(0.0)*w).sum(("lat","lon"))
    return num/den.where(den > 0), den/float(weights.sum())

def _to_year_month(s, years):
    out = np.full((len(years), 12), np.nan)
    t = pd.DatetimeIndex(s["time"].values)
    yi = pd.Index(years).get_indexer(t.year); ok = yi >= 0
    out[yi[ok], t.month.to_numpy()[ok]-1] = np.asarray(s.values, float)[ok]
    return out

def _season_offsets(months):
    """Year offset of each month relative to the season's label year: a season
    is labelled by the year of its LAST month, so DJF is (-1,0,0)."""
    off = [0]; rev = list(months)[::-1]
    for p, c in zip(rev[:-1], rev[1:]):
        off.append(off[-1] - (1 if c > p else 0))
    return list(reversed(off))

def season_mean(monthly, months, years):
    """day-weighted seasonal mean; NaN unless every month of the season is there."""
    n = len(years)
    lengths = np.array([[calendar.monthrange(int(y), m)[1] for m in range(1,13)]
                        for y in years])
    tot = np.zeros(n); wt = np.zeros(n); complete = np.ones(n, bool)
    for m, off in zip(months, _season_offsets(months)):
        src = np.arange(n) + off; inside = (src >= 0) & (src < n)
        idx = np.clip(src, 0, n-1)
        vals = np.where(inside, monthly[idx, m-1], np.nan)
        w = np.where(inside, lengths[idx, m-1], 0)
        complete &= np.isfinite(vals)
        tot += np.where(np.isfinite(vals), vals*w, 0.0)
        wt  += np.where(np.isfinite(vals), w, 0)
    with np.errstate(invalid="ignore"):
        out = tot/np.where(wt > 0, wt, np.nan)
    return np.where(complete, out, np.nan)

# ══ STATIONS: anomaly-first, then IDW ═══════════════════════════════════════
def _xyz(lat, lon):
    la = np.radians(np.asarray(lat, float)); lo = np.radians(np.asarray(lon, float))
    return np.column_stack([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)])

class IDWGrid:
    def __init__(self, lat_c, lon_c, power=2.0, k=8, radius_km=150.0):
        self.lat = np.asarray(lat_c, float); self.lon = np.asarray(lon_c, float)
        self.power, self.k, self.radius = power, k, radius_km
        LON, LAT = np.meshgrid(self.lon, self.lat)
        self.shape = LAT.shape; self.cells = _xyz(LAT.ravel(), LON.ravel())
    def grid(self, values, slat, slon):
        v = np.asarray(values, float); ok = np.isfinite(v)
        if not ok.any(): return np.full(self.shape, np.nan)
        tree = cKDTree(_xyz(np.asarray(slat)[ok], np.asarray(slon)[ok]))
        k = min(self.k, int(ok.sum()))
        chord = 2.0*np.sin(0.5*self.radius/EARTH_R)
        dist, idx = tree.query(self.cells, k=k, distance_upper_bound=chord)
        if k == 1: dist = dist[:, None]; idx = idx[:, None]
        good = np.isfinite(dist); idxs = np.where(good, idx, 0)
        arc = 2.0*EARTH_R*np.arcsin(np.clip(dist/2.0, 0, 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(good, np.power(np.maximum(arc, 1e-6), -self.power), 0.0)
        exact = good & (arc < 1e-3); rows = exact.any(axis=1)
        w[rows] = np.where(exact[rows], 1.0, 0.0)
        ws = w.sum(1); vv = v[ok]
        out = np.where(ws > 0, (w*vv[idxs]).sum(1)/np.where(ws > 0, ws, 1.0), np.nan)
        return out.reshape(self.shape)

def station_anomalies(sm, min_base=MIN_BASE_STN):
    b = sm[sm["year"].between(*BASELINE)]
    g = b.groupby(["id","month"])["v"].agg(["mean","count"])
    clim = g[g["count"] >= min_base]["mean"].rename("clim")
    out = sm.join(clim, on=["id","month"]).dropna(subset=["clim"]).copy()
    out["v"] = out["v"] - out["clim"]
    return out.drop(columns="clim")

def grid_station_field(sm, gridder, years):
    piv = sm.pivot_table(index="id", columns=["year","month"], values="v")
    meta = sm.groupby("id")[["lat","lon"]].first().reindex(piv.index)
    slat, slon = meta["lat"].values, meta["lon"].values
    nt = len(years)*12
    field = np.full((nt, *gridder.shape), np.nan, np.float32)
    t = 0
    for y in years:
        for m in range(1, 13):
            if (y, m) in piv.columns:
                field[t] = gridder.grid(piv[(y, m)].values, slat, slon)
            t += 1
    time = pd.date_range(f"{years[0]}-01-01", periods=nt, freq="MS")
    return xr.DataArray(field, dims=("time","lat","lon"),
                        coords=dict(time=time, lat=gridder.lat, lon=gridder.lon))

def ghcnd_station_months(years):
    """One pass over the yearly GHCN-Daily parquets -> CONUS station-months.

    The span is in the filename: the table is built by looping over it, so a
    cache from a shorter record must not be handed back silently -- that is how
    an extended record quietly keeps ending at the old last year."""
    fp = CACHE_ANOM / f"ghcnd_conus_station_months_{years[0]}_{years[-1]}.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    parts = []
    for y in range(years[0], years[-1]+1):
        f = GHCND_GLOBAL / f"ghcn_global_{y}.parquet"
        if not f.exists(): continue
        d = pd.read_parquet(f, columns=["station_id","date","tmax_c","tmin_c","lat","lon"])
        d = d[(d.lat.between(BOX["lat_min"], BOX["lat_max"])) &
              (d.lon.between(BOX["lon_min"], BOX["lon_max"]))]
        if d.empty: continue
        d = d.assign(year=d.date.dt.year, month=d.date.dt.month)
        d["tavg_c"] = np.where(d.tmax_c.notna() & d.tmin_c.notna(),
                               (d.tmax_c+d.tmin_c)/2, np.nan)
        parts.append(d.groupby(["station_id","year","month"]).agg(
            tmax_s=("tmax_c","sum"), tmax_n=("tmax_c","count"),
            tmin_s=("tmin_c","sum"), tmin_n=("tmin_c","count"),
            tavg_s=("tavg_c","sum"), tavg_n=("tavg_c","count"),
            lat=("lat","first"), lon=("lon","first")).reset_index())
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(fp, index=False)
    return out

def _complete_months(d, cs, cn):
    d = d.rename(columns={cs:"s", cn:"n"})
    dim = np.array([calendar.monthrange(int(y), int(m))[1]
                    for y, m in zip(d["year"].values, d["month"].values)])
    d = d[(d["n"] >= MIN_DAYS_FRAC*dim) & (d["n"] > 0)].copy()
    d["v"] = d["s"]/d["n"]
    return d[np.isfinite(d["v"]) & (d["v"] > -60) & (d["v"] < 60)]

def station_months(dataset, elem, years):
    if dataset == "ghcnd":
        df = ghcnd_station_months(years).rename(columns={"station_id":"id"})
        return _complete_months(df[["id","year","month","lat","lon",
                                      f"{elem}_s", f"{elem}_n"]], f"{elem}_s", f"{elem}_n")
    which = "raw" if dataset == "ushcn" else "adj"
    df = pd.read_parquet(UH_PAIRED).rename(columns={"ushcn_id":"id"})
    return _complete_months(df[["id","year","month","lat","lon",
                                  f"{elem}_{which}_sum", f"{elem}_{which}_n"]],
                              f"{elem}_{which}_sum", f"{elem}_{which}_n")

# ══ GRIDDED PRODUCTS ════════════════════════════════════════════════════════
def _std(da):
    return da.rename({"latitude":"lat","longitude":"lon"}) if "latitude" in da.dims else da


def check_pipeline():
    """CHECK -- the pipeline on synthetic numbers, so a failure here is a code
    bug, not a data problem. Needs no files."""
    _yrs = np.arange(2000, 2011)

    # 1. a season is labelled by the year of its LAST month: DJF 2005 = Dec 2004 + Jan/Feb 2005
    assert _season_offsets((12, 1, 2)) == [-1, 0, 0], _season_offsets((12, 1, 2))
    assert _season_offsets((6, 7, 8)) == [0, 0, 0]

    # 2. day-weighted seasonal mean, and "NaN unless all three months are there"
    _m = np.full((len(_yrs), 12), np.nan)
    _m[:, 5], _m[:, 6], _m[:, 7] = 0.0, 0.0, 3.0            # Jun, Jul, Aug
    _js = season_mean(_m, (6, 7, 8), _yrs)
    _expect = (30*0 + 31*0 + 31*3) / 92
    assert np.allclose(_js, _expect), (_js[0], _expect)
    _m2 = _m.copy(); _m2[3, 6] = np.nan                     # drop one July
    assert np.isnan(season_mean(_m2, (6, 7, 8), _yrs)[3])
    print(f"season_mean: day-weighted JJA of (0, 0, 3) = {_js[0]:.4f} (hand value {_expect:.4f}), "
          f"incomplete season -> NaN")

    # 3. cell anomalies are taken per calendar month against 1951-1980
    _t = pd.date_range("1951-01-01", "1980-12-31", freq="MS")
    _seasonal = np.tile(np.arange(12, dtype=float), len(_t) // 12)
    _da = xr.DataArray(np.stack([_seasonal + 10, _seasonal - 5], axis=1)[:, :, None],
                       dims=("time", "lat", "lon"),
                       coords=dict(time=_t, lat=[35.0, 45.0], lon=[-100.0]))
    _an = cell_anomalies(_da)
    assert np.allclose(np.asarray(_an), 0.0), "a field equal to its own climatology must give 0"
    print("cell_anomalies: a field equal to its own climatology gives exactly 0")

    # 4. area_mean weights by the cells that reported, and reports the sampled fraction
    _w = xr.DataArray([[1.0, 3.0]], dims=("lat", "lon"),
                      coords=dict(lat=[35.0], lon=[-100.0, -90.0]))
    _f = xr.DataArray([[[2.0, 6.0]]], dims=("time", "lat", "lon"),
                      coords=dict(time=[pd.Timestamp("2000-01-01")], lat=[35.0], lon=[-100.0, -90.0]))
    # the notebook wrote float(_mean) here; area_mean returns a length-1 time
    # axis, and numpy 2 refuses float() on a size-1 1-D array, so take the one
    # time step explicitly. Same numbers, works on old and new numpy alike.
    _mean, _frac = (v.isel(time=0) for v in area_mean(_f, _w))
    assert np.isclose(float(_mean), (2*1 + 6*3) / 4) and np.isclose(float(_frac), 1.0)
    _f2 = _f.copy(); _f2[0, 0, 1] = np.nan
    _mean2, _frac2 = (v.isel(time=0) for v in area_mean(_f2, _w))
    assert np.isclose(float(_mean2), 2.0) and np.isclose(float(_frac2), 0.25)
    print(f"area_mean: full coverage {float(_mean):.2f} (area fraction {float(_frac):.2f}), "
          f"one cell missing {float(_mean2):.2f} (fraction {float(_frac2):.2f})")

    # 5. IDW: a cell with a station sitting on it takes that station's value exactly
    _g = IDWGrid(np.array([40.0, 41.0]), np.array([-100.0, -99.0]), **dict(power=2.0, k=8, radius_km=150.0))
    _out = _g.grid([5.0, -3.0], [40.0, 41.0], [-100.0, -99.0])
    assert np.isclose(_out[0, 0], 5.0) and np.isclose(_out[1, 1], -3.0), _out
    assert _out.shape == (2, 2)
    print(f"IDWGrid: colocated stations reproduced exactly ({_out[0, 0]:.1f}, {_out[1, 1]:.1f}); "
          f"interpolated corner {_out[0, 1]:.3f}")

    # 6. station anomalies drop stations without enough baseline years
    _rows = [dict(id="A", year=y, month=7, lat=40.0, lon=-100.0, v=20.0 + (y - 1965))
             for y in range(1951, 1981)]
    _rows += [dict(id="B", year=y, month=7, lat=41.0, lon=-99.0, v=15.0) for y in range(1951, 1961)]
    _sm = pd.DataFrame(_rows)
    _sa = station_anomalies(_sm)
    assert set(_sa["id"]) == {"A"}, "station B has only 10 of 30 baseline years and must be dropped"
    assert np.isclose(_sa["v"].mean(), 0.0), "over its own baseline a station must average to zero"
    assert np.isclose(_sa.loc[_sa["year"] == 1965, "v"].iloc[0], -0.5)   # baseline centres on 1965.5
    print(f"station_anomalies: kept {_sa['id'].nunique()} of 2 stations "
          f"(>= {MIN_BASE_STN} of 30 baseline years required); "
          f"station A averages {_sa['v'].mean():+.3f} over its own baseline")
    print("\nall pipeline checks passed")


# ══════════════════════════════════════════════════════════════════════════════
# GRIDDED PRODUCTS THAT FIGURES 1 AND 2 SHARE
# Each loader reads the archive once and caches the field it returns.
# ══════════════════════════════════════════════════════════════════════════════

def noaa_field():
    def build():
        ds = xr.open_dataset(NOAA_FP)
        da = ds["anom"].sel(z=0, lat=slice(24,51), lon=slice(235,295)).load()
        da = da.assign_coords(lon=((da["lon"]+180)%360)-180).sortby("lon")
        ds.close(); return da.drop_vars("z", errors="ignore")
    return _cache("field_noaa", build)

def crutem_field():
    def build():
        ds = xr.open_dataset(CRUTEM_FP)
        da = _std(ds["tas"].sel(latitude=slice(24,51), longitude=slice(-126,-65))).load()
        ds.close(); return da
    return _cache("field_crutem5", build)


def check_gridded():
    """CHECK -- read the two gridded products and confirm they cover CONUS and
    the right years. First run reads the archive (slow); later runs read the cache."""
    for _nm, _da in (("NOAAGlobalTemp", noaa_field()), ("CRUTEM5", crutem_field())):
        _t = pd.DatetimeIndex(_da["time"].values)
        print(f"{_nm:<16}{_da.shape}  lat {float(_da.lat.min()):.1f}..{float(_da.lat.max()):.1f}  "
              f"lon {float(_da.lon.min()):.1f}..{float(_da.lon.max()):.1f}  "
              f"{_t.year.min()}-{_t.year.max()}  "
              f"{100 * float(np.isfinite(_da).mean()):.0f}% of cells report")


# ══════════════════════════════════════════════════════════════════════════════
# THE MAY-SEPTEMBER GHCN-DAILY CONUS CUBE -- shared by Figures 2 and 5.
#
# In the notebook this lived in section 7 (Figure 2) and section 10 (Figure 5)
# read the .npz it left behind, which is why the notebook had to be run in
# order. Here it is one function, so figure2.py and figure5.py can each be run
# on its own; the cache paths are unchanged, so an existing cache is reused.
# ══════════════════════════════════════════════════════════════════════════════
Y0_F2, Y1_F2 = 1900, 2025
MONTHS   = [5, 6, 7, 8, 9]
NDAYS    = 153
YEARS_F2 = np.arange(Y0_F2, Y1_F2 + 1)
MIN_YEARS, MIN_FRAC = 100, 0.80
TMAX_BOUNDS_F2 = (-40.0, 57.0)      # degC; 56.7 is the highest reliable US reading
FORCE_F2 = False
# the year span belongs in the tag: the cube, the coverage table and the record
# counts are all built over it, so extending the record has to miss the cache.
TAG_F2   = (f"y{MIN_YEARS}_f{MIN_FRAC:g}_b{TMAX_BOUNDS_F2[0]:g}-"
            f"{TMAX_BOUNDS_F2[1]:g}_{Y0_F2}-{Y1_F2}")
COV_FP   = CACHE_REC / f"ghcnd_coverage_mjjas_conus_{TAG_F2}.parquet"
DAILY_FP = CACHE_REC / f"ghcnd_daily_mjjas_conus_{TAG_F2}.npz"

# the canonical May-September calendar: len(YEARS_F2) years x 153 days
_t = pd.date_range(f"{Y0_F2}-01-01", f"{Y1_F2}-12-31", freq="D")
_t = _t[np.isin(_t.month, MONTHS)]
KEYS = (_t.year * 10000 + _t.month * 100 + _t.day).to_numpy(np.int64)
KPOS = pd.Index(KEYS)
YR, MO, DY = _t.year.values, _t.month.values, _t.day.values
assert len(KEYS) == len(YEARS_F2) * NDAYS, "calendar is the wrong length"


def coverage(vals, yr):
    """(n_years_with_data, frac_of_possible_days) per unit."""
    fin = np.isfinite(vals)
    has = np.zeros((len(YEARS_F2), vals.shape[1]), bool)
    for i, y in enumerate(YEARS_F2):
        s = yr == y
        if s.any():
            has[i] = fin[s].any(axis=0)
    return has.sum(axis=0), fin.sum(axis=0) / float(len(YEARS_F2) * NDAYS)

def complete(ny, fr):
    return (ny >= MIN_YEARS) & (fr >= MIN_FRAC)


def ghcnd_conus_positions():
    """CONUS TMAX stations from the GHCN-Daily inventory, point-in-polygon tested."""
    inv = pd.read_fwf(str(GHCND_INV),
                      colspecs=[(0, 11), (12, 20), (21, 30), (31, 35), (36, 40), (41, 45)],
                      names=["id", "lat", "lon", "elem", "y0", "y1"])
    inv = inv[inv["elem"] == "TMAX"]
    inv = inv[inv["lat"].between(BOX["lat_min"], BOX["lat_max"])
              & inv["lon"].between(BOX["lon_min"], BOX["lon_max"])]
    pos = inv.groupby("id")[["lat", "lon"]].first()
    return pos[points_inside(pos["lat"].to_numpy(), pos["lon"].to_numpy())]


def ghcnd_daily_cube():
    """(IDS, V, POS): the MJJAS TMAX cube for the CONUS stations that pass the
    completeness rule, on the canonical calendar. Built once, then cached."""
    POS = ghcnd_conus_positions()
    CANDIDATES = set(POS.index)

    yrs_present = [y for y in range(Y0_F2, Y1_F2 + 1)
                   if (GHCND_GLOBAL / f"ghcn_global_{y}.parquet").exists()]

    qcols = []
    first = True
    def read_year(y, keep_ids=None):
        """MJJAS TMAX for one year: QC'd, deduplicated, CONUS stations only."""
        nonlocal qcols, first
        fp = GHCND_GLOBAL / f"ghcn_global_{y}.parquet"
        if first:
            avail = set(pd.read_parquet(fp).head(0).columns)
            qcols = [c for c in ("qflag", "q_flag", "tmax_qflag", "quality_flag")
                     if c in avail]
            first = False
        d = pd.read_parquet(fp, columns=["station_id", "date", "tmax_c"] + qcols)
        d = d[d["tmax_c"].notna()]
        d = d[d["station_id"].isin(keep_ids if keep_ids is not None else CANDIDATES)]
        if d.empty:
            return d
        for c in qcols:                    # a non-empty flag means the value failed
            d = d[d[c].fillna("").astype(str).str.strip() == ""]
        d = d[d["tmax_c"].between(*TMAX_BOUNDS_F2)]
        dt = pd.DatetimeIndex(d["date"])
        d = d[np.isin(dt.month, MONTHS)]
        if d.empty:
            return d
        return d.drop_duplicates(subset=["station_id", "date"], keep="first")

    if COV_FP.exists() and not FORCE_F2:
        cov = pd.read_parquet(COV_FP)
    else:
        per_year = {}
        for y in yrs_present:
            d = read_year(y)
            per_year[y] = (d.groupby("station_id").size().astype("int32")
                           if not d.empty else pd.Series(dtype="int32"))
        cov = pd.DataFrame(per_year).fillna(0).astype("int32")
        cov.to_parquet(COV_FP)

    _ny = (cov > 0).sum(axis=1)
    _fr = cov.sum(axis=1) / float(len(YEARS_F2) * NDAYS)
    keep_ids = cov.index[complete(_ny, _fr)]

    if DAILY_FP.exists() and not FORCE_F2:
        z = np.load(DAILY_FP, allow_pickle=False)
        IDS, V = z["ids"], z["values"]
    else:
        IDS = np.asarray(sorted(keep_ids), dtype="<U11")
        IPOS = pd.Index(IDS); want = set(IDS.tolist())
        V = np.full((len(KEYS), len(IDS)), np.nan, np.float32)
        for y in yrs_present:
            d = read_year(y, keep_ids=want)
            if d.empty:
                continue
            dt = pd.DatetimeIndex(d["date"])
            key = (dt.year * 10000 + dt.month * 100 + dt.day).to_numpy(np.int64)
            r = KPOS.get_indexer(key); c = IPOS.get_indexer(d["station_id"].to_numpy())
            g = (r >= 0) & (c >= 0)
            V[r[g], c[g]] = d["tmax_c"].to_numpy(np.float32)[g]
        np.savez_compressed(DAILY_FP, ids=IDS, values=V)

    return IDS, V, POS


# ── the same cube, for the 24-50N band at every longitude ────────────────────
# Figure 5's bottom row counts heat waves AT STATIONS and grids the counts, the
# way Christy (2026) does, so it needs the band as stations rather than as the
# 2 deg field of daily temperatures prepare_data.py builds. Same archive, same
# QC, same calendar as ghcnd_daily_cube -- only the domain and the screen differ.
BAND_COV_FP = CACHE_REC / "global" / f"ghcnd_band_coverage_mjjas_{TAG_F2}.parquet"
_BAND_CUBE  = CACHE_REC / "global" / "ghcnd_band_mjjas"     # + _{screen}.npy

def ghcnd_band_cube(min_total_days=0, min_good_years=0, min_days_in_year=0):
    """(IDS, V, POS) for the 24-50N band: V is (len(KEYS), n_stn) float32 of MJJAS
    TMAX in degC, POS a lat/lon frame indexed by station id.

    The screen is the caller's, because what may be discarded depends on the
    caller's threshold rule, not on anything intrinsic to the data:
      min_total_days   finite MJJAS days a station needs over the whole record
      min_days_in_year  days a year needs to count as "good"
      min_good_years   how many such years a station needs

    Positions come from the parquets rather than from ghcnd-inventory.txt, which
    the archive only carries for the CONUS.

    The cube is written uncompressed so np.load(mmap_mode="r") stays available:
    the band is an order of magnitude more stations than the CONUS, and callers
    that only ever touch a station block at a time should not have to hold it
    all. Budget about 1.5 GB."""
    (CACHE_REC / "global").mkdir(parents=True, exist_ok=True)
    yrs = [y for y in range(Y0_F2, Y1_F2 + 1)
           if (GHCND_GLOBAL / f"ghcn_global_{y}.parquet").exists()]
    assert yrs, f"no band parquets under {GHCND_GLOBAL} -- run prepare_data.py"

    qcols, first = [], True
    def read_year(y, keep=None):
        nonlocal qcols, first
        fp = GHCND_GLOBAL / f"ghcn_global_{y}.parquet"
        if first:
            avail = set(pd.read_parquet(fp).head(0).columns)
            qcols = [c for c in ("qflag", "q_flag", "tmax_qflag", "quality_flag")
                     if c in avail]
            first = False
        d = pd.read_parquet(fp, columns=["station_id", "date", "tmax_c",
                                         "lat", "lon"] + qcols)
        d = d[d["tmax_c"].notna()]
        if keep is not None:
            d = d[d["station_id"].isin(keep)]
        if d.empty:
            return d
        for c in qcols:
            d = d[d[c].fillna("").astype(str).str.strip() == ""]
        d = d[d["tmax_c"].between(*TMAX_BOUNDS_F2)]
        dt = pd.DatetimeIndex(d["date"])
        d = d[np.isin(dt.month, MONTHS)]
        if d.empty:
            return d
        return d.drop_duplicates(subset=["station_id", "date"], keep="first")

    pos_fp = CACHE_REC / "global" / "ghcnd_band_positions.parquet"
    if BAND_COV_FP.exists() and pos_fp.exists() and not FORCE_F2:
        cov = pd.read_parquet(BAND_COV_FP)
        POS = pd.read_parquet(pos_fp)
    else:
        per_year, pos = {}, []
        for y in yrs:
            d = read_year(y)
            per_year[y] = (d.groupby("station_id").size().astype("int32")
                           if not d.empty else pd.Series(dtype="int32"))
            if not d.empty:
                pos.append(d.groupby("station_id")[["lat", "lon"]].first())
        cov = pd.DataFrame(per_year).fillna(0).astype("int32")
        POS = pd.concat(pos).groupby(level=0).first()
        cov.to_parquet(BAND_COV_FP); POS.to_parquet(pos_fp)

    tot = cov.sum(axis=1)
    good = (cov >= int(min_days_in_year)).sum(axis=1)
    keep_ids = cov.index[(tot >= min_total_days) & (good >= max(min_good_years, 1))]
    keep_ids = pd.Index(sorted(set(keep_ids) & set(POS.index)))

    tag = f"t{int(min_total_days)}_g{int(min_good_years)}x{int(min_days_in_year)}"
    cube_fp = Path(f"{_BAND_CUBE}_{TAG_F2}_{tag}.npy")
    if cube_fp.exists() and not FORCE_F2:
        V = np.load(cube_fp, allow_pickle=False)
        IDS = np.asarray(keep_ids, dtype="<U11")
        assert V.shape == (len(KEYS), len(IDS)), \
            f"{cube_fp.name} is {V.shape}, the screen now wants " \
            f"{(len(KEYS), len(IDS))} -- delete it"
    else:
        IDS = np.asarray(keep_ids, dtype="<U11")
        IPOS = pd.Index(IDS); want = set(IDS.tolist())
        V = np.full((len(KEYS), len(IDS)), np.nan, np.float32)
        for y in yrs:                      # year-major: the parquets are too
            d = read_year(y, keep=want)
            if d.empty:
                continue
            dt = pd.DatetimeIndex(d["date"])
            key = (dt.year * 10000 + dt.month * 100 + dt.day).to_numpy(np.int64)
            r = KPOS.get_indexer(key); c = IPOS.get_indexer(d["station_id"].to_numpy())
            g = (r >= 0) & (c >= 0)
            V[r[g], c[g]] = d["tmax_c"].to_numpy(np.float32)[g]
        np.save(cube_fp, V)
    return IDS, V, POS.reindex(pd.Index(IDS))


# ══════════════════════════════════════════════════════════════════════════════
# THE SMOOTHER -- LOWESS, shared by Figures 1, 2, 4 and 5.
#
# LOWESS (Cleveland 1979) in its plainest form. The smooth at a year is the
# intercept of a straight line, fit by weighted least squares to the
# LOWESS_YEARS nearest years and read at that year. The weights are the tricube
# (1 - (d/h)^3)^3, with d the distance in years and h the distance to the
# farthest year in the window, so they fall smoothly to zero at its edge.
#
# Four choices decide what the line does:
#
#   THE WHOLE LENGTH. Near an end the nearest years all lie on one side, so the
#   window SLIDES against the end of the record instead of shrinking: every fit,
#   first year to last, rests on the same number of years and there is no
#   separate end rule to choose. It is a local LINE and not a local mean, so a
#   trend is carried to the endpoint rather than flattened or lagged there.
#   Nothing is invented past the data: a year the series lacks stays blank, so a
#   record that starts in 1940 or stops in 2024 is smoothed over exactly those
#   years and needs no trimming first.
#
#   LOWESS_YEARS = 17 gives the smoothness of the 11-year centered mean that the
#   figures used before. A 17-year tricube window has an interior kernel of
#   standard deviation 3.04 yr (3.16 for an 11-year mean) and averages 11.3
#   effective observations (11), so inside the record the two curves are equally
#   smooth and what differs is the ends. Odd, so that every window has a centre
#   year. check_smoother() prints both numbers.
#
#   LOWESS_IT = 0, NO ROBUSTNESS ITERATIONS. statsmodels defaults to 3, which
#   down-weights years whose residual is large. Here the large residuals are the
#   result -- the record-setting 1930s -- and a record count is skewed right, so
#   a robust fit scores real heat as outliers: three iterations lowered the Dust
#   Bowl peak of Figure 2's smooths by 12 to 51%, and Berkeley Earth's by half.
#   Figure 2 prints that comparison on every run.
#
#   HOLES ARE KEPT. A year the series lacks is never bridged. Each unbroken run
#   of consecutive years is smoothed as a record of its own, so nothing is drawn
#   across a blank year, and a run shorter than the window is left blank. The
#   hole is the blank year itself, narrower than the 11-year hole the centered
#   mean left around it, because LOWESS does not need a full window.
#
# This replaced the 11-year centered mean carried to the ends of each record by
# a shrinking local linear fit (`roll`, with `end_uncertainty`, `end_band` and
# the END_METHOD options), removed 2026-09-30. The notebook still carries it.
# ══════════════════════════════════════════════════════════════════════════════
LOWESS_YEARS = 17        # years in each local fit
LOWESS_IT    = 0         # robustness iterations; 0 = none
assert LOWESS_YEARS % 2 == 1, "LOWESS_YEARS must be odd"


def lowess_smooth(s, years=None, it=None):
    """LOWESS of an annual series, returned on the series' own index.

    A year the series lacks stays NaN, so the line ends where the data end and
    never crosses a blank year inside the record: each unbroken run of
    consecutive years is smoothed on its own, and a run shorter than the window
    is left blank. The window is `years` years, i.e. frac = years / (years in the
    run) in statsmodels' terms.

    `years` and `it` override LOWESS_YEARS and LOWESS_IT for one call; Figure 2
    passes `it` to show what statsmodels' default robustness would do."""
    years = LOWESS_YEARS if years is None else years
    it = LOWESS_IT if it is None else it
    if not (s.index.is_unique and s.index.is_monotonic_increasing):
        raise ValueError("lowess_smooth needs a sorted, unique year index")
    out = pd.Series(np.nan, index=s.index)
    d = s.dropna()
    if d.empty:
        return out
    x, y = d.index.values.astype(float), d.values.astype(float)
    for run in np.split(np.arange(d.size), np.nonzero(np.diff(x) != 1)[0] + 1):
        if run.size < years:
            continue                                # too short for one window
        fit = lowess(y[run], x[run], frac=years / run.size, it=it, delta=0.0,
                     is_sorted=True, return_sorted=True)
        assert np.array_equal(fit[:, 0], x[run])
        out.loc[d.index[run]] = fit[:, 1]
    return out


def end_revision(s, years=None, it=None, n_min=33, lags=8):
    """How much the smooth's last few values move as later years arrive, measured
    on this series and not asserted. Cut the record at each year T, smooth the
    cut record, and compare its value at T, T-1, ... with the value the full
    record gives that year. Returns the RMSE by distance from the cut. T stops
    far enough from the real end that the full-record value is itself a centred
    fit -- otherwise both sides would be end fits and agree too easily -- so the
    revision is exactly zero from lag `half` on and the table shows it decaying
    there. The record must be unbroken."""
    years = LOWESS_YEARS if years is None else years
    y, half = s.dropna(), (years - 1) // 2
    if not (np.diff(y.index.values) == 1).all():
        raise ValueError("end_revision needs an unbroken record")
    full = lowess_smooth(y, years, it)
    err = {l: [] for l in range(lags)}
    for T in range(n_min, y.size - half):
        est = lowess_smooth(y.iloc[:T + 1], years, it)
        for l in range(lags):
            err[l].append(est.iloc[T - l] - full.iloc[T - l])
    return np.array([np.sqrt(np.mean(np.square(err[l]))) for l in range(lags)])


def check_smoother():
    """CHECK -- the smoother, on synthetic numbers. Needs no data files."""
    yrs = np.arange(1900, 2026)
    g = pd.Series(np.random.default_rng(1).gamma(2.0, 1.0, size=yrs.size), index=yrs)

    # 1. It is the textbook definition: tricube-weighted local linear regression
    #    on the LOWESS_YEARS nearest years, written out longhand, agrees with the
    #    library to rounding at every year -- inside the record and at both ends.
    def longhand(y, k):
        x, out = np.arange(len(y), dtype=float), np.empty(len(y))
        for i in range(len(y)):
            lo = min(max(i - (k - 1) // 2, 0), len(y) - k)   # centred inside, slid at an end
            xs, ys = x[lo:lo + k] - x[i], y[lo:lo + k]
            w = (1 - (np.abs(xs) / np.abs(xs).max()) ** 3) ** 3
            out[i] = np.polyfit(xs, ys, 1, w=np.sqrt(w))[1]  # polyfit squares its weights
        return out
    dev = np.abs(lowess_smooth(g, it=0).values - longhand(g.values, LOWESS_YEARS)).max()
    assert dev < 1e-8, f"lowess_smooth() is not the longhand tricube local line: {dev:.2e}"

    # 2. A straight line comes back exactly, at both ends as well as inside: the
    #    fit is a local LINE, so a trend reaches the endpoint instead of being
    #    flattened there.
    ln = pd.Series(0.04 * (yrs - yrs[0]) + 0.5, index=yrs)
    assert np.abs(lowess_smooth(ln) - ln).max() < 1e-8, "a straight line was not reproduced"

    # 3. The line runs the whole length of the data and no further. A record that
    #    starts late or stops early is smoothed over its own years and nothing
    #    appears in the years it lacks.
    cut = g.copy(); cut.iloc[:3] = np.nan; cut.iloc[-2:] = np.nan
    sc = lowess_smooth(cut)
    assert (sc.notna() == cut.notna()).all(), "the line does not span exactly the data"
    assert sc.first_valid_index() == yrs[3] and sc.last_valid_index() == yrs[-3]

    # 4. Holes are kept. A blank year inside a record is never bridged: each side
    #    is smoothed as a record of its own, untouched by the other, and a run too
    #    short for one window stays blank.
    gap = g.copy(); gap.loc[yrs[60]] = np.nan
    sg = lowess_smooth(gap)
    assert (sg.notna() == gap.notna()).all(), "a blank year was bridged"
    assert np.allclose(sg.loc[:yrs[59]], lowess_smooth(g.loc[:yrs[59]]))
    assert np.allclose(sg.loc[yrs[61]:], lowess_smooth(g.loc[yrs[61]:]))
    sh = g.copy(); sh.loc[yrs[10]] = np.nan                   # leaves a 10-year run
    ss = lowess_smooth(sh)
    assert ss.loc[:yrs[9]].isna().all() and ss.loc[yrs[11]:].notna().all()

    k = LOWESS_YEARS
    dk = np.arange(k) - (k - 1) / 2
    wk = (1 - (np.abs(dk) / np.abs(dk).max()) ** 3) ** 3; wk /= wk.sum()
    print(f"lowess_smooth(): {k}-year tricube window agrees with the longhand fit to "
          f"{dev:.0e}; straight-line, whole-length and hole checks passed\n"
          f"  interior kernel: sd {np.sqrt((wk * dk ** 2).sum()):.2f} yr, "
          f"{1 / (wk ** 2).sum():.1f} effective observations")


__all__ = [
    # modules and objects the figure scripts use directly
    "os", "re", "calendar", "warnings", "pickle", "Path", "lru_cache",
    "np", "pd", "xr", "mpl", "plt", "mticker", "mpatches", "Line2D", "GridSpec",
    "blended_transform_factory", "TwoSlopeNorm", "cKDTree", "ccrs", "cfeature",
    # paths
    "ROOT", "WORK", "FIGD", "DATA",
    "GHCND_DIR", "GHCND_BY_YEAR", "GHCND_STATION_DAILY", "GHCND_INV",
    "GHCND_STATIONS_TXT", "GHCND_GLOBAL", "GHCND_GLOBAL_GRID",
    "BE_DIR", "BE_PROC", "BE_RAW", "BE_TMAX", "BE_TMIN", "BE_TAVG",
    "ERA5_DIR", "NOAA_DIR", "NOAA_FP", "CRUTEM_DIR", "CRUTEM_FP",
    "NCLIMDIV_DIR", "nclimdiv_file",
    "USHCN_DIR", "USHCN_OFFSETS_FP", "USHCN_CROSSWALK_FP", "USHCN_STATIONS_TXT",
    "USREG_FP", "UH_PAIRED", "UH_PIV_DIR", "UH_WORKDIR", "UH_STN_FP",
    "UH_DAILY_DIR", "BAND_LAT_PREP", "Y0_PREP", "Y1_PREP",
    "CACHE_ANOM", "CACHE_REC", "CACHE_MAPS", "CACHE_BAND", "CACHE_HW",
    "STAGE", "STAGE_ENABLED", "stage",
    # shared settings
    "BASELINE", "BOX", "MIN_BASE_STN", "MIN_BASE_CELL", "MIN_DAYS_FRAC",
    "MIN_DAYS_MON", "EARTH_R", "G05_LAT_EDGES", "G05_LON_EDGES",
    # geometry
    "conus_polygon", "land_polygon", "_edges", "cell_weights", "g05_centers",
    "points_inside", "points_in_grid",
    # pipeline
    "_cache", "cell_anomalies", "area_mean", "_to_year_month", "_season_offsets",
    "season_mean", "_xyz", "IDWGrid", "station_anomalies", "grid_station_field",
    "ghcnd_station_months", "_complete_months", "station_months", "_std",
    # gridded products
    "noaa_field", "crutem_field",
    # the MJJAS cube shared by Figures 2 and 5
    "Y0_F2", "Y1_F2", "MONTHS", "NDAYS", "YEARS_F2", "MIN_YEARS", "MIN_FRAC",
    "TMAX_BOUNDS_F2", "FORCE_F2", "TAG_F2", "COV_FP", "DAILY_FP",
    "KEYS", "KPOS", "YR", "MO", "DY",
    "coverage", "complete", "ghcnd_conus_positions", "ghcnd_daily_cube",
    "ghcnd_band_cube", "BAND_COV_FP",
    # the LOWESS smoother shared by Figures 1, 2, 4 and 5
    "LOWESS_YEARS", "LOWESS_IT", "lowess_smooth", "end_revision",
    # checks
    "check_paths", "check_geometry", "check_pipeline", "check_gridded",
    "check_smoother",
]

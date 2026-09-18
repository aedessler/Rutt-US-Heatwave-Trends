#!/usr/bin/env python3
"""Figure 4 -- Berkeley Earth JJA exceedance frequency maps.

Share of JJA days above each cell's own day-of-year 95th percentile, with the
threshold taken from the whole record rather than 1951-1980, across the
24-50N band. Notebook section 8.

    python figures/figure4.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *          # noqa: F401,F403

check_paths()
print()

# ══ SETTINGS AND HELPERS ═════════════════════════════════════════════════════
warnings.filterwarnings('ignore')
FIGD.mkdir(exist_ok=True)
CACHE_MAPS.mkdir(parents=True, exist_ok=True)

# ─── constants ────────────────────────────────────────────────────────────────
_WARM   = [6, 7, 8]           # JJA only
_WIN    = 31
_DUST_YRS = list(range(1930, 1940))
_CLIM_THRESH = 5.0   # expected exceedance rate for p95 threshold; shown in grey
_DIFF_VMIN   = -20.0
_DIFF_VMAX   =  35.0

# ─── auto-detect available record span ────────────────────────────────────────
_BE_FILES = sorted(BE_RAW.glob('Complete_TMAX_Daily_LatLong1_*.nc'))
if _BE_FILES:
    with xr.open_dataset(_BE_FILES[-1]) as _ds_last:
        _max_year = int(_ds_last['year'].values.max())
    with xr.open_dataset(_BE_FILES[0]) as _ds_first:
        _min_year = int(_ds_first['year'].values.min())
else:
    _min_year, _max_year = 1900, 2024
_MOD_YRS = list(range(_max_year - 9, _max_year + 1))

# ── the change: baseline is the ENTIRE record, not 1951-1980 ──────────────────
_BASE_S, _BASE_E = _min_year, _max_year

# ─── coordinate helpers ───────────────────────────────────────────────────────
def _std_maps(obj):
    rename = {old: new for old, new in [('latitude', 'lat'), ('longitude', 'lon')]
              if old in obj.dims or old in obj.coords}
    return obj.rename(rename) if rename else obj

def _negpos(obj):
    if 'lon' in obj.coords and float(obj['lon'].max()) > 180:
        obj = obj.assign_coords(lon=((obj['lon'] + 180) % 360) - 180).sortby('lon')
    return obj

# ─── target grid ──────────────────────────────────────────────────────────────
_LATS = np.arange(24.5, 50.0, 1.0)
_LONS = np.arange(-179.5, 180.0, 1.0)

# ─── land mask ────────────────────────────────────────────────────────────────
with xr.open_dataset(BE_RAW / 'Complete_TMAX_Daily_LatLong1_1900.nc') as _ds0:
    _lm = _std_maps(_negpos(_ds0['land_mask']))
    _LMASK = (_lm.interp(lat=_LATS, lon=_LONS, method='nearest')
               .fillna(0).values >= 0.5)

# ─── DOY-p95 threshold builder ───────────────────────────────────────────────
# (identical to the 1951-80 version, including the +1 DOY window convention, so
#  the only difference between the two figures is the years that go in)
def _build_thr(data_b, doy_b, cp, doy_cp):
    half = _WIN // 2
    thr_arr, doy_list = [], []
    for d in np.unique(doy_b):
        window = [int((d - half + k) % 366 + 1) for k in range(_WIN)
                  if (d - half + k) % 366 + 1 != 0]
        mw = np.isin(doy_b, window)
        t = (np.nanpercentile(data_b[mw], 95, axis=0).astype(np.float32)
             if mw.sum() > 0 else np.full(data_b.shape[1:], np.nan, np.float32))
        thr_arr.append(t)
        doy_list.append(int(d))
    np.save(cp, np.stack(thr_arr))
    np.save(doy_cp, np.array(doy_list))
    return dict(zip(doy_list, thr_arr))

# ─── Berkeley Earth thresholds (cached, JJA-specific, FULL RECORD) ───────────
_cp_be     = CACHE_MAPS / f'thr_be_jja_fullrecord_{_BASE_S}_{_BASE_E}.npy'
_doy_cp_be = CACHE_MAPS / f'thr_be_jja_fullrecord_{_BASE_S}_{_BASE_E}_doys.npy'

if _cp_be.exists() and _doy_cp_be.exists():
    _THR = dict(zip(np.load(_doy_cp_be).astype(int),
                    np.load(_cp_be, allow_pickle=False)))
else:
    _all_data, _all_doys = [], []
    for fp in _BE_FILES:
        with xr.open_dataset(fp) as ds:
            mon_arr = ds['month'].values.astype(int)
            doy_arr = ds['day_of_year'].values.astype(int)
            be_lats = ds['latitude'].values
            sel = np.isin(mon_arr, _WARM)          # every year in the file
            if not sel.any():
                continue
            t_idx   = np.where(sel)[0]
            lat_idx = np.clip(np.searchsorted(be_lats, _LATS), 0, len(be_lats) - 1)
            # read only the JJA days and only the target latitude band, so the
            # whole-record stack stays small enough to hold in memory
            temp = ds['temperature'].isel(time=t_idx, latitude=lat_idx).values
            clim = ds['climatology'].isel(latitude=lat_idx).values
        cidx  = np.clip(doy_arr[sel] - 1, 0, 364)
        _all_data.append((temp + clim[cidx]).astype(np.float32))
        _all_doys.append(doy_arr[sel])
        del temp, clim
    _THR = _build_thr(np.concatenate(_all_data), np.concatenate(_all_doys),
                      _cp_be, _doy_cp_be)
    del _all_data, _all_doys

# ─── Berkeley Earth year loader ───────────────────────────────────────────────
def _load_be(yr):
    decade = (yr // 10) * 10
    fp = BE_RAW / f'Complete_TMAX_Daily_LatLong1_{decade}.nc'
    if not fp.exists():
        return None
    with xr.open_dataset(fp) as ds:
        yr_arr  = ds['year'].values.astype(int)
        mon_arr = ds['month'].values.astype(int)
        doy_arr = ds['day_of_year'].values.astype(int)
        temp    = ds['temperature'].values
        clim    = ds['climatology'].values
        be_lats = ds['latitude'].values
    sel = (yr_arr == yr) & np.isin(mon_arr, _WARM)
    if not sel.any():
        return None
    cidx    = np.clip(doy_arr[sel] - 1, 0, 364)
    abs_t   = temp[sel] + clim[cidx]
    lat_idx = np.clip(np.searchsorted(be_lats, _LATS), 0, len(be_lats) - 1)
    return abs_t[:, lat_idx, :], doy_arr[sel]

# ─── spatial exceedance computation ──────────────────────────────────────────
def _spatial(years, cache_path, force=False):
    if cache_path.exists() and not force:
        return np.load(cache_path)
    yearly = []
    for yr in years:
        res = _load_be(yr)
        if res is None:
            continue
        data_yr, doys = res
        exc_sum = np.zeros((len(_LATS), len(_LONS)), dtype=float)
        n = 0
        for t_i, d in enumerate(doys):
            dk = int(d)
            if dk not in _THR:
                continue
            t   = _THR[dk]
            exc = (data_yr[t_i] > t).astype(float)
            exc[np.isnan(t)] = np.nan
            exc_sum += np.where(np.isnan(exc), 0, exc)
            n += 1
        if n > 0:
            yearly.append(exc_sum / n * 100)
    result = np.nanmean(yearly, axis=0) if yearly else np.full(
        (len(_LATS), len(_LONS)), np.nan)
    result[~_LMASK] = np.nan
    np.save(cache_path, result)
    return result

_tag = f'fullrec{_BASE_S}_{_BASE_E}'
_dust_cache = CACHE_MAPS / f'spatial_be_jja_dustbowl_{_DUST_YRS[0]}_{_DUST_YRS[-1]}_{_tag}.npy'
_mod_cache  = CACHE_MAPS / f'spatial_be_jja_modern_{_MOD_YRS[0]}_{_MOD_YRS[-1]}_{_tag}.npy'

_map_dust = _spatial(_DUST_YRS, _dust_cache)
_map_mod  = _spatial(_MOD_YRS,  _mod_cache)
_map_diff = _map_mod - _map_dust

# ─── plot ─────────────────────────────────────────────────────────────────────
lon2d, lat2d = np.meshgrid(_LONS, _LATS)

valid = np.isfinite(_map_dust) | np.isfinite(_map_mod)
extent = ([lon2d[valid].min() - 2, lon2d[valid].max() + 2,
           lat2d[valid].min() - 1, lat2d[valid].max() + 1]
          if valid.any() else [-180, 180, float(_LATS.min()), float(_LATS.max())])

_vmax = float(np.nanpercentile(
    np.concatenate([_map_dust[np.isfinite(_map_dust)],
                    _map_mod[np.isfinite(_map_mod)]]), 99))

cmap_abs  = plt.cm.YlOrRd.copy()
cmap_abs.set_bad('none')
cmap_abs.set_under('lightgrey')

cmap_diff = plt.cm.RdBu_r.copy()
cmap_diff.set_bad('none')
_norm_diff = TwoSlopeNorm(vmin=_DIFF_VMIN, vcenter=0.0, vmax=_DIFF_VMAX)

proj = ccrs.PlateCarree()
fig, axes = plt.subplots(3, 1, figsize=(13, 9),
                          subplot_kw={'projection': proj},
                          gridspec_kw={'hspace': 0.04})

rows = [
    (_map_dust, cmap_abs,  _CLIM_THRESH, _vmax,      None,       f'Dust Bowl ({_DUST_YRS[0]}-{_DUST_YRS[-1]})'),
    (_map_mod,  cmap_abs,  _CLIM_THRESH, _vmax,      None,       f'Modern ({_MOD_YRS[0]}-{_MOD_YRS[-1]})'),
    (_map_diff, cmap_diff, _DIFF_VMIN,   _DIFF_VMAX, _norm_diff, 'Modern - Dust Bowl'),
]

_pcm_abs = _pcm_diff = None
for ax, (data, cmap, vmin_, vmax_, norm_, title) in zip(axes, rows):
    ax.set_extent(extent, crs=proj)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'),     facecolor='#cfe2f3', zorder=0)
    ax.add_feature(cfeature.LAND.with_scale('50m'),      facecolor='#ede8e0', zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'),  linewidth=0.35,      zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'),   linewidth=0.15, ls=':', zorder=2)
    if norm_ is not None:
        pcm = ax.pcolormesh(lon2d, lat2d, data, cmap=cmap,
                             norm=norm_, transform=proj,
                             zorder=1, shading='auto')
    else:
        pcm = ax.pcolormesh(lon2d, lat2d, data, cmap=cmap,
                             vmin=vmin_, vmax=vmax_, transform=proj,
                             zorder=1, shading='auto')
    ax.set_title(title, fontsize=11, fontweight='bold', fontfamily='serif', pad=3)
    if cmap is cmap_abs:
        _pcm_abs = pcm
    else:
        _pcm_diff = pcm
    ax.set_xticks([]); ax.set_yticks([])

ax_cb_abs  = fig.add_axes([0.15, 0.05, 0.42, 0.018])
ax_cb_diff = fig.add_axes([0.60, 0.05, 0.32, 0.018])

cb_a = fig.colorbar(_pcm_abs,  cax=ax_cb_abs,  orientation='horizontal', extend='max')
cb_a.set_label(
    f'% of JJA days exceeding local DOY-p95  (baseline: full record {_BASE_S}-{_BASE_E})',
    fontsize=9)
cb_a.ax.tick_params(labelsize=8)

cb_d = fig.colorbar(_pcm_diff, cax=ax_cb_diff, orientation='horizontal', extend='both')
cb_d.set_label(
    f'% change  (Modern {_MOD_YRS[0]}-{_MOD_YRS[-1]}  -  Dust Bowl {_DUST_YRS[0]}-{_DUST_YRS[-1]})',
    fontsize=9)
cb_d.ax.tick_params(labelsize=8)

fig.suptitle(
    f'Berkeley Earth - exceedance frequency maps\n'
    f'% summer (JJA) days per cell above local DOY-p95  '
    f'(baseline: full record {_BASE_S}-{_BASE_E})',
    fontsize=12, fontweight='bold', fontfamily='serif', y=0.99,
)

out = FIGD / (f'Fig_spatial_BE_dustbowl_vs_modern_JJA_grey5pct_'
                 f'{_MOD_YRS[0]}_{_MOD_YRS[-1]}_fixeddiff_FULLRECORDp95.png')
fig.savefig(out, bbox_inches='tight', dpi=150)
plt.show()
plt.close()

print(f"\nwrote {out}")

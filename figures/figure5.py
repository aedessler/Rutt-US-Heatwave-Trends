#!/usr/bin/env python3
"""Figure 5 -- the northern mid-latitude band, 24-50N, JJA.

Berkeley Earth and ERA5 JJA anomalies averaged over ALL LAND in the band, at
every longitude. Notebook section 9.

Both series are plain cos(lat)-weighted means over land cells on each dataset's
OWN native grid -- Berkeley at 1 degree, ERA5 at 0.25 degree. There is no
co-sampling and no CONUS thinning: the GHCN-Daily co-sample that earlier
versions of this figure used has been removed, so nothing about the station
network decides which cells enter the average (2026-09-23 request).

The two curves cover the same geography. Berkeley's daily TMAX/TMIN product is
land-only -- it carries no data at all where its land_mask is absent -- so its
land_mask IS its footprint, and ERA5 (which is global, land and ocean) is
masked to that same Berkeley land footprint rather than to a second, independent
coastline. ERA5 still starts in 1940; Berkeley runs from 1900. Each smoothed
curve is drawn over its own record, so the ERA5 line begins at 1940 rather than
five years into it.

    python figures/figure5.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *          # noqa: F401,F403

check_paths()
print()

# ══ SETTINGS AND HELPERS ═══════════════════════════════════════════════════
warnings.filterwarnings("ignore", category=RuntimeWarning)
for _d in (FIGD, CACHE_BAND): _d.mkdir(parents=True, exist_ok=True)

# -- Settings ---------------------------------------------------------------
Y0_F5, Y1_F5  = 1900, 2025
JJA           = [6, 7, 8]
BAND_LAT      = (24.0, 50.0)
MIN_JJA_DAYS  = 75                  # of 92; a cell needs this many in a year
MIN_BASE_YRS  = 15                  # of 30, per cell
BE_LAND_MIN   = 0.0                 # a Berkeley 1-deg cell is land if
                                    # land_mask > this. Berkeley's daily product
                                    # has no data off its own mask, so 0.0 means
                                    # "every cell Berkeley reports land for".
                                    # the window is common.py's ROLL = 11: ODD, so
                                    # the centred window is symmetric
MAD_K         = 8.0                 # light guard on a single absurd year
SMOOTH_FROM_RECORD_START = True     # loclin over a series' OWN first ROLL//2
                                    # years, not the axis's. ERA5 starts in 1940,
                                    # so its smooth starts in 1940 rather than
                                    # 1945 (2026-09-23 request). Set False for
                                    # the old behaviour, where a late-starting
                                    # record's first five years were blank.
DUSTBOWL      = (1930, 1940)
SHOW_ANNUAL_F5   = True                # thin annual JJA values under the smoothed curve ...
ANNUAL_SETS_F5   = {"Berkeley Earth"}  # ... for these datasets only (2026-09-21 request)
LW_ANN_F5, ALPHA_ANN_F5 = 0.9, 0.28    # Figure 1's thin-line weight and alpha
YPAD_FRAC     = 0.08                # padding as a fraction of the drawn range
FORCE_F5      = False
COL_BE_F5, COL_ERA5_F5 = "#009E73", "#D55E00"
ERA5_Y0_F5, ERA5_Y1_F5 = 1940, 2025
TAG_F5 = f"land{BE_LAND_MIN:g}_d{MIN_JJA_DAYS}_b{MIN_BASE_YRS}_{Y0_F5}-{Y1_F5}"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 16, "axes.titlesize": 20, "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5, "savefig.dpi": 300,
})

YEARS_F5 = np.arange(Y0_F5, Y1_F5 + 1)
BASE_OK  = (YEARS_F5 >= BASELINE[0]) & (YEARS_F5 <= BASELINE[1])


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
    return pd.Series(out, index=YEARS_F5), pd.Series(cov, index=YEARS_F5)


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
    no temperature at all -- so `> BE_LAND_MIN` with BE_LAND_MIN = 0 selects
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
        return la[keep], lo, np.isfinite(lm) & (lm > BE_LAND_MIN)
    raise FileNotFoundError(f"no Berkeley NH decade files under {BE_PROC}")


# ══ (2) BERKELEY: JJA ANOMALY ON ITS OWN GRID, THEN THE LAND BAND MEAN ═════
def berkeley_band():
    """Per-year JJA band mean anomaly over all Berkeley land in 24-50N."""
    fp = CACHE_BAND / f"berkeley_land_band_{TAG_F5}.npz"
    if fp.exists() and not FORCE_F5:
        z = np.load(fp)
        return ({v: pd.Series(z[v], index=YEARS_F5) for v in ("tmax", "tmin")},
                {v: pd.Series(z[f"cov_{v}"], index=YEARS_F5) for v in ("tmax", "tmin")})
    blat, blon, land = berkeley_grid()
    ser, cov = {}, {}
    for var in ("TMAX", "TMIN"):
        ann = np.full((len(YEARS_F5), blat.size, blon.size), np.nan, np.float32)
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
                if not (Y0_F5 <= y <= Y1_F5):
                    continue
                k = yrs == y
                cnt = np.isfinite(v[k]).sum(axis=0)
                ann[y - Y0_F5] = np.where(cnt >= MIN_JJA_DAYS,
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
    fp = CACHE_BAND / f"era5_land_band_{TAG_F5}.npz"
    if fp.exists() and not FORCE_F5:
        z = np.load(fp)
        return ({v: pd.Series(z[v], index=YEARS_F5) for v in ("tmax", "tmin")},
                {v: pd.Series(z[f"cov_{v}"], index=YEARS_F5) for v in ("tmax", "tmin")})
    blat, blon, bland = berkeley_grid()
    elat = elon = order = land = None
    sum_tx = sum_tn = cnt = None
    missing = []
    for y in range(ERA5_Y0_F5, ERA5_Y1_F5 + 1):
        for mo in JJA:
            f = ERA5_DIR / f"era5_2t_{y}{mo:02d}_daily.nc"
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
                shape = (len(YEARS_F5), elat.size, elon.size)
                sum_tx = np.zeros(shape); sum_tn = np.zeros(shape)
                cnt = np.zeros(shape, np.int32)
            yi = y - Y0_F5
            tx = ds["t2m_max"].values[:, :, order].astype(np.float64)
            tn = ds["t2m_min"].values[:, :, order].astype(np.float64)
            ds.close()
            sum_tx[yi] += np.nansum(tx, axis=0)
            sum_tn[yi] += np.nansum(tn, axis=0)
            cnt[yi] += np.isfinite(tx).sum(axis=0)
    if missing:
        print(f"ERA5 band field: missing {len(missing)} of "
              f"{(ERA5_Y1_F5 - ERA5_Y0_F5 + 1) * len(JJA)} JJA months "
              f"(e.g. {missing[:3]})")
    ser, cov = {}, {}
    for s, key in ((sum_tx, "tmax"), (sum_tn, "tmin")):
        ann = np.where(cnt >= MIN_JJA_DAYS, s / np.maximum(cnt, 1), np.nan)
        anom = np.where(land[None, :, :], anomalise(ann), np.nan)
        ser[key], cov[key] = band_mean_native(anom, elat, land)
    np.savez_compressed(fp, tmax=ser["tmax"].to_numpy(), tmin=ser["tmin"].to_numpy(),
                        cov_tmax=cov["tmax"].to_numpy(), cov_tmin=cov["tmin"].to_numpy())
    return ser, cov


BE, BE_COV = berkeley_band()
ER, ER_COV = era5_band()

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
              f"{list(YEARS_F5[bad])}")
        m[bad] = np.nan
    return pd.Series(m, index=YEARS_F5)


S = {f"be_{v}": finish(BE[v], f"Berkeley {v.upper()}") for v in ("tmax", "tmin")}
S.update({f"era5_{v}": finish(ER[v], f"ERA5 {v.upper()}") for v in ("tmax", "tmin")})
S["be_tavg"]   = 0.5 * (S["be_tmax"] + S["be_tmin"])
S["era5_tavg"] = 0.5 * (S["era5_tmax"] + S["era5_tmin"])


def smooth(s):
    """The smoother Figures 1, 3 and 5 share: an 11-year centered mean with a
    local-linear fit standing in over the outer ROLL//2 years at each end, so
    the curves reach the ends of the record instead of stopping five years
    short. strict_interior keeps the plain full-window rule inside the record,
    so a blanked year still leaves a hole.

    The series is cut to its OWN first and last year before smoothing. The end
    region is otherwise measured from the ends of the AXIS, which runs 1900-2025
    here, so a record that starts or stops somewhere else has its own edge years
    treated as interior: they want a full 11-year window, it reaches into empty
    years, and the curve comes out five years short of the data with a detached
    loclin segment beyond it. Trimming the tail fixed that at 2024 for Berkeley;
    trimming the head does the same for ERA5, whose record starts in 1940 and
    whose smooth used to start in 1945. A mid-record hole is still a hole."""
    first, last = s.first_valid_index(), s.last_valid_index()
    if last is None:
        return s
    if not SMOOTH_FROM_RECORD_START:
        first = s.index[0]
    return roll(s.loc[first:last], strict_interior=True).reindex(s.index)


# ══ CHECK -- Figure 5 series, before plotting. JJA anomaly over band land. ══
print(f"\n{'series':<28}{'1930s':>8}{'2010-24':>9}{'trend/century':>15}{'years':>7}")
for _lbl, _k in (("Berkeley TMAX", "be_tmax"), ("Berkeley TMIN", "be_tmin"),
                 ("Berkeley TAVG", "be_tavg"),
                 ("ERA5 TMAX", "era5_tmax"), ("ERA5 TMIN", "era5_tmin"),
                 ("ERA5 TAVG", "era5_tavg")):
    _s = S[_k].dropna()
    _tr = np.polyfit(_s.index, _s.values, 1)[0] * 100
    _d30 = _s.reindex(range(1930, 1940)).mean()
    print(f"  {_lbl:<26}{_d30:>+8.2f}{_s.reindex(range(2010, 2025)).mean():>+9.2f}"
          f"{_tr:>+15.2f}{len(_s):>7}")

print(f"\nland coverage (share of the band's land area actually averaged)")
print(f"  {'decade':<10}{'Berkeley TMAX':>16}{'ERA5 TMAX':>12}")
for _d0 in range(1900, 2030, 10):
    _yy = range(_d0, min(_d0 + 10, Y1_F5 + 1))
    _b = BE_COV["tmax"].reindex(_yy).mean()
    _e = ER_COV["tmax"].reindex(_yy).mean()
    if np.isfinite(_b) or np.isfinite(_e):
        print(f"  {_d0}s{'':<5}{_b:>15.1%}{_e:>12.1%}")

# ══ PLOT — smoothed curves (plus Berkeley's annual values), scaled to what is drawn ══
PANELS = [("(a) TMAX: JJA", "be_tmax", "era5_tmax"),
          ("(b) TMIN: JJA", "be_tmin", "era5_tmin"),
          ("(c) TAVG: JJA", "be_tavg", "era5_tavg")]

# precompute everything that will appear on the axes, and scale to THAT
SM = {k: smooth(S[k]) for _, b, e in PANELS for k in (b, e)}
_drawn_parts = [v.to_numpy(float) for v in SM.values()]
if SHOW_ANNUAL_F5:                    # the thin annual traces must fit on the axes too
    _drawn_parts += [S[k].to_numpy(float) for _, kb, ke in PANELS
                     for lbl, k in (("Berkeley Earth", kb), ("ERA5", ke))
                     if lbl in ANNUAL_SETS_F5]
_drawn = np.concatenate(_drawn_parts)
_drawn = _drawn[np.isfinite(_drawn)]
_pad = YPAD_FRAC * (np.nanmax(_drawn) - np.nanmin(_drawn))
_lo, _hi = np.nanmin(_drawn) - _pad, np.nanmax(_drawn) + _pad

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
for ax, (title, kb, ke) in zip(axes, PANELS):
    ax.axvspan(*DUSTBOWL, color="#caa472", alpha=0.18, lw=0, zorder=0)
    for lbl, k, col in (("Berkeley Earth", kb, COL_BE_F5),
                        ("ERA5", ke, COL_ERA5_F5)):
        if SHOW_ANNUAL_F5 and lbl in ANNUAL_SETS_F5:
            ax.plot(YEARS_F5, S[k].values, color=col, lw=LW_ANN_F5, alpha=ALPHA_ANN_F5,
                    zorder=2)
        ax.plot(YEARS_F5, SM[k].values, color=col, lw=2.4, label=lbl, zorder=3)
    ax.axhline(0, color="0.2", lw=0.9, alpha=0.8, zorder=0)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=18)
    ax.set_ylabel("Anomaly (°C)")
    ax.grid(True, axis="y", color="0.85", lw=0.8); ax.grid(False, axis="x")
    ax.set_ylim(_lo, _hi); ax.set_xlim(1900, 2026)
    ax.tick_params(axis="both", labelsize=14)
axes[2].set_xlabel("Year", fontsize=17)
axes[2].xaxis.set_major_locator(mticker.MultipleLocator(20))
axes[2].xaxis.set_minor_locator(mticker.MultipleLocator(10))

h, l, seen = [], [], set()
for ax in axes:
    for hh, ll in zip(*ax.get_legend_handles_labels()):
        if ll not in seen:
            h.append(hh); l.append(ll); seen.add(ll)
fig.legend(h, l, loc="upper center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, 1.06), handlelength=2.4, columnspacing=2, fontsize=20)
OUT_F5 = FIGD / "Figure5.png"
fig.savefig(OUT_F5, bbox_inches="tight", dpi=300, facecolor="white")
plt.show()
plt.close(fig)

print(f"\nwrote {OUT_F5}")

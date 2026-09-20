#!/usr/bin/env python3
"""Figure 5 -- the northern mid-latitude band, 24-50N, JJA.

GHCN-Daily and Berkeley Earth JJA anomalies over the band, with CONUS thinned
to match the rest of the band's sampling. Notebook section 9.

    python figures/figure5.py
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
for _d in (FIGD, CACHE_BAND): _d.mkdir(parents=True, exist_ok=True)

# -- Settings ---------------------------------------------------------------
Y0_F5, Y1_F5        = 1900, 2024
JJA           = [6, 7, 8]
BAND_LAT      = (24.0, 50.0)
GRID_DEG      = 2.0                 # analysis grid for the co-sample
MIN_JJA_DAYS  = 75                  # of 92; a cell/station needs this many
MIN_BASE_YRS  = 15                  # of 30, per station and per cell
TMAX_BOUNDS_F5   = (-60.0, 60.0)
N_DRAWS       = 20                  # random co-sample realizations per year
RNG_SEED      = 42
                                    # the window is common.py's ROLL = 11: ODD, so
                                    # the centred window is symmetric
TRIM_TERMINAL = False               # see note 7; reports either way
MAD_K         = 8.0
DUSTBOWL      = (1930, 1940)
SHOW_ANNUAL_F5   = False               # v9b: panels show smoothed curves only
YPAD_FRAC     = 0.08                # padding as a fraction of the drawn range
FORCE_F5         = False
COL_GH_F5, COL_BE_F5 = "#E63946", "#009E73"
TAG_F5 = f"{int(GRID_DEG*10)}deg_d{MIN_JJA_DAYS}_b{MIN_BASE_YRS}"

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
LAT_EDGES = np.arange(BAND_LAT[0], BAND_LAT[1] + 1e-9, GRID_DEG)
LON_EDGES = np.arange(-180.0, 180.0 + 1e-9, GRID_DEG)
LATC = 0.5 * (LAT_EDGES[:-1] + LAT_EDGES[1:])
LONC = 0.5 * (LON_EDGES[:-1] + LON_EDGES[1:])
NLAT, NLON = LATC.size, LONC.size
CELLW = np.repeat(np.cos(np.deg2rad(LATC))[:, None], NLON, axis=1)   # cos(lat)

def conus_cell_mask():
    """True where a cell's centre is inside CONUS. Cached."""
    fp = CACHE_BAND / f"conus_cells_{TAG_F5}.npy"
    if fp.exists():
        return np.load(fp)
    poly = conus_polygon()
    LON2, LAT2 = np.meshgrid(LONC, LATC)
    try:
        import shapely
        m = np.asarray(shapely.contains_xy(poly, LON2.ravel(), LAT2.ravel()))
    except Exception:
        from shapely.geometry import Point
        from shapely.prepared import prep
        pp = prep(poly)
        m = np.array([pp.contains(Point(o, a))
                      for a, o in zip(LAT2.ravel(), LON2.ravel())])
    m = m.reshape(LAT2.shape)
    np.save(fp, m)
    return m

CONUS = conus_cell_mask()

# ══ (1) GHCN-DAILY FROM THE STATION ARCHIVE ════════════════════════════════
STN_FP = CACHE_BAND / f"ghcn_band_station_jja_{Y0_F5}_{Y1_F5}.parquet"

def build_station_jja():
    """Per station and year: JJA mean TMAX and TMIN, from the global parquets.
    A station-year is kept only if BOTH elements have >= MIN_JJA_DAYS days, so
    TAVG is formed from the same days as TMAX and TMIN."""
    if STN_FP.exists() and not FORCE_F5:
        return pd.read_parquet(STN_FP)
    qcols = None
    parts = []
    for y in range(Y0_F5, Y1_F5 + 1):
        fp = GHCND_GLOBAL / f"ghcn_global_{y}.parquet"
        if not fp.exists():
            continue
        if qcols is None:
            avail = set(pd.read_parquet(fp).head(0).columns)
            qcols = [c for c in ("qflag", "q_flag", "quality_flag") if c in avail]
        d = pd.read_parquet(fp, columns=["station_id", "date", "tmax_c", "tmin_c",
                                         "lat", "lon"] + qcols)
        d = d[d["lat"].between(*BAND_LAT)]
        if d.empty:
            continue
        for c in qcols:
            d = d[d[c].fillna("").astype(str).str.strip() == ""]
        dt = pd.DatetimeIndex(d["date"])
        d = d[np.isin(dt.month, JJA)]
        if d.empty:
            continue
        for c in ("tmax_c", "tmin_c"):
            d.loc[~d[c].between(*TMAX_BOUNDS_F5), c] = np.nan
        d = d.drop_duplicates(subset=["station_id", "date"], keep="first")
        g = d.groupby("station_id").agg(
            tmax=("tmax_c", "mean"), n_tmax=("tmax_c", "count"),
            tmin=("tmin_c", "mean"), n_tmin=("tmin_c", "count"),
            lat=("lat", "first"), lon=("lon", "first")).reset_index()
        g = g[(g["n_tmax"] >= MIN_JJA_DAYS) & (g["n_tmin"] >= MIN_JJA_DAYS)]
        if g.empty:
            continue
        g["year"] = y
        parts.append(g[["station_id", "year", "lat", "lon", "tmax", "tmin"]])
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(STN_FP, index=False)
    return out

STN = build_station_jja()

# ══ (2) STATION ANOMALIES, THEN BIN TO THE GRID ════════════════════════════
def station_anomaly_field():
    """(year, lat, lon) GHCN-Daily JJA anomaly, and the count of stations per
    cell. Every station is anomalised against ITS OWN 1951-1980 JJA mean BEFORE
    any spatial averaging, so a changing station set cannot inject climatology."""
    fp = CACHE_BAND / f"ghcn_band_field_{TAG_F5}.npz"
    if fp.exists() and not FORCE_F5:
        z = np.load(fp)
        return z["tmax"], z["tmin"], z["n"]
    d = STN.copy()
    base = d[d["year"].between(*BASELINE)]
    clim = base.groupby("station_id").agg(
        cx=("tmax", "mean"), cn=("tmin", "mean"), k=("year", "count"))
    clim = clim[clim["k"] >= MIN_BASE_YRS]
    d = d.join(clim, on="station_id", how="inner")
    d["ax"] = d["tmax"] - d["cx"]
    d["an"] = d["tmin"] - d["cn"]
    d["il"] = np.clip(((d["lat"] - LAT_EDGES[0]) / GRID_DEG).astype(int), 0, NLAT - 1)
    d["io"] = np.clip(((d["lon"] + 180.0) / GRID_DEG).astype(int), 0, NLON - 1)
    d["yi"] = d["year"] - Y0_F5
    tx = np.full((len(YEARS_F5), NLAT, NLON), np.nan, np.float32)
    tn = np.full((len(YEARS_F5), NLAT, NLON), np.nan, np.float32)
    nn = np.zeros((len(YEARS_F5), NLAT, NLON), np.int16)
    g = d.groupby(["yi", "il", "io"]).agg(ax=("ax", "mean"), an=("an", "mean"),
                                          n=("ax", "size")).reset_index()
    tx[g["yi"], g["il"], g["io"]] = g["ax"].to_numpy(np.float32)
    tn[g["yi"], g["il"], g["io"]] = g["an"].to_numpy(np.float32)
    nn[g["yi"], g["il"], g["io"]] = g["n"].to_numpy(np.int16)
    np.savez_compressed(fp, tmax=tx, tmin=tn, n=nn)
    return tx, tn, nn

GH_TX, GH_TN, GH_N = station_anomaly_field()

# ══ (4) BERKELEY: CELL ANOMALIES, AGGREGATED ONTO THE SAME GRID ════════════
def berkeley_band_field():
    """(year, lat, lon) Berkeley JJA anomaly on the analysis grid. Each 1-degree
    cell is anomalised against its own 1951-1980 JJA mean, then the analysis
    cell takes the cos(lat)-weighted mean of the 1-degree cells inside it."""
    fp = CACHE_BAND / f"berkeley_band_field_{TAG_F5}.npz"
    if fp.exists() and not FORCE_F5:
        z = np.load(fp)
        return z["tmax"], z["tmin"]
    out = {}
    for var in ("TMAX", "TMIN"):
        ann, blat, blon = None, None, None
        for dec in range(1900, 2030, 10):
            f = BE_PROC / f"processed_nh_{var}_Complete_{var}_Daily_LatLong1_{dec}.nc"
            if not f.exists():
                continue
            ds = xr.open_dataset(f)
            t = ds["temperature"]
            la = ds["latitude"].values.astype(float)
            lo = ds["longitude"].values.astype(float)
            if lo.max() > 180.0:
                lo = np.where(lo > 180.0, lo - 360.0, lo)
            keep_la = (la >= BAND_LAT[0]) & (la <= BAND_LAT[1])   # direction-safe
            tt = pd.DatetimeIndex(t["time"].values)
            sel = np.isin(tt.month, JJA)
            v = t.values[sel][:, keep_la, :].astype(np.float32)
            v = np.where(np.isfinite(v) & (np.abs(v) < 1e4), v, np.nan)
            yrs = tt.year.values[sel]
            lm = ds["land_mask"]
            lm = (lm.isel(time=0) if "time" in lm.dims else lm).values[keep_la, :]
            ds.close()
            if ann is None:
                blat, blon = la[keep_la], lo
                ann = np.full((len(YEARS_F5), blat.size, blon.size), np.nan, np.float32)
                land = lm > 0
            for y in np.unique(yrs):
                if not (Y0_F5 <= y <= Y1_F5):
                    continue
                k = yrs == y
                if k.sum() < MIN_JJA_DAYS:
                    continue
                m = np.nanmean(v[k], axis=0)
                cnt = np.isfinite(v[k]).sum(axis=0)
                ann[y - Y0_F5] = np.where(cnt >= MIN_JJA_DAYS, m, np.nan)
        ann = np.where(land[None, :, :], ann, np.nan)
        b = (YEARS_F5 >= BASELINE[0]) & (YEARS_F5 <= BASELINE[1])
        nb = np.isfinite(ann[b]).sum(axis=0)
        clim = np.where(nb >= MIN_BASE_YRS, np.nanmean(ann[b], axis=0), np.nan)
        anom = ann - clim[None, :, :]
        # aggregate 1 deg -> analysis grid, cos(lat) weighted
        il = np.clip(((blat - LAT_EDGES[0]) / GRID_DEG).astype(int), 0, NLAT - 1)
        io = np.clip(((blon + 180.0) / GRID_DEG).astype(int), 0, NLON - 1)
        w1 = np.cos(np.deg2rad(blat))[:, None] * np.ones(blon.size)[None, :]
        num = np.zeros((len(YEARS_F5), NLAT, NLON)); den = np.zeros_like(num)
        good = np.isfinite(anom)
        aw = np.where(good, anom * w1[None, :, :], 0.0)
        ww = np.where(good, w1[None, :, :], 0.0)
        for i in range(blat.size):
            np.add.at(num, (slice(None), il[i], io), aw[:, i, :])
            np.add.at(den, (slice(None), il[i], io), ww[:, i, :])
        out[var] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan).astype(np.float32)
    np.savez_compressed(fp, tmax=out["TMAX"], tmin=out["TMIN"])
    return out["TMAX"], out["TMIN"]

BE_TX, BE_TN = berkeley_band_field()

# ══ (3)(7) THE CO-SAMPLE: identical cells for both datasets, N_DRAWS times ══
rng = np.random.default_rng(RNG_SEED)
res = {k: np.full((N_DRAWS, len(YEARS_F5)), np.nan)
       for k in ("gh_tx", "gh_tn", "be_tx", "be_tn")}
full = {k: np.full(len(YEARS_F5), np.nan) for k in ("gh_tx", "gh_tn", "be_tx", "be_tn")}
diag = []
for yi, y in enumerate(YEARS_F5):
    # a cell counts only if BOTH datasets have it -- this is what makes it a
    # co-sample, and what v8j silently violated
    ok = (np.isfinite(GH_TX[yi]) & np.isfinite(GH_TN[yi])
          & np.isfinite(BE_TX[yi]) & np.isfinite(BE_TN[yi]))
    c_idx = np.argwhere(ok & CONUS)
    r_idx = np.argwhere(ok & ~CONUS)
    if len(c_idx) == 0 or len(r_idx) == 0:
        continue
    # unsampled full-band reference, for context
    fi = np.argwhere(ok)
    fw = CELLW[fi[:, 0], fi[:, 1]]
    for k, F in (("gh_tx", GH_TX), ("gh_tn", GH_TN), ("be_tx", BE_TX), ("be_tn", BE_TN)):
        full[k][yi] = np.sum(F[yi][fi[:, 0], fi[:, 1]] * fw) / np.sum(fw)
    n_draw = min(len(r_idx), len(c_idx))
    for dch in range(N_DRAWS):
        cs = c_idx[rng.choice(len(c_idx), size=n_draw, replace=False)] \
             if n_draw < len(c_idx) else c_idx
        sel = np.concatenate([cs, r_idx], axis=0)
        w = CELLW[sel[:, 0], sel[:, 1]]
        for k, F in (("gh_tx", GH_TX), ("gh_tn", GH_TN),
                     ("be_tx", BE_TX), ("be_tn", BE_TN)):
            res[k][dch, yi] = np.sum(F[yi][sel[:, 0], sel[:, 1]] * w) / np.sum(w)
    diag.append((y, len(c_idx), len(r_idx), n_draw))

# ══ SERIES, LIGHT FILTERING, DIAGNOSTICS ═══════════════════════════════════
def mad_flag(v, k=MAD_K):
    v = np.asarray(v, float)
    med = np.nanmedian(v); mad = np.nanmedian(np.abs(v - med))
    if not np.isfinite(mad) or mad == 0:
        return np.zeros(v.shape, bool)
    return np.isfinite(v) & (np.abs(v - med) > k * 1.4826 * mad)

def finish(mat, label):
    """Ensemble mean over draws, one MAD pass (not two), terminal trim reported."""
    m = np.nanmean(mat, axis=0) if mat.ndim == 2 else np.asarray(mat, float)
    bad = mad_flag(m)
    if bad.any():
        m = np.where(bad, np.nan, m)
    fi = np.where(np.isfinite(m))[0]
    if len(fi) >= 15:
        hist, tail = fi[:-2], fi[-2:]
        med = np.nanmedian(m[hist]); mad = np.nanmedian(np.abs(m[hist] - med))
        if np.isfinite(mad) and mad > 0:
            drop = [i for i in tail if abs(m[i] - med) > 6.0 * 1.4826 * mad]
            if drop:
                if TRIM_TERMINAL:
                    m[drop] = np.nan
    return pd.Series(m, index=YEARS_F5)

S   = {k: finish(res[k], k) for k in res}
SF  = {k: finish(full[k], f"{k} (full band)") for k in full}
SPR = {k: pd.DataFrame(res[k], columns=YEARS_F5) for k in res}
S["gh_ta"] = 0.5 * (S["gh_tx"] + S["gh_tn"])
S["be_ta"] = 0.5 * (S["be_tx"] + S["be_tn"])
SF["gh_ta"] = 0.5 * (SF["gh_tx"] + SF["gh_tn"])
SF["be_ta"] = 0.5 * (SF["be_tx"] + SF["be_tn"])
for k in ("gh_ta", "be_ta"):
    SPR[k] = 0.5 * (SPR[k.replace("_ta", "_tx")] + SPR[k.replace("_ta", "_tn")])

def smooth(s):
    """The smoother Figures 1, 3 and 5 now share: an 11-year centered mean with
    a local-linear fit standing in over the outer ROLL//2 years at each end of
    the record, so the curves reach 2024 instead of stopping in 2019.
    strict_interior keeps the plain full-window rule inside the record, so the
    years finish() blanks as MAD outliers still leave holes rather than having a
    six-point line drawn through them."""
    return roll(s, strict_interior=True)

# ══ CHECK -- Figure 5 series, before plotting. JJA anomaly on the band. ════
print(f"{'series':<28}{'1930s':>8}{'2010-24':>9}{'trend/century':>15}{'years':>7}")
for _lbl, _k in (("GHCN-Daily TMAX", "gh_tx"), ("GHCN-Daily TMIN", "gh_tn"),
                 ("Berkeley TMAX", "be_tx"), ("Berkeley TMIN", "be_tn")):
    _s = S[_k].dropna()
    _tr = np.polyfit(_s.index, _s.values, 1)[0] * 100
    print(f"  {_lbl:<26}{_s.loc[1930:1939].mean():>+8.2f}{_s.loc[2010:2024].mean():>+9.2f}"
          f"{_tr:>+15.2f}{len(_s):>7}")
print(f"\nequal-region sampling: {N_DRAWS} draws, spread (1 s.d. across draws) "
      f"{float(np.nanmean(np.nanstd(res['gh_tx'], axis=0))):.3f} degC")
# the panels no longer shade this, so it is reported rather than drawn
print(f"{'series':<28}{'2.5-97.5 spread across draws, degC':>38}")
for _k in ("gh_tx", "be_tx", "gh_tn", "be_tn", "gh_ta", "be_ta"):
    _w = (SPR[_k].quantile(0.975) - SPR[_k].quantile(0.025))
    print(f"  {_k:<26}{f'mean {_w.mean():.4f}   max {_w.max():.4f}':>38}")

# ══ PLOT — smoothed curves only, scaled to what is actually drawn ══════════
PANELS = [("(a) TMAX: JJA", "gh_tx", "be_tx"),
          ("(b) TMIN: JJA", "gh_tn", "be_tn"),
          ("(c) TAVG: JJA", "gh_ta", "be_ta")]

# precompute everything that will appear on the axes, and scale to THAT
SM   = {k: smooth(S[k]) for _, a, b in PANELS for k in (a, b)}
# The panels carry the smoothed curve alone. The 2.5/97.5 spread across the
# N_DRAWS co-samples is still computed and reported in the diagnostic above; it
# is simply not shaded, so nothing on the axes competes with the curve.
_drawn = np.concatenate([v.to_numpy(float) for v in SM.values()])
_drawn = _drawn[np.isfinite(_drawn)]
_pad = YPAD_FRAC * (np.nanmax(_drawn) - np.nanmin(_drawn))
_lo, _hi = np.nanmin(_drawn) - _pad, np.nanmax(_drawn) + _pad

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
for ax, (title, kg, kb) in zip(axes, PANELS):
    ax.axvspan(*DUSTBOWL, color="#caa472", alpha=0.18, lw=0, zorder=0)
    for lbl, k, col in (
        # ("GHCN-Daily", kg, COL_GH_F5),  # suppressed from the plot; kg is
        # still computed and checked above
        ("Berkeley Earth", kb, COL_BE_F5),
    ):
        if SHOW_ANNUAL_F5:
            ax.plot(YEARS_F5, S[k].values, color=col, lw=0.9, alpha=0.28, zorder=2)
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

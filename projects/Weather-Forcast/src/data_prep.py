"""
data_prep.py -- Fetch real weather data via Open-Meteo + preprocess
=====================================================================
- Auto-sets end_date to today (extends dataset over time)
- Incremental update: only fetches new days since last run
- Saves metadata cache in data/meta.json
- On each run, downloads missing data, appends, re-processes

Usage:
    python data_prep.py                    (fetch to today)
    python data_prep.py --end-date 2026-06-01   (fixed end)
    python data_prep.py --fake             (synthetic data)
"""

import json
import os
import pickle
import sys
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yaml
from sklearn.preprocessing import MinMaxScaler


def load_config():
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
META_PATH = os.path.join(DATA_DIR, "meta.json")

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

VARIABLE_MAP = {
    "tavg": "temperature_2m_mean",
    "tmin": "temperature_2m_min",
    "tmax": "temperature_2m_max",
    "prcp": "precipitation_sum",
    "rhum": "relative_humidity_2m_mean",
    "wspd": "wind_speed_10m_mean",
    "pres": "surface_pressure_mean",
}


# ─── metadata cache ─────────────────────────────────────────────

def load_meta():
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            return json.load(f)
    return {"last_fetch": None, "last_end_date": None, "total_days": 0}


def save_meta(meta):
    os.makedirs(DATA_DIR, exist_ok=True)
    meta["updated_at"] = datetime.now().isoformat()
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


# ─── real data: Open-Meteo API ──────────────────────────────────

def fetch_city_meteo(lat, lon, start, end):
    variables = list(VARIABLE_MAP.values())
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": str(start), "end_date": str(end),
        "daily": ",".join(variables),
        "timezone": "Asia/Shanghai",
    }
    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=60)
        if resp.status_code == 429:
            print("RATE-LIMITED, waiting 5s ...", end=" ", flush=True)
            time.sleep(5)
            resp = requests.get(OPEN_METEO_URL, params=params, timeout=60)
        resp.raise_for_status()
        body = resp.json()
        daily = body.get("daily", {})
        if not daily or "time" not in daily:
            return None
        df = pd.DataFrame(daily)
        df = df.rename(columns={"time": "date"})
        reverse_map = {v: k for k, v in VARIABLE_MAP.items()}
        df = df.rename(columns=reverse_map)
        df["date"] = pd.to_datetime(df["date"])
        return df[["date"] + list(VARIABLE_MAP.keys())]
    except Exception as e:
        print(f"[ERROR] {e}")
        return None


def download_real_data(cfg, start_date, end_date):
    cities = cfg["data"]["cities"]
    print(f"[*] Fetching {start_date} -> {end_date} from Open-Meteo ...")
    all_dfs = []

    for name, info in cities.items():
        print(f"    {name:10s} ...", end=" ", flush=True)
        df = fetch_city_meteo(info["lat"], info["lon"], start_date, end_date)
        if df is None or df.empty:
            print("FAILED")
            continue
        df["city"] = name
        df["lat"] = info["lat"]
        df["lon"] = info["lon"]
        df["elev"] = info["elev"]
        all_dfs.append(df)
        print(f"OK ({len(df)} days)")
        time.sleep(1.5)

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return None


def incremental_fetch(cfg):
    """
    Smart fetch: only download new days since last run.
    Returns full DataFrame with all available data.
    """
    meta = load_meta()
    start_date = cfg["data"]["start_date"]
    today = date.today()

    # Parse CLI --end-date override
    for i, a in enumerate(sys.argv):
        if a == "--end-date" and i + 1 < len(sys.argv):
            today = date.fromisoformat(sys.argv[i + 1])
            break

    if meta["last_end_date"]:
        last_end = date.fromisoformat(meta["last_end_date"])
    else:
        last_end = None

    # Check if we need to fetch anything
    if last_end and last_end >= today:
        print(f"[*] Data already up-to-date (covers up to {last_end}).")
        cached = os.path.join(DATA_DIR, "all_raw.csv")
        if os.path.exists(cached):
            df = pd.read_csv(cached, parse_dates=["date"])
            print(f"    Loaded {len(df)} rows from cache.")
            return df

    if last_end and last_end < today:
        # Incremental: only fetch from day after last_end to today
        fetch_start = last_end + timedelta(days=1)
        print(f"[*] Incremental update: {fetch_start} -> {today}")
        new_df = download_real_data(cfg, fetch_start, today)
        if new_df is None:
            print("[WARN] Incremental fetch failed, trying full re-download.")
            new_df = None

        # Merge with existing cache if available
        cached = os.path.join(DATA_DIR, "all_raw.csv")
        if new_df is not None and os.path.exists(cached):
            old_df = pd.read_csv(cached, parse_dates=["date"])
            df = pd.concat([old_df, new_df], ignore_index=True)
            df = df.drop_duplicates(subset=["city", "date"]).sort_values(["city", "date"])
        elif new_df is not None:
            df = new_df
        else:
            if os.path.exists(cached):
                df = pd.read_csv(cached, parse_dates=["date"])
                print(f"    Loaded {len(df)} rows from cache.")
                return df
            return None
    else:
        # Full download (no cache or cache reset)
        print(f"[*] Full download: {start_date} -> {today}")
        df = download_real_data(cfg, start_date, today)

    if df is None or df.empty:
        return None

    # Save cache
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_DIR, "all_raw.csv"), index=False)
    save_meta({
        "last_fetch": date.today().isoformat(),
        "last_end_date": today.isoformat(),
        "total_days": len(df),
        "cities": sorted(df["city"].unique().tolist()),
        "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
    })
    return df


# ─── fallback: synthetic data ───────────────────────────────────

def generate_synthetic(cfg, end_date):
    print("[*] Generating synthetic data ...")
    rng = np.random.default_rng(cfg["train"]["seed"])
    start = pd.Timestamp(cfg["data"]["start_date"])
    end = pd.Timestamp(end_date)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    doy = dates.dayofyear.values
    cities = cfg["data"]["cities"]

    CLIMATE = {
        "Kunming":   {"tavg": 15.5, "tamp": 7.5, "tphase": 190, "prcp": 2.8, "prcp_skew": 0.3, "rhum": 72, "wspd": 2.5, "pres": 810},
        "Guiyang":   {"tavg": 15.3, "tamp": 10.0, "tphase": 195, "prcp": 3.2, "prcp_skew": 0.4, "rhum": 78, "wspd": 2.2, "pres": 880},
        "Chengdu":   {"tavg": 16.2, "tamp": 8.5, "tphase": 200, "prcp": 2.5, "prcp_skew": 0.5, "rhum": 82, "wspd": 1.5, "pres": 950},
        "Chongqing": {"tavg": 18.3, "tamp": 10.0, "tphase": 205, "prcp": 3.2, "prcp_skew": 0.5, "rhum": 80, "wspd": 1.8, "pres": 985},
    }
    df_list = []
    for city_name, info in cities.items():
        c = CLIMATE[city_name]
        ek = info["elev"] / 1000.0
        t = c["tavg"] + c["tamp"] * np.sin(2 * np.pi * (doy - c["tphase"]) / 365.0) - 6.5 * (ek - 0.5) + rng.normal(0, 2, n)
        tmin = t - rng.uniform(3, 6, n)
        tmax = t + rng.uniform(3, 6, n)
        prcp = rng.gamma(shape=c["prcp_skew"] * 3, scale=c["prcp"] / 3 * (1 + 0.6 * np.sin(2 * np.pi * (doy - 160) / 365.0)))
        prcp[rng.random(n) < 0.35] = 0
        rhum = np.clip(c["rhum"] + 8 * np.sin(2 * np.pi * (doy - 160) / 365.0) + rng.normal(0, 5, n), 30, 100)
        wspd = c["wspd"] + rng.gamma(shape=2, scale=1.0, size=n)
        pres = c["pres"] + 5 * np.sin(2 * np.pi * (doy - 150) / 365.0) + rng.normal(0, 3, n)
        df_list.append(pd.DataFrame({
            "city": city_name, "lat": info["lat"], "lon": info["lon"], "elev": info["elev"],
            "date": dates, "tavg": t, "tmin": tmin, "tmax": tmax, "prcp": prcp, "rhum": rhum, "wspd": wspd, "pres": pres,
        }))
    df = pd.concat(df_list, ignore_index=True)
    print(f"  {len(df)} rows, {df['city'].nunique()} cities")
    return df


# ─── preprocessing ─────────────────────────────────────────────

def preprocess(df, cfg, today):
    features = cfg["data"]["features"]
    target = cfg["data"]["target"]
    wlen = cfg["window"]["window_len"]
    horizon = cfg["window"]["horizon"]

    print("[*] Preprocessing: clean + time features + scale + windows + split ...")

    df = df.sort_values(["city", "date"]).copy()
    for col in features:
        df[col] = df[col].astype("float64")
        df[col] = df.groupby("city")[col].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both").ffill().bfill()
        )

    # Add cyclical time features (day of year -> sin/cos)
    doy = df["date"].dt.dayofyear.values
    df["day_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["day_cos"] = np.cos(2 * np.pi * doy / 365.25)
    features = features + ["day_sin", "day_cos"]
    print(f"    Features ({len(features)}): {features}")

    missing = df[features].isna().sum()
    if missing.sum() > 0:
        print(f"    WARNING: {missing.sum()} missing values, dropping rows.")
        df = df.dropna(subset=features)

    # Scale per city
    scalers = {}
    for city in df["city"].unique():
        scalers[city] = MinMaxScaler()
        m = df["city"] == city
        df.loc[m, features] = scalers[city].fit_transform(df.loc[m, features])

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "scalers.pkl"), "wb") as f:
        pickle.dump(scalers, f)

    # -- Sliding windows --
    print("    Building windows ...")
    segments = {}
    for city in df["city"].unique():
        cdf = df[df["city"] == city].sort_values("date")
        vals = cdf[features].values.astype(np.float32)
        y_all = cdf[target].values.astype(np.float32)
        d_all = cdf["date"].values
        total = len(vals) - wlen - horizon + 1
        if total <= 0:
            print(f"    {city}: too few days ({len(vals)}), skipping")
            continue
        X_wins = np.lib.stride_tricks.sliding_window_view(vals, (wlen, vals.shape[1]))
        X_wins = X_wins.squeeze(1).transpose(0, 1, 2)
        y_win = y_all[wlen + horizon - 1:]
        d_win = d_all[wlen + horizon - 1:]
        min_len = min(len(X_wins), len(y_win))
        segments[city] = {
            "X": X_wins[:min_len].astype(np.float32),
            "y": y_win[:min_len].astype(np.float32),
            "dates": d_win[:min_len],
        }
        print(f"    {city:10s}  {min_len} windows")

    # -- Dynamic split (relative to today) --
    ty = cfg["split"]["train_years_back"]
    vy = cfg["split"]["val_years_back"]
    testy = cfg["split"]["test_years_back"]

    train_end = today.year - ty
    val_end = today.year - vy
    test_end = today.year - testy

    print(f"\n    Split: train <= {train_end} | val ~{train_end+1}-{val_end} | test > {val_end}")

    def split_by_range(seg, lo_year, hi_year):
        d = pd.to_datetime(seg["dates"])
        m = (d.year >= lo_year) & (d.year <= hi_year)
        return seg["X"][m], seg["y"][m], seg["dates"][m]

    splits = {
        "train": (None, train_end),
        "val":   (train_end + 1, val_end),
        "test":  (val_end + 1, today.year),
    }

    for split_name, (lo, hi) in splits.items():
        Xs, ys, ds, cs = [], [], [], []
        lo = lo or int(df["date"].dt.year.min())
        hi = hi or today.year
        for city, seg in segments.items():
            Xp, yp, dp = split_by_range(seg, lo, hi)
            if len(Xp) == 0:
                continue
            Xs.append(Xp); ys.append(yp); ds.append(dp)
            cs.append(np.array([city] * len(yp)))
        if Xs:
            path = os.path.join(DATA_DIR, f"{split_name}.npz")
            np.savez_compressed(
                path, X=np.concatenate(Xs).astype(np.float32),
                y=np.concatenate(ys).astype(np.float32),
                dates=np.concatenate(ds), city=np.concatenate(cs),
            )
            print(f"    -> {path}  X:{np.concatenate(Xs).shape}")


# ─── main ──────────────────────────────────────────────────────

def main():
    use_fake = "--fake" in sys.argv
    cfg = load_config()
    today = date.today()
    for i, a in enumerate(sys.argv):
        if a == "--end-date" and i + 1 < len(sys.argv):
            today = date.fromisoformat(sys.argv[i + 1])
            break

    print("=" * 60)
    print(f"  weather_sw Data Pipeline")
    print(f"  Source: {'SYNTHETIC' if use_fake else 'Open-Meteo API (real)'}")
    print(f"  Target end date: {today}")
    print("=" * 60)

    if use_fake:
        df = generate_synthetic(cfg, today)
    else:
        df = incremental_fetch(cfg)

    if df is None:
        print("[FALLBACK] API failed. Run with --fake for synthetic data.")
        sys.exit(1)

    # Quick stats
    print(f"\n  Dataset: {len(df)} rows | "
          f"{df['date'].min().date()} ~ {df['date'].max().date()} | "
          f"{df['city'].nunique()} cities")
    for city in cfg["data"]["cities"]:
        cdf = df[df["city"] == city]
        if len(cdf):
            print(f"    {city:10s}  tavg={cdf['tavg'].mean():.1f}C  "
                  f"missing={cdf[cfg['data']['features']].isna().sum().sum()}")

    preprocess(df, cfg, today)
    print(f"\n[DONE] Data ready. Next: python src/train.py")
    print(f"  Cached:  {os.path.join(DATA_DIR, 'all_raw.csv')}")
    print(f"  Meta:    {META_PATH}")


if __name__ == "__main__":
    main()

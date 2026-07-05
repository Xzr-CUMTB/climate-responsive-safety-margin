from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.utils.project import ensure_dir, log_event, rel


WQ_FILE = ROOT / "data" / "raw" / "water_quality" / "full_dataset.csv"
CIPS_OBJECTS = ROOT / "data" / "processed" / "CIPs_objects" / "cips_objects.parquet"
OUT_DIR = ROOT / "data" / "processed" / "water_quality"
PARK_WQ = OUT_DIR / "park_water_quality_state_2000_2020.parquet"
PARK_STATION = OUT_DIR / "park_nearest_water_station.csv"
STATION_STATS = OUT_DIR / "water_quality_station_indicator_stats.csv"
REPORT = ROOT / "outputs" / "reports" / "WATER_QUALITY_PREPROCESS_REPORT.md"


HIGH_BAD = {
    "BOD",
    "COD",
    "CODMn",
    "DIN",
    "DIP",
    "DOC",
    "NH4N",
    "NO2N",
    "NO3N",
    "TDP",
    "TOC",
    "TP",
    "TPH",
    "TSSs",
}
LOW_BAD = {"DO", "DOSAT"}
EPS = 1e-6


def haversine_km(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    r = 6371.0088
    lon1 = np.deg2rad(lon1)
    lat1 = np.deg2rad(lat1)
    lon2 = np.deg2rad(lon2)
    lat2 = np.deg2rad(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def load_water_quality() -> pd.DataFrame:
    if not WQ_FILE.exists():
        raise SystemExit(f"Missing water-quality file: {WQ_FILE}")
    usecols = [
        "MonitoringLocationIdentifier",
        "LongitudeMeasure_WGS84",
        "LatitudeMeasure_WGS84",
        "MonitoringDate",
        "IndicatorsName",
        "Value",
        "Unit",
        "SourceProvider",
    ]
    df = pd.read_csv(WQ_FILE, usecols=usecols)
    df = df.rename(
        columns={
            "MonitoringLocationIdentifier": "station_id",
            "LongitudeMeasure_WGS84": "station_lon",
            "LatitudeMeasure_WGS84": "station_lat",
            "MonitoringDate": "date",
            "IndicatorsName": "indicator",
            "Value": "value",
            "Unit": "unit",
            "SourceProvider": "source_provider",
        }
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed").dt.normalize()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["station_id", "station_lon", "station_lat", "date", "indicator", "value"])
    df = df[df["indicator"].isin(HIGH_BAD | LOW_BAD)].copy()
    return df


def compute_indicator_stress(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = (
        df.groupby(["station_id", "indicator"])["value"]
        .quantile([0.05, 0.50, 0.95])
        .unstack()
        .rename(columns={0.05: "q05", 0.50: "q50", 0.95: "q95"})
        .reset_index()
    )
    out = df.merge(stats, on=["station_id", "indicator"], how="left")
    high_mask = out["indicator"].isin(HIGH_BAD)
    low_mask = out["indicator"].isin(LOW_BAD)
    stress = np.zeros(len(out), dtype=np.float32)
    stress[high_mask.to_numpy()] = (
        (out.loc[high_mask, "value"] - out.loc[high_mask, "q50"])
        / (out.loc[high_mask, "q95"] - out.loc[high_mask, "q50"] + EPS)
    ).clip(lower=0).astype("float32")
    stress[low_mask.to_numpy()] = (
        (out.loc[low_mask, "q50"] - out.loc[low_mask, "value"])
        / (out.loc[low_mask, "q50"] - out.loc[low_mask, "q05"] + EPS)
    ).clip(lower=0).astype("float32")
    out["indicator_stress"] = stress
    stats.to_csv(STATION_STATS, index=False, encoding="utf-8-sig")
    return out, stats


def build_station_daily(obs: pd.DataFrame, station_ids: np.ndarray) -> pd.DataFrame:
    daily_obs = (
        obs[obs["station_id"].isin(station_ids)]
        .groupby(["station_id", "date"], as_index=False)
        .agg(
            receptor_stress_obs=("indicator_stress", "max"),
            indicators_observed=("indicator", lambda x: ";".join(sorted(set(map(str, x))))),
        )
    )
    date_index = pd.date_range("2000-01-01", "2020-12-31", freq="D")
    frames = []
    for station_id, group in daily_obs.groupby("station_id"):
        s = group.set_index("date").sort_index()
        stress = s["receptor_stress_obs"].reindex(date_index)
        present = stress.notna().astype("int16")
        rolling_stress = stress.rolling(30, min_periods=1).max()
        rolling_count = present.rolling(30, min_periods=1).sum()
        frame = pd.DataFrame(
            {
                "station_id": station_id,
                "date": date_index,
                "water_receptor_stress": rolling_stress.fillna(0).astype("float32").to_numpy(),
                "wq_recent_obs_count_30d": rolling_count.astype("int16").to_numpy(),
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def match_parks_to_stations(objects: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    park_lon = objects["centroid_lon"].to_numpy()[:, None]
    park_lat = objects["centroid_lat"].to_numpy()[:, None]
    st_lon = stations["station_lon"].to_numpy()[None, :]
    st_lat = stations["station_lat"].to_numpy()[None, :]
    dist = haversine_km(park_lon, park_lat, st_lon, st_lat)
    nearest = dist.argmin(axis=1)
    matched = objects[["component_id", "centroid_lon", "centroid_lat"]].rename(columns={"component_id": "park_id"}).copy()
    matched["station_id"] = stations.iloc[nearest]["station_id"].to_numpy()
    matched["station_lon"] = stations.iloc[nearest]["station_lon"].to_numpy()
    matched["station_lat"] = stations.iloc[nearest]["station_lat"].to_numpy()
    matched["water_station_distance_km"] = dist[np.arange(len(objects)), nearest].astype("float32")
    return matched


def write_report(obs: pd.DataFrame, matched: pd.DataFrame, park_daily: pd.DataFrame) -> None:
    ensure_dir(REPORT.parent)
    recent_share = (park_daily["wq_recent_obs_count_30d"] > 0).mean()
    lines = [
        "# WATER_QUALITY_PREPROCESS_REPORT",
        "",
        f"- Source file: `{rel(WQ_FILE)}`",
        f"- Stress-capable observations retained: {len(obs):,}",
        f"- Matched parks: {matched['park_id'].nunique():,}",
        f"- Unique matched monitoring stations: {matched['station_id'].nunique():,}",
        f"- Median park-to-station distance: {matched['water_station_distance_km'].median():.2f} km",
        f"- Daily park receptor-state rows: {len(park_daily):,}",
        f"- Share of park-days with at least one observation in prior 30 days: {recent_share:.3f}",
        "",
        "## Indicator Direction",
        "",
        f"- High-is-worse indicators: {', '.join(sorted(HIGH_BAD))}",
        f"- Low-is-worse indicators: {', '.join(sorted(LOW_BAD))}",
        "",
        "## Scope",
        "",
        "Water quality is used as external receptor-state context and consistency evidence. The generated receptor stress is not causal evidence of chemical-park emissions and is not accident validation.",
        "",
        "## Outputs",
        "",
        f"- `{rel(PARK_WQ)}`",
        f"- `{rel(PARK_STATION)}`",
        f"- `{rel(STATION_STATS)}`",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not CIPS_OBJECTS.exists():
        raise SystemExit(f"Missing CIPs objects: {CIPS_OBJECTS}")
    ensure_dir(OUT_DIR)
    log_event("03_preprocess_water_quality", "started")
    objects = pd.read_parquet(CIPS_OBJECTS)
    wq = load_water_quality()
    obs, _ = compute_indicator_stress(wq)
    stations = (
        wq.groupby("station_id", as_index=False)
        .agg(station_lon=("station_lon", "median"), station_lat=("station_lat", "median"))
    )
    matched = match_parks_to_stations(objects, stations)
    matched.to_csv(PARK_STATION, index=False, encoding="utf-8-sig")
    station_daily = build_station_daily(obs, matched["station_id"].unique())
    park_daily = matched[["park_id", "station_id", "water_station_distance_km"]].merge(
        station_daily, on="station_id", how="left"
    )
    park_daily["park_id"] = park_daily["park_id"].astype("int32")
    park_daily.to_parquet(PARK_WQ, index=False)
    write_report(obs, matched, park_daily)
    log_event("03_preprocess_water_quality", f"wrote {len(park_daily):,} rows to {rel(PARK_WQ)}")


if __name__ == "__main__":
    main()


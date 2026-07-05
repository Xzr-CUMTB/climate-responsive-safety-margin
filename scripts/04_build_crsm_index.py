from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.utils.project import ensure_dir, log_event, rel


PARK_DAILY = ROOT / "data" / "processed" / "park_daily_table" / "park_daily_2000_2020.parquet"
PARK_WQ = ROOT / "data" / "processed" / "water_quality" / "park_water_quality_state_2000_2020.parquet"
OUT_DIR = ROOT / "data" / "processed" / "C_RSM"
OUT_FILE = OUT_DIR / "c_rsm_daily.parquet"
THRESHOLDS = ROOT / "outputs" / "tables" / "crsm_thresholds_by_park.csv"
SUMMARY = ROOT / "outputs" / "tables" / "crsm_baseline_summary.csv"
COMPONENT_QUANTILES = ROOT / "outputs" / "tables" / "crsm_component_quantiles.csv"
REPORT = ROOT / "outputs" / "reports" / "CRSM_BUILD_REPORT.md"


ETA = 0.3
EA_J_MOL = 60_000.0
R_J_MOL_K = 8.314462618
T0_K = 298.15
EPS = 1e-6
EPS_WIND_MS = 0.1
RAW_COMPONENTS = ["S_T", "S_P", "S_V", "S_W"]
DEMAND_COMPONENTS = ["D_T", "D_P", "D_V", "D_W"]
DENOM_FLOORS = {"S_T": EPS, "S_P": EPS, "S_V": EPS, "S_W": 1.0}


def compute_raw_stress(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["park_id", "date"]).copy()
    area_factor = np.power(np.maximum(df["park_area_km2"].to_numpy(dtype=np.float64), EPS), ETA)

    temp_k = df["tmax_c"].to_numpy(dtype=np.float64) + 273.15
    exponent = (EA_J_MOL / R_J_MOL_K) * ((1.0 / T0_K) - (1.0 / np.maximum(temp_k, EPS)))
    df["S_T"] = (area_factor * np.maximum(np.exp(exponent) - 1.0, 0.0)).astype("float32")

    df["precip_3d_mm"] = (
        df.groupby("park_id", sort=False)["precip_mm"]
        .rolling(3, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .astype("float32")
    )
    precip_q95 = df.groupby("park_id")["precip_3d_mm"].transform(lambda x: x.quantile(0.95)).astype("float32")
    df["S_P"] = (
        area_factor * df["precip_3d_mm"].to_numpy(dtype=np.float64) / (precip_q95.to_numpy(dtype=np.float64) + EPS)
    ).astype("float32")

    wind_q05 = df.groupby("park_id")["wind_ms"].transform(lambda x: x.quantile(0.05)).astype("float32")
    df["S_V"] = (
        area_factor
        * (wind_q05.to_numpy(dtype=np.float64) + EPS_WIND_MS)
        / (df["wind_ms"].to_numpy(dtype=np.float64) + EPS_WIND_MS)
    ).astype("float32")

    # scripts/03_preprocess_water_quality.py already implements the station-level
    # max_m [(C-Q50)/(Q95-Q50+eps)]_+ receptor-state transform.
    df["S_W"] = df["water_receptor_stress"].fillna(0).astype("float32")
    return df


def normalize_demands(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    q95 = (
        df.groupby("park_id")[RAW_COMPONENTS]
        .quantile(0.95)
        .rename(columns={c: f"q95_{c}" for c in RAW_COMPONENTS})
        .reset_index()
    )
    raw_aux = (
        df.groupby("park_id")
        .agg(
            precip_3d_q95_mm=("precip_3d_mm", lambda x: x.quantile(0.95)),
            wind_q05_ms=("wind_ms", lambda x: x.quantile(0.05)),
            n_baseline_days=("date", "size"),
        )
        .reset_index()
    )
    theta = q95.merge(raw_aux, on="park_id", how="left")
    theta["eta"] = ETA
    theta["Ea_J_mol"] = EA_J_MOL
    theta["T0_K"] = T0_K
    theta["eps"] = EPS
    theta["eps_wind_ms"] = EPS_WIND_MS
    theta["normalization_basis"] = "D_i,k,t = S_i,k,t / (Q0.95(S_i,k,baseline) + eps)"
    for raw in RAW_COMPONENTS:
        theta[f"denom_{raw}"] = np.maximum(theta[f"q95_{raw}"].to_numpy(dtype=np.float64), DENOM_FLOORS[raw])

    df = df.merge(theta, on="park_id", how="left")
    for raw, demand in zip(RAW_COMPONENTS, DEMAND_COMPONENTS):
        denom = df[f"denom_{raw}"].to_numpy(dtype=np.float64) + EPS
        df[demand] = (df[raw].to_numpy(dtype=np.float64) / denom).astype("float32")

    margin_cols = []
    for demand, suffix in zip(DEMAND_COMPONENTS, ["T", "P", "V", "W"]):
        col = f"margin_{suffix}"
        df[col] = (1.0 - df[demand].to_numpy(dtype=np.float64)).astype("float32")
        margin_cols.append(col)

    margins = df[margin_cols].to_numpy(dtype=np.float32)
    idx = margins.argmin(axis=1)
    drivers = np.array(["T", "P", "V", "W"])
    df["C_RSM"] = margins[np.arange(len(df)), idx].astype("float32")
    df["dominant_driver"] = drivers[idx]
    df["margin_compression_episode"] = df["C_RSM"] < 0
    return df, theta


def write_summaries(df: pd.DataFrame, theta: pd.DataFrame) -> None:
    ensure_dir(THRESHOLDS.parent)
    theta.to_csv(THRESHOLDS, index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(
        [
            {
                "n_parks": df["park_id"].nunique(),
                "n_days": df["date"].nunique(),
                "n_rows": len(df),
                "mean_c_rsm": df["C_RSM"].mean(),
                "median_c_rsm": df["C_RSM"].median(),
                "p05_c_rsm": df["C_RSM"].quantile(0.05),
                "collapse_day_share": df["margin_compression_episode"].mean(),
                "wq_recent_obs_share": (df["wq_recent_obs_count_30d"] > 0).mean(),
                "mean_D_T": df["D_T"].mean(),
                "mean_D_P": df["D_P"].mean(),
                "mean_D_V": df["D_V"].mean(),
                "mean_D_W": df["D_W"].mean(),
            }
        ]
    )
    driver_share = df["dominant_driver"].value_counts(normalize=True).rename_axis("driver").reset_index(name="share")
    quantiles = []
    for col in RAW_COMPONENTS + DEMAND_COMPONENTS + ["C_RSM"]:
        quantiles.append(
            {
                "variable": col,
                "q01": df[col].quantile(0.01),
                "q05": df[col].quantile(0.05),
                "q50": df[col].quantile(0.50),
                "q95": df[col].quantile(0.95),
                "q99": df[col].quantile(0.99),
                "mean": df[col].mean(),
            }
        )
    summary.to_csv(SUMMARY, index=False, encoding="utf-8-sig")
    driver_share.to_csv(ROOT / "outputs" / "tables" / "crsm_driver_share_baseline.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(quantiles).to_csv(COMPONENT_QUANTILES, index=False, encoding="utf-8-sig")


def write_report(df: pd.DataFrame) -> None:
    ensure_dir(REPORT.parent)
    lines = [
        "# CRSM_BUILD_REPORT",
        "",
        f"- Output: `{rel(OUT_FILE)}`",
        f"- Rows: {len(df):,}",
        f"- Parks: {df['park_id'].nunique():,}",
        f"- Date range: {df['date'].min().date()} to {df['date'].max().date()}",
        f"- Mean C-RSM: {df['C_RSM'].mean():.4f}",
        f"- Median C-RSM: {df['C_RSM'].median():.4f}",
        f"- Margin-compression row share: {df['margin_compression_episode'].mean():.4f}",
        "",
        "## Formula Implemented",
        "",
        "This rebuild follows the prompt-specified two-step formulation: first compute raw forcing stresses `S_T`, `S_P`, `S_V` and `S_W`; then normalize each park/component by its historical 2000-2020 baseline 95th percentile, `D_i,k,t = S_i,k,t / (Q0.95(S_i,k,.) + eps)`. The reported margin is `C-RSM_i,t = min_k[1 - D_i,k,t]` over `k={T,P,V,W}`.",
        "",
        "Thermal stress uses the Arrhenius sensitivity term. Rainfall-runoff stress uses rolling 3-day precipitation normalized by the park's historical 95th percentile 3-day rainfall. Low-wind dispersion stress uses the park's 5th percentile wind speed and a 0.1 m s^-1 numerical offset. Water-quality receptor stress is imported from the station-level indicator transform in `scripts/03_preprocess_water_quality.py`.",
        "",
        "Because water-quality observations are sparse, 212 parks have `Q0.95(S_W)=0` under the rolling receptor-state series. For this component only, the effective denominator is `max(Q0.95(S_W), 1.0)`. This keeps receptor-state stress on its original station-quantile scale and prevents a zero-denominator artefact from dominating the margin.",
        "",
        "## Interpretation Guardrail",
        "",
        "`C-RSM < 0` denotes a relative margin-compression episode under this open-data screening boundary. It is not an accident label, not a plant-level HAZOP/LOPA replacement and not post-2021 real-time risk.",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not PARK_DAILY.exists():
        raise SystemExit(f"Missing CDMet park table: {PARK_DAILY}")
    if not PARK_WQ.exists():
        raise SystemExit(f"Missing water-quality park table: {PARK_WQ}")
    ensure_dir(OUT_DIR)
    log_event("04_build_crsm_index", "started")
    daily = pd.read_parquet(PARK_DAILY)
    wq = pd.read_parquet(PARK_WQ)
    df = daily.merge(
        wq[["park_id", "date", "station_id", "water_station_distance_km", "water_receptor_stress", "wq_recent_obs_count_30d"]],
        on=["park_id", "date"],
        how="left",
    )
    df["water_receptor_stress"] = df["water_receptor_stress"].fillna(0).astype("float32")
    df["wq_recent_obs_count_30d"] = df["wq_recent_obs_count_30d"].fillna(0).astype("int16")
    df = compute_raw_stress(df)
    df, theta = normalize_demands(df)
    df.to_parquet(OUT_FILE, index=False)
    write_summaries(df, theta)
    write_report(df)
    log_event("04_build_crsm_index", f"wrote {len(df):,} rows to {rel(OUT_FILE)}")


if __name__ == "__main__":
    main()

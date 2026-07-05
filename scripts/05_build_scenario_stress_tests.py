from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.project import ensure_dir, log_event, rel


BASELINE = ROOT / "data" / "processed" / "C_RSM" / "c_rsm_daily.parquet"
THRESHOLDS = ROOT / "outputs" / "tables" / "crsm_thresholds_by_park.csv"
OUT_DIR = ROOT / "data" / "processed" / "C_RSM"
SCENARIO_DAILY = OUT_DIR / "scenario_c_rsm_daily.parquet"
SUMMARY_BY_PARK = OUT_DIR / "scenario_summary_by_park.csv"
SUMMARY_OVERALL = OUT_DIR / "scenario_summary_overall.csv"
TABLE_RESULTS = ROOT / "outputs" / "tables" / "scenario_stress_test_results.csv"
REPORT = ROOT / "outputs" / "reports" / "SCENARIO_STRESS_TEST_REPORT.md"


ETA = 0.3
EA_J_MOL = 60_000.0
R_J_MOL_K = 8.314462618
T0_K = 298.15
EPS = 1e-6
EPS_WIND_MS = 0.1


SCENARIOS = {
    "S0": {"label": "historical 2000-2020 design-basis baseline", "delta_t": 0.0, "precip_mult": 1.0, "wind_mult": 1.0},
    "S1": {"label": "Tmax +1 degC", "delta_t": 1.0, "precip_mult": 1.0, "wind_mult": 1.0},
    "S2": {"label": "Tmax +2 degC", "delta_t": 2.0, "precip_mult": 1.0, "wind_mult": 1.0},
    "S3": {"label": "precipitation +10%", "delta_t": 0.0, "precip_mult": 1.10, "wind_mult": 1.0},
    "S4": {"label": "precipitation +20%", "delta_t": 0.0, "precip_mult": 1.20, "wind_mult": 1.0},
    "S5": {"label": "low-wind enhancement, wind speed -10%", "delta_t": 0.0, "precip_mult": 1.0, "wind_mult": 0.90},
    "S6": {"label": "compound Tmax +2 degC, precipitation +20%, wind speed -10%", "delta_t": 2.0, "precip_mult": 1.20, "wind_mult": 0.90},
}


def scenario_components(base: pd.DataFrame, spec: dict[str, float], theta: pd.DataFrame) -> pd.DataFrame:
    cols = ["park_id", "date", "park_area_km2", "tmax_c", "precip_3d_mm", "wind_ms", "S_W"]
    df = base[cols].copy()
    df = df.merge(
        theta[["park_id", "precip_3d_q95_mm", "wind_q05_ms", "denom_S_T", "denom_S_P", "denom_S_V", "denom_S_W"]],
        on="park_id",
        how="left",
    )
    area_factor = np.power(np.maximum(df["park_area_km2"].to_numpy(dtype=np.float64), EPS), ETA)

    temp_c = df["tmax_c"].to_numpy(dtype=np.float64) + spec["delta_t"]
    temp_k = temp_c + 273.15
    exponent = (EA_J_MOL / R_J_MOL_K) * ((1.0 / T0_K) - (1.0 / np.maximum(temp_k, EPS)))
    s_t = area_factor * np.maximum(np.exp(exponent) - 1.0, 0.0)

    precip_3d = df["precip_3d_mm"].to_numpy(dtype=np.float64) * spec["precip_mult"]
    s_p = area_factor * precip_3d / (df["precip_3d_q95_mm"].to_numpy(dtype=np.float64) + EPS)

    wind = df["wind_ms"].to_numpy(dtype=np.float64) * spec["wind_mult"]
    s_v = area_factor * (df["wind_q05_ms"].to_numpy(dtype=np.float64) + EPS_WIND_MS) / (wind + EPS_WIND_MS)
    s_w = df["S_W"].to_numpy(dtype=np.float64)

    raw = np.column_stack([s_t, s_p, s_v, s_w]).astype("float32")
    denom = df[["denom_S_T", "denom_S_P", "denom_S_V", "denom_S_W"]].to_numpy(dtype=np.float64) + EPS
    demand = (raw.astype(np.float64) / denom).astype("float32")
    margins = (1.0 - demand).astype("float32")
    idx = margins.argmin(axis=1)
    drivers = np.array(["T", "P", "V", "W"])
    out = pd.DataFrame(
        {
            "park_id": df["park_id"].to_numpy(dtype=np.int32),
            "date": df["date"].to_numpy(),
            "S_T": raw[:, 0],
            "S_P": raw[:, 1],
            "S_V": raw[:, 2],
            "S_W": raw[:, 3],
            "D_T": demand[:, 0],
            "D_P": demand[:, 1],
            "D_V": demand[:, 2],
            "D_W": demand[:, 3],
            "C_RSM": margins[np.arange(len(df)), idx],
            "dominant_driver": drivers[idx],
        }
    )
    out["margin_compression_episode"] = out["C_RSM"] < 0
    return out


def summarize_scenario(
    df: pd.DataFrame,
    scenario: str,
    label: str,
    baseline_by_park: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    by_park = (
        df.groupby("park_id")
        .agg(
            mean_c_rsm=("C_RSM", "mean"),
            p05_c_rsm=("C_RSM", lambda x: x.quantile(0.05)),
            collapse_days=("margin_compression_episode", "sum"),
            collapse_day_share=("margin_compression_episode", "mean"),
            dominant_T_share=("dominant_driver", lambda x: (x == "T").mean()),
            dominant_P_share=("dominant_driver", lambda x: (x == "P").mean()),
            dominant_V_share=("dominant_driver", lambda x: (x == "V").mean()),
            dominant_W_share=("dominant_driver", lambda x: (x == "W").mean()),
        )
        .reset_index()
    )
    by_park.insert(0, "scenario", scenario)
    by_park.insert(1, "scenario_label", label)
    if baseline_by_park is not None:
        merged = by_park.merge(
            baseline_by_park[["park_id", "mean_c_rsm", "collapse_days"]].rename(
                columns={"mean_c_rsm": "baseline_mean_c_rsm", "collapse_days": "baseline_collapse_days"}
            ),
            on="park_id",
            how="left",
        )
        by_park["delta_mean_c_rsm_vs_S0"] = merged["mean_c_rsm"] - merged["baseline_mean_c_rsm"]
        by_park["delta_collapse_days_vs_S0"] = merged["collapse_days"] - merged["baseline_collapse_days"]
    else:
        by_park["delta_mean_c_rsm_vs_S0"] = 0.0
        by_park["delta_collapse_days_vs_S0"] = 0

    driver_share = df["dominant_driver"].value_counts(normalize=True)
    overall = {
        "scenario": scenario,
        "scenario_label": label,
        "mean_c_rsm": float(df["C_RSM"].mean()),
        "median_c_rsm": float(df["C_RSM"].median()),
        "p05_c_rsm": float(df["C_RSM"].quantile(0.05)),
        "collapse_day_share": float(df["margin_compression_episode"].mean()),
        "dominant_T_share": float(driver_share.get("T", 0.0)),
        "dominant_P_share": float(driver_share.get("P", 0.0)),
        "dominant_V_share": float(driver_share.get("V", 0.0)),
        "dominant_W_share": float(driver_share.get("W", 0.0)),
    }
    return by_park, overall


def add_elasticities(by_park: pd.DataFrame) -> pd.DataFrame:
    by_park = by_park.copy()
    by_park["elasticity_type"] = ""
    by_park["scenario_elasticity"] = np.nan
    masks = {
        "S1": ("temperature_degC", 1.0),
        "S2": ("temperature_degC", 2.0),
        "S3": ("precipitation_fraction", 0.10),
        "S4": ("precipitation_fraction", 0.20),
        "S5": ("wind_fraction", -0.10),
    }
    for scenario, (etype, denom) in masks.items():
        mask = by_park["scenario"] == scenario
        by_park.loc[mask, "elasticity_type"] = etype
        by_park.loc[mask, "scenario_elasticity"] = by_park.loc[mask, "delta_mean_c_rsm_vs_S0"] / denom
    by_park.loc[by_park["scenario"] == "S6", "elasticity_type"] = "compound_not_single_elasticity"
    return by_park


def write_report(overall: pd.DataFrame) -> None:
    ensure_dir(REPORT.parent)
    s0 = overall[overall["scenario"] == "S0"].iloc[0]
    s6 = overall[overall["scenario"] == "S6"].iloc[0]
    lines = [
        "# SCENARIO_STRESS_TEST_REPORT",
        "",
        f"- Daily output: `{rel(SCENARIO_DAILY)}`",
        f"- Park summary: `{rel(SUMMARY_BY_PARK)}`",
        f"- Overall summary: `{rel(SUMMARY_OVERALL)}`",
        f"- S0 mean C-RSM: {s0['mean_c_rsm']:.4f}",
        f"- S6 mean C-RSM: {s6['mean_c_rsm']:.4f}",
        f"- S6 collapse-day share: {s6['collapse_day_share']:.4f}",
        "",
        "## Scenario Definitions",
        "",
    ]
    for scenario, spec in SCENARIOS.items():
        lines.append(f"- {scenario}: {spec['label']}")
    lines.extend(
        [
            "",
            "## Formula Implemented",
            "",
            "Scenario perturbations are applied to the historical 2000-2020 forcing baseline as `T'=T+deltaT`, `P'=P(1+deltaP)` and `u'=u(1-deltau)`. Raw scenario stresses are then normalized against the baseline `Q0.95(S_i,k,.)` denominators, not against each scenario's own distribution. This preserves analogue-stress sensitivity.",
            "",
            "The water-quality receptor-state denominator uses the same sparse-data guardrail as the baseline build: `max(Q0.95(S_W), 1.0)`.",
            "",
            "## Interpretation Guardrail",
            "",
            "These are climate analogue stress tests. They perturb the historical design-basis forcing baseline and do not represent true future climate projections or post-2021 real-time chemical-park risk.",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not BASELINE.exists():
        raise SystemExit(f"Missing baseline C-RSM file: {BASELINE}")
    if not THRESHOLDS.exists():
        raise SystemExit(f"Missing threshold table: {THRESHOLDS}")
    ensure_dir(OUT_DIR)
    ensure_dir(TABLE_RESULTS.parent)
    log_event("05_build_scenario_stress_tests", "started")
    base = pd.read_parquet(BASELINE).sort_values(["park_id", "date"])
    theta = pd.read_csv(THRESHOLDS)

    if SCENARIO_DAILY.exists():
        SCENARIO_DAILY.unlink()
    writer = None
    summaries = []
    overall_rows = []
    baseline_by_park = None
    try:
        for scenario, spec in SCENARIOS.items():
            scen = scenario_components(base, spec, theta)
            scen.insert(0, "scenario", scenario)
            scen.insert(1, "scenario_label", spec["label"])
            by_park, overall = summarize_scenario(scen, scenario, spec["label"], baseline_by_park)
            if scenario == "S0":
                baseline_by_park = by_park.copy()
            summaries.append(by_park)
            overall_rows.append(overall)
            table = pa.Table.from_pandas(scen, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(SCENARIO_DAILY, table.schema, compression="snappy")
            writer.write_table(table)
            log_event("05_build_scenario_stress_tests", f"wrote scenario={scenario} rows={len(scen):,}")
    finally:
        if writer is not None:
            writer.close()

    by_park_all = pd.concat(summaries, ignore_index=True)
    by_park_all = add_elasticities(by_park_all)
    overall = pd.DataFrame(overall_rows)
    s0 = overall[overall["scenario"] == "S0"].iloc[0]
    overall["delta_mean_c_rsm_vs_S0"] = overall["mean_c_rsm"] - float(s0["mean_c_rsm"])
    overall["delta_collapse_day_share_vs_S0"] = overall["collapse_day_share"] - float(s0["collapse_day_share"])
    by_park_all.to_csv(SUMMARY_BY_PARK, index=False, encoding="utf-8-sig")
    overall.to_csv(SUMMARY_OVERALL, index=False, encoding="utf-8-sig")
    overall.to_csv(TABLE_RESULTS, index=False, encoding="utf-8-sig")
    write_report(overall)
    log_event("05_build_scenario_stress_tests", "completed S0-S6 scenario stress tests")


if __name__ == "__main__":
    main()

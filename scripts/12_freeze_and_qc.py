from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_pinball_loss,
    mean_squared_error,
    r2_score,
    recall_score,
    roc_auc_score,
)


REPORT_DIR = ROOT / "outputs" / "reports"
TABLE_DIR = ROOT / "outputs" / "tables"
FIGURE_DIR = ROOT / "figures"

REQUIRED_FILES = [
    ROOT / "data" / "processed" / "CIPs_objects" / "cips_objects.parquet",
    ROOT / "data" / "processed" / "CIPs_objects" / "cips_summary.csv",
    ROOT / "data" / "processed" / "park_daily_table" / "park_daily_2000_2020.parquet",
    ROOT / "data" / "processed" / "C_RSM" / "c_rsm_daily.parquet",
    ROOT / "data" / "processed" / "C_RSM" / "scenario_c_rsm_daily.parquet",
    ROOT / "data" / "processed" / "C_RSM" / "scenario_summary_by_park.csv",
    ROOT / "data" / "processed" / "C_RSM" / "scenario_summary_overall.csv",
    ROOT / "outputs" / "models" / "crsmamba_best.pt",
    ROOT / "outputs" / "tables" / "model_performance_main.csv",
    ROOT / "outputs" / "tables" / "model_performance_by_horizon.csv",
    ROOT / "outputs" / "tables" / "ablation_results.csv",
    ROOT / "outputs" / "tables" / "calibration_results.csv",
    ROOT / "outputs" / "tables" / "scenario_stress_test_results.csv",
    ROOT / "REFERENCES_AUDIT.csv",
]

CIPS_OBJECTS = ROOT / "data" / "processed" / "CIPs_objects" / "cips_objects.parquet"
CIPS_SUMMARY = ROOT / "data" / "processed" / "CIPs_objects" / "cips_summary.csv"
PARK_DAILY = ROOT / "data" / "processed" / "park_daily_table" / "park_daily_2000_2020.parquet"
PARK_WQ = ROOT / "data" / "processed" / "water_quality" / "park_water_quality_state_2000_2020.parquet"
CRSM_DAILY = ROOT / "data" / "processed" / "C_RSM" / "c_rsm_daily.parquet"
SCENARIO_DAILY = ROOT / "data" / "processed" / "C_RSM" / "scenario_c_rsm_daily.parquet"
SCENARIO_OVERALL = ROOT / "data" / "processed" / "C_RSM" / "scenario_summary_overall.csv"
SCENARIO_BY_PARK = ROOT / "data" / "processed" / "C_RSM" / "scenario_summary_by_park.csv"
THRESHOLDS = ROOT / "outputs" / "tables" / "crsm_thresholds_by_park.csv"
PREDICTIONS = ROOT / "data" / "processed" / "modeling" / "test_predictions.parquet"
FEATURE_SCHEMA = ROOT / "data" / "processed" / "modeling" / "feature_schema.txt"
SCALER_META = ROOT / "data" / "processed" / "modeling" / "feature_scaler_main.json"

EPS = 1e-6
ETA = 0.3
EA_J_MOL = 60_000.0
R_J_MOL_K = 8.314462618
T0_K = 298.15
EPS_WIND_MS = 0.1
SEQ_LEN = 30
HORIZONS = [1, 3, 7]
QUANTILES = [0.10, 0.50, 0.90]
SPLITS = {
    "main": {
        "train_start": "2000-01-01",
        "train_end": "2015-12-31",
        "val_start": "2016-01-01",
        "val_end": "2017-12-31",
        "test_start": "2018-01-01",
        "test_end": "2020-12-31",
    },
    "pseudo_prospective": {
        "train_start": "2000-01-01",
        "train_end": "2017-12-31",
        "val_start": "2018-01-01",
        "val_end": "2019-12-31",
        "test_start": "2020-01-01",
        "test_end": "2020-12-31",
    },
}


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def clean_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return None if math.isnan(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, dict):
        return {str(k): clean_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_value(v) for v in value]
    return value


def md_table(rows: list[dict[str, Any]], columns: list[str] | None = None, max_cell: int = 120) -> str:
    if not rows:
        return "_No rows._"
    columns = columns or list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.6g}"
            text = str(value).replace("\n", " ")
            if len(text) > max_cell:
                text = text[: max_cell - 3] + "..."
            cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def flatten_json(prefix: str, obj: Any, rows: list[dict[str, Any]]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            flatten_json(f"{prefix}.{key}" if prefix else str(key), value, rows)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            flatten_json(f"{prefix}[{i}]", value, rows)
    else:
        rows.append({"metric_path": prefix, "value": obj})


def count_csv_rows(path: Path) -> int | None:
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace") as fh:
            n = sum(1 for _ in fh)
        return max(n - 1, 0)
    except Exception:
        return None


def parquet_date_minmax(path: Path, date_col: str = "date") -> tuple[str | None, str | None]:
    try:
        pf = pq.ParquetFile(path)
        if date_col not in pf.schema_arrow.names:
            return None, None
        mins = []
        maxs = []
        for batch in pf.iter_batches(columns=[date_col], batch_size=750_000):
            s = batch.to_pandas()[date_col]
            if len(s):
                mins.append(pd.to_datetime(s).min())
                maxs.append(pd.to_datetime(s).max())
        if not mins:
            return None, None
        return min(mins).date().isoformat(), max(maxs).date().isoformat()
    except Exception:
        return None, None


def audit_table_file(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "file_path": rel(path),
        "exists": path.exists(),
        "file_size_bytes": path.stat().st_size if path.exists() else None,
        "readable": False,
        "rows": None,
        "columns": "",
        "date_range": "",
        "critical_problems": "",
    }
    if not path.exists():
        info["critical_problems"] = "MISSING"
        return info
    try:
        if path.suffix.lower() == ".parquet":
            pf = pq.ParquetFile(path)
            columns = pf.schema_arrow.names
            info["rows"] = int(pf.metadata.num_rows)
            info["columns"] = ", ".join(columns)
            start, end = parquet_date_minmax(path) if "date" in columns else (None, None)
            if start and end:
                info["date_range"] = f"{start} to {end}"
        elif path.suffix.lower() == ".csv":
            head = pd.read_csv(path, nrows=5)
            info["rows"] = count_csv_rows(path)
            info["columns"] = ", ".join(map(str, head.columns))
            for candidate in ["date", "target_date"]:
                if candidate in head.columns:
                    dates = pd.read_csv(path, usecols=[candidate])
                    dates[candidate] = pd.to_datetime(dates[candidate], errors="coerce")
                    info["date_range"] = f"{dates[candidate].min().date()} to {dates[candidate].max().date()}"
                    break
        else:
            with path.open("rb") as fh:
                fh.read(16)
        info["readable"] = True
    except Exception as exc:
        info["critical_problems"] = f"UNREADABLE: {type(exc).__name__}: {exc}"
    return info


def derive_cips_summary_if_missing() -> bool:
    if CIPS_SUMMARY.exists() or not CIPS_OBJECTS.exists():
        return False
    cips = pd.read_parquet(CIPS_OBJECTS)
    summary = pd.DataFrame(
        [
            {
                "n_cips_objects": len(cips),
                "total_area_km2": cips["park_area_km2"].sum(),
                "median_area_km2": cips["park_area_km2"].median(),
                "p05_area_km2": cips["park_area_km2"].quantile(0.05),
                "p95_area_km2": cips["park_area_km2"].quantile(0.95),
                "total_pixels": cips["pixel_count"].sum() if "pixel_count" in cips.columns else np.nan,
                "geometry_representation": ";".join(sorted(map(str, cips["geometry_representation"].dropna().unique())))
                if "geometry_representation" in cips.columns
                else "",
                "derived_by": "scripts/12_freeze_and_qc.py",
            }
        ]
    )
    summary.to_csv(CIPS_SUMMARY, index=False, encoding="utf-8-sig")
    return True


def file_audit() -> tuple[list[dict[str, Any]], bool]:
    derived_summary = derive_cips_summary_if_missing()
    rows = [audit_table_file(path) for path in REQUIRED_FILES]
    lines = [
        "# EXPERIMENT_FILE_AUDIT",
        "",
        f"- Project root: `{ROOT}`",
        f"- Audit mode: read-only experiment validation, except derived missing `cips_summary.csv`: `{derived_summary}`",
        f"- Critical missing/unreadable files: {sum((not r['exists']) or (not r['readable']) for r in rows)}",
        "",
        md_table(rows, ["file_path", "exists", "file_size_bytes", "readable", "rows", "columns", "date_range", "critical_problems"], max_cell=180),
        "",
        "## Notes",
        "",
        "- File sizes and row counts are read from filesystem metadata, CSV line counts or parquet metadata.",
        "- Date ranges are read from table date columns when present.",
    ]
    (REPORT_DIR / "EXPERIMENT_FILE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    return rows, derived_summary


def pinball_mean(y: np.ndarray, q10: np.ndarray, q50: np.ndarray, q90: np.ndarray) -> tuple[float, float, float, float]:
    p10 = float(mean_pinball_loss(y, q10, alpha=0.10))
    p50 = float(mean_pinball_loss(y, q50, alpha=0.50))
    p90 = float(mean_pinball_loss(y, q90, alpha=0.90))
    return p10, p50, p90, float(np.mean([p10, p50, p90]))


def safe_auc(y_bin: np.ndarray, prob: np.ndarray) -> float | None:
    if len(np.unique(y_bin)) < 2:
        return None
    return float(roc_auc_score(y_bin, prob))


def safe_pr_auc(y_bin: np.ndarray, prob: np.ndarray) -> float | None:
    if len(np.unique(y_bin)) < 2:
        return None
    return float(average_precision_score(y_bin, prob))


def safe_f1(y_bin: np.ndarray, prob: np.ndarray) -> float | None:
    pred = (prob >= 0.5).astype(int)
    if y_bin.sum() == 0 and pred.sum() == 0:
        return None
    return float(f1_score(y_bin, pred, zero_division=0))


def safe_recall(y_bin: np.ndarray, prob: np.ndarray) -> float | None:
    pred = (prob >= 0.5).astype(int)
    if y_bin.sum() == 0:
        return None
    return float(recall_score(y_bin, pred, zero_division=0))


def recompute_model_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, model, horizon), g in preds.groupby(["split", "model", "horizon_days"], sort=True):
        y = g["target_c_rsm"].to_numpy(dtype=np.float64)
        q10 = g["q10"].to_numpy(dtype=np.float64)
        q50 = g["q50"].to_numpy(dtype=np.float64)
        q90 = g["q90"].to_numpy(dtype=np.float64)
        q = np.sort(np.column_stack([q10, q50, q90]), axis=1)
        q10, q50, q90 = q[:, 0], q[:, 1], q[:, 2]
        prob = np.clip(g["collapse_probability"].to_numpy(dtype=np.float64), 0, 1)
        y_bin = (y < 0).astype(int)
        p10, p50, p90, pmean = pinball_mean(y, q10, q50, q90)
        row = {
            "split": split,
            "model": model,
            "horizon_days": int(horizon),
            "n": int(len(g)),
            "mae": float(mean_absolute_error(y, q50)),
            "rmse": float(mean_squared_error(y, q50) ** 0.5),
            "r2": float(r2_score(y, q50)) if len(y) > 1 else None,
            "pinball_loss": pmean,
            "pinball_p10": p10,
            "pinball_p50": p50,
            "pinball_p90": p90,
            "roc_auc": safe_auc(y_bin, prob),
            "pr_auc": safe_pr_auc(y_bin, prob),
            "brier_score": float(brier_score_loss(y_bin, prob)),
            "event_f1": safe_f1(y_bin, prob),
            "low_margin_recall": safe_recall(y_bin, prob),
            "p10_p90_coverage": float(((y >= q10) & (y <= q90)).mean()),
            "interval_width_p10_p90": float(np.mean(q90 - q10)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def compare_existing_metrics(recomputed: pd.DataFrame) -> list[dict[str, Any]]:
    report_rows: list[dict[str, Any]] = []
    comparisons = [
        (TABLE_DIR / "model_performance_main.csv", "main"),
        (TABLE_DIR / "model_performance_pseudo_prospective.csv", "pseudo_prospective"),
    ]
    mapping = {
        "mae_p50": "mae",
        "rmse_p50": "rmse",
        "pinball_p10": "pinball_p10",
        "pinball_p50": "pinball_p50",
        "pinball_p90": "pinball_p90",
        "p10_p90_coverage": "p10_p90_coverage",
        "interval_width_p10_p90": "interval_width_p10_p90",
        "brier_collapse": "brier_score",
        "roc_auc_collapse": "roc_auc",
    }
    for path, split in comparisons:
        if not path.exists():
            continue
        current = pd.read_csv(path)
        merged = current.merge(
            recomputed[recomputed["split"] == split],
            on=["split", "model", "horizon_days"],
            suffixes=("_file", "_recomputed"),
            how="left",
        )
        for source_col, new_col in mapping.items():
            if source_col not in merged.columns or new_col not in merged.columns:
                continue
            diff = (merged[source_col] - merged[new_col]).abs()
            bad = merged[diff > 1e-6]
            for row in bad.head(20).to_dict("records"):
                report_rows.append(
                    {
                        "file": rel(path),
                        "split": row["split"],
                        "model": row["model"],
                        "horizon_days": row["horizon_days"],
                        "metric": source_col,
                        "file_value": row[source_col],
                        "recomputed_value": row[new_col],
                        "abs_diff": abs(row[source_col] - row[new_col]),
                    }
                )
    return report_rows


def compute_results_freeze(file_rows: list[dict[str, Any]], derived_cips_summary: bool) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    cips = pd.read_parquet(CIPS_OBJECTS)
    park = pd.read_parquet(PARK_DAILY, columns=["park_id", "date", "cdmet_lon", "cdmet_lat"])
    wq = pd.read_parquet(PARK_WQ, columns=["station_id", "wq_recent_obs_count_30d"])
    crsm_cols = [
        "park_id",
        "date",
        "C_RSM",
        "dominant_driver",
        "margin_compression_episode",
        "S_T",
        "S_P",
        "S_V",
        "S_W",
        "D_T",
        "D_P",
        "D_V",
        "D_W",
        "margin_T",
        "margin_P",
        "margin_V",
        "margin_W",
        "tmax_c",
        "precip_mm",
        "precip_3d_mm",
        "wind_ms",
        "park_area_km2",
        "water_receptor_stress",
        "wq_recent_obs_count_30d",
        "denom_S_T",
        "denom_S_P",
        "denom_S_V",
        "denom_S_W",
        "precip_3d_q95_mm",
        "wind_q05_ms",
        "Ea_J_mol",
        "T0_K",
        "eps",
        "eps_wind_ms",
    ]
    crsm = pd.read_parquet(CRSM_DAILY, columns=crsm_cols)
    crsm["date"] = pd.to_datetime(crsm["date"])
    scenario_overall = pd.read_csv(SCENARIO_OVERALL)
    scenario_by_park = pd.read_csv(SCENARIO_BY_PARK)
    preds = pd.read_parquet(PREDICTIONS)
    model_metrics = recompute_model_metrics(preds)
    metric_conflicts = compare_existing_metrics(model_metrics)

    cips_count = int(len(cips))
    cips_area = float(cips["park_area_km2"].sum())
    park_day_rows = int(len(park))
    cdmet_date_range = [
        pd.to_datetime(park["date"]).min().date().isoformat(),
        pd.to_datetime(park["date"]).max().date().isoformat(),
    ]
    met_grid_count = int(park[["cdmet_lon", "cdmet_lat"]].drop_duplicates().shape[0])
    wq_station_count = int(wq["station_id"].nunique())
    wq_coverage = float((wq["wq_recent_obs_count_30d"] > 0).mean())

    baseline = {
        "mean_c_rsm": float(crsm["C_RSM"].mean()),
        "median_c_rsm": float(crsm["C_RSM"].median()),
        "p05_c_rsm": float(crsm["C_RSM"].quantile(0.05)),
        "p25_c_rsm": float(crsm["C_RSM"].quantile(0.25)),
        "p75_c_rsm": float(crsm["C_RSM"].quantile(0.75)),
        "p95_c_rsm": float(crsm["C_RSM"].quantile(0.95)),
        "compression_park_day_share": float((crsm["C_RSM"] < 0).mean()),
    }
    driver_shares = {driver: float(share) for driver, share in crsm["dominant_driver"].value_counts(normalize=True).to_dict().items()}
    for driver in ["T", "P", "V", "W"]:
        driver_shares.setdefault(driver, 0.0)

    low_margin = (
        scenario_by_park.groupby("scenario")
        .agg(
            low_margin_park_count=("mean_c_rsm", lambda x: int((x < 0).sum())),
            any_compression_park_count=("collapse_days", lambda x: int((x > 0).sum())),
        )
        .reset_index()
    )
    scenario = scenario_overall.merge(low_margin, on="scenario", how="left")
    scenario_records = scenario.to_dict("records")
    s0 = scenario[scenario["scenario"] == "S0"].iloc[0]
    s6 = scenario[scenario["scenario"] == "S6"].iloc[0]
    s6_delta = {
        "delta_mean_c_rsm": float(s6["mean_c_rsm"] - s0["mean_c_rsm"]),
        "delta_compression_day_share": float(s6["collapse_day_share"] - s0["collapse_day_share"]),
        "delta_low_margin_park_count": int(s6["low_margin_park_count"] - s0["low_margin_park_count"]),
    }

    ablation = pd.read_csv(TABLE_DIR / "ablation_results.csv")
    ablation_map = {
        "no_water": "water_features_masked",
        "no_climate_stress": "climate_stress_components_masked",
        "no_static_geometry": "static_area_masked",
        "no_area_exposure": "static_area_masked",
        "no_quantile_head": None,
    }
    ablation_records: dict[str, Any] = {}
    for required, stored in ablation_map.items():
        if stored is None:
            ablation_records[required] = {"available": False, "reason": "not implemented in current masking-ablation table"}
        else:
            rows = ablation[ablation["ablation"] == stored]
            ablation_records[required] = {
                "available": not rows.empty,
                "source_ablation": stored,
                "metrics": rows.to_dict("records"),
            }

    pseudo = model_metrics[model_metrics["split"] == "pseudo_prospective"].to_dict("records")
    main_metrics = model_metrics[model_metrics["split"] == "main"].to_dict("records")
    crsmamba_main = model_metrics[(model_metrics["split"] == "main") & (model_metrics["model"] == "C-RSMamba")].to_dict("records")
    baselines = model_metrics[
        (model_metrics["split"] == "main")
        & (~model_metrics["model"].isin(["C-RSMamba", "LSTM", "GRU", "TCN", "Transformer", "PatchTST-lite"]))
    ].to_dict("records")

    freeze = {
        "metadata": {
            "project_root": str(ROOT),
            "generated_by": "scripts/12_freeze_and_qc.py",
            "source_rule": "Use this JSON as the only allowed core-number source for future figure drawing.",
            "environment_requirement": "conda environment pytorch",
            "cips_summary_derived_during_qc": derived_cips_summary,
        },
        "file_audit": file_rows,
        "core_counts": {
            "cips_object_count": cips_count,
            "cips_total_area_km2": cips_area,
            "park_day_rows": park_day_rows,
            "cdmet_date_range": cdmet_date_range,
            "meteorological_grid_count": met_grid_count,
            "water_quality_station_count": wq_station_count,
            "water_quality_30d_coverage_share": wq_coverage,
        },
        "baseline_c_rsm": baseline,
        "baseline_dominant_driver_share": driver_shares,
        "scenario_s0_s6": scenario_records,
        "s6_vs_s0_delta": s6_delta,
        "model_metrics_main_all_models": main_metrics,
        "model_metrics_main_crsmamba": crsmamba_main,
        "model_metrics_main_flat_baselines": baselines,
        "ablation_results": ablation_records,
        "pseudo_prospective_2020_metrics": pseudo,
        "metric_table_conflicts": metric_conflicts,
    }
    return clean_value(freeze), model_metrics, metric_conflicts


def write_results_freeze_report(freeze: dict[str, Any], model_metrics: pd.DataFrame, metric_conflicts: list[dict[str, Any]]) -> None:
    (REPORT_DIR / "RESULTS_FREEZE.json").write_text(json.dumps(freeze, indent=2, ensure_ascii=False), encoding="utf-8")
    rows: list[dict[str, Any]] = []
    flatten_json("", freeze, rows)
    pd.DataFrame(rows).to_csv(TABLE_DIR / "results_freeze_summary.csv", index=False, encoding="utf-8-sig")

    counts = freeze["core_counts"]
    baseline = freeze["baseline_c_rsm"]
    scenario_rows = freeze["scenario_s0_s6"]
    crsmamba = model_metrics[(model_metrics["split"] == "main") & (model_metrics["model"] == "C-RSMamba")]
    best_by_h = (
        model_metrics[model_metrics["split"] == "main"]
        .sort_values(["horizon_days", "mae"])
        .groupby("horizon_days", as_index=False)
        .first()[["horizon_days", "model", "mae", "rmse", "r2", "roc_auc", "pr_auc"]]
        .to_dict("records")
    )
    lines = [
        "# RESULTS_FREEZE_REPORT",
        "",
        "## Frozen Source Rule",
        "",
        "`outputs/reports/RESULTS_FREEZE.json` is the sole permitted source for core numbers in subsequent figure drawing. Values below were recomputed from parquet/csv outputs, not copied from Word/manuscript text.",
        "",
        "## Core Counts",
        "",
        md_table(
            [
                {"metric": "CIPs object count", "value": counts["cips_object_count"]},
                {"metric": "CIPs total area km2", "value": counts["cips_total_area_km2"]},
                {"metric": "park-day rows", "value": counts["park_day_rows"]},
                {"metric": "CDMet date range", "value": " to ".join(counts["cdmet_date_range"])},
                {"metric": "meteorological grid count", "value": counts["meteorological_grid_count"]},
                {"metric": "water-quality station count", "value": counts["water_quality_station_count"]},
                {"metric": "water-quality 30d coverage share", "value": counts["water_quality_30d_coverage_share"]},
            ]
        ),
        "",
        "## Baseline C-RSM",
        "",
        md_table([{"metric": k, "value": v} for k, v in baseline.items()]),
        "",
        "## Dominant Driver Share",
        "",
        md_table([{"driver": k, "share": v} for k, v in freeze["baseline_dominant_driver_share"].items()]),
        "",
        "## Scenario S0-S6",
        "",
        md_table(
            scenario_rows,
            [
                "scenario",
                "mean_c_rsm",
                "median_c_rsm",
                "collapse_day_share",
                "low_margin_park_count",
                "dominant_T_share",
                "dominant_P_share",
                "dominant_V_share",
                "dominant_W_share",
            ],
        ),
        "",
        "## S6 versus S0",
        "",
        md_table([{"metric": k, "value": v} for k, v in freeze["s6_vs_s0_delta"].items()]),
        "",
        "## C-RSMamba Main-Split Metrics",
        "",
        md_table(crsmamba.to_dict("records"), ["horizon_days", "mae", "rmse", "r2", "pinball_loss", "roc_auc", "pr_auc", "brier_score", "event_f1", "low_margin_recall"]),
        "",
        "## Best Model By Horizon, Main Split",
        "",
        md_table(best_by_h),
        "",
        "## Ablation Availability",
        "",
        md_table(
            [
                {
                    "required_ablation": name,
                    "available": value.get("available"),
                    "source_ablation": value.get("source_ablation", ""),
                    "reason": value.get("reason", ""),
                }
                for name, value in freeze["ablation_results"].items()
            ]
        ),
        "",
        "## Metric Table Consistency",
        "",
        f"- Recomputed metric conflicts against existing CSV metrics: {len(metric_conflicts)}",
    ]
    (REPORT_DIR / "RESULTS_FREEZE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    conflict_lines = [
        "# RESULTS_CONFLICT_REPORT",
        "",
        f"- Conflict detected in existing metric CSV tables: {'YES' if metric_conflicts else 'NO'}",
        "- Word/manuscript text was not used as a source of truth.",
        "- Any future manuscript or figure label must be updated from `outputs/reports/RESULTS_FREEZE.json`.",
        "",
    ]
    if metric_conflicts:
        conflict_lines.extend(["## Metric CSV Conflicts", "", md_table(metric_conflicts)])
    else:
        conflict_lines.append("No deterministic numeric conflict was detected between recomputed prediction metrics and the existing machine-readable metric CSV tables.")
    (REPORT_DIR / "RESULTS_CONFLICT_REPORT.md").write_text("\n".join(conflict_lines), encoding="utf-8")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def formula_implementation_audit() -> dict[str, Any]:
    crsm_cols = [
        "park_id",
        "date",
        "park_area_km2",
        "tmax_c",
        "precip_mm",
        "precip_3d_mm",
        "wind_ms",
        "water_receptor_stress",
        "S_T",
        "S_P",
        "S_V",
        "S_W",
        "D_T",
        "D_P",
        "D_V",
        "D_W",
        "margin_T",
        "margin_P",
        "margin_V",
        "margin_W",
        "C_RSM",
        "dominant_driver",
        "denom_S_T",
        "denom_S_P",
        "denom_S_V",
        "denom_S_W",
        "precip_3d_q95_mm",
        "wind_q05_ms",
        "Ea_J_mol",
        "T0_K",
        "eps",
        "eps_wind_ms",
    ]
    crsm = pd.read_parquet(CRSM_DAILY, columns=crsm_cols)
    crsm["date"] = pd.to_datetime(crsm["date"])
    sample = crsm.sample(n=min(250_000, len(crsm)), random_state=42).copy()

    area_factor = np.power(np.maximum(sample["park_area_km2"].to_numpy(dtype=np.float64), EPS), ETA)
    temp_k = sample["tmax_c"].to_numpy(dtype=np.float64) + 273.15
    exponent = (EA_J_MOL / R_J_MOL_K) * ((1.0 / T0_K) - (1.0 / np.maximum(temp_k, EPS)))
    s_t = area_factor * np.maximum(np.exp(exponent) - 1.0, 0.0)
    s_p = area_factor * sample["precip_3d_mm"].to_numpy(dtype=np.float64) / (sample["precip_3d_q95_mm"].to_numpy(dtype=np.float64) + EPS)
    s_v = area_factor * (sample["wind_q05_ms"].to_numpy(dtype=np.float64) + EPS_WIND_MS) / (sample["wind_ms"].to_numpy(dtype=np.float64) + EPS_WIND_MS)
    s_w = sample["water_receptor_stress"].fillna(0).to_numpy(dtype=np.float64)
    raw_recomputed = {"S_T": s_t, "S_P": s_p, "S_V": s_v, "S_W": s_w}

    raw_max_abs_error = {
        col: float(np.max(np.abs(sample[col].to_numpy(dtype=np.float64) - values))) for col, values in raw_recomputed.items()
    }
    d_max_abs_error = {}
    for raw, demand in zip(["S_T", "S_P", "S_V", "S_W"], ["D_T", "D_P", "D_V", "D_W"]):
        denom = sample[f"denom_{raw}"].to_numpy(dtype=np.float64) + EPS
        recomputed = sample[raw].to_numpy(dtype=np.float64) / denom
        d_max_abs_error[demand] = float(np.max(np.abs(sample[demand].to_numpy(dtype=np.float64) - recomputed)))

    margin_cols = ["margin_T", "margin_P", "margin_V", "margin_W"]
    demand_cols = ["D_T", "D_P", "D_V", "D_W"]
    margins = 1.0 - sample[demand_cols].to_numpy(dtype=np.float64)
    stored_margins = sample[margin_cols].to_numpy(dtype=np.float64)
    margin_max_abs_error = float(np.max(np.abs(margins - stored_margins)))
    c_recomputed = margins.min(axis=1)
    c_max_abs_error = float(np.max(np.abs(sample["C_RSM"].to_numpy(dtype=np.float64) - c_recomputed)))
    drivers = np.array(["T", "P", "V", "W"])
    driver_recomputed = drivers[np.argmin(margins, axis=1)]
    driver_mismatch_rate = float((driver_recomputed != sample["dominant_driver"].to_numpy()).mean())

    precip_check = crsm[["park_id", "date", "precip_mm", "precip_3d_mm"]].sort_values(["park_id", "date"]).copy()
    precip_check["precip_3d_recomputed"] = (
        precip_check.groupby("park_id", sort=False)["precip_mm"]
        .rolling(3, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    precip_3d_max_abs_error = float(np.max(np.abs(precip_check["precip_3d_mm"] - precip_check["precip_3d_recomputed"])))

    script04 = read_text(ROOT / "scripts" / "04_build_crsm_index.py")
    script05 = read_text(ROOT / "scripts" / "05_build_scenario_stress_tests.py")
    script03 = read_text(ROOT / "scripts" / "03_preprocess_water_quality.py")
    scenario_overall = pd.read_csv(SCENARIO_OVERALL)
    s0_mean = float(scenario_overall.loc[scenario_overall["scenario"] == "S0", "mean_c_rsm"].iloc[0])
    baseline_mean = float(crsm["C_RSM"].mean())

    checks = {
        "raw_S_formula_sample_max_abs_error": raw_max_abs_error,
        "D_formula_sample_max_abs_error": d_max_abs_error,
        "margin_sample_max_abs_error": margin_max_abs_error,
        "C_RSM_min_margin_sample_max_abs_error": c_max_abs_error,
        "driver_argmin_margin_mismatch_rate": driver_mismatch_rate,
        "precip_3d_max_abs_error": precip_3d_max_abs_error,
        "tmax_c_range": [float(crsm["tmax_c"].min()), float(crsm["tmax_c"].max())],
        "ea_j_mol_unique": sorted(map(float, crsm["Ea_J_mol"].dropna().unique())),
        "t0_k_unique": sorted(map(float, crsm["T0_K"].dropna().unique())),
        "eps_wind_ms_unique": sorted(map(float, crsm["eps_wind_ms"].dropna().unique())),
        "scenario_s0_baseline_mean_abs_diff": abs(s0_mean - baseline_mean),
        "code_checks": {
            "tmax_plus_273_15_in_baseline_code": "+ 273.15" in script04,
            "tmax_plus_273_15_in_scenario_code": "+ 273.15" in script05,
            "Ea_60000_in_code": "EA_J_MOL = 60_000.0" in script04 and "EA_J_MOL = 60_000.0" in script05,
            "R_unit_j_mol_k_in_code": "R_J_MOL_K" in script04 and "8.314462618" in script04,
            "rolling_3day_precip_in_code": "rolling(3" in script04,
            "low_wind_inverse_in_code": "wind_q05" in script04 and "/ (df[\"wind_ms\"" in script04,
            "eps_wind_0_1_in_code": "EPS_WIND_MS = 0.1" in script04 and "EPS_WIND_MS = 0.1" in script05,
            "water_receptor_context_in_code": "water_receptor_stress" in script04 and "receptor-state" in script04,
            "wq_missing_fill_zero_in_code": "fillna(0)" in script03 and "rolling(30" in script03,
            "scenario_dw_keeps_baseline_sw": "\"S_W\"" in script05 and "s_w = df[\"S_W\"]" in script05,
            "scenario_uses_baseline_denominator": "denom_S_T" in script05 and "scenario's own distribution" in script05,
        },
    }
    tolerance_pass = (
        max(raw_max_abs_error.values()) < 5e-4
        and max(d_max_abs_error.values()) < 5e-6
        and margin_max_abs_error < 5e-6
        and c_max_abs_error < 5e-6
        and driver_mismatch_rate == 0.0
        and precip_3d_max_abs_error < 5e-5
        and checks["scenario_s0_baseline_mean_abs_diff"] < 5e-6
        and all(checks["code_checks"].values())
    )
    checks["formula_pass"] = bool(tolerance_pass)

    lines = [
        "# FORMULA_IMPLEMENTATION_AUDIT",
        "",
        f"- Formula pass: {'YES' if tolerance_pass else 'NO'}",
        f"- Sample rows checked: {len(sample):,}",
        "",
        "## Numerical Recalculation Checks",
        "",
        md_table(
            [
                {"check": "raw S max abs error", "value": raw_max_abs_error},
                {"check": "D max abs error", "value": d_max_abs_error},
                {"check": "margin max abs error", "value": margin_max_abs_error},
                {"check": "C-RSM min-margin max abs error", "value": c_max_abs_error},
                {"check": "driver mismatch rate", "value": driver_mismatch_rate},
                {"check": "3-day precipitation max abs error", "value": precip_3d_max_abs_error},
                {"check": "S0 mean versus baseline mean abs diff", "value": checks["scenario_s0_baseline_mean_abs_diff"]},
            ],
            max_cell=220,
        ),
        "",
        "## Required Formula Items",
        "",
        md_table([{"item": k, "pass": v} for k, v in checks["code_checks"].items()]),
        "",
        "## Interpretation Guardrail",
        "",
        "`C-RSM < 0` is frozen as a relative margin-compression episode under an open-data screening boundary. It must not be interpreted as an accident, causal event, HAZOP/LOPA substitute, or real-time post-2021 risk label.",
    ]
    (REPORT_DIR / "FORMULA_IMPLEMENTATION_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    if not tolerance_pass:
        (REPORT_DIR / "NEED_USER_CONFIRMATION_FORMULA.md").write_text(
            "# NEED_USER_CONFIRMATION_FORMULA\n\nFormula implementation mismatch detected. Do not proceed to figure drawing until resolved.\n",
            encoding="utf-8",
        )
    return clean_value(checks)


def leakage_and_split_audit(model_metrics: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    script06 = read_text(ROOT / "scripts" / "06_train_models.py")
    script03 = read_text(ROOT / "scripts" / "03_preprocess_water_quality.py")
    schema = read_text(FEATURE_SCHEMA).splitlines() if FEATURE_SCHEMA.exists() else []
    scaler = json.loads(read_text(SCALER_META)) if SCALER_META.exists() else {}

    df = pd.read_parquet(CRSM_DAILY, columns=["park_id", "date", "C_RSM"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["park_id", "date"]).copy()
    row_in_park = df.groupby("park_id", sort=False).cumcount().to_numpy()
    leakage_rows = []
    for split_name, spec in SPLITS.items():
        for horizon in HORIZONS:
            target_date = df.groupby("park_id", sort=False)["date"].shift(-horizon)
            eligible = (row_in_park >= SEQ_LEN - 1) & target_date.notna()
            for segment in ["train", "val", "test"]:
                start = pd.Timestamp(spec[f"{segment}_start"])
                end = pd.Timestamp(spec[f"{segment}_end"])
                origin_mask = eligible & (df["date"] >= start) & (df["date"] <= end)
                count = int(origin_mask.sum())
                target_after_segment = int((origin_mask & (target_date > end)).sum())
                input_window_start_after_segment_start = int((origin_mask & (df["date"] - pd.Timedelta(days=SEQ_LEN - 1) < start)).sum())
                leakage_rows.append(
                    {
                        "split": split_name,
                        "segment": segment,
                        "horizon_days": horizon,
                        "origin_window_count_before_sampling": count,
                        "target_date_after_segment_end_count": target_after_segment,
                        "input_window_starts_before_segment_start_count": input_window_start_after_segment_start,
                    }
                )

    best_by_horizon = (
        model_metrics[model_metrics["split"] == "main"]
        .sort_values(["horizon_days", "mae"])
        .groupby("horizon_days", as_index=False)
        .first()[["horizon_days", "model", "mae", "rmse", "r2"]]
    )
    crsmamba_rank = []
    for horizon, g in model_metrics[model_metrics["split"] == "main"].groupby("horizon_days"):
        ordered = g.sort_values("mae").reset_index(drop=True)
        match = ordered[ordered["model"] == "C-RSMamba"]
        crsmamba_rank.append({"horizon_days": int(horizon), "mae_rank": int(match.index[0] + 1) if not match.empty else None, "n_models": int(len(ordered))})

    checks = {
        "main_split_declared": all(token in script06 for token in ["2000-01-01", "2015-12-31", "2016-01-01", "2017-12-31", "2018-01-01", "2020-12-31"]),
        "pseudo_split_declared": all(token in script06 for token in ["2017-12-31", "2018-01-01", "2019-12-31", "2020-01-01"]),
        "scaler_fit_train_only": "standardize_features" in script06 and "train_mask" in script06 and "np.nanmean(train_values" in script06,
        "target_created_by_forward_shift": "shift(-horizon)" in script06,
        "sequence_window_uses_past_to_origin": "x[start : end + 1]" in script06 and "start = end - SEQ_LEN + 1" in script06,
        "static_features_no_target_stats": not any("target" in feature.lower() for feature in schema),
        "scenario_not_mixed_into_training": "scenario_c_rsm_daily" not in script06.lower(),
        "wq_rolling_no_future_shift": "rolling(30" in script03 and "shift(-" not in script03,
        "mamba_inspired_recorded": "Mamba-inspired" in script06 and "mamba_ssm" in script06,
        "crsmamba_not_universally_best_recordable": any(row["mae_rank"] != 1 for row in crsmamba_rank),
    }
    fatal = not all(v for k, v in checks.items() if k != "crsmamba_not_universally_best_recordable")
    audit = {
        "checks": checks,
        "split_window_boundary_counts": leakage_rows,
        "fatal_leakage_detected": bool(fatal),
        "origin_date_split_policy_note": "Targets near split boundaries can fall after the segment end under the implemented forecast-origin split. This is not future information in the input window, but should be disclosed if a strict target-date split is desired.",
    }
    interpretation = {
        "best_by_horizon_mae": best_by_horizon.to_dict("records"),
        "crsmamba_mae_rank_by_horizon": crsmamba_rank,
        "can_say": "C-RSMamba establishes a trainable probabilistic sequence-forecasting workflow for C-RSM and is competitive under the frozen main split, with horizon-specific performance that should be reported exactly from RESULTS_FREEZE.json.",
        "cannot_say": "Do not claim that C-RSMamba universally outperforms all baselines at all horizons, do not imply official mamba-ssm CUDA kernel use, and do not treat C-RSM forecasts as accident forecasts.",
    }

    lines = [
        "# LEAKAGE_AND_SPLIT_AUDIT",
        "",
        f"- Fatal leakage detected: {'YES' if fatal else 'NO'}",
        "- Split policy: forecast-origin date split. Input windows use historical rows through the forecast origin.",
        "",
        "## Required Checks",
        "",
        md_table([{"check": k, "pass": v} for k, v in checks.items()]),
        "",
        "## Boundary Counts",
        "",
        md_table(leakage_rows, max_cell=160),
        "",
        "## Note",
        "",
        audit["origin_date_split_policy_note"],
    ]
    (REPORT_DIR / "LEAKAGE_AND_SPLIT_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")

    lines2 = [
        "# MODEL_RESULT_INTERPRETATION_AUDIT",
        "",
        "## One-Sentence Conclusion",
        "",
        f"- Can say: {interpretation['can_say']}",
        f"- Cannot say: {interpretation['cannot_say']}",
        "",
        "## Best Model By MAE",
        "",
        md_table(best_by_horizon.to_dict("records")),
        "",
        "## C-RSMamba MAE Rank",
        "",
        md_table(crsmamba_rank),
    ]
    (REPORT_DIR / "MODEL_RESULT_INTERPRETATION_AUDIT.md").write_text("\n".join(lines2), encoding="utf-8")
    return clean_value(audit), clean_value(interpretation)


def figure_data_readiness_audit() -> tuple[list[dict[str, Any]], bool]:
    fig_plan = [
        {
            "figure": "Fig. 1 Conceptual and computational framework",
            "required_source_data": "CIPs objects, C-RSM formula components, model task metadata, figure source table",
            "paths": [CIPS_OBJECTS, CRSM_DAILY, FEATURE_SCHEMA, ROOT / "figures" / "source_data" / "Fig1_source_data.csv"],
        },
        {
            "figure": "Fig. 2 Data construction and temporal harmonization",
            "required_source_data": "CIPs objects, CDMet park-day table, water-quality state table, Fig2 source table",
            "paths": [CIPS_OBJECTS, PARK_DAILY, PARK_WQ, ROOT / "figures" / "source_data" / "Fig2_source_data.csv"],
        },
        {
            "figure": "Fig. 3 Design-basis climate forcing landscape",
            "required_source_data": "park-day meteorology, annual climate summaries, driver source data",
            "paths": [PARK_DAILY, ROOT / "figures" / "source_data" / "Fig3_annual_source_data.csv", ROOT / "figures" / "source_data" / "Fig3_driver_source_data.csv"],
        },
        {
            "figure": "Fig. 4 C-RSM components and margin boundary",
            "required_source_data": "daily C-RSM component columns and quantile source data",
            "paths": [CRSM_DAILY, TABLE_DIR / "crsm_component_quantiles.csv", ROOT / "figures" / "source_data" / "Fig4_source_data.csv"],
        },
        {
            "figure": "Fig. 5 Baseline margin-compression mechanism attribution",
            "required_source_data": "baseline C-RSM, driver shares, component quantiles",
            "paths": [CRSM_DAILY, TABLE_DIR / "crsm_driver_share_baseline.csv", ROOT / "figures" / "source_data" / "Fig5_source_data.csv"],
        },
        {
            "figure": "Fig. 6 C-RSMamba architecture and learning task",
            "required_source_data": "feature schema, model checkpoint, training audit, station/coverage source tables",
            "paths": [FEATURE_SCHEMA, ROOT / "outputs" / "models" / "crsmamba_best.pt", REPORT_DIR / "MODEL_TRAINING_AUDIT.md", ROOT / "figures" / "source_data" / "Fig6_station_source_data.csv", ROOT / "figures" / "source_data" / "Fig6_coverage_source_data.csv"],
        },
        {
            "figure": "Fig. 7 Forecasting skill and calibration",
            "required_source_data": "prediction parquet, model-performance tables, calibration table, Fig7 source data",
            "paths": [PREDICTIONS, TABLE_DIR / "model_performance_main.csv", TABLE_DIR / "calibration_results.csv", ROOT / "figures" / "source_data" / "Fig7_source_data.csv"],
        },
        {
            "figure": "Fig. 8 Climate analogue scenario extension and screening implications",
            "required_source_data": "scenario daily table, scenario summaries, Fig8 source data",
            "paths": [SCENARIO_DAILY, SCENARIO_OVERALL, SCENARIO_BY_PARK, ROOT / "figures" / "source_data" / "Fig8_source_data.csv"],
        },
    ]
    rows = []
    for item in fig_plan:
        missing = [rel(p) for p in item["paths"] if not p.exists()]
        rows.append(
            {
                "Figure": item["figure"],
                "required source data": item["required_source_data"],
                "available yes/no": "yes" if not missing else "no",
                "source file path": "; ".join(rel(p) for p in item["paths"] if p.exists()),
                "missing data": "; ".join(missing),
                "can derive from existing outputs yes/no": "yes" if missing else "n/a",
                "need rerun experiment yes/no": "no" if not missing else "no, derive source table from frozen processed outputs",
            }
        )
    all_ready = all(row["available yes/no"] == "yes" for row in rows)
    lines = [
        "# FIGURE_DATA_READINESS_AUDIT",
        "",
        f"- All planned Fig. 1-Fig. 8 source data ready: {'YES' if all_ready else 'NO'}",
        "- No figures were redrawn in this QC step.",
        "",
        md_table(rows, ["Figure", "required source data", "available yes/no", "source file path", "missing data", "can derive from existing outputs yes/no", "need rerun experiment yes/no"], max_cell=200),
    ]
    (FIGURE_DIR / "FIGURE_DATA_READINESS_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    return rows, all_ready


def go_no_go(
    file_rows: list[dict[str, Any]],
    formula: dict[str, Any],
    leakage: dict[str, Any],
    figure_ready: bool,
    freeze: dict[str, Any],
) -> str:
    critical_file_ok = all(row["exists"] and row["readable"] for row in file_rows)
    formula_ok = bool(formula.get("formula_pass"))
    leakage_ok = not leakage.get("fatal_leakage_detected", True)
    no_metric_conflicts = len(freeze.get("metric_table_conflicts", [])) == 0
    go = critical_file_ok and formula_ok and leakage_ok and figure_ready and no_metric_conflicts
    caveats = []
    if not freeze["ablation_results"]["no_quantile_head"]["available"]:
        caveats.append("No dedicated no-quantile-head ablation is available; future figures must not claim this ablation.")
    if any(row.get("target_date_after_segment_end_count", 0) for row in leakage["split_window_boundary_counts"]):
        caveats.append("Forecast-origin split allows target dates near split edges to fall in the next calendar segment; disclose this policy or trim edges before strict target-date claims.")
    if not critical_file_ok:
        caveats.append("At least one required experiment file is missing or unreadable.")
    if not formula_ok:
        caveats.append("Formula audit failed.")
    if not leakage_ok:
        caveats.append("Fatal leakage condition detected.")
    if not figure_ready:
        caveats.append("At least one planned figure source-data input is missing.")
    if not no_metric_conflicts:
        caveats.append("Recomputed metrics conflict with existing machine-readable tables.")

    lines = [
        "# EXPERIMENT_GO_NO_GO_FOR_FIGURES",
        "",
        f"GO_FOR_FIGURES = {'YES' if go else 'NO'}",
        "",
        "## Basis",
        "",
        md_table(
            [
                {"gate": "Required files exist/readable", "pass": critical_file_ok},
                {"gate": "Formula implementation audit", "pass": formula_ok},
                {"gate": "Leakage/split audit", "pass": leakage_ok},
                {"gate": "Fig. 1-Fig. 8 source data readiness", "pass": figure_ready},
                {"gate": "Metric CSV consistency", "pass": no_metric_conflicts},
            ]
        ),
        "",
        "## Frozen Result Files",
        "",
        "- `outputs/reports/RESULTS_FREEZE.json`",
        "- `outputs/reports/RESULTS_FREEZE_REPORT.md`",
        "- `outputs/tables/results_freeze_summary.csv`",
        "- `outputs/reports/FORMULA_IMPLEMENTATION_AUDIT.md`",
        "- `outputs/reports/LEAKAGE_AND_SPLIT_AUDIT.md`",
        "- `outputs/reports/MODEL_RESULT_INTERPRETATION_AUDIT.md`",
        "- `figures/FIGURE_DATA_READINESS_AUDIT.md`",
        "",
        "## Caveats",
        "",
        "\n".join(f"- {c}" for c in caveats) if caveats else "- None.",
    ]
    (REPORT_DIR / "EXPERIMENT_GO_NO_GO_FOR_FIGURES.md").write_text("\n".join(lines), encoding="utf-8")
    return "YES" if go else "NO"


def main() -> None:
    ensure_dirs()
    file_rows, derived_cips_summary = file_audit()
    freeze, model_metrics, metric_conflicts = compute_results_freeze(file_rows, derived_cips_summary)
    write_results_freeze_report(freeze, model_metrics, metric_conflicts)
    formula = formula_implementation_audit()
    leakage, interpretation = leakage_and_split_audit(model_metrics)
    figure_rows, figure_ready = figure_data_readiness_audit()
    freeze["formula_implementation_audit"] = formula
    freeze["leakage_and_split_audit"] = leakage
    freeze["model_result_interpretation_audit"] = interpretation
    freeze["figure_data_readiness"] = figure_rows
    go = go_no_go(file_rows, formula, leakage, figure_ready, freeze)
    freeze["go_for_figures"] = go
    (REPORT_DIR / "RESULTS_FREEZE.json").write_text(json.dumps(clean_value(freeze), indent=2, ensure_ascii=False), encoding="utf-8")

    summary_rows: list[dict[str, Any]] = []
    flatten_json("", clean_value(freeze), summary_rows)
    pd.DataFrame(summary_rows).to_csv(TABLE_DIR / "results_freeze_summary.csv", index=False, encoding="utf-8-sig")
    print(f"RESULTS_FREEZE written. GO_FOR_FIGURES={go}")


if __name__ == "__main__":
    main()

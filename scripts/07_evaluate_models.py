from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.utils.project import ensure_dir, log_event, rel


METRICS = ROOT / "outputs" / "tables" / "model_evaluation_metrics.csv"
REPORT = ROOT / "outputs" / "reports" / "MODEL_EVALUATION_REPORT.md"


def main() -> None:
    if not METRICS.exists():
        raise SystemExit(f"Missing model metrics: {METRICS}. Run scripts/06_train_models.py first.")
    log_event("07_evaluate_models", "started")
    df = pd.read_csv(METRICS)
    best = df.sort_values(["horizon_days", "mae_p50"]).groupby("horizon_days").first().reset_index()
    lines = [
        "# MODEL_EVALUATION_REPORT",
        "",
        f"- Metrics table: `{rel(METRICS)}`",
        "",
        "## Best Median-Forecast MAE By Horizon",
        "",
        "| Horizon | Best model | MAE P50 | RMSE P50 | Brier collapse | ROC AUC collapse |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in best.itertuples(index=False):
        lines.append(
            f"| {row.horizon_days} | {row.model} | {row.mae_p50:.4f} | {row.rmse_p50:.4f} | {row.brier_collapse:.4f} | {row.roc_auc_collapse:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Scope Guardrail",
            "",
            "These metrics are generated from the 2000-2020 design-basis baseline using historical and pseudo-prospective splits. They do not establish post-2021 real-time warning performance. `C-RSMamba` denotes the trained PyTorch selective state-space encoder implemented in this repository; the official `mamba-ssm` CUDA package is not claimed because it was not installable in the local Windows toolchain.",
        ]
    )
    ensure_dir(REPORT.parent)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log_event("07_evaluate_models", f"wrote {rel(REPORT)}")


if __name__ == "__main__":
    main()

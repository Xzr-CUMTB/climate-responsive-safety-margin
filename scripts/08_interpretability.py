from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd

from src.utils.project import ensure_dir, log_event, rel


MODEL_DIR = ROOT / "outputs" / "models"
FEATURE_SCHEMA = ROOT / "data" / "processed" / "modeling" / "feature_schema.txt"
OUT = ROOT / "outputs" / "tables" / "ridge_feature_coefficients.csv"
REPORT = ROOT / "outputs" / "reports" / "INTERPRETABILITY_REPORT.md"


def main() -> None:
    if not FEATURE_SCHEMA.exists():
        raise SystemExit(f"Missing feature schema: {FEATURE_SCHEMA}")
    log_event("08_interpretability", "started")
    features = FEATURE_SCHEMA.read_text(encoding="utf-8").splitlines()
    rows = []
    for horizon in [1, 3, 7]:
        model_path = MODEL_DIR / f"ridge_h{horizon}.joblib"
        if not model_path.exists():
            continue
        pipe = joblib.load(model_path)
        ridge = pipe.named_steps["ridge"]
        for feature, coef in zip(features, ridge.coef_):
            rows.append(
                {
                    "horizon_days": horizon,
                    "feature": feature,
                    "ridge_scaled_coefficient": float(coef),
                    "abs_coefficient": float(abs(coef)),
                }
            )
    df = pd.DataFrame(rows).sort_values(["horizon_days", "abs_coefficient"], ascending=[True, False])
    ensure_dir(OUT.parent)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    lines = [
        "# INTERPRETABILITY_REPORT",
        "",
        f"- Output: `{rel(OUT)}`",
        "",
        "The current interpretability output reports scaled Ridge coefficients for reproducible linear baselines. Histogram gradient boosting permutation importance is not run in this pass to keep the pipeline lightweight. Do not interpret these coefficients as causal effects.",
    ]
    ensure_dir(REPORT.parent)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log_event("08_interpretability", f"wrote {rel(OUT)}")


if __name__ == "__main__":
    main()


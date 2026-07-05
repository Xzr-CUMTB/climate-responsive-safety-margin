from __future__ import annotations

from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = ROOT / "outputs" / "logs" / "pipeline.log"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_event(step: str, message: str) -> None:
    ensure_dir(LOG_PATH.parent)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{stamp}] {step}: {message}\n")


def human_size(num_bytes: int | float | None) -> str:
    if num_bytes is None:
        return "NA"
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(value) < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


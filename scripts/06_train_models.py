from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_pinball_loss, mean_squared_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.utils.project import ensure_dir, log_event, rel


CRSM = ROOT / "data" / "processed" / "C_RSM" / "c_rsm_daily.parquet"
MODEL_DIR = ROOT / "outputs" / "models"
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT = ROOT / "outputs" / "reports" / "MODEL_TRAINING_AUDIT.md"
METRICS = TABLE_DIR / "model_evaluation_metrics.csv"
PERF_MAIN = TABLE_DIR / "model_performance_main.csv"
PERF_BY_HORIZON = TABLE_DIR / "model_performance_by_horizon.csv"
PERF_PSEUDO = TABLE_DIR / "model_performance_pseudo_prospective.csv"
ABLATION = TABLE_DIR / "ablation_results.csv"
CALIBRATION = TABLE_DIR / "calibration_results.csv"
PREDICTIONS = ROOT / "data" / "processed" / "modeling" / "test_predictions.parquet"
FEATURE_SCHEMA = ROOT / "data" / "processed" / "modeling" / "feature_schema.txt"
SCALER_META = ROOT / "data" / "processed" / "modeling" / "feature_scaler_main.json"


RANDOM_STATE = 42
SEQ_LEN = 30
HORIZONS = [1, 3, 7]
QUANTILES = [0.10, 0.50, 0.90]
MAX_TRAIN_WINDOWS = 60_000
MAX_VAL_WINDOWS = 15_000
MAX_TEST_WINDOWS = 50_000
HGB_TRAIN_ROWS = 80_000
BATCH_SIZE = 768
EPS = 1e-6

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

BASE_FEATURES = [
    "C_RSM",
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
    "relative_humidity_pct",
    "water_receptor_stress",
    "wq_recent_obs_count_30d",
    "park_area_km2",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "C_RSM_roll3",
    "C_RSM_roll7",
    "D_T_roll7",
    "D_P_roll7",
    "D_V_roll7",
    "D_W_roll7",
]


def module_present(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def set_seeds(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["park_id", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["dayofyear"] = df["date"].dt.dayofyear
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype("float32")
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype("float32")
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 366).astype("float32")
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 366).astype("float32")
    for col in ["C_RSM", "D_T", "D_P", "D_V", "D_W"]:
        df[f"{col}_roll7"] = (
            df.groupby("park_id", sort=False)[col]
            .rolling(7, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .astype("float32")
        )
    df["C_RSM_roll3"] = (
        df.groupby("park_id", sort=False)["C_RSM"]
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .astype("float32")
    )
    for horizon in HORIZONS:
        df[f"target_c_rsm_h{horizon}"] = df.groupby("park_id", sort=False)["C_RSM"].shift(-horizon)
        df[f"target_date_h{horizon}"] = df.groupby("park_id", sort=False)["date"].shift(-horizon)
        df[f"target_collapse_h{horizon}"] = (df[f"target_c_rsm_h{horizon}"] < 0).astype("float32")
    return df


def sample_indices(indices: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    if len(indices) <= max_rows:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=max_rows, replace=False))


@dataclass
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def build_split_indices(df: pd.DataFrame, split_name: str) -> SplitIndices:
    split = SPLITS[split_name]
    row_in_park = df.groupby("park_id", sort=False).cumcount().to_numpy()
    target_cols = [f"target_c_rsm_h{h}" for h in HORIZONS]
    valid_targets = df[target_cols].notna().all(axis=1).to_numpy()
    eligible = (row_in_park >= SEQ_LEN - 1) & valid_targets
    dates = df["date"]
    train_mask = eligible & (dates >= pd.Timestamp(split["train_start"])) & (dates <= pd.Timestamp(split["train_end"]))
    val_mask = eligible & (dates >= pd.Timestamp(split["val_start"])) & (dates <= pd.Timestamp(split["val_end"]))
    test_mask = eligible & (dates >= pd.Timestamp(split["test_start"])) & (dates <= pd.Timestamp(split["test_end"]))
    train = sample_indices(np.flatnonzero(train_mask), MAX_TRAIN_WINDOWS, RANDOM_STATE)
    val = sample_indices(np.flatnonzero(val_mask), MAX_VAL_WINDOWS, RANDOM_STATE + 1)
    test = sample_indices(np.flatnonzero(test_mask), MAX_TEST_WINDOWS, RANDOM_STATE + 2)
    return SplitIndices(train=train, val=val, test=test)


def standardize_features(df: pd.DataFrame, split_name: str, feature_cols: list[str]) -> tuple[np.ndarray, dict[str, list[float]]]:
    split = SPLITS[split_name]
    train_mask = (df["date"] >= pd.Timestamp(split["train_start"])) & (df["date"] <= pd.Timestamp(split["train_end"]))
    train_values = df.loc[train_mask, feature_cols].to_numpy(dtype=np.float32)
    mean = np.nanmean(train_values, axis=0)
    std = np.nanstd(train_values, axis=0)
    std[std < EPS] = 1.0
    values = df[feature_cols].to_numpy(dtype=np.float32)
    values = np.nan_to_num((values - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    return values, {"features": feature_cols, "mean": mean.tolist(), "std": std.tolist()}


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, df: pd.DataFrame, end_indices: np.ndarray):
        self.x = x
        self.end_indices = end_indices.astype(np.int64)
        self.y = df[[f"target_c_rsm_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
        self.c = df[[f"target_collapse_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.end_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        end = self.end_indices[idx]
        start = end - SEQ_LEN + 1
        return (
            torch.from_numpy(self.x[start : end + 1]),
            torch.from_numpy(self.y[end]),
            torch.from_numpy(self.c[end]),
        )


def make_loader(x: np.ndarray, df: pd.DataFrame, indices: np.ndarray, shuffle: bool) -> DataLoader:
    return DataLoader(
        WindowDataset(x, df, indices),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


class ForecastHead(nn.Module):
    def __init__(self, hidden_dim: int, n_horizons: int):
        super().__init__()
        self.quantile = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, n_horizons * 3))
        self.collapse = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, n_horizons))
        self.n_horizons = n_horizons

    def forward(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.quantile(pooled).view(-1, self.n_horizons, 3)
        q50 = raw[:, :, 1]
        q10 = q50 - torch.nn.functional.softplus(raw[:, :, 0])
        q90 = q50 + torch.nn.functional.softplus(raw[:, :, 2])
        q = torch.stack([q10, q50, q90], dim=-1)
        logits = self.collapse(pooled)
        return q, logits


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True, dropout=0.0)
        self.head = ForecastHead(hidden_dim, len(HORIZONS))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, (h, _) = self.rnn(x)
        return self.head(h[-1])


class GRUForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True, dropout=0.0)
        self.head = ForecastHead(hidden_dim, len(HORIZONS))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, h = self.rnn(x)
        return self.head(h[-1])


class TCNForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        layers = []
        for dilation in [1, 2, 4]:
            layers.extend(
                [
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation),
                    nn.GELU(),
                    nn.Dropout(0.05),
                ]
            )
        self.conv = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = ForecastHead(hidden_dim, len(HORIZONS))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.proj(x).transpose(1, 2)
        z = self.conv(z).transpose(1, 2)[:, -x.shape[1] :, :]
        return self.head(self.norm(z[:, -1]))


class TransformerForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, hidden_dim))
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2, dropout=0.05, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = ForecastHead(hidden_dim, len(HORIZONS))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.proj(x) + self.pos[:, : x.shape[1]]
        z = self.encoder(z)
        return self.head(z[:, -1])


class PatchTSTLiteForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96, patch_len: int = 5):
        super().__init__()
        self.patch = nn.Conv1d(input_dim, hidden_dim, kernel_size=patch_len, stride=patch_len)
        n_patches = math.ceil((SEQ_LEN - patch_len + 1) / patch_len)
        self.pos = nn.Parameter(torch.zeros(1, max(1, n_patches), hidden_dim))
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2, dropout=0.05, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = ForecastHead(hidden_dim, len(HORIZONS))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.patch(x.transpose(1, 2)).transpose(1, 2)
        z = z + self.pos[:, : z.shape[1]]
        z = self.encoder(z)
        return self.head(z[:, -1])


class SelectiveStateSpaceBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int = 5, gated: bool = True):
        super().__init__()
        self.gated = gated
        self.dwconv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=hidden_dim)
        self.delta = nn.Linear(hidden_dim, hidden_dim)
        self.b_proj = nn.Linear(hidden_dim, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.skip = nn.Parameter(torch.ones(hidden_dim))
        self.a_log = nn.Parameter(torch.zeros(hidden_dim))
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        u = self.dwconv(x.transpose(1, 2)).transpose(1, 2)[:, : x.shape[1], :]
        delta = torch.nn.functional.softplus(self.delta(u)) + 1e-3
        b_t = torch.tanh(self.b_proj(u))
        c_t = torch.tanh(self.c_proj(u))
        gate = torch.sigmoid(self.gate(u)) if self.gated else 1.0
        a = -torch.nn.functional.softplus(self.a_log).view(1, -1)
        state = torch.zeros(x.shape[0], x.shape[2], device=x.device, dtype=x.dtype)
        outs = []
        for t in range(x.shape[1]):
            decay = torch.exp(delta[:, t, :] * a)
            state = decay * state + (1.0 - decay) * b_t[:, t, :] * u[:, t, :]
            y = c_t[:, t, :] * state + self.skip.view(1, -1) * u[:, t, :]
            outs.append(y * gate[:, t, :] if self.gated else y)
        z = torch.stack(outs, dim=1)
        return self.norm(residual + self.out(z))


class CRSMambaForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96, n_layers: int = 2, gated: bool = True):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([SelectiveStateSpaceBlock(hidden_dim, gated=gated) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = ForecastHead(hidden_dim, len(HORIZONS))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.embed(x)
        for block in self.blocks:
            z = block(z)
        z = self.norm(z)
        pooled = 0.7 * z[:, -1] + 0.3 * z.mean(dim=1)
        return self.head(pooled)


def pinball_loss(q_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    qs = torch.tensor(QUANTILES, device=y.device, dtype=y.dtype).view(1, 1, 3)
    err = y.unsqueeze(-1) - q_pred
    return torch.maximum(qs * err, (qs - 1.0) * err).mean()


def combined_loss(q_pred: torch.Tensor, logits: torch.Tensor, y: torch.Tensor, collapse: torch.Tensor) -> torch.Tensor:
    return pinball_loss(q_pred, y) + 0.20 * torch.nn.functional.binary_cross_entropy_with_logits(logits, collapse)


@torch.no_grad()
def collect_model_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    qs = []
    probs = []
    ys = []
    for xb, yb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        q, logits = model(xb)
        qs.append(q.cpu().numpy())
        probs.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(qs), np.concatenate(probs), np.concatenate(ys)


def train_torch_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    save_path: Path | None = None,
) -> dict[str, float | str]:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for xb, yb, cb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            cb = cb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            q, logits = model(xb)
            loss = combined_loss(q, logits, yb, cb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(xb)
            n += len(xb)
        model.eval()
        val_total = 0.0
        val_n = 0
        with torch.no_grad():
            for xb, yb, cb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                cb = cb.to(device, non_blocking=True)
                q, logits = model(xb)
                loss = combined_loss(q, logits, yb, cb)
                val_total += float(loss.item()) * len(xb)
                val_n += len(xb)
        val_loss = val_total / max(val_n, 1)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        log_event("06_train_models", f"{model_name} epoch={epoch} train_loss={total/max(n,1):.5f} val_loss={val_loss:.5f}")
    if best_state is not None:
        model.load_state_dict(best_state)
        if save_path is not None:
            ensure_dir(save_path.parent)
            torch.save(
                {
                    "model_name": model_name,
                    "state_dict": best_state,
                    "seq_len": SEQ_LEN,
                    "horizons": HORIZONS,
                    "quantiles": QUANTILES,
                    "mamba_ssm_available": module_present("mamba_ssm"),
                    "encoder_description": "trainable Mamba-inspired selective state-space encoder" if "C-RSMamba" in model_name else model_name,
                },
                save_path,
            )
    return {"model": model_name, "best_val_loss": best_val, "best_epoch": best_epoch, "epochs": epochs}


def evaluate_arrays(
    y: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    prob: np.ndarray,
    y_bin: np.ndarray,
) -> dict[str, float]:
    q_stack = np.sort(np.column_stack([q10, q50, q90]).astype(np.float32), axis=1)
    q10, q50, q90 = q_stack[:, 0], q_stack[:, 1], q_stack[:, 2]
    out = {
        "mae_p50": float(mean_absolute_error(y, q50)),
        "rmse_p50": float(mean_squared_error(y, q50) ** 0.5),
        "pinball_p10": float(mean_pinball_loss(y, q10, alpha=0.10)),
        "pinball_p50": float(mean_pinball_loss(y, q50, alpha=0.50)),
        "pinball_p90": float(mean_pinball_loss(y, q90, alpha=0.90)),
        "p10_p90_coverage": float(((y >= q10) & (y <= q90)).mean()),
        "interval_width_p10_p90": float(np.mean(q90 - q10)),
        "brier_collapse": float(brier_score_loss(y_bin, np.clip(prob, 0, 1))),
    }
    try:
        out["roc_auc_collapse"] = float(roc_auc_score(y_bin, prob))
    except ValueError:
        out["roc_auc_collapse"] = np.nan
    return out


def rows_from_multi_predictions(
    model_name: str,
    split_name: str,
    df: pd.DataFrame,
    indices: np.ndarray,
    q: np.ndarray,
    prob: np.ndarray,
) -> tuple[list[dict[str, float | str | int]], pd.DataFrame]:
    metrics = []
    pred_frames = []
    for h_i, horizon in enumerate(HORIZONS):
        y = df.iloc[indices][f"target_c_rsm_h{horizon}"].to_numpy(dtype=np.float32)
        y_bin = (y < 0).astype(np.int8)
        q_h = np.sort(q[:, h_i, :], axis=1)
        prob_h = np.clip(prob[:, h_i], 0, 1)
        row = evaluate_arrays(y, q_h[:, 0], q_h[:, 1], q_h[:, 2], prob_h, y_bin)
        row.update({"split": split_name, "model": model_name, "horizon_days": horizon, "test_rows": len(indices)})
        metrics.append(row)
        pred_frames.append(
            pd.DataFrame(
                {
                    "split": split_name,
                    "model": model_name,
                    "horizon_days": horizon,
                    "park_id": df.iloc[indices]["park_id"].to_numpy(dtype=np.int32),
                    "date": df.iloc[indices]["date"].to_numpy(),
                    "target_date": df.iloc[indices][f"target_date_h{horizon}"].to_numpy(),
                    "target_c_rsm": y,
                    "q10": q_h[:, 0],
                    "q50": q_h[:, 1],
                    "q90": q_h[:, 2],
                    "collapse_probability": prob_h,
                }
            )
        )
    return metrics, pd.concat(pred_frames, ignore_index=True)


def train_flat_baselines(
    df: pd.DataFrame,
    feature_cols: list[str],
    indices: SplitIndices,
    split_name: str,
) -> tuple[list[dict[str, float | str | int]], pd.DataFrame]:
    metrics = []
    pred_frames = []
    train_s = sample_indices(indices.train, HGB_TRAIN_ROWS, RANDOM_STATE + 11)
    x_train = df.iloc[train_s][feature_cols].to_numpy(dtype=np.float32)
    x_test = df.iloc[indices.test][feature_cols].to_numpy(dtype=np.float32)
    current = df.iloc[indices.test]["C_RSM"].to_numpy(dtype=np.float32)

    for horizon in HORIZONS:
        y_train = df.iloc[train_s][f"target_c_rsm_h{horizon}"].to_numpy(dtype=np.float32)
        y_train_bin = (y_train < 0).astype(np.int8)
        y_test = df.iloc[indices.test][f"target_c_rsm_h{horizon}"].to_numpy(dtype=np.float32)
        y_test_bin = (y_test < 0).astype(np.int8)

        prob_persist = (current < 0).astype("float32")
        row = evaluate_arrays(y_test, current, current, current, prob_persist, y_test_bin)
        row.update({"split": split_name, "model": "persistence", "horizon_days": horizon, "train_rows": len(train_s), "test_rows": len(indices.test)})
        metrics.append(row)
        pred_frames.append(make_prediction_frame(df, indices.test, split_name, "persistence", horizon, y_test, current, current, current, prob_persist))

        ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=RANDOM_STATE))
        ridge.fit(x_train, y_train)
        train_pred = ridge.predict(x_train).astype("float32")
        resid = y_train - train_pred
        r10, r90 = np.quantile(resid, [0.10, 0.90])
        q50 = ridge.predict(x_test).astype("float32")
        q10 = (q50 + r10).astype("float32")
        q90 = (q50 + r90).astype("float32")
        prob = (1.0 / (1.0 + np.exp(10.0 * q50))).astype("float32")
        row = evaluate_arrays(y_test, q10, q50, q90, prob, y_test_bin)
        row.update({"split": split_name, "model": "ridge_empirical_interval", "horizon_days": horizon, "train_rows": len(train_s), "test_rows": len(indices.test)})
        metrics.append(row)
        if split_name == "main":
            joblib.dump(ridge, MODEL_DIR / f"ridge_h{horizon}.joblib")
        pred_frames.append(make_prediction_frame(df, indices.test, split_name, "ridge_empirical_interval", horizon, y_test, q10, q50, q90, prob))

        quantile_preds = {}
        for alpha, name in [(0.10, "q10"), (0.50, "q50"), (0.90, "q90")]:
            model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=alpha,
                max_iter=60,
                learning_rate=0.06,
                max_leaf_nodes=31,
                l2_regularization=0.01,
                random_state=RANDOM_STATE + horizon + int(alpha * 100),
            )
            model.fit(x_train, y_train)
            quantile_preds[name] = model.predict(x_test).astype("float32")
            if split_name == "main":
                joblib.dump(model, MODEL_DIR / f"hgb_{name}_h{horizon}.joblib")
        clf = HistGradientBoostingClassifier(
            max_iter=60,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=RANDOM_STATE + horizon,
        )
        if len(np.unique(y_train_bin)) > 1:
            clf.fit(x_train, y_train_bin)
            prob = clf.predict_proba(x_test)[:, 1].astype("float32")
            if split_name == "main":
                joblib.dump(clf, MODEL_DIR / f"hgb_collapse_h{horizon}.joblib")
        else:
            prob = np.full(len(x_test), float(y_train_bin.mean()), dtype=np.float32)
        row = evaluate_arrays(y_test, quantile_preds["q10"], quantile_preds["q50"], quantile_preds["q90"], prob, y_test_bin)
        row.update({"split": split_name, "model": "hist_gradient_boosting_quantile", "horizon_days": horizon, "train_rows": len(train_s), "test_rows": len(indices.test)})
        metrics.append(row)
        pred_frames.append(
            make_prediction_frame(
                df,
                indices.test,
                split_name,
                "hist_gradient_boosting_quantile",
                horizon,
                y_test,
                quantile_preds["q10"],
                quantile_preds["q50"],
                quantile_preds["q90"],
                prob,
            )
        )
    return metrics, pd.concat(pred_frames, ignore_index=True)


def make_prediction_frame(
    df: pd.DataFrame,
    indices: np.ndarray,
    split_name: str,
    model: str,
    horizon: int,
    y: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    prob: np.ndarray,
) -> pd.DataFrame:
    q_stack = np.sort(np.column_stack([q10, q50, q90]).astype(np.float32), axis=1)
    return pd.DataFrame(
        {
            "split": split_name,
            "model": model,
            "horizon_days": horizon,
            "park_id": df.iloc[indices]["park_id"].to_numpy(dtype=np.int32),
            "date": df.iloc[indices]["date"].to_numpy(),
            "target_date": df.iloc[indices][f"target_date_h{horizon}"].to_numpy(),
            "target_c_rsm": y,
            "q10": q_stack[:, 0],
            "q50": q_stack[:, 1],
            "q90": q_stack[:, 2],
            "collapse_probability": np.clip(prob, 0, 1),
        }
    )


def build_model(model_name: str, input_dim: int) -> nn.Module:
    if model_name == "LSTM":
        return LSTMForecaster(input_dim)
    if model_name == "GRU":
        return GRUForecaster(input_dim)
    if model_name == "TCN":
        return TCNForecaster(input_dim)
    if model_name == "Transformer":
        return TransformerForecaster(input_dim)
    if model_name == "PatchTST-lite":
        return PatchTSTLiteForecaster(input_dim)
    if model_name == "C-RSMamba":
        return CRSMambaForecaster(input_dim)
    raise ValueError(model_name)


def run_torch_models(
    df: pd.DataFrame,
    feature_array: np.ndarray,
    indices: SplitIndices,
    split_name: str,
    model_names: list[str],
    device: torch.device,
) -> tuple[list[dict[str, float | str | int]], list[dict[str, float | str]], pd.DataFrame, nn.Module | None]:
    train_loader = make_loader(feature_array, df, indices.train, shuffle=True)
    val_loader = make_loader(feature_array, df, indices.val, shuffle=False)
    test_loader = make_loader(feature_array, df, indices.test, shuffle=False)
    all_metrics = []
    train_records = []
    pred_frames = []
    crsmamba_model = None
    epochs = {"C-RSMamba": 3, "LSTM": 1, "GRU": 1, "TCN": 1, "Transformer": 1, "PatchTST-lite": 1}
    for model_name in model_names:
        set_seeds(RANDOM_STATE)
        model = build_model(model_name, feature_array.shape[1])
        save_path = MODEL_DIR / "crsmamba_best.pt" if (split_name == "main" and model_name == "C-RSMamba") else None
        if split_name == "pseudo_prospective" and model_name == "C-RSMamba":
            save_path = MODEL_DIR / "crsmamba_pseudo_prospective_best.pt"
        record = train_torch_model(model_name, model, train_loader, val_loader, device, epochs[model_name], save_path=save_path)
        record["split"] = split_name
        train_records.append(record)
        q, prob, _ = collect_model_predictions(model.to(device), test_loader, device)
        rows, preds = rows_from_multi_predictions(model_name, split_name, df, indices.test, q, prob)
        all_metrics.extend(rows)
        pred_frames.append(preds)
        if split_name == "main" and model_name == "C-RSMamba":
            crsmamba_model = model.to(device)
    return all_metrics, train_records, pd.concat(pred_frames, ignore_index=True), crsmamba_model


@torch.no_grad()
def build_mask_ablation(
    model: nn.Module,
    feature_array: np.ndarray,
    df: pd.DataFrame,
    indices: SplitIndices,
    feature_cols: list[str],
    device: torch.device,
) -> pd.DataFrame:
    if model is None:
        return pd.DataFrame()
    masks = {
        "full": [],
        "water_features_masked": ["S_W", "D_W", "margin_W", "water_receptor_stress", "wq_recent_obs_count_30d"],
        "climate_stress_components_masked": ["S_T", "S_P", "S_V", "D_T", "D_P", "D_V", "margin_T", "margin_P", "margin_V"],
        "static_area_masked": ["park_area_km2"],
    }
    rows = []
    for ablation_name, cols in masks.items():
        x = feature_array.copy()
        for col in cols:
            if col in feature_cols:
                x[:, feature_cols.index(col)] = 0.0
        loader = make_loader(x, df, indices.test, shuffle=False)
        q, prob, _ = collect_model_predictions(model, loader, device)
        metrics, _ = rows_from_multi_predictions("C-RSMamba", "main", df, indices.test, q, prob)
        for row in metrics:
            row["ablation"] = ablation_name
            rows.append(row)
    return pd.DataFrame(rows)


def calibration_table(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    crs = preds[(preds["split"] == "main") & (preds["model"] == "C-RSMamba")].copy()
    if crs.empty:
        return pd.DataFrame()
    for horizon, group in crs.groupby("horizon_days"):
        group = group.copy()
        group["collapse_observed"] = (group["target_c_rsm"] < 0).astype(int)
        group["bin"] = pd.cut(group["collapse_probability"], bins=np.linspace(0, 1, 11), include_lowest=True)
        for bin_label, b in group.groupby("bin", observed=False):
            if len(b) == 0:
                continue
            rows.append(
                {
                    "model": "C-RSMamba",
                    "horizon_days": horizon,
                    "probability_bin": str(bin_label),
                    "n": len(b),
                    "mean_predicted_probability": b["collapse_probability"].mean(),
                    "observed_collapse_rate": b["collapse_observed"].mean(),
                    "p10_p90_coverage": ((b["target_c_rsm"] >= b["q10"]) & (b["target_c_rsm"] <= b["q90"])).mean(),
                    "mean_interval_width": (b["q90"] - b["q10"]).mean(),
                }
            )
    return pd.DataFrame(rows)


def write_report(metrics: pd.DataFrame, train_records: pd.DataFrame, split_counts: dict[str, dict[str, int]], device: torch.device) -> None:
    ensure_dir(REPORT.parent)
    main = metrics[metrics["split"] == "main"].copy()
    best = main.sort_values(["horizon_days", "mae_p50"]).groupby("horizon_days").first().reset_index()
    lines = [
        "# MODEL_TRAINING_AUDIT",
        "",
        f"- Python executable: `{sys.executable}`",
        f"- Torch version: {torch.__version__}",
        f"- Torch CUDA version: {torch.version.cuda}",
        f"- CUDA available: {torch.cuda.is_available()}",
        f"- Device used: {device}",
        f"- GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}",
        f"- official mamba_ssm available: {module_present('mamba_ssm')}",
        "- Encoder used for C-RSMamba: trainable Mamba-inspired selective state-space encoder with input-dependent decay, depthwise temporal convolution and gated state updates.",
        f"- Sequence length: {SEQ_LEN} days",
        f"- Horizons: {HORIZONS}",
        "",
        "## Split Counts",
        "",
        "| Split | Train windows | Validation windows | Test windows |",
        "|---|---:|---:|---:|",
    ]
    for split, counts in split_counts.items():
        lines.append(f"| {split} | {counts['train']} | {counts['val']} | {counts['test']} |")
    lines.extend(["", "## Best Main-Split Models By Median MAE", "", "| Horizon | Model | MAE | RMSE | Brier | ROC AUC |", "|---:|---|---:|---:|---:|---:|"])
    for row in best.itertuples(index=False):
        lines.append(f"| {row.horizon_days} | {row.model} | {row.mae_p50:.4f} | {row.rmse_p50:.4f} | {row.brier_collapse:.4f} | {row.roc_auc_collapse:.4f} |")
    lines.extend(
        [
            "",
            "## Training Records",
            "",
            "```text",
            train_records.to_string(index=False) if not train_records.empty else "No training records.",
            "```",
            "",
            "## Scope Guardrail",
            "",
            "The official `mamba-ssm` package was attempted but was not installable in this Windows environment because the build required `nvcc`. The submitted model is therefore named honestly as C-RSMamba with a Mamba-inspired selective state-space encoder, not as the official CUDA Mamba kernel implementation.",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not CRSM.exists():
        raise SystemExit(f"Missing C-RSM file: {CRSM}")
    set_seeds()
    ensure_dir(MODEL_DIR)
    ensure_dir(TABLE_DIR)
    ensure_dir(PREDICTIONS.parent)
    log_event("06_train_models", "started")
    df = pd.read_parquet(CRSM)
    df = make_features(df)
    missing = [col for col in BASE_FEATURES if col not in df.columns]
    if missing:
        raise SystemExit(f"Missing required feature columns: {missing}")
    FEATURE_SCHEMA.write_text("\n".join(BASE_FEATURES), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_metrics = []
    all_predictions = []
    train_records = []
    split_counts = {}

    for split_name in ["main", "pseudo_prospective"]:
        indices = build_split_indices(df, split_name)
        split_counts[split_name] = {"train": len(indices.train), "val": len(indices.val), "test": len(indices.test)}
        feature_array, scaler = standardize_features(df, split_name, BASE_FEATURES)
        if split_name == "main":
            SCALER_META.write_text(json.dumps(scaler, indent=2), encoding="utf-8")

        flat_metrics, flat_preds = train_flat_baselines(df, BASE_FEATURES, indices, split_name)
        all_metrics.extend(flat_metrics)
        all_predictions.append(flat_preds)

        model_names = ["C-RSMamba", "LSTM", "GRU", "TCN", "Transformer", "PatchTST-lite"] if split_name == "main" else ["C-RSMamba"]
        torch_metrics, records, torch_preds, crsmamba_model = run_torch_models(df, feature_array, indices, split_name, model_names, device)
        all_metrics.extend(torch_metrics)
        train_records.extend(records)
        all_predictions.append(torch_preds)

        if split_name == "main":
            ablation = build_mask_ablation(crsmamba_model, feature_array, df, indices, BASE_FEATURES, device)
            ablation.to_csv(ABLATION, index=False, encoding="utf-8-sig")

    metrics_df = pd.DataFrame(all_metrics)
    preds_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df.to_csv(METRICS, index=False, encoding="utf-8-sig")
    metrics_df[metrics_df["split"] == "main"].to_csv(PERF_MAIN, index=False, encoding="utf-8-sig")
    metrics_df.groupby(["split", "model", "horizon_days"], as_index=False).first().to_csv(PERF_BY_HORIZON, index=False, encoding="utf-8-sig")
    metrics_df[metrics_df["split"] == "pseudo_prospective"].to_csv(PERF_PSEUDO, index=False, encoding="utf-8-sig")
    preds_df.to_parquet(PREDICTIONS, index=False)
    calibration_table(preds_df).to_csv(CALIBRATION, index=False, encoding="utf-8-sig")
    train_records_df = pd.DataFrame(train_records)
    train_records_df.to_csv(TABLE_DIR / "torch_training_records.csv", index=False, encoding="utf-8-sig")
    write_report(metrics_df, train_records_df, split_counts, device)
    log_event("06_train_models", f"wrote metrics to {rel(METRICS)} and model to {rel(MODEL_DIR / 'crsmamba_best.pt')}")


if __name__ == "__main__":
    main()

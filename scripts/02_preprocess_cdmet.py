from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import xarray as xr

from src.utils.project import ensure_dir, log_event, rel


CDMET = ROOT / "data" / "raw" / "CDMet"
CIPS_OBJECTS = ROOT / "data" / "processed" / "CIPs_objects" / "cips_objects.parquet"
OUT_DIR = ROOT / "data" / "processed" / "park_daily_table"
OUT_FILE = OUT_DIR / "park_daily_2000_2020.parquet"
GRID_MAP = OUT_DIR / "park_grid_mapping.csv"
REPORT = ROOT / "outputs" / "reports" / "CDMET_PREPROCESS_REPORT.md"


VARIABLES = {
    "maxtmp": {
        "folder": "Maximum Temperature",
        "pattern": "CDMet_maxtmp_{year}.nc",
        "var": "maxtmp",
        "output": "tmax_c",
        "unit": "degC",
        "transform": "kelvin_to_celsius",
    },
    "pre": {
        "folder": "Total Precipitation",
        "pattern": "CDMet_pre_{year}.nc",
        "var": "pre",
        "output": "precip_mm",
        "unit": "mm d^-1",
        "transform": "identity",
    },
    "win": {
        "folder": "Wind",
        "pattern": "CDMet_win_{year}.nc",
        "var": "win",
        "output": "wind_ms",
        "unit": "m s^-1",
        "transform": "identity",
    },
    "rhu": {
        "folder": "Relative humidity",
        "pattern": "CDMet_rhu_{year}.nc",
        "var": "rhu",
        "output": "relative_humidity_pct",
        "unit": "%",
        "transform": "identity",
    },
}


def nearest_indices(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    grid = np.asarray(grid, dtype=float)
    return np.abs(grid[:, None] - values[None, :]).argmin(axis=0).astype(np.int32)


def load_grid() -> tuple[np.ndarray, np.ndarray]:
    sample = CDMET / "Maximum Temperature" / "CDMet_maxtmp_2000.nc"
    if not sample.exists():
        raise SystemExit(f"Missing CDMet sample file: {sample}")
    ds = xr.open_dataset(sample, decode_times=False)
    lon = ds["lon"].to_numpy()
    lat = ds["lat"].to_numpy()
    ds.close()
    return lon, lat


def build_grid_mapping(objects: pd.DataFrame) -> pd.DataFrame:
    lon_grid, lat_grid = load_grid()
    lon_idx = nearest_indices(objects["centroid_lon"].to_numpy(), lon_grid)
    lat_idx = nearest_indices(objects["centroid_lat"].to_numpy(), lat_grid)
    mapping = objects[
        ["component_id", "centroid_lon", "centroid_lat", "park_area_km2", "pixel_count"]
    ].copy()
    mapping = mapping.rename(columns={"component_id": "park_id"})
    mapping["cdmet_lon_idx"] = lon_idx
    mapping["cdmet_lat_idx"] = lat_idx
    mapping["cdmet_lon"] = lon_grid[lon_idx]
    mapping["cdmet_lat"] = lat_grid[lat_idx]
    mapping["cdmet_grid_id"] = mapping["cdmet_lat_idx"].astype(str) + "_" + mapping["cdmet_lon_idx"].astype(str)
    ensure_dir(OUT_DIR)
    mapping.to_csv(GRID_MAP, index=False, encoding="utf-8-sig")
    return mapping


def extract_variable(year: int, spec: dict[str, str], unique_grids: pd.DataFrame, inverse: np.ndarray) -> np.ndarray:
    path = CDMET / spec["folder"] / spec["pattern"].format(year=year)
    if not path.exists():
        raise FileNotFoundError(path)
    ds = xr.open_dataset(path, decode_times=False)
    series = []
    data_array = ds[spec["var"]]
    # The scipy NetCDF backend used in this environment does not support xarray
    # vectorized indexing. Point-wise extraction keeps memory bounded and avoids
    # loading a full lat-lon rectangle.
    for row in unique_grids.itertuples(index=False):
        series.append(
            data_array.isel(lat=int(row.cdmet_lat_idx), lon=int(row.cdmet_lon_idx)).to_numpy()
        )
    arr = np.stack(series, axis=1)
    ds.close()
    # arr shape: time x unique_grid. Map back to parks.
    arr = arr[:, inverse]
    arr = arr.astype("float32", copy=False)
    if spec["transform"] == "kelvin_to_celsius":
        arr = arr - np.float32(273.15)
    return arr


def process_year(year: int, mapping: pd.DataFrame) -> pd.DataFrame:
    unique_grids = mapping[["cdmet_grid_id", "cdmet_lat_idx", "cdmet_lon_idx"]].drop_duplicates("cdmet_grid_id").reset_index(drop=True)
    grid_to_pos = {grid_id: pos for pos, grid_id in enumerate(unique_grids["cdmet_grid_id"])}
    inverse = mapping["cdmet_grid_id"].map(grid_to_pos).to_numpy(dtype=np.int32)
    arrays = {}
    time_len = None
    for name, spec in VARIABLES.items():
        arr = extract_variable(year, spec, unique_grids, inverse)
        arrays[spec["output"]] = arr
        time_len = arr.shape[0]
        log_event("02_preprocess_cdmet", f"year={year} variable={name} shape={arr.shape}")
    assert time_len is not None
    dates = pd.date_range(f"{year}-01-01", periods=time_len, freq="D")
    n_parks = len(mapping)
    data = {
        "park_id": np.repeat(mapping["park_id"].to_numpy(dtype=np.int32), time_len),
        "date": np.tile(dates.to_numpy(), n_parks),
        "centroid_lon": np.repeat(mapping["centroid_lon"].to_numpy(dtype=np.float32), time_len),
        "centroid_lat": np.repeat(mapping["centroid_lat"].to_numpy(dtype=np.float32), time_len),
        "park_area_km2": np.repeat(mapping["park_area_km2"].to_numpy(dtype=np.float32), time_len),
        "cdmet_lon": np.repeat(mapping["cdmet_lon"].to_numpy(dtype=np.float32), time_len),
        "cdmet_lat": np.repeat(mapping["cdmet_lat"].to_numpy(dtype=np.float32), time_len),
    }
    for output_name, arr in arrays.items():
        data[output_name] = arr.T.reshape(-1)
    return pd.DataFrame(data)


def write_report(df: pd.DataFrame, mapping: pd.DataFrame) -> None:
    ensure_dir(REPORT.parent)
    lines = [
        "# CDMET_PREPROCESS_REPORT",
        "",
        f"- CIPs objects used: {len(mapping):,}",
        f"- Unique CDMet grid cells sampled: {mapping['cdmet_grid_id'].nunique():,}",
        f"- Daily park rows: {len(df):,}",
        f"- Date range: {df['date'].min().date()} to {df['date'].max().date()}",
        f"- Output: `{rel(OUT_FILE)}`",
        f"- Grid mapping: `{rel(GRID_MAP)}`",
        "",
        "## Units",
        "",
        "- `tmax_c`: maximum temperature converted from Kelvin to degrees Celsius.",
        "- `precip_mm`: daily total precipitation in millimetres.",
        "- `wind_ms`: 10 m wind speed in m s^-1.",
        "- `relative_humidity_pct`: relative humidity in percent.",
        "",
        "## Scope",
        "",
        "The extracted CDMet table is a 2000-2020 design-basis climate forcing baseline under a fixed 2021 CIPs exposure layer. It is not a post-2021 real-time risk table.",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not CIPS_OBJECTS.exists():
        raise SystemExit(f"Missing CIPs objects. Run scripts/01_preprocess_cips.py first: {CIPS_OBJECTS}")
    log_event("02_preprocess_cdmet", "started")
    objects = pd.read_parquet(CIPS_OBJECTS)
    mapping = build_grid_mapping(objects)
    frames = []
    for year in range(2000, 2021):
        frames.append(process_year(year, mapping))
        log_event("02_preprocess_cdmet", f"completed year={year}")
    df = pd.concat(frames, ignore_index=True)
    ensure_dir(OUT_DIR)
    df.to_parquet(OUT_FILE, index=False)
    write_report(df, mapping)
    log_event("02_preprocess_cdmet", f"wrote {len(df):,} rows to {rel(OUT_FILE)}")


if __name__ == "__main__":
    main()

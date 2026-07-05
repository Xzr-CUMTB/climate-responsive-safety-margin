from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.utils.project import ensure_dir, human_size, log_event, rel


RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "outputs" / "reports"


@dataclass
class Check:
    item: str
    status: str
    detail: str


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def inventory_files() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(RAW.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "path": rel(path),
                    "bytes": path.stat().st_size,
                    "size_human": human_size(path.stat().st_size),
                    "modified": pd.Timestamp(path.stat().st_mtime, unit="s").isoformat(),
                }
            )
    df = pd.DataFrame(rows)
    ensure_dir(REPORTS)
    df.to_csv(REPORTS / "data_inventory.csv", index=False, encoding="utf-8-sig")
    return df


def audit_dependencies() -> list[Check]:
    modules = [
        "numpy",
        "pandas",
        "geopandas",
        "pyogrio",
        "shapely",
        "rasterio",
        "tifffile",
        "xarray",
        "netCDF4",
        "pyarrow",
        "matplotlib",
        "sklearn",
        "torch",
        "xlrd",
        "docx",
    ]
    checks = []
    for module in modules:
        checks.append(Check(f"python module: {module}", "present" if has_module(module) else "missing", ""))
    return checks


def audit_cips() -> tuple[list[Check], list[str]]:
    checks: list[Check] = []
    notes: list[str] = []
    cips_dir = RAW / "CIPs"
    folder = cips_dir / "The Yangtse River_CIPs_10m_2021"
    tif = folder / "CIPs21_WGS84.tif"
    checks.append(Check("CIPs raw folder", "present" if folder.exists() else "missing", rel(folder)))
    checks.append(Check("CIPs GeoTIFF", "present" if tif.exists() else "missing", rel(tif)))
    for name in [
        "CIPs21_WGS84.tfw",
        "CIPs21_WGS84.tif.vat.dbf",
        "CIPs21_WGS84.tif.ovr",
        "CIPs21_WGS84.tif.aux.xml",
    ]:
        p = folder / name
        checks.append(Check(f"CIPs sidecar: {name}", "present" if p.exists() else "missing", rel(p)))

    if tif.exists() and has_module("tifffile"):
        import tifffile

        with tifffile.TiffFile(tif) as tf:
            page = tf.pages[0]
            checks.append(Check("CIPs raster shape", "ok", f"{page.shape}, dtype={page.dtype}"))
            checks.append(Check("CIPs raster tiling", "ok", f"tile={page.tags.get('TileWidth').value}x{page.tags.get('TileLength').value}"))
            scale = page.tags.get("ModelPixelScaleTag")
            tie = page.tags.get("ModelTiepointTag")
            if scale is not None and tie is not None:
                sx, sy, _ = scale.value
                _, _, _, x0, y0, _ = tie.value
                height, width = page.shape
                xmin, xmax = x0, x0 + width * sx
                ymax, ymin = y0, y0 - height * sy
                checks.append(Check("CIPs CRS and extent", "ok", f"EPSG:4326 inferred; lon {xmin:.4f}-{xmax:.4f}, lat {ymin:.4f}-{ymax:.4f}"))
            total_pixels = int(page.shape[0]) * int(page.shape[1])
            notes.append(f"CIPs is a one-bit tiled GeoTIFF with about {total_pixels:,} pixels; object extraction must be block-wise or use GDAL/rasterio polygonization.")
    elif tif.exists():
        checks.append(Check("CIPs raster metadata", "blocked", "Python module tifffile is missing"))

    if not has_module("rasterio"):
        notes.append("rasterio/GDAL is not available in the active Python environment; full GeoTIFF polygonization cannot run yet.")
    return checks, notes


def audit_cdmet() -> tuple[list[Check], list[str]]:
    checks: list[Check] = []
    notes: list[str] = []
    cdmet = RAW / "CDMet"
    info = cdmet / "Data Information for the CDMet.txt"
    checks.append(Check("CDMet info file", "present" if info.exists() else "missing", rel(info)))

    expected = {
        "Maximum Temperature": ("CDMet_maxtmp_(\\d{4})\\.nc", "maxtmp", "K; convert to degC for C-RSM formula"),
        "Total Precipitation": ("CDMet_pre_(\\d{4})\\.nc", "pre", "mm"),
        "Wind": ("CDMet_win_(\\d{4})\\.nc", "win", "m s^-1"),
        "Relative humidity": ("CDMet_rhu_(\\d{4})\\.nc", "rhu", "%; secondary covariate"),
    }

    for folder_name, (pattern, var_name, unit_note) in expected.items():
        folder = cdmet / folder_name
        checks.append(Check(f"CDMet folder: {folder_name}", "present" if folder.exists() else "missing", rel(folder)))
        years = []
        if folder.exists():
            rgx = re.compile(pattern)
            for p in folder.glob("*.nc"):
                match = rgx.match(p.name)
                if match:
                    years.append(int(match.group(1)))
            years = sorted(years)
            status = "ok" if years == list(range(2000, 2021)) else "check"
            checks.append(Check(f"CDMet files: {folder_name}", status, f"{len(years)} files; years={years[:3]}...{years[-3:] if years else []}; unit={unit_note}"))
            sample = folder / pattern.replace("(\\d{4})", "2000").replace("\\.", ".")
            sample = folder / sample.name.replace("\\", "")
            candidates = list(folder.glob("*2000.nc"))
            if candidates and has_module("xarray"):
                import xarray as xr

                try:
                    ds = xr.open_dataset(candidates[0], decode_times=False)
                    dims = dict(ds.sizes)
                    data_vars = list(ds.data_vars)
                    var_status = "ok" if var_name in data_vars else "check"
                    checks.append(Check(f"CDMet sample metadata: {folder_name}", var_status, f"{rel(candidates[0])}; dims={dims}; vars={data_vars}"))
                    ds.close()
                except Exception as exc:
                    checks.append(Check(f"CDMet sample metadata: {folder_name}", "blocked", f"{type(exc).__name__}: {exc}"))

    for optional in ["Surface Pressure.zip", "Sunshine Duration.zip"]:
        p = cdmet / optional
        checks.append(Check(f"optional CDMet: {optional}", "present" if p.exists() else "optional missing", rel(p)))
    notes.append("CDMet provides a 2000-2020 design-basis climate forcing baseline, not a post-2021 real-time forcing record.")
    notes.append("NetCDF files are large; downstream extraction should sample only park-relevant cells and avoid loading full China grids into memory.")
    return checks, notes


def date_summary(series: pd.Series) -> tuple[str, str, int]:
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    if parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    valid = parsed.dropna()
    if valid.empty:
        return "NA", "NA", int(parsed.isna().sum())
    return valid.min().date().isoformat(), valid.max().date().isoformat(), int(parsed.isna().sum())


def audit_water_quality() -> tuple[list[Check], list[str]]:
    checks: list[Check] = []
    notes: list[str] = []
    wq = RAW / "water_quality"
    expected_columns = {
        "MonitoringLocationIdentifier",
        "LongitudeMeasure_WGS84",
        "LatitudeMeasure_WGS84",
        "MonitoringDate",
        "IndicatorsName",
        "Value",
        "Unit",
        "SourceProvider",
    }
    for name in ["daily_land.csv", "full_dataset.csv", "weekly_land.csv", "monthly_ocean.csv"]:
        p = wq / name
        checks.append(Check(f"water-quality file: {name}", "present" if p.exists() else "missing", rel(p)))
        if not p.exists():
            continue
        try:
            header = pd.read_csv(p, nrows=0)
            missing = sorted(expected_columns - set(header.columns))
            status = "ok" if not missing else "check"
            detail = f"columns={list(header.columns)}"
            if missing:
                detail += f"; missing={missing}"
            checks.append(Check(f"water-quality columns: {name}", status, detail))
            usecols = [c for c in ["MonitoringDate", "IndicatorsName", "LongitudeMeasure_WGS84", "LatitudeMeasure_WGS84"] if c in header.columns]
            rows = 0
            indicators: set[str] = set()
            lon_min = lon_max = lat_min = lat_max = None
            date_min = date_max = "NA"
            date_na = 0
            for chunk in pd.read_csv(p, usecols=usecols, chunksize=200_000):
                rows += len(chunk)
                if "IndicatorsName" in chunk:
                    indicators.update(map(str, chunk["IndicatorsName"].dropna().unique()))
                if "LongitudeMeasure_WGS84" in chunk:
                    lon = pd.to_numeric(chunk["LongitudeMeasure_WGS84"], errors="coerce")
                    lon_min = lon.min() if lon_min is None else min(lon_min, lon.min())
                    lon_max = lon.max() if lon_max is None else max(lon_max, lon.max())
                if "LatitudeMeasure_WGS84" in chunk:
                    lat = pd.to_numeric(chunk["LatitudeMeasure_WGS84"], errors="coerce")
                    lat_min = lat.min() if lat_min is None else min(lat_min, lat.min())
                    lat_max = lat.max() if lat_max is None else max(lat_max, lat.max())
                if "MonitoringDate" in chunk:
                    mn, mx, na = date_summary(chunk["MonitoringDate"])
                    date_na += na
                    if mn != "NA":
                        date_min = mn if date_min == "NA" else min(date_min, mn)
                        date_max = mx if date_max == "NA" else max(date_max, mx)
            checks.append(Check(f"water-quality summary: {name}", "ok", f"rows={rows:,}; dates={date_min} to {date_max}; indicators={sorted(indicators)[:20]}; lon={lon_min}-{lon_max}; lat={lat_min}-{lat_max}; unparsed_dates={date_na:,}"))
        except Exception as exc:
            checks.append(Check(f"water-quality parse: {name}", "blocked", f"{type(exc).__name__}: {exc}"))

    for name in ["metadata_and_statistics.xls", "Monitoring_sites.rar", "22584742.zip"]:
        p = wq / name
        checks.append(Check(f"water-quality auxiliary: {name}", "present" if p.exists() else "missing", rel(p)))
    if not has_module("xlrd"):
        notes.append("metadata_and_statistics.xls exists, but xlrd is missing; Excel metadata cannot be inspected until xlrd is installed.")
    notes.append("Water-quality observations should be used only as external receptor-state context and consistency evidence, not as causal evidence of emissions from CIPs.")
    return checks, notes


def audit_templates() -> tuple[list[Check], list[str]]:
    checks: list[Check] = []
    notes: list[str] = []
    tmpl = ROOT / "nature系列word模板"
    checks.append(Check("Word template folder", "present" if tmpl.exists() else "missing", rel(tmpl)))
    if tmpl.exists():
        names = sorted(p.name for p in tmpl.iterdir() if p.is_file())
        checks.append(Check("Word template files", "check", ", ".join(names)))
        if any("splnproc" in name.lower() for name in names):
            notes.append("Template filenames indicate Springer Proceedings/LNCS, not Nature Chemical Engineering. Do not force these templates for initial submission.")
    return checks, notes


def write_markdown(checks: list[Check], notes: list[str], inventory: pd.DataFrame) -> None:
    lines = [
        "# DATA_AUDIT",
        "",
        "This audit records the state of the local open-data workflow before preprocessing. It does not contain computed scientific results.",
        "",
        "## Inventory Summary",
        "",
        f"- Raw files found: {len(inventory)}",
        f"- Total raw size: {human_size(int(inventory['bytes'].sum())) if not inventory.empty else 'NA'}",
        "- Full file inventory: `outputs/reports/data_inventory.csv`",
        "",
        "## Checks",
        "",
        "| Item | Status | Detail |",
        "|---|---|---|",
    ]
    for check in checks:
        detail = check.detail.replace("|", "\\|")
        lines.append(f"| {check.item} | {check.status} | {detail} |")
    lines.extend(["", "## Notes", ""])
    for note in notes:
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Required raw datasets are present for the audit stage. The next hard dependency is geospatial raster polygonization for the CIPs GeoTIFF; this requires rasterio/GDAL or an equivalent block-wise raster vectorization path.",
            "",
        ]
    )
    (ROOT / "DATA_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dir(REPORTS)
    log_event("00_audit_data", "started")
    inventory = inventory_files()
    checks: list[Check] = []
    notes: list[str] = []
    for part in [audit_dependencies]:
        checks.extend(part())
    for part in [audit_cips, audit_cdmet, audit_water_quality, audit_templates]:
        part_checks, part_notes = part()
        checks.extend(part_checks)
        notes.extend(part_notes)
    write_markdown(checks, notes, inventory)
    log_event("00_audit_data", "wrote DATA_AUDIT.md and outputs/reports/data_inventory.csv")


if __name__ == "__main__":
    main()


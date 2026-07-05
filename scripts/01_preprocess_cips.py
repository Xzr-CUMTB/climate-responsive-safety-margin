from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.utils.project import ensure_dir, log_event, rel


RAW_TIF = ROOT / "data" / "raw" / "CIPs" / "The Yangtse River_CIPs_10m_2021" / "CIPs21_WGS84.tif"
OUT_DIR = ROOT / "data" / "processed" / "CIPs_objects"
REPORT = ROOT / "outputs" / "reports" / "CIPS_PREPROCESS_REPORT.md"


def require_modules() -> None:
    missing = []
    for name in ["tifffile", "geopandas", "shapely", "pyarrow"]:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise SystemExit(f"Missing required Python modules for CIPs preprocessing: {', '.join(missing)}")


def read_true_pixels() -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    import tifffile

    rows_parts: list[np.ndarray] = []
    cols_parts: list[np.ndarray] = []
    with tifffile.TiffFile(RAW_TIF) as tif:
        page = tif.pages[0]
        height, width = page.shape
        scale = page.tags["ModelPixelScaleTag"].value
        tie = page.tags["ModelTiepointTag"].value
        sx, sy, _ = scale
        _, _, _, x0, y0, _ = tie
        meta = {"height": height, "width": width, "sx": sx, "sy": sy, "x0": x0, "y0": y0}

        for idx, (data, indices, _) in enumerate(page.segments(maxworkers=1)):
            if data is None:
                continue
            row0 = int(indices[2])
            col0 = int(indices[3])
            tile = data[0, :, :, 0]
            tile = tile[: max(0, min(tile.shape[0], height - row0)), : max(0, min(tile.shape[1], width - col0))]
            if not tile.any():
                continue
            rr, cc = np.nonzero(tile)
            rows_parts.append((rr + row0).astype(np.uint32))
            cols_parts.append((cc + col0).astype(np.uint32))
            if idx and idx % 50_000 == 0:
                log_event("01_preprocess_cips", f"scanned {idx:,} tiles; true pixels so far={sum(len(x) for x in rows_parts):,}")

    rows = np.concatenate(rows_parts) if rows_parts else np.array([], dtype=np.uint32)
    cols = np.concatenate(cols_parts) if cols_parts else np.array([], dtype=np.uint32)
    return rows, cols, meta


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.uint8)

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return int(x)

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        rank = self.rank
        parent = self.parent
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1


def label_components(rows: np.ndarray, cols: np.ndarray, width: int) -> np.ndarray:
    order = np.lexsort((cols, rows))
    rows = rows[order]
    cols = cols[order]
    lin = rows.astype(np.uint64) * np.uint64(width) + cols.astype(np.uint64)
    n = len(lin)
    uf = UnionFind(n)
    index_by_lin = {int(v): i for i, v in enumerate(lin)}
    for i, value in enumerate(lin):
        r = int(rows[i])
        c = int(cols[i])
        candidates = []
        if c > 0:
            candidates.append(int(value - 1))
        if r > 0:
            candidates.append(int(value - width))
            if c > 0:
                candidates.append(int(value - width - 1))
            if c + 1 < width:
                candidates.append(int(value - width + 1))
        for nb in candidates:
            j = index_by_lin.get(nb)
            if j is not None:
                uf.union(i, j)
        if i and i % 250_000 == 0:
            log_event("01_preprocess_cips", f"union-find processed {i:,}/{n:,} true pixels")
    roots = np.array([uf.find(i) for i in range(n)], dtype=np.int32)
    dense_codes, dense = pd.factorize(roots, sort=True)
    labels_sorted = dense_codes.astype(np.int32) + 1
    labels = np.empty_like(labels_sorted)
    labels[order] = labels_sorted
    return labels


def pixel_area_m2(lat: pd.Series, sx: float, sy: float) -> pd.Series:
    # WGS84 approximate cell area at centroid latitude.
    lat_rad = np.deg2rad(lat.astype(float))
    m_per_deg_lat = 111_132.92 - 559.82 * np.cos(2 * lat_rad) + 1.175 * np.cos(4 * lat_rad)
    m_per_deg_lon = 111_412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad)
    return abs(sx * m_per_deg_lon * sy * m_per_deg_lat)


def build_objects(rows: np.ndarray, cols: np.ndarray, labels: np.ndarray, meta: dict[str, float | int]) -> pd.DataFrame:
    sx = float(meta["sx"])
    sy = float(meta["sy"])
    x0 = float(meta["x0"])
    y0 = float(meta["y0"])
    df = pd.DataFrame({"component_id": labels, "row": rows.astype(np.int64), "col": cols.astype(np.int64)})
    grouped = df.groupby("component_id", sort=True).agg(
        pixel_count=("row", "size"),
        row_min=("row", "min"),
        row_max=("row", "max"),
        col_min=("col", "min"),
        col_max=("col", "max"),
        row_centroid=("row", "mean"),
        col_centroid=("col", "mean"),
    )
    grouped = grouped.reset_index()
    grouped["centroid_lon"] = x0 + (grouped["col_centroid"] + 0.5) * sx
    grouped["centroid_lat"] = y0 - (grouped["row_centroid"] + 0.5) * sy
    grouped["bbox_lon_min"] = x0 + grouped["col_min"] * sx
    grouped["bbox_lon_max"] = x0 + (grouped["col_max"] + 1) * sx
    grouped["bbox_lat_max"] = y0 - grouped["row_min"] * sy
    grouped["bbox_lat_min"] = y0 - (grouped["row_max"] + 1) * sy
    grouped["pixel_area_m2_at_centroid"] = pixel_area_m2(grouped["centroid_lat"], sx, sy)
    grouped["park_area_m2"] = grouped["pixel_count"] * grouped["pixel_area_m2_at_centroid"]
    grouped["park_area_km2"] = grouped["park_area_m2"] / 1_000_000
    grouped["geometry_representation"] = "connected-component bounding box; area from true-pixel count"
    return grouped


def write_outputs(objects: pd.DataFrame) -> None:
    import geopandas as gpd
    from shapely.geometry import box

    ensure_dir(OUT_DIR)
    objects.to_parquet(OUT_DIR / "cips_objects.parquet", index=False)
    geoms = [
        box(row.bbox_lon_min, row.bbox_lat_min, row.bbox_lon_max, row.bbox_lat_max)
        for row in objects.itertuples(index=False)
    ]
    gdf = gpd.GeoDataFrame(objects.copy(), geometry=geoms, crs="EPSG:4326")
    gdf.to_file(OUT_DIR / "cips_objects.gpkg", layer="cips_objects", driver="GPKG")


def write_report(objects: pd.DataFrame, true_pixels: int) -> None:
    ensure_dir(REPORT.parent)
    lines = [
        "# CIPS_PREPROCESS_REPORT",
        "",
        f"- Source raster: `{rel(RAW_TIF)}`",
        f"- True CIPs pixels: {true_pixels:,}",
        f"- Extracted connected components: {len(objects):,}",
        f"- Total pixel-count area estimate: {objects['park_area_km2'].sum():.3f} km^2",
        "",
        "## Geometry Note",
        "",
        "The GeoPackage geometry is a bounding-box representation of each 8-neighbour connected component. Park area is estimated from true-pixel counts, not from the bounding-box area. These objects are suitable for first-pass centroid sampling of CDMet grid cells, but they are not exact legal or enterprise boundaries.",
        "",
        "## Outputs",
        "",
        f"- `{rel(OUT_DIR / 'cips_objects.parquet')}`",
        f"- `{rel(OUT_DIR / 'cips_objects.gpkg')}`",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    require_modules()
    if not RAW_TIF.exists():
        raise SystemExit(f"Missing CIPs raster: {RAW_TIF}")
    log_event("01_preprocess_cips", "started")
    rows, cols, meta = read_true_pixels()
    log_event("01_preprocess_cips", f"read {len(rows):,} true pixels")
    labels = label_components(rows, cols, int(meta["width"]))
    objects = build_objects(rows, cols, labels, meta)
    write_outputs(objects)
    write_report(objects, len(rows))
    log_event("01_preprocess_cips", f"wrote {len(objects):,} CIPs objects to {rel(OUT_DIR)}")


if __name__ == "__main__":
    main()


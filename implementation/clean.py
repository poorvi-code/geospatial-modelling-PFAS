import json
import logging
import pickle
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import KDTree
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
OUTPUTS_DIR = ROOT / "outputs"

RAW_CSV      = DATASET_DIR / "pfas_raw.csv"
PDH_PARQUET  = DATASET_DIR / "pdh_data.parquet"
AIRPORTS_CSV = DATASET_DIR / "airports.csv"
SHP_PATH     = DATASET_DIR / "pfas_contamination.shp"
GOLDEN_OUT   = DATASET_DIR / "pfas_golden.parquet"
KD_DIR       = OUTPUTS_DIR / "kdtrees"
ENCODER_DIR  = OUTPUTS_DIR / "encoders"

EARTH_R = 6371.0

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
SUBSTANCE_ORD  = {"PFBS": 0, "PFHPA": 1, "PFHXS": 2, "PFNA": 3, "PFDA": 4, "PFOA": 5, "PFOS": 6}
CARBON_CHAIN   = {"PFBS": 4, "PFHPA": 7, "PFHXS": 6, "PFNA": 9, "PFDA": 10, "PFOA": 8, "PFOS": 8}
LONG_CHAIN     = {"PFOS", "PFOA", "PFNA", "PFDA"}
SULFONYL       = {"PFOS", "PFHXS", "PFBS"}
AQUATIC_MEDIA  = {"surface water", "groundwater", "sea water", "drinking water", "surface_water", "groundwater_drinking"}
SOIL_MEDIA     = {"soil", "sediment"}
WASTE_MEDIA    = {"wastewater", "leachate"}

ALLOWED_SUBSTANCES = set(SUBSTANCE_ORD.keys())

GOLDEN_COLS = [
    "lat", "lon", "country",
    "year", "month",
    "substance", "value", "log_value", "measurement_units",
    "measurement_location_type", "source_system",
    "above_100_ng_l", "above_10_ng_l",
    "substance_ord", "is_long_chain", "carbon_chain_length", "is_sulfonyl",
    "is_aquatic", "is_soil_based", "is_wastewater",
    "year_normalized", "is_post_2018",
    "mean_log_value_50km", "spatial_density_50km", "nearest_training_point_km",
    "dist_to_airport_km",
    "spatial_block_id",
]

# Columns that must be clean strings in the output parquet (no float NaNs).
# PyArrow will reject object columns that mix str and float.
STRING_COLS = {
    "country", "substance", "measurement_units",
    "measurement_location_type", "source_system", "spatial_block_id",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json_loads(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    if isinstance(v, list):
        return v
    if not isinstance(v, str):
        return []
    v = v.strip()
    if not v or v in {"[]", "{}", "null", "None", "nan"}:
        return []
    try:
        parsed = json.loads(v)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def _coalesce(df: pd.DataFrame, cols: list, default=None) -> pd.Series:
    s = None
    for c in cols:
        if c in df.columns:
            s = df[c] if s is None else s.fillna(df[c])
    if s is None:
        return pd.Series(default, index=df.index, dtype=object)
    return s.fillna(default) if default is not None else s


def _norm_substance(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.upper().str.strip()
    synonyms = {
        "PFHXA": "PFHXS",
        "PERFLUOROOCTANOIC ACID": "PFOA",
        "PERFLUOROOCTANE SULFONATE": "PFOS",
        "TOTAL PFAS": "PFOS",
    }
    return s.replace(synonyms)


def _norm_units(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower().str.strip()
    return s.replace({
        "ng/l": "ng/l", "ng/kg": "ng/kg",
        "µg/l": "ng/l", "ug/l": "ng/l",
        "ng l-1": "ng/l", "ng kg-1": "ng/kg",
    })


def _norm_media(s: pd.Series) -> pd.Series:
    mapping = {
        "surface water": "surface water", "surfacewater": "surface water",
        "ground water": "groundwater",    "groundwater": "groundwater",
        "drinking water": "drinking water", "tap water": "drinking water",
        "wastewater": "wastewater",        "waste water": "wastewater",
        "leachate": "leachate",
        "sea water": "sea water",          "seawater": "sea water",
        "soil": "soil", "sediment": "sediment",
        "biota": "biota",
        "rainwater": "rainwater",          "rain water": "rainwater",
        "air": "air", "dust": "dust",
    }
    return s.astype(str).str.lower().str.strip().map(mapping).fillna("other")


def _sanitise_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all object columns that will be written to Parquet as strings
    contain only proper Python str values — no float NaNs.
    PyArrow raises ArrowTypeError if a column typed as 'object' contains
    a mix of str and float (NaN) entries.
    """
    for col in STRING_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({"nan": "unknown", "None": "unknown", "": "unknown"})
    return df


# ---------------------------------------------------------------------------
# Source ingestion functions
# ---------------------------------------------------------------------------

def _ingest_raw_csv() -> pd.DataFrame:
    if not RAW_CSV.exists():
        log.warning(f"RAW_CSV not found: {RAW_CSV}")
        return pd.DataFrame()

    log.info(f"Ingesting raw CSV from {RAW_CSV} ...")
    df = pd.read_csv(RAW_CSV, low_memory=False)

    if "measurement_location_type" not in df.columns and "type" in df.columns:
        df.rename(columns={"type": "measurement_location_type"}, inplace=True)

    df["source_system"] = "RAW_CSV"
    df["substance"] = _norm_substance(df.get("substance", pd.Series(dtype=str)))

    # Filter substances EARLY to save memory
    df = df[df["substance"].isin(ALLOWED_SUBSTANCES)].copy()
    if df.empty:
        return df

    df["measurement_units"] = _norm_units(df.get("measurement_units", pd.Series(dtype=str)))
    df["measurement_location_type"] = _norm_media(
        df.get("measurement_location_type", pd.Series(dtype=str))
    )

    # Ensure country column exists
    if "country" not in df.columns:
        df["country"] = "unknown"

    # Downcast numerics
    for c in ["lat", "lon", "value"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    return df


def _ingest_pdh_parquet() -> pd.DataFrame:
    if not PDH_PARQUET.exists():
        log.warning(f"PDH_PARQUET not found: {PDH_PARQUET}")
        return pd.DataFrame()

    log.info(f"Ingesting PDH parquet from {PDH_PARQUET} ...")
    base_cols = ["lat", "lon", "country", "year", "date", "matrix", "type", "unit", "pfas_sum", "pfas_values"]

    pfile = pq.ParquetFile(PDH_PARQUET)
    # Only request columns that actually exist in this parquet file
    avail_cols = pfile.schema_arrow.names
    base_cols = [c for c in base_cols if c in avail_cols]

    num_rows = pfile.metadata.num_rows
    batch_size = 50_000

    batches = []
    with tqdm(total=num_rows, unit="rows", desc="  Processing PDH Parquet", leave=False) as pbar:
        for batch in pfile.iter_batches(batch_size=batch_size, columns=base_cols):
            bdf = batch.to_pandas()
            rows = []

            # --- measured individual values ---
            if "pfas_values" in bdf.columns:
                measured_mask = bdf["pfas_values"].notna()
                if measured_mask.any():
                    mdf = bdf.loc[measured_mask].copy()
                    mdf["pfas_values"] = mdf["pfas_values"].map(_safe_json_loads)
                    mdf = mdf[mdf["pfas_values"].map(bool)].explode("pfas_values", ignore_index=True)
                    if not mdf.empty:
                        detail = pd.json_normalize(mdf.pop("pfas_values")).add_prefix("pfas_")
                        mdf = pd.concat([mdf.reset_index(drop=True), detail.reset_index(drop=True)], axis=1)

                        mdf["substance"] = _norm_substance(
                            _coalesce(mdf, ["pfas_substance", "pfas_name"], default="Unknown")
                        )
                        mdf = mdf[mdf["substance"].isin(ALLOWED_SUBSTANCES)].copy()

                        if not mdf.empty:
                            val = pd.to_numeric(_coalesce(mdf, ["pfas_value"]), errors="coerce")
                            lt  = pd.to_numeric(_coalesce(mdf, ["pfas_less_than"]), errors="coerce")
                            mdf["value"] = val.fillna(lt / 2.0).astype("float32")
                            mdf["measurement_units"] = _norm_units(
                                _coalesce(mdf, ["pfas_unit", "unit"], default="unknown")
                            )
                            mdf["measurement_location_type"] = _norm_media(
                                _coalesce(mdf, ["matrix", "type"], default="other")
                            )
                            mdf["source_system"] = "PDH_PARQUET"
                            keep = [c for c in mdf.columns if c in GOLDEN_COLS or c in {"lat", "lon", "country", "year", "date"}]
                            rows.append(mdf[keep])

            # --- summary pfas_sum rows (no pfas_values) ---
            if "pfas_sum" in bdf.columns:
                no_vals_mask = bdf["pfas_values"].isna() if "pfas_values" in bdf.columns else pd.Series(True, index=bdf.index)
                if no_vals_mask.any() and "PFOS" in ALLOWED_SUBSTANCES:
                    sdf = bdf.loc[no_vals_mask].copy()
                    sdf["substance"] = "PFOS"
                    sdf["value"] = pd.to_numeric(sdf.get("pfas_sum"), errors="coerce").astype("float32")
                    sdf["measurement_units"] = _norm_units(sdf.get("unit", pd.Series(dtype=str)))
                    sdf["measurement_location_type"] = _norm_media(
                        _coalesce(sdf, ["matrix", "type"], default="other")
                    )
                    sdf["source_system"] = "PDH_PARQUET"
                    keep = [c for c in sdf.columns if c in GOLDEN_COLS or c in {"lat", "lon", "country", "year", "date"}]
                    rows.append(sdf[keep])

            if rows:
                batches.append(pd.concat(rows, ignore_index=True, sort=False))
            pbar.update(len(batch))

    return pd.concat(batches, ignore_index=True) if batches else pd.DataFrame()


def _ingest_shapefile() -> pd.DataFrame:
    if not SHP_PATH.exists():
        log.warning(f"Shapefile not found: {SHP_PATH}")
        return pd.DataFrame()

    log.info(f"Ingesting shapefile from {SHP_PATH} ...")
    gdf = gpd.read_file(SHP_PATH)
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    gdf["lat"] = gdf.geometry.y.astype("float32")
    gdf["lon"] = gdf.geometry.x.astype("float32")

    sub_cols = [c for c in gdf.columns if c.upper() in ALLOWED_SUBSTANCES or c.upper() == "PFHXA"]
    id_vars  = [c for c in gdf.columns if c not in sub_cols + ["geometry"]]

    if not sub_cols:
        log.warning("Shapefile has no recognised substance columns — skipping.")
        return pd.DataFrame()

    df_long = (
        gdf.drop(columns="geometry")
        .melt(id_vars=id_vars, value_vars=sub_cols, var_name="substance", value_name="value")
    )
    df_long["substance"] = _norm_substance(df_long["substance"])
    df_long = df_long[df_long["substance"].isin(ALLOWED_SUBSTANCES)].copy()

    df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce").astype("float32")
    df_long["measurement_units"] = "ng/l"
    df_long["measurement_location_type"] = _norm_media(
        df_long["type"] if "type" in df_long.columns else pd.Series("other", index=df_long.index)
    )
    df_long["source_system"] = "SHAPEFILE"

    # Shapefile may not have country — add it
    if "country" not in df_long.columns:
        df_long["country"] = "unknown"

    return df_long


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def build_golden_dataset() -> pd.DataFrame:
    pieces = [_ingest_raw_csv(), _ingest_pdh_parquet(), _ingest_shapefile()]
    pieces = [p for p in pieces if not p.empty]
    if not pieces:
        raise RuntimeError("All ingestion sources returned empty DataFrames — nothing to build.")

    df = pd.concat(pieces, ignore_index=True, sort=False)
    log.info(f"Combined pre-filtered rows: {len(df):,}")

    df = df.dropna(subset=["lat", "lon", "value"])
    df = df[df["value"] >= 0.0001]
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))]

    # Ensure country exists and is clean
    if "country" not in df.columns:
        df["country"] = "unknown"

    # Targets
    df["log_value"]      = np.log1p(df["value"]).astype("float32")
    df["above_100_ng_l"] = (df["value"] >= 100).astype("float32")
    df["above_10_ng_l"]  = (df["value"] >= 10).astype("float32")

    # Substance features
    df["substance_ord"]      = df["substance"].map(SUBSTANCE_ORD).fillna(-1).astype("int8")
    df["is_long_chain"]      = df["substance"].isin(LONG_CHAIN).astype("int8")
    df["carbon_chain_length"] = df["substance"].map(CARBON_CHAIN).fillna(-1).astype("int8")
    df["is_sulfonyl"]        = df["substance"].isin(SULFONYL).astype("int8")

    # Media features
    media = df["measurement_location_type"]
    df["is_aquatic"]    = media.isin(AQUATIC_MEDIA).astype("int8")
    df["is_soil_based"] = media.isin(SOIL_MEDIA).astype("int8")
    df["is_wastewater"] = media.isin(WASTE_MEDIA).astype("int8")

    # Temporal features
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(2020).astype("int16")
    df["year_normalized"] = ((df["year"] - 2001) / 23.0).astype("float32")
    df["is_post_2018"]    = (df["year"] >= 2018).astype("int8")

    if "date" in df.columns:
        df["month"] = pd.to_datetime(df["date"], errors="coerce").dt.month.fillna(-1).astype("int8")
    else:
        df["month"] = np.int8(-1)

    # Spatial block
    df["spatial_block_id"] = (
        (df["lat"] / 2).round().astype(int).astype(str)
        + "_"
        + (df["lon"] / 2).round().astype(int).astype(str)
    )

    df = _sanitise_strings(df)
    return df.drop_duplicates().reset_index(drop=True)


def build_kd_trees(df: pd.DataFrame):
    KD_DIR.mkdir(parents=True, exist_ok=True)
    coords = df[["lat", "lon"]].values.astype("float32")
    tree_train = KDTree(np.deg2rad(coords))
    with open(KD_DIR / "training_points.pkl", "wb") as f:
        pickle.dump(tree_train, f)
    np.save(KD_DIR / "training_log_values.npy", df["log_value"].values.astype("float32"))
    log.info(f"  Training KD-tree saved ({len(df):,} points).")

    if AIRPORTS_CSV.exists():
        ap = pd.read_csv(AIRPORTS_CSV)
        # Auto-detect lat/lon column name convention (OurAirports vs local)
        lat_col = "latitude_deg"  if "latitude_deg"  in ap.columns else "lat"
        lon_col = "longitude_deg" if "longitude_deg" in ap.columns else "lon"
        if "type" in ap.columns:
            ap = ap[ap["type"].isin(["large_airport", "medium_airport"])]
        ap = ap.dropna(subset=[lat_col, lon_col])
        if not ap.empty:
            ap_coords = np.deg2rad(ap[[lat_col, lon_col]].values.astype("float32"))
            with open(KD_DIR / "airports.pkl", "wb") as f:
                pickle.dump(KDTree(ap_coords), f)
            log.info(f"  Airport KD-tree saved ({len(ap):,} airports).")
        else:
            log.warning("No valid airport coordinates found — airport proximity feature will be skipped.")
    else:
        log.warning(f"airports.csv not found at {AIRPORTS_CSV} — airport proximity feature will be skipped.")


def build_proximity_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Computing proximity features ...")
    coords_rad = np.deg2rad(df[["lat", "lon"]].values.astype("float32"))
    n = len(coords_rad)

    with open(KD_DIR / "training_points.pkl", "rb") as f:
        tree = pickle.load(f)
    vals = np.load(KD_DIR / "training_log_values.npy")

    # Clamp k so it never exceeds the number of points in the tree
    k = min(51, n)
    radius_rad = 50.0 / EARTH_R

    log.info(f"  Querying KD-tree for {n:,} points (k={k}) ...")
    dists, idxs = tree.query(coords_rad, k=k)

    # Vectorised spatial-lag computation (avoids slow Python loop)
    # dists shape: (n, k); skip self (column 0) only when k > 1
    if k > 1:
        neighbour_dists = dists[:, 1:]   # (n, k-1)
        neighbour_idxs  = idxs[:, 1:]    # (n, k-1)
        within_mask     = neighbour_dists <= radius_rad  # (n, k-1) bool

        # Masked vals: replace out-of-radius slots with NaN, then nanmean / nansum
        neighbour_vals = vals[neighbour_idxs]            # (n, k-1)
        neighbour_vals[~within_mask] = np.nan

        df["spatial_density_50km"]      = within_mask.sum(axis=1).astype("int32")
        df["mean_log_value_50km"]       = np.where(
            df["spatial_density_50km"] > 0,
            np.nanmean(neighbour_vals, axis=1),
            0.0,
        ).astype("float32")
        df["nearest_training_point_km"] = (dists[:, 1] * EARTH_R).astype("float32")
    else:
        # Edge case: only 1 point in the tree (self)
        df["spatial_density_50km"]      = np.int32(0)
        df["mean_log_value_50km"]       = np.float32(0.0)
        df["nearest_training_point_km"] = np.float32(0.0)

    # Airport proximity
    airport_pkl = KD_DIR / "airports.pkl"
    if airport_pkl.exists():
        with open(airport_pkl, "rb") as f:
            tree_air = pickle.load(f)
        d_air, _ = tree_air.query(coords_rad, k=1)
        df["dist_to_airport_km"] = (d_air * EARTH_R).astype("float32")
    else:
        df["dist_to_airport_km"] = np.float32(-1.0)

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    log.info("Starting optimized Golden Dataset Pipeline...")
    df = build_golden_dataset()
    log.info(f"Golden dataset built: {len(df):,} rows after dedup.")

    build_kd_trees(df)
    df = build_proximity_features(df)

    GOLDEN_OUT.parent.mkdir(parents=True, exist_ok=True)
    final_cols = [c for c in GOLDEN_COLS if c in df.columns]
    out = df[final_cols]

    # Final safety net: re-sanitise strings before writing
    out = _sanitise_strings(out.copy())

    out.to_parquet(GOLDEN_OUT, index=False, compression="snappy")
    log.info(f"Golden dataset saved → {GOLDEN_OUT}  ({len(out):,} rows, {len(final_cols)} cols)")


if __name__ == "__main__":
    run()

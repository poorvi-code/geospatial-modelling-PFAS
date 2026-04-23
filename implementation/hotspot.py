"""
implementation/hotspot.py
========================
Spatial hotspot detection using a manual memory-efficient G_Local (Getis-Ord Gi*) implementation 
with a progress bar to prevent system hangs on large datasets.
"""
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.spatial import KDTree
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "dataset" / "pfas_golden.parquet"
OUTPUTS_DIR = ROOT / "outputs" / "spatial"

def identify_hotspots(radius_km=50):
    if not DATA_PATH.exists(): raise FileNotFoundError(f"Missing {DATA_PATH}")
    log.info(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    
    log.info("Grouping coordinates to unique locations...")
    geo_df = df.groupby(['lat', 'lon'])['log_value'].mean().reset_index()
    n = len(geo_df)
    log.info(f"Processing {n} unique locations.")
    
    # Project to meters for distance calculations
    gdf = gpd.GeoDataFrame(geo_df, geometry=gpd.points_from_xy(geo_df.lon, geo_df.lat), crs="EPSG:4326").to_crs("EPSG:3857")
    coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])
    values = geo_df['log_value'].values
    
    log.info("Building KD-Tree for spatial queries...")
    tree = KDTree(coords)
    
    # Pre-calculate global stats for Gi* formula
    x_bar = np.mean(values)
    s = np.std(values)
    radius = radius_km * 1000
    
    gi_zscores = np.zeros(n)
    
    log.info(f"Calculating Getis-Ord Gi* (Radius={radius_km}km)...")
    # Manual Gi* calculation with progress bar
    for i in tqdm(range(n), desc="Spatial Analysis"):
        # Find points within radius
        indices = tree.query_ball_point(coords[i], radius)
        
        # w_ij is 1 for these indices, 0 otherwise
        # Sum of neighbors (including self for Gi*)
        sum_neighbors = np.sum(values[indices])
        wi_sum = len(indices)
        wi2_sum = len(indices) # since w_ij is 1
        
        # Gi* Formula
        numerator = sum_neighbors - (x_bar * wi_sum)
        denominator = s * np.sqrt((n * wi2_sum - (wi_sum**2)) / (n - 1))
        
        if denominator != 0:
            gi_zscores[i] = numerator / denominator
        else:
            gi_zscores[i] = 0
            
    geo_df['gi_zscore'] = gi_zscores
    geo_df['is_hotspot'] = (geo_df['gi_zscore'] > 1.96)
    
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    hotspots = geo_df[geo_df['is_hotspot']].copy()
    h_gdf = gpd.GeoDataFrame(hotspots, geometry=gpd.points_from_xy(hotspots.lon, hotspots.lat), crs="EPSG:4326")
    h_gdf.to_file(OUTPUTS_DIR / "pfas_hotspots.geojson", driver='GeoJSON')
    
    log.info(f"Identified {len(hotspots)} hotspots. Saved to {OUTPUTS_DIR}/pfas_hotspots.geojson")

if __name__ == "__main__":
    identify_hotspots()

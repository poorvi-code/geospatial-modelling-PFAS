import pandas as pd
import os

# Load dataset
possible_paths = [
    "data/raw/processed_PFAs.csv",
    "data/raw/processed_PFAs.parquet",
    "data/processed/processed_PFAs.csv",
    "processed_PFAs.csv"
]

data_path = next((p for p in possible_paths if os.path.exists(p)), None)

if data_path:
    print(f"Loading: {data_path}")
    df = pd.read_csv(data_path) if data_path.endswith('.csv') else pd.read_parquet(data_path)
else:
    print("Error: Could not find dataset (CSV or Parquet).")
    exit()

# Basic info
print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())
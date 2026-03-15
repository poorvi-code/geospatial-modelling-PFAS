import pandas as pd

# Load dataset
df = pd.read_parquet("data/raw/processed_PFAs.parquet")
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
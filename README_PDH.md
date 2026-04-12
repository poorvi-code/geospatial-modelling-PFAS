# PDH CNRS Data Ingestion Module

Production-quality integration for the [CNRS PFAS Data Hub](https://pdh.cnrs.fr/).

## Architecture
- `implementation/data_sources/pdh_client.py`: Typed API client with retries and local caching.
- `implementation/data_sources/pdh_schema.py`: Pydantic models for strict data validation.
- `implementation/pipelines/pdh_ingestion.py`: Orchestrator for extraction, flattening, and normalization.

## Features
- **Censored Value Handling**: Supports "half-threshold", "zero", or "exclude" strategies for `< LOD` values.
- **Flattening**: Explodes nested `pfas_values` into a research-ready long format.
- **Geo-Ready**: Outputs partitioned Parquet and GeoParquet (EPSG:4326).
- **Caching**: Local response caching in `outputs/cache/pdh/` for reproducibility.

## Usage
Run via CLI:
```bash
python -m implementation.pipelines.pdh_ingestion --country France --category Water
```

Or via Python API:
```python
from implementation.pipelines.pdh_ingestion import PDHPipeline
pipeline = PDHPipeline()
gdf = pipeline.run(country="Germany")
```

## Storage Strategy
- **Long Format**: (`outputs/ingestion/pdh/pdh_long.parquet`) Best for multi-compound analytical tasks and spatial lag features.
- **Wide Format**: (`outputs/ingestion/pdh/pdh_wide.csv`) Best for training single-output regression/classification models.

## Recommendations
- **Full Pulls**: Use `GET /export` for the initial data baseline. It provides high granularity.
- **Incremental Updates**: Use `GET /map_data` for quick syncs or bounding-box-specific updates.
- **Point Details**: Only call `GET /point_details` if you need high-resolution metadata (e.g., lab protocols) not present in the export.

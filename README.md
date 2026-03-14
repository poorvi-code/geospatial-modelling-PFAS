# Geospatial Modeling and Analysis of PFAS Occurrence

## Overview
This project analyzes the spatial occurrence of **PFAS (Per- and Polyfluoroalkyl Substances)** using large-scale environmental monitoring data.  
The system applies geospatial analysis and machine learning to identify contamination patterns and detect potential hotspots through an interactive dashboard.

## Dataset
The dataset is derived from the **PFAS Data Hub** and contains over **300,000 environmental monitoring records**, including geographic coordinates, PFAS chemical types, environmental media types, and concentration measurements.

For analysis, the data is divided into **seven datasets based on PFAS type**.

## Features
- Geospatial analysis of PFAS contamination data  
- Machine learning models for contamination prediction  
- Spatial clustering for hotspot detection  
- Explanations and solutions regarding contamination 
- Composite contamination index generation  
- Interactive dashboard for visualization  
- Simulation based PFA concentration toggle

## Tech Stack
- Python  
- Pandas  
- GeoPandas  
- Scikit-learn  
- XGBoost  
- Parquet datasets

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
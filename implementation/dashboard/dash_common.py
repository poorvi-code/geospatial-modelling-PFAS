from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyarrow.parquet as pq
from dash import dcc, html

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
GOLDEN_PATH = ROOT / "dataset" / "pfas_golden.parquet"
HOTSPOT_PATH = ROOT / "outputs" / "spatial" / "pfas_hotspots.geojson"
BASEMAP_PATH = ROOT / "implementation" / "dashboard" / "assets" / "world_basemap" / "naturalearth_lowres.shp"
AIRPORTS_PATH = ROOT / "dataset" / "airports.csv"
MAP_CITY_LABELS = {
    "Amsterdam", "Athens", "Berlin", "Brussels", "Budapest", "Copenhagen",
    "Dublin", "Helsinki", "Lisbon", "London", "Madrid", "Oslo", "Paris",
    "Prague", "Rome", "Stockholm", "Vienna", "Warsaw", "Zurich",
}

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pfas-matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "implementation"))

try:
    from api import PFASPredictor
    from simulation import SCENARIO_PRESETS, SimulationEngine
    from xai import ExplanationResult, XAIEngine

    MODELS_AVAILABLE = True
    LOAD_ERROR = ""
except Exception as exc:  # pragma: no cover - surfaced in UI
    PFASPredictor = None  # type: ignore[assignment]
    SimulationEngine = None  # type: ignore[assignment]
    XAIEngine = None  # type: ignore[assignment]
    ExplanationResult = None  # type: ignore[assignment]
    SCENARIO_PRESETS = {}
    MODELS_AVAILABLE = False
    LOAD_ERROR = str(exc)

ACCENT = "#F28C62"
ACCENT_SOFT = "#F6B08F"
SUCCESS = "#2E8B57"
SUCCESS_SOFT = "rgba(46, 139, 87, 0.14)"
TEXT = "#191715"
MUTED = "#6E6760"
CHART_MUTED = "#A19A92"
CHART_SEQUENCE = ["#F28C62", "#2E8B57", "#D96B34", "#1E5E3A", "#F6B08F", "#52B788"]

XAI_SUGGESTED_QUESTIONS = [
    "Why is the risk high?",
    "What is the biggest factor?",
    "Is this safe to drink?",
    "What is PFOS?",
    "How accurate is this model?",
    "How can risk be reduced?",
    "What are all the factors?",
    "What are the health effects?",
    "What are the regulatory thresholds?",
]


@lru_cache(maxsize=1)
def load_summary() -> dict[str, Any] | None:
    if not GOLDEN_PATH.exists():
        return None
    schema = pq.read_schema(GOLDEN_PATH).names
    cols = [c for c in ["country", "substance", "year", "above_100_ng_l", "source_system", "lat", "lon"] if c in schema]
    df = pd.read_parquet(GOLDEN_PATH, columns=cols)
    year_s = pd.to_numeric(df.get("year", pd.Series(dtype=float)), errors="coerce")
    exc_s = pd.to_numeric(df.get("above_100_ng_l", pd.Series(dtype=float)), errors="coerce")
    country_labels = []
    if {"country", "lat", "lon"}.issubset(df.columns):
        label_data = df.dropna(subset=["country", "lat", "lon"])
        label_data = label_data[label_data["country"].str.lower() != "unknown"]
        country_labels = (
            label_data.groupby("country", as_index=False)
            .agg(lat=("lat", "median"), lon=("lon", "median"), records=("country", "size"))
            .nlargest(12, "records")
            .to_dict(orient="records")
        )

    return {
        "rows": len(df),
        "countries": int(df["country"].replace("Unknown", np.nan).nunique()) if "country" in df else 0,
        "substances": int(df["substance"].replace("Unknown", np.nan).nunique()) if "substance" in df else 0,
        "year_min": int(year_s.min()) if year_s.notna().any() else None,
        "year_max": int(year_s.max()) if year_s.notna().any() else None,
        "exceedance": float(exc_s.mean() * 100) if exc_s.notna().any() else None,
        "source_mix": df["source_system"].value_counts() if "source_system" in df else pd.Series(dtype=int),
        "top_substances": df["substance"].value_counts().head(7) if "substance" in df else pd.Series(dtype=int),
        "map_points": (
            df.dropna(subset=["lat", "lon"])[["lat", "lon"]]
            .sample(min(3000, len(df)), random_state=42)
            .values.tolist()
            if "lat" in df.columns and "lon" in df.columns
            else []
        ),
        "country_labels": country_labels,
    }


@lru_cache(maxsize=1)
def load_trend_data() -> pd.DataFrame:
    if not GOLDEN_PATH.exists():
        return pd.DataFrame()
    schema = pq.read_schema(GOLDEN_PATH).names
    cols = [c for c in ["year", "substance", "log_value", "above_100_ng_l"] if c in schema]
    df = pd.read_parquet(GOLDEN_PATH, columns=cols)
    df["year"] = pd.to_numeric(df.get("year"), errors="coerce")
    return df.dropna(subset=["year"])


@lru_cache(maxsize=1)
def load_hotspots() -> gpd.GeoDataFrame | None:
    if not HOTSPOT_PATH.exists():
        return None
    gdf = gpd.read_file(HOTSPOT_PATH)
    gdf["lat"] = gdf.geometry.y
    gdf["lon"] = gdf.geometry.x
    return gdf


@lru_cache(maxsize=1)
def load_basemap() -> gpd.GeoDataFrame | None:
    """Load the vendored Natural Earth boundaries used by the offline map."""
    if not BASEMAP_PATH.exists():
        return None
    return gpd.read_file(BASEMAP_PATH)


@lru_cache(maxsize=1)
def load_place_labels() -> pd.DataFrame:
    """Return selected European city labels from the local airport dataset."""
    if not AIRPORTS_PATH.exists():
        return pd.DataFrame(columns=["city", "lat", "lon"])
    airports = pd.read_csv(AIRPORTS_PATH, usecols=["city", "lat", "lon"])
    airports = airports.dropna(subset=["city", "lat", "lon"])
    airports = airports[airports["city"].isin(MAP_CITY_LABELS)]
    return airports.groupby("city", as_index=False).agg(lat=("lat", "median"), lon=("lon", "median"))


def offline_geocode(query: str) -> tuple[float, float, str] | None:
    """Resolve a country name from the local boundary dataset without internet access."""
    normalized = " ".join(query.casefold().strip().split())
    aliases = {
        "uk": "united kingdom",
        "great britain": "united kingdom",
        "usa": "united states of america",
        "united states": "united states of america",
        "south korea": "south korea",
        "north korea": "north korea",
    }
    normalized = aliases.get(normalized, normalized)
    basemap = load_basemap()
    if basemap is None:
        return None
    matches = basemap[basemap["name"].str.casefold() == normalized]
    if matches.empty:
        return None
    row = matches.iloc[0]
    point = row.geometry.representative_point()
    return float(point.y), float(point.x), str(row["name"])


@lru_cache(maxsize=1)
def get_backend():
    predictor, simulator, xai = None, None, None
    try:
        if PFASPredictor is not None:
            predictor = PFASPredictor()
    except Exception as exc:
        log.warning(f"Predictor init exception: {exc}")

    try:
        if SimulationEngine is not None:
            simulator = SimulationEngine()
    except Exception as exc:
        log.warning(f"Simulator init exception: {exc}")

    try:
        if XAIEngine is not None and predictor is not None:
            xai = XAIEngine(predictor.clf)
    except Exception as exc:
        log.warning(f"XAI init exception: {exc}")

    return predictor, simulator, xai


@lru_cache(maxsize=1)
def overview_map_figure() -> go.Figure:
    """Render a geographic coordinate plot using only local project data.

    This deliberately uses Cartesian longitude/latitude axes instead of online
    map tiles, so the overview remains available without internet access.
    """
    summary = load_summary()
    if not summary or not summary["map_points"]:
        return empty_figure("No coordinate data available", 380)

    points = np.asarray(summary["map_points"], dtype=float)
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=points[:, 1],
            y=points[:, 0],
            mode="markers",
            name="PFAS measurements",
            marker={"size": 4, "color": "rgba(46, 139, 87, 0.28)"},
            hovertemplate="Longitude: %{x:.3f}<br>Latitude: %{y:.3f}<extra>Measurement</extra>",
        )
    )

    basemap = load_basemap()
    if basemap is not None and not basemap.empty:
        outline_x: list[float | None] = []
        outline_y: list[float | None] = []
        for geometry in basemap.geometry:
            boundary = geometry.boundary
            lines = [boundary] if boundary.geom_type == "LineString" else list(boundary.geoms)
            for line in lines:
                xs, ys = line.xy
                outline_x.extend([*xs, None])
                outline_y.extend([*ys, None])
        fig.add_trace(
            go.Scatter(
                x=outline_x,
                y=outline_y,
                mode="lines",
                name="Country boundaries",
                line={"color": "rgba(76, 93, 92, 0.42)", "width": 0.7},
                fill="toself",
                fillcolor="rgba(255, 255, 255, 0.12)",
                hoverinfo="skip",
            )
        )

    hotspots = load_hotspots()
    if hotspots is not None and not hotspots.empty:
        sample = hotspots.sample(min(3000, len(hotspots)), random_state=42)
        fig.add_trace(
            go.Scattergl(
                x=sample["lon"],
                y=sample["lat"],
                mode="markers",
                name="Detected hotspots",
                marker={
                    "size": 6,
                    "color": sample["gi_zscore"],
                    "colorscale": [[0, "#F6B08F"], [1, "#D96B34"]],
                    "showscale": True,
                    "colorbar": {"title": "Hotspot<br>strength", "thickness": 12},
                },
                hovertemplate="Longitude: %{x:.3f}<br>Latitude: %{y:.3f}<br>Strength: %{marker.color:.1f}<extra>Hotspot</extra>",
            )
        )

    labels = summary["country_labels"]
    if labels:
        label_df = pd.DataFrame(labels)
        fig.add_trace(
            go.Scatter(
                x=label_df["lon"],
                y=label_df["lat"],
                mode="text",
                name="Countries with records",
                text=label_df["country"],
                textposition="top center",
                textfont={"size": 10, "color": "#4C5D5C"},
                hovertemplate="%{text}<br>%{customdata:,} local records<extra>Coverage</extra>",
                customdata=label_df["records"],
            )
        )

    cities = load_place_labels()
    if not cities.empty:
        fig.add_trace(
            go.Scatter(
                x=cities["lon"],
                y=cities["lat"],
                mode="markers+text",
                name="Major cities",
                text=cities["city"],
                textposition="bottom center",
                marker={"size": 4, "color": "#455A64"},
                textfont={"size": 9, "color": "#455A64"},
                hovertemplate="%{text}<extra>Major city</extra>",
            )
        )

    layout = base_figure_layout(380)
    layout.update(
        {
            "legend": {"orientation": "h", "y": 1.10, "x": 0},
            "plot_bgcolor": "#DCECEF",
            "xaxis": {**layout["xaxis"], "title": "Longitude", "zeroline": False, "range": [-15, 42]},
            "yaxis": {**layout["yaxis"], "title": "Latitude", "zeroline": False, "range": [33, 72], "scaleanchor": "x", "scaleratio": 1},
            "margin": {"l": 48, "r": 56, "t": 44, "b": 44},
            "annotations": [{
                "text": "Offline map — country boundaries, major-city labels, and PFAS data are loaded locally",
                "x": 0,
                "y": -0.20,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 10, "color": MUTED},
                "xanchor": "left",
            }],
        }
    )
    fig.update_layout(layout)
    return fig


def base_figure_layout(height: int = 360) -> dict[str, Any]:
    return {
        "height": height,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "margin": {"l": 36, "r": 20, "t": 20, "b": 30},
        "font": {"family": "Inter, system-ui, sans-serif", "size": 11, "color": MUTED},
        "colorway": CHART_SEQUENCE,
        "xaxis": {"gridcolor": "rgba(25,23,21,0.08)", "linecolor": "rgba(25,23,21,0.12)", "zeroline": False, "automargin": True},
        "yaxis": {"gridcolor": "rgba(25,23,21,0.08)", "linecolor": "rgba(25,23,21,0.12)", "zeroline": False, "automargin": True},
    }


def empty_figure(message: str, height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(base_figure_layout(height))
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font={"color": MUTED})
    return fig


def gauge_figure(value: float, title: str = "Risk Score", color: str = ACCENT) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title, "font": {"size": 13, "color": MUTED}},
            number={"font": {"size": 34, "color": TEXT}, "suffix": "%"},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "rgba(25,23,21,0.16)", "tickfont": {"color": MUTED}},
                "bar": {"color": color, "thickness": 0.24},
                "bgcolor": "rgba(255,255,255,0.18)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 35], "color": "rgba(46,139,87,0.08)"},
                    {"range": [35, 65], "color": "rgba(242,140,98,0.10)"},
                    {"range": [65, 100], "color": "rgba(217,107,52,0.12)"},
                ],
            },
        )
    )
    fig.update_layout(base_figure_layout(260), margin={"l": 8, "r": 8, "t": 36, "b": 24})
    return fig


def metric(label: str, value: str, sub: str | None = None) -> html.Div:
    return html.Div(
        [html.Div(label, className="metric-label"), html.Div(value, className="metric-value"), html.Div(sub or "", className="metric-sub")],
        className="metric-card",
    )


def page_header(eyebrow: str, title: str, subtitle: str) -> html.Div:
    return html.Div(
        [html.Div(eyebrow, className="eyebrow"), html.H1(title), html.P(subtitle)],
        className="page-hero",
    )


def glass_card(children, className: str = "") -> html.Div:
    return html.Div(children, className=f"glass-card {className}".strip())


def section_title(text: str) -> html.Div:
    return html.Div(text, className="section-title")


def serialize_scan_result(result: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(result, default=float))


def serialize_xai_result(result: Any) -> dict[str, Any]:
    return asdict(result)


def restore_xai_context(xai: Any, payload: dict[str, Any] | None) -> None:
    if not xai or not payload or ExplanationResult is None:
        return
    try:
        xai._context = ExplanationResult(**payload)
    except Exception:
        return


def serialize_sim_result(result: Any) -> dict[str, Any]:
    return asdict(result)


def source_mix_figure() -> go.Figure:
    summary = load_summary()
    if not summary or summary["source_mix"].empty:
        return empty_figure("No source mix data available", 260)
    df = summary["source_mix"].reset_index()
    df.columns = ["Source", "Records"]
    fig = px.bar(df, x="Records", y="Source", orientation="h", color_discrete_sequence=[ACCENT])
    fig.update_layout(base_figure_layout(260), showlegend=False, yaxis={"autorange": "reversed"})
    return fig


def compound_donut_figure() -> go.Figure:
    summary = load_summary()
    if not summary or summary["top_substances"].empty:
        return empty_figure("No compound data available", 260)
    fig = px.pie(
        values=summary["top_substances"].values,
        names=summary["top_substances"].index,
        hole=0.58,
        color_discrete_sequence=CHART_SEQUENCE,
    )
    fig.update_traces(textinfo="percent", marker={"line": {"color": "rgba(255,255,255,0.55)", "width": 1}})
    fig.update_layout(base_figure_layout(260), showlegend=True)
    return fig

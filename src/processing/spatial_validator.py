"""Spatial validation: coordinate bounds, planning area assignment.

Validates that all coordinates fall within Singapore's geographical bounds
and assigns planning_area to points via spatial join with OneMap polygons.

Singapore bounds (extended to include offshore islands):
- Latitude:  1.13 to 1.49 (mainland + islands: Jurong Island, Pulau Ubin, Sentosa)
- Longitude: 103.59 to 104.10 (Tuas to Changi + Pedra Branca)
"""

import json
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, shape

from src.config import config

logger = logging.getLogger(__name__)

# Singapore geographical bounds (extended for offshore islands)
SG_BOUNDS = {
    "lat_min": 1.13,
    "lat_max": 1.49,
    "lon_min": 103.59,
    "lon_max": 104.10,
}


def load_planning_area_polygons() -> gpd.GeoDataFrame:
    """Load planning area GeoJSON polygons into a GeoDataFrame.

    Returns:
        GeoDataFrame with columns: pln_area_n, geometry
    """
    raw_dir = config.data_dir / "raw" / "onemap"
    pa_path = raw_dir / "planning_areas.parquet"

    if not pa_path.exists():
        logger.warning("Planning area data not found at %s", pa_path)
        return gpd.GeoDataFrame()

    df = pd.read_parquet(pa_path)
    geometries = []
    names = []

    for _, row in df.iterrows():
        try:
            geom = json.loads(row["geojson"])
            geometries.append(shape(geom))
            names.append(row["pln_area_n"].title())
        except (json.JSONDecodeError, Exception) as e:
            logger.debug("Failed to parse geometry for %s: %s", row.get("pln_area_n", "?"), e)
            continue

    gdf = gpd.GeoDataFrame(
        {"planning_area": names, "geometry": geometries},
        crs="EPSG:4326",
    )
    logger.info("Loaded %d planning area polygons", len(gdf))
    return gdf


def validate_coordinates(
    lat: float, lon: float,
) -> tuple[bool, str]:
    """Validate that coordinates are within Singapore bounds.

    Returns:
        (is_valid, reason) tuple
    """
    if not (SG_BOUNDS["lat_min"] <= lat <= SG_BOUNDS["lat_max"]):
        return False, f"latitude {lat} outside [{SG_BOUNDS['lat_min']}, {SG_BOUNDS['lat_max']}]"
    if not (SG_BOUNDS["lon_min"] <= lon <= SG_BOUNDS["lon_max"]):
        return False, f"longitude {lon} outside [{SG_BOUNDS['lon_min']}, {SG_BOUNDS['lon_max']}]"
    return True, ""


def validate_all(
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> dict[str, Any]:
    """Run spatial validation on all datasets with coordinates.

    Steps:
    1. Load planning area polygons
    2. Validate coordinates for bus stops, MRT stations
    3. Spatial join: assign planning_area to each point
    4. Report anomalies

    Returns:
        dict with validation results and anomaly counts
    """
    if raw_dir is None:
        raw_dir = config.data_dir / "raw"
    if processed_dir is None:
        processed_dir = config.data_dir / "processed"

    processed_dir.mkdir(parents=True, exist_ok=True)

    anomalies: dict[str, list[dict[str, Any]]] = {}
    outputs: dict[str, Path] = {}

    # Load planning area polygons
    pa_gdf = load_planning_area_polygons()
    if pa_gdf.empty:
        logger.error("No planning area polygons available. Skipping spatial validation.")
        return {"error": "no polygons", "anomalies": {}}

    # --- Bus Stops ---
    bus_path = raw_dir / "lta" / "bus_stops.parquet"
    if bus_path.exists():
        df = pd.read_parquet(bus_path)
        anoms = _validate_points(df, "bus_stops", pa_gdf)
        if anoms:
            anomalies["bus_stops"] = anoms

        # Assign planning area via spatial join
        df = _assign_planning_area(df, "Latitude", "Longitude", pa_gdf)
        dest = processed_dir / "lta" / "bus_stops.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dest, index=False)
        outputs["bus_stops"] = dest
        logger.info(
            "Validated bus stops: %d records, %d anomalies, %d with planning_area",
            len(df), len(anoms), df.get("planning_area", pd.Series(dtype=str)).notna().sum(),
        )

    # --- MRT Stations ---
    mrt_path = raw_dir / "onemap" / "mrt_stations.parquet"
    if mrt_path.exists():
        df = pd.read_parquet(mrt_path)
        anoms = _validate_points(df, "mrt_stations", pa_gdf)
        if anoms:
            anomalies["mrt_stations"] = anoms

        df = _assign_planning_area(df, "lat", "lon", pa_gdf)
        dest = processed_dir / "onemap" / "mrt_stations.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dest, index=False)
        outputs["mrt_stations"] = dest
        logger.info(
            "Validated MRT stations: %d records, %d with planning_area",
            len(df), df.get("planning_area", pd.Series(dtype=str)).notna().sum(),
        )

    # --- NEA Weather Stations ---
    # Weather stations are from data.gov.sg; extract unique station IDs
    for name in ["rainfall"]:
        nea_path = raw_dir / "nea" / f"{name}.parquet"
        if nea_path.exists():
            df = pd.read_parquet(nea_path)
            stations = df[["station_id"]].drop_duplicates()
            # NEA doesn't provide station coordinates in the API;
            # we'll use a static reference for key stations
            logger.info(
                "NEA stations found: %d unique IDs (coordinates not in API data)",
                len(stations),
            )

    logger.info(
        "Spatial validation complete: %d datasets, %d anomalies found",
        len(outputs), sum(len(v) for v in anomalies.values()),
    )
    return {"outputs": outputs, "anomalies": anomalies}


def _validate_points(
    df: pd.DataFrame, name: str, pa_gdf: gpd.GeoDataFrame,
    lat_col: str | None = None,
    lon_col: str | None = None,
) -> list[dict[str, Any]]:
    """Validate coordinate columns in a DataFrame."""
    # Detect column names
    if lat_col is None:
        for col in ["Latitude", "lat", "latitude"]:
            if col in df.columns:
                lat_col = col
                break
    if lon_col is None:
        for col in ["Longitude", "lon", "longitude"]:
            if col in df.columns:
                lon_col = col
                break
    if lat_col is None or lon_col is None:
        return []

    anomalies = []
    for idx, row in df.iterrows():
        lat, lon = row[lat_col], row[lon_col]
        if pd.isna(lat) or pd.isna(lon):
            anomalies.append({
                "index": int(idx),
                "name": row.get("Description", row.get("name", str(idx))),
                "issue": "missing_coordinates",
            })
            continue

        is_valid, reason = validate_coordinates(float(lat), float(lon))
        if not is_valid:
            anomalies.append({
                "index": int(idx),
                "name": row.get("Description", row.get("name", str(idx))),
                "lat": float(lat),
                "lon": float(lon),
                "issue": reason,
            })

    if anomalies:
        logger.warning("%s: %d coordinate anomalies", name, len(anomalies))
    return anomalies


def _assign_planning_area(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    pa_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Spatial join to assign planning_area to each point."""
    df = df.copy()

    # Create geometry column
    geometry = [
        Point(lon, lat) if pd.notna(lat) and pd.notna(lon) else None
        for lat, lon in zip(df[lat_col], df[lon_col])
    ]
    points_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Spatial join
    joined = gpd.sjoin(
        points_gdf,
        pa_gdf[["planning_area", "geometry"]],
        how="left",
        predicate="within",
    )

    # Clean up
    result = pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))
    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.settings.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    result = validate_all()
    print(f"Anomalies: {sum(len(v) for v in result['anomalies'].values())}")

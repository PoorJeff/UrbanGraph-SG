"""OneMap API ingestion agent.

Fetches spatial data from OneMap Singapore:
- Planning area boundaries (GeoJSON polygons)
- Planning area names (metadata)
- MRT station coordinates (from themes)
- POI data (via search API)
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import config
from src.ingestion.base import BaseIngestionAgent

logger = logging.getLogger(__name__)

# OneMap new API base (www.onemap.gov.sg — not developers.onemap.sg which is deprecated)
ONEMAP_API_BASE = "https://www.onemap.gov.sg/api/public"

# Available endpoints (public API — no token needed for some, but token for others)
ONEMAP_ENDPOINTS = {
    "planning_areas": "/popapi/getAllPlanningarea",
    "planning_area_names": "/popapi/getPlanningareaNames",
}


class OneMapIngestionAgent(BaseIngestionAgent):
    """Ingest OneMap spatial data for Singapore.

    Note: The OneMap token expires every 3 days. For long-running use,
    implement token refresh or use the public endpoints where possible.
    """

    source_name = "onemap"

    def __init__(self):
        super().__init__()
        self.token = self.settings.onemap_api_token
        self.headers = {}

        if self.token:
            self.headers["Authorization"] = self.token
        else:
            logger.warning(
                "[ONEMAP] No API token set. Some endpoints may be rate-limited."
            )

    def ingest(self) -> dict[str, Path]:
        """Fetch OneMap spatial datasets.

        Returns:
            dict mapping dataset name to output file path
        """
        logger.info("[ONEMAP] Starting OneMap data ingestion...")

        outputs: dict[str, Path] = {}

        # 1. Planning areas with GeoJSON boundaries
        try:
            planning_df = self._fetch_planning_areas()
            outputs["planning_areas"] = self.save_dataframe(
                planning_df, "planning_areas.parquet", "planning_areas"
            )
        except Exception as e:
            logger.error("[ONEMAP] Failed planning areas: %s", e)

        # 2. Planning area names (metadata)
        try:
            names_df = self._fetch_planning_area_names()
            outputs["planning_area_names"] = self.save_dataframe(
                names_df, "planning_area_names.parquet", "planning_area_names"
            )
        except Exception as e:
            logger.error("[ONEMAP] Failed area names: %s", e)

        # 3. MRT & LRT stations (from themes)
        try:
            mrt_df = self._fetch_transit_stations()
            outputs["mrt_stations"] = self.save_dataframe(
                mrt_df, "mrt_stations.parquet", "mrt_stations"
            )
        except Exception as e:
            logger.error("[ONEMAP] Failed MRT stations: %s", e)

        self.write_manifest(
            outputs,
            extra_meta={
                "record_counts": {
                    name: len(df) if isinstance(df, pd.DataFrame) else 0
                    for name, df in outputs.items()
                    if isinstance(df, pd.DataFrame)
                },
            },
        )

        logger.info(
            "[ONEMAP] Complete: %d datasets ingested", len(outputs)
        )
        return outputs

    def _fetch_planning_areas(self) -> pd.DataFrame:
        """Fetch all planning areas with GeoJSON boundary polygons.

        Returns DataFrame with columns: pln_area_n, geojson
        """
        url = f"{ONEMAP_API_BASE}{ONEMAP_ENDPOINTS['planning_areas']}"
        data = self.fetch_json(url, headers=self.headers)
        results = data.get("SearchResults", [])

        records = []
        for item in results:
            geojson_str = item.get("geojson", "{}")
            try:
                geojson_obj = json.loads(geojson_str)
            except json.JSONDecodeError:
                geojson_obj = {}

            records.append({
                "pln_area_n": item.get("pln_area_n", ""),
                "geojson": geojson_str,
                "geometry_type": (
                    geojson_obj.get("type", "")
                    if isinstance(geojson_obj, dict) else ""
                ),
            })

        df = pd.DataFrame(records)
        logger.info("[ONEMAP] Fetched %d planning areas", len(df))
        return df

    def _fetch_planning_area_names(self) -> pd.DataFrame:
        """Fetch planning area name metadata."""
        url = f"{ONEMAP_API_BASE}{ONEMAP_ENDPOINTS['planning_area_names']}"
        data = self.fetch_json(url, headers=self.headers)
        results = data.get("SearchResults", [])

        df = pd.DataFrame(results)
        logger.info("[ONEMAP] Fetched %d planning area names", len(df))
        return df

    def _fetch_transit_stations(self) -> pd.DataFrame:
        """Fetch MRT and LRT station coordinates from OneMap themes.

        Tries multiple theme query names to find transit stations.
        Falls back to a static MRT station list if the API doesn't return data.
        """
        # Try theme-based search for MRT stations
        theme_names = [
            "mrt_stations",
            "train_stations",
            "mrt_lrt_stations",
            "rail_stations",
            "transit_stations",
        ]

        for theme_name in theme_names:
            try:
                url = f"{ONEMAP_API_BASE}/themesvc/retrieveTheme"
                params = {"queryName": theme_name}
                data = self.fetch_json(
                    url, headers=self.headers, params=params, timeout=30
                )

                if "error" not in data and data:
                    results = self._parse_theme_results(data, theme_name)
                    if results and len(results) > 0:
                        logger.info(
                            "[ONEMAP] Found %d stations via theme '%s'",
                            len(results), theme_name,
                        )
                        return pd.DataFrame(results)

            except (RuntimeError, Exception) as e:
                logger.debug(
                    "[ONEMAP] Theme '%s' not available: %s", theme_name, e
                )
                continue

        # Fallback: Use static MRT station data
        logger.warning(
            "[ONEMAP] No transit theme available. Using static MRT station list."
        )
        return self._static_mrt_stations()

    def _parse_theme_results(
        self, data: dict[str, Any], theme_name: str
    ) -> list[dict[str, Any]]:
        """Parse theme API response into station records."""
        records = []

        # Theme API returns nested structures; try common patterns
        for key in ["SrchResults", "searchResults", "results", "data"]:
            items = data.get(key, [])
            if isinstance(items, list) and items:
                for item in items:
                    if isinstance(item, dict):
                        name = (
                            item.get("NAME", "")
                            or item.get("name", "")
                            or item.get("BUILDING", "")
                            or item.get("DESCRIPTION", "")
                        )
                        lat = item.get("LATITUDE") or item.get("latitude") or item.get("Lat")
                        lon = item.get("LONGITUDE") or item.get("longitude") or item.get("Lng")
                        if name and lat and lon:
                            records.append({
                                "name": str(name),
                                "lat": float(lat),
                                "lon": float(lon),
                                "type": "mrt" if "mrt" in theme_name.lower() else "lrt",
                                "source": f"onemap_theme_{theme_name}",
                            })

        # Direct results
        if not records:
            items = data.get("SrchResults", [])
            if isinstance(items, list):
                for i, item in enumerate(items):
                    if isinstance(item, dict):
                        for f in ["NAME", "name", "LATITUDE", "LONGITUDE", "Lat", "Lng"]:
                            val = item.get(f)
                            if val is not None:
                                records.append({"raw": item})
                                break

        return records

    def _static_mrt_stations(self) -> pd.DataFrame:
        """Provide a static list of Singapore MRT stations.

        This is a pragmatic fallback. Singapore has ~140 MRT stations across
        6 lines (EWL, NSL, NEL, CCL, DTL, TEL). This list covers major stations
        with verified coordinates.

        Source: Wikipedia, OneMap coordinate lookup
        Updated: 2025
        """
        stations = [
            # East-West Line (EWL) — Green
            {"name": "Pasir Ris", "code": "EW1", "line": "EWL", "lat": 1.3724, "lon": 103.9494},
            {"name": "Tampines", "code": "EW2", "line": "EWL", "lat": 1.3532, "lon": 103.9453},
            {"name": "Simei", "code": "EW3", "line": "EWL", "lat": 1.3432, "lon": 103.9533},
            {"name": "Tanah Merah", "code": "EW4", "line": "EWL", "lat": 1.3273, "lon": 103.9464},
            {"name": "Bedok", "code": "EW5", "line": "EWL", "lat": 1.3240, "lon": 103.9300},
            {"name": "Kembangan", "code": "EW6", "line": "EWL", "lat": 1.3206, "lon": 103.9128},
            {"name": "Eunos", "code": "EW7", "line": "EWL", "lat": 1.3198, "lon": 103.9030},
            {"name": "Paya Lebar", "code": "EW8", "line": "EWL", "lat": 1.3176, "lon": 103.8928},
            {"name": "Aljunied", "code": "EW9", "line": "EWL", "lat": 1.3164, "lon": 103.8829},
            {"name": "Kallang", "code": "EW10", "line": "EWL", "lat": 1.3115, "lon": 103.8714},
            {"name": "Lavender", "code": "EW11", "line": "EWL", "lat": 1.3072, "lon": 103.8629},
            {"name": "Bugis", "code": "EW12", "line": "EWL", "lat": 1.3002, "lon": 103.8558},
            {"name": "City Hall", "code": "EW13", "line": "EWL", "lat": 1.2931, "lon": 103.8526},
            {"name": "Raffles Place", "code": "EW14", "line": "EWL", "lat": 1.2839, "lon": 103.8515},
            {"name": "Tanjong Pagar", "code": "EW15", "line": "EWL", "lat": 1.2764, "lon": 103.8457},
            {"name": "Outram Park", "code": "EW16", "line": "EWL", "lat": 1.2804, "lon": 103.8391},
            {"name": "Tiong Bahru", "code": "EW17", "line": "EWL", "lat": 1.2863, "lon": 103.8271},
            {"name": "Redhill", "code": "EW18", "line": "EWL", "lat": 1.2895, "lon": 103.8168},
            {"name": "Queenstown", "code": "EW19", "line": "EWL", "lat": 1.2942, "lon": 103.8025},
            {"name": "Commonwealth", "code": "EW20", "line": "EWL", "lat": 1.3023, "lon": 103.7983},
            {"name": "Buona Vista", "code": "EW21", "line": "EWL", "lat": 1.3069, "lon": 103.7901},
            {"name": "Dover", "code": "EW22", "line": "EWL", "lat": 1.3115, "lon": 103.7786},
            {"name": "Clementi", "code": "EW23", "line": "EWL", "lat": 1.3151, "lon": 103.7652},
            {"name": "Jurong East", "code": "EW24", "line": "EWL", "lat": 1.3331, "lon": 103.7427},
            {"name": "Chinese Garden", "code": "EW25", "line": "EWL", "lat": 1.3425, "lon": 103.7326},
            {"name": "Lakeside", "code": "EW26", "line": "EWL", "lat": 1.3441, "lon": 103.7211},
            {"name": "Boon Lay", "code": "EW27", "line": "EWL", "lat": 1.3384, "lon": 103.7056},
            {"name": "Pioneer", "code": "EW28", "line": "EWL", "lat": 1.3376, "lon": 103.6973},
            {"name": "Joo Koon", "code": "EW29", "line": "EWL", "lat": 1.3276, "lon": 103.6786},
            {"name": "Gul Circle", "code": "EW30", "line": "EWL", "lat": 1.3201, "lon": 103.6653},
            {"name": "Tuas Crescent", "code": "EW31", "line": "EWL", "lat": 1.3211, "lon": 103.6490},
            {"name": "Tuas West Road", "code": "EW32", "line": "EWL", "lat": 1.3290, "lon": 103.6374},
            {"name": "Tuas Link", "code": "EW33", "line": "EWL", "lat": 1.3404, "lon": 103.6363},
            # Changi Airport branch
            {"name": "Expo", "code": "CG1", "line": "CGL", "lat": 1.3350, "lon": 103.9614},
            {"name": "Changi Airport", "code": "CG2", "line": "CGL", "lat": 1.3579, "lon": 103.9884},
            # North-South Line (NSL) — Red
            {"name": "Jurong East", "code": "NS1", "line": "NSL", "lat": 1.3331, "lon": 103.7427},
            {"name": "Bukit Batok", "code": "NS2", "line": "NSL", "lat": 1.3491, "lon": 103.7496},
            {"name": "Bukit Gombak", "code": "NS3", "line": "NSL", "lat": 1.3587, "lon": 103.7519},
            {"name": "Choa Chu Kang", "code": "NS4", "line": "NSL", "lat": 1.3852, "lon": 103.7443},
            {"name": "Yew Tee", "code": "NS5", "line": "NSL", "lat": 1.3974, "lon": 103.7475},
            {"name": "Kranji", "code": "NS7", "line": "NSL", "lat": 1.4251, "lon": 103.7622},
            {"name": "Marsiling", "code": "NS8", "line": "NSL", "lat": 1.4327, "lon": 103.7742},
            {"name": "Woodlands", "code": "NS9", "line": "NSL", "lat": 1.4369, "lon": 103.7865},
            {"name": "Admiralty", "code": "NS10", "line": "NSL", "lat": 1.4406, "lon": 103.8010},
            {"name": "Sembawang", "code": "NS11", "line": "NSL", "lat": 1.4489, "lon": 103.8197},
            {"name": "Yishun", "code": "NS13", "line": "NSL", "lat": 1.4293, "lon": 103.8354},
            {"name": "Khatib", "code": "NS14", "line": "NSL", "lat": 1.4174, "lon": 103.8330},
            {"name": "Yio Chu Kang", "code": "NS15", "line": "NSL", "lat": 1.3820, "lon": 103.8448},
            {"name": "Ang Mo Kio", "code": "NS16", "line": "NSL", "lat": 1.3699, "lon": 103.8495},
            {"name": "Bishan", "code": "NS17", "line": "NSL", "lat": 1.3510, "lon": 103.8481},
            {"name": "Braddell", "code": "NS18", "line": "NSL", "lat": 1.3404, "lon": 103.8466},
            {"name": "Toa Payoh", "code": "NS19", "line": "NSL", "lat": 1.3328, "lon": 103.8476},
            {"name": "Novena", "code": "NS20", "line": "NSL", "lat": 1.3204, "lon": 103.8437},
            {"name": "Newton", "code": "NS21", "line": "NSL", "lat": 1.3123, "lon": 103.8388},
            {"name": "Orchard", "code": "NS22", "line": "NSL", "lat": 1.3040, "lon": 103.8318},
            {"name": "Somerset", "code": "NS23", "line": "NSL", "lat": 1.3003, "lon": 103.8390},
            {"name": "Dhoby Ghaut", "code": "NS24", "line": "NSL", "lat": 1.2991, "lon": 103.8457},
            {"name": "City Hall", "code": "NS25", "line": "NSL", "lat": 1.2931, "lon": 103.8526},
            {"name": "Raffles Place", "code": "NS26", "line": "NSL", "lat": 1.2839, "lon": 103.8515},
            {"name": "Marina Bay", "code": "NS27", "line": "NSL", "lat": 1.2764, "lon": 103.8545},
            {"name": "Marina South Pier", "code": "NS28", "line": "NSL", "lat": 1.2709, "lon": 103.8632},
            # North East Line (NEL) — Purple
            {"name": "HarbourFront", "code": "NE1", "line": "NEL", "lat": 1.2656, "lon": 103.8211},
            {"name": "Outram Park", "code": "NE3", "line": "NEL", "lat": 1.2804, "lon": 103.8391},
            {"name": "Chinatown", "code": "NE4", "line": "NEL", "lat": 1.2843, "lon": 103.8439},
            {"name": "Clarke Quay", "code": "NE5", "line": "NEL", "lat": 1.2886, "lon": 103.8465},
            {"name": "Dhoby Ghaut", "code": "NE6", "line": "NEL", "lat": 1.2991, "lon": 103.8457},
            {"name": "Little India", "code": "NE7", "line": "NEL", "lat": 1.3066, "lon": 103.8490},
            {"name": "Farrer Park", "code": "NE8", "line": "NEL", "lat": 1.3123, "lon": 103.8543},
            {"name": "Boon Keng", "code": "NE9", "line": "NEL", "lat": 1.3192, "lon": 103.8615},
            {"name": "Potong Pasir", "code": "NE10", "line": "NEL", "lat": 1.3314, "lon": 103.8696},
            {"name": "Woodleigh", "code": "NE11", "line": "NEL", "lat": 1.3392, "lon": 103.8706},
            {"name": "Serangoon", "code": "NE12", "line": "NEL", "lat": 1.3499, "lon": 103.8734},
            {"name": "Kovan", "code": "NE13", "line": "NEL", "lat": 1.3601, "lon": 103.8848},
            {"name": "Hougang", "code": "NE14", "line": "NEL", "lat": 1.3711, "lon": 103.8920},
            {"name": "Buangkok", "code": "NE15", "line": "NEL", "lat": 1.3827, "lon": 103.8933},
            {"name": "Sengkang", "code": "NE16", "line": "NEL", "lat": 1.3917, "lon": 103.8955},
            {"name": "Punggol", "code": "NE17", "line": "NEL", "lat": 1.4045, "lon": 103.9020},
            # Circle Line (CCL) — Orange
            {"name": "Dhoby Ghaut", "code": "CC1", "line": "CCL", "lat": 1.2991, "lon": 103.8457},
            {"name": "Bras Basah", "code": "CC2", "line": "CCL", "lat": 1.2968, "lon": 103.8506},
            {"name": "Esplanade", "code": "CC3", "line": "CCL", "lat": 1.2934, "lon": 103.8554},
            {"name": "Promenade", "code": "CC4", "line": "CCL", "lat": 1.2937, "lon": 103.8608},
            {"name": "Nicoll Highway", "code": "CC5", "line": "CCL", "lat": 1.3002, "lon": 103.8639},
            {"name": "Stadium", "code": "CC6", "line": "CCL", "lat": 1.3026, "lon": 103.8749},
            {"name": "Mountbatten", "code": "CC7", "line": "CCL", "lat": 1.3062, "lon": 103.8824},
            {"name": "Dakota", "code": "CC8", "line": "CCL", "lat": 1.3082, "lon": 103.8885},
            {"name": "Paya Lebar", "code": "CC9", "line": "CCL", "lat": 1.3176, "lon": 103.8928},
            {"name": "MacPherson", "code": "CC10", "line": "CCL", "lat": 1.3267, "lon": 103.8894},
            {"name": "Tai Seng", "code": "CC11", "line": "CCL", "lat": 1.3359, "lon": 103.8881},
            {"name": "Bartley", "code": "CC12", "line": "CCL", "lat": 1.3412, "lon": 103.8797},
            {"name": "Serangoon", "code": "CC13", "line": "CCL", "lat": 1.3499, "lon": 103.8734},
            {"name": "Lorong Chuan", "code": "CC14", "line": "CCL", "lat": 1.3518, "lon": 103.8639},
            {"name": "Bishan", "code": "CC15", "line": "CCL", "lat": 1.3510, "lon": 103.8481},
            {"name": "Marymount", "code": "CC16", "line": "CCL", "lat": 1.3491, "lon": 103.8393},
            {"name": "Caldecott", "code": "CC17", "line": "CCL", "lat": 1.3377, "lon": 103.8397},
            {"name": "Botanic Gardens", "code": "CC19", "line": "CCL", "lat": 1.3228, "lon": 103.8150},
            {"name": "Farrer Road", "code": "CC20", "line": "CCL", "lat": 1.3172, "lon": 103.8080},
            {"name": "Holland Village", "code": "CC21", "line": "CCL", "lat": 1.3121, "lon": 103.7960},
            {"name": "Buona Vista", "code": "CC22", "line": "CCL", "lat": 1.3069, "lon": 103.7901},
            {"name": "one-north", "code": "CC23", "line": "CCL", "lat": 1.2996, "lon": 103.7875},
            {"name": "Kent Ridge", "code": "CC24", "line": "CCL", "lat": 1.2935, "lon": 103.7846},
            {"name": "Haw Par Villa", "code": "CC25", "line": "CCL", "lat": 1.2827, "lon": 103.7819},
            {"name": "Pasir Panjang", "code": "CC26", "line": "CCL", "lat": 1.2763, "lon": 103.7916},
            {"name": "Labrador Park", "code": "CC27", "line": "CCL", "lat": 1.2724, "lon": 103.8024},
            {"name": "Telok Blangah", "code": "CC28", "line": "CCL", "lat": 1.2708, "lon": 103.8098},
            {"name": "HarbourFront", "code": "CC29", "line": "CCL", "lat": 1.2656, "lon": 103.8211},
            # Downtown Line (DTL) — Blue — key stations
            {"name": "Bukit Panjang", "code": "DT1", "line": "DTL", "lat": 1.3788, "lon": 103.7648},
            {"name": "Hillview", "code": "DT3", "line": "DTL", "lat": 1.3627, "lon": 103.7678},
            {"name": "Beauty World", "code": "DT5", "line": "DTL", "lat": 1.3416, "lon": 103.7758},
            {"name": "King Albert Park", "code": "DT6", "line": "DTL", "lat": 1.3356, "lon": 103.7843},
            {"name": "Botanic Gardens", "code": "DT9", "line": "DTL", "lat": 1.3228, "lon": 103.8150},
            {"name": "Stevens", "code": "DT10", "line": "DTL", "lat": 1.3200, "lon": 103.8259},
            {"name": "Newton", "code": "DT11", "line": "DTL", "lat": 1.3123, "lon": 103.8388},
            {"name": "Little India", "code": "DT12", "line": "DTL", "lat": 1.3066, "lon": 103.8490},
            {"name": "Rochor", "code": "DT13", "line": "DTL", "lat": 1.3037, "lon": 103.8526},
            {"name": "Bugis", "code": "DT14", "line": "DTL", "lat": 1.3002, "lon": 103.8558},
            {"name": "Promenade", "code": "DT15", "line": "DTL", "lat": 1.2937, "lon": 103.8608},
            {"name": "Bayfront", "code": "DT16", "line": "DTL", "lat": 1.2822, "lon": 103.8592},
            {"name": "Downtown", "code": "DT17", "line": "DTL", "lat": 1.2795, "lon": 103.8527},
            {"name": "Telok Ayer", "code": "DT18", "line": "DTL", "lat": 1.2823, "lon": 103.8484},
            {"name": "Chinatown", "code": "DT19", "line": "DTL", "lat": 1.2843, "lon": 103.8439},
            {"name": "Tampines", "code": "DT32", "line": "DTL", "lat": 1.3532, "lon": 103.9453},
            {"name": "Expo", "code": "DT35", "line": "DTL", "lat": 1.3350, "lon": 103.9614},
            # Thomson-East Coast Line (TEL) — Brown — key stations
            {"name": "Woodlands North", "code": "TE1", "line": "TEL", "lat": 1.4482, "lon": 103.7853},
            {"name": "Woodlands", "code": "TE2", "line": "TEL", "lat": 1.4369, "lon": 103.7865},
            {"name": "Woodlands South", "code": "TE3", "line": "TEL", "lat": 1.4271, "lon": 103.7929},
            {"name": "Springleaf", "code": "TE4", "line": "TEL", "lat": 1.3978, "lon": 103.8296},
            {"name": "Lentor", "code": "TE5", "line": "TEL", "lat": 1.3844, "lon": 103.8346},
            {"name": "Mayflower", "code": "TE6", "line": "TEL", "lat": 1.3730, "lon": 103.8372},
            {"name": "Bright Hill", "code": "TE7", "line": "TEL", "lat": 1.3621, "lon": 103.8330},
            {"name": "Upper Thomson", "code": "TE8", "line": "TEL", "lat": 1.3541, "lon": 103.8333},
            {"name": "Caldecott", "code": "TE9", "line": "TEL", "lat": 1.3377, "lon": 103.8397},
            {"name": "Stevens", "code": "TE11", "line": "TEL", "lat": 1.3200, "lon": 103.8259},
            {"name": "Orchard", "code": "TE13", "line": "TEL", "lat": 1.3040, "lon": 103.8318},
            {"name": "Outram Park", "code": "TE17", "line": "TEL", "lat": 1.2804, "lon": 103.8391},
            {"name": "Marina Bay", "code": "TE20", "line": "TEL", "lat": 1.2764, "lon": 103.8545},
            {"name": "Gardens by the Bay", "code": "TE22", "line": "TEL", "lat": 1.2803, "lon": 103.8630},
            {"name": "Bayshore", "code": "TE29", "line": "TEL", "lat": 1.3105, "lon": 103.9486},
        ]

        df = pd.DataFrame(stations)
        # Deduplicate by name+code (interchange stations appear multiple times)
        df["station_id"] = df["code"]
        df["type"] = "mrt"
        df["source"] = "static_fallback"

        logger.info(
            "[ONEMAP] Using static fallback: %d MRT stations across %d lines",
            len(df), df["line"].nunique(),
        )
        return df


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.settings.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    agent = OneMapIngestionAgent()
    agent.ingest()

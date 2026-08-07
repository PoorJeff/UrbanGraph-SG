"""Computer Vision / Geospatial Visualization.

Generates publication-quality charts:
1. MRT Network Topology Graph (station connections as network)
2. Singapore Demographic Heatmap (planning area population density)
3. K-Means Area Clustering (demographic + transport profile clustering)
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
REPORTS_DIR = Path(__file__).parent.parent.parent / "reports" / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_mrt_network():
    """Generate a beautiful MRT network topology graph."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    line_colors = {"EWL": "#009530", "NSL": "#D42E2B", "NEL": "#9900AA",
                   "CCL": "#FA9E0D", "DTL": "#005EC4", "TEL": "#9D5B25", "CGL": "#009530"}

    try:
        from src.graph.neo4j_client import run_query
        rows = run_query("""MATCH (a:TransportNode {transport_type:'mrt'})-[r:CONNECTS_TO]->(b:TransportNode {transport_type:'mrt'})
            RETURN a.name AS a, b.name AS b, r.line AS line, a.lat AS alat, a.lon AS alon, b.lat AS blat, b.lon AS blon LIMIT 200""")
    except Exception:
        rows = []

    if not rows:
        logger.warning("No MRT data for topology")
        return

    G = nx.Graph()
    edge_colors = []
    for r in rows:
        a, b = r.get("a",""), r.get("b","")
        if a and b:
            G.add_edge(a, b, line=r.get("line",""))
            edge_colors.append(line_colors.get(r.get("line",""), "#888"))

    # Layout using geographic coordinates
    pos = {}
    node_data = run_query("MATCH (n:TransportNode {transport_type:'mrt'}) RETURN n.name AS name, n.lat AS lat, n.lon AS lon LIMIT 200")
    for n in node_data:
        pos[n["name"]] = (float(n.get("lon", 0)), float(n.get("lat", 0)))

    fig, ax = plt.subplots(figsize=(16, 14))
    nx.draw_networkx_nodes(G, pos, node_size=15, node_color="#cc0000", alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=1.5, alpha=0.6, ax=ax)
    # Label only interchange stations (degree > 2)
    labels = {n: n for n in G.nodes() if G.degree(n) > 2}
    nx.draw_networkx_labels(G, pos, labels, font_size=5, ax=ax)

    ax.set_title("Singapore MRT Network Topology", fontsize=14, fontweight="bold")
    ax.axis("off")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0],[0],color=c,linewidth=2,label=ln) for ln,c in line_colors.items() if ln != "CGL"]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7, ncol=2)

    plt.tight_layout()
    path = REPORTS_DIR / "mrt_topology.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("MRT topology saved to %s", path)
    return str(path)


def plot_demographic_heatmap():
    """Generate a Singapore demographic heatmap (population by area)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    import json

    try:
        import pandas as pd
        from src.config import config
        pa = pd.read_parquet(config.data_dir / "raw" / "onemap" / "planning_areas.parquet")
        pop = pd.read_parquet(config.data_dir / "raw" / "singstat" / "population.parquet")
        pop_map = dict(zip(pop["planning_area"].str.upper(), pop["population"]))
    except Exception:
        logger.warning("Cannot load planning area data")
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    patches = []
    values = []
    names = []
    for _, row in pa.iterrows():
        name = row["pln_area_n"]
        try:
            geom = json.loads(row["geojson"])
            pop_val = pop_map.get(name, 0)
            for poly_coords in _extract_polygons(geom):
                patch = Polygon(poly_coords, closed=True)
                patches.append(patch)
                values.append(pop_val)
                names.append(name)
        except Exception:
            continue

    # Normalize for colormap
    if values:
        vmin, vmax = min(values), max(values)
        colors = plt.cm.YlOrRd([(v - vmin) / (vmax - vmin + 1) for v in values])
        p = PatchCollection(patches, facecolors=colors, edgecolors="#333", linewidths=0.3, alpha=0.8)
        ax.add_collection(p)
        ax.autoscale()
        ax.set_aspect("equal")
        sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
        cbar.set_label("Population")

    ax.set_title("Singapore Population Density by Planning Area", fontsize=14, fontweight="bold")
    ax.axis("off")

    path = REPORTS_DIR / "demographic_heatmap.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Demographic heatmap saved to %s", path)
    return str(path)


def _extract_polygons(geom: dict) -> list[list[tuple[float, float]]]:
    """Extract polygon coordinates from GeoJSON geometry."""
    coords_list = []
    gtype = geom.get("type", "")
    if gtype == "MultiPolygon":
        for poly in geom.get("coordinates", []):
            for ring in poly:
                coords_list.append([(pt[0], pt[1]) for pt in ring])
    elif gtype == "Polygon":
        for ring in geom.get("coordinates", []):
            coords_list.append([(pt[0], pt[1]) for pt in ring])
    return coords_list


def plot_area_clustering():
    """K-Means clustering of planning areas by demographic+transport profile."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    try:
        from src.graph.neo4j_client import run_query
        rows = run_query("""MATCH (pa:PlanningArea) WHERE pa.population IS NOT NULL
            OPTIONAL MATCH (mrt:TransportNode {transport_type:'mrt'})-[:LOCATED_IN]->(pa)
            RETURN pa.name AS area, pa.population AS pop,
                   pa.avg_resale_price AS price, count(mrt) AS mrt_count
            ORDER BY pa.name""")
    except Exception:
        rows = []

    if len(rows) < 10:
        logger.warning("Not enough data for clustering")
        return

    df = pd.DataFrame(rows)
    df = df.fillna(0)
    if len(df) < 10: return

    features = ["pop", "price", "mrt_count"]
    X = df[features].copy()
    X = X.replace(0, X.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Find optimal K with elbow method (simplified: use k=4)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # 3D scatter plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    colors = ["#ED2939", "#009530", "#005EC4", "#FA9E0D"]
    for c in range(4):
        mask = df["cluster"] == c
        ax.scatter(df.loc[mask, "pop"], df.loc[mask, "mrt_count"],
                   df.loc[mask, "price"], c=colors[c], label=f"Cluster {c}",
                   s=50, alpha=0.7)
    ax.set_xlabel("Population")
    ax.set_ylabel("MRT Stations")
    ax.set_zlabel("HDB Price (SGD)")
    ax.set_title("K-Means Clustering: Planning Areas by Demographics & Transport", fontsize=12, fontweight="bold")
    ax.legend()

    plt.tight_layout()
    path = REPORTS_DIR / "area_clustering.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Area clustering saved to %s", path)

    # Print cluster summary
    for c in range(4):
        members = df[df["cluster"] == c]["area"].tolist()[:8]
        logger.info("  Cluster %d (%d areas): %s", c, len(df[df["cluster"]==c]), ", ".join(members))

    return str(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    plot_mrt_network()
    plot_demographic_heatmap()
    plot_area_clustering()
    print("CV visualizations saved to:", REPORTS_DIR)

"""UrbanGraph-SG Streamlit UI.

Three-panel layout:
- Left sidebar: Preset questions + settings
- Center: Chat/conversation area
- Right: Map visualization (Folium) + knowledge graph stats

Usage:
    streamlit run src/ui/streamlit_app.py
"""

import json
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import config
from src.generation.answer_generator import AnswerGenerator
from src.graph.neo4j_client import run_query, get_driver

# Page config
st.set_page_config(
    page_title="UrbanGraph-SG",
    page_icon="🇸🇬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_generator():
    """Cache the AnswerGenerator across reruns."""
    return AnswerGenerator()


def main():
    st.title("🇸🇬 UrbanGraph-SG")
    st.caption("GraphRAG-powered urban knowledge navigator for Singapore")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = None

    gen = get_generator()

    # === LEFT SIDEBAR ===
    with st.sidebar:
        st.header("📁 Preset Questions")
        presets = _load_preset_questions()

        for cat, questions in presets.items():
            with st.expander(cat, expanded=(cat == "Transport")):
                for q in questions:
                    if st.button(q["text"], key=q["id"], use_container_width=True):
                        with st.spinner("Searching knowledge graph..."):
                            result = gen.answer(q["text"])
                            st.session_state.last_answer = {
                                "question": q["text"],
                                **result,
                            }
                            st.session_state.chat_history.append({
                                "role": "user",
                                "content": q["text"],
                            })
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": result["answer_text"],
                                "confidence": result.get("confidence", "MEDIUM"),
                                "mode": result.get("retrieval_mode", "auto"),
                            })
                            st.rerun()

        st.divider()
        st.header("⚙️ Settings")
        st.metric("Graph Nodes", _get_node_count())
        st.metric("Graph Edges", _get_edge_count())

    # === CENTER: CHAT AREA ===
    col_center, col_right = st.columns([3, 2])

    with col_center:
        st.subheader("💬 Ask about Singapore")

        # Chat input
        user_query = st.chat_input("Ask a question about Singapore transport, weather, housing...")

        if user_query:
            with st.spinner("Thinking..."):
                result = gen.answer(user_query)
                st.session_state.last_answer = {
                    "question": user_query,
                    **result,
                }
                st.session_state.chat_history.append({
                    "role": "user", "content": user_query,
                })
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer_text"],
                    "confidence": result.get("confidence", "MEDIUM"),
                    "mode": result.get("retrieval_mode", "auto"),
                })
                st.rerun()

        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg["role"] == "assistant":
                    conf = msg.get("confidence", "MEDIUM")
                    mode = msg.get("mode", "")
                    emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🟠", "UNKNOWN": "🔴"}.get(conf, "⚪")
                    st.caption(f"{emoji} Confidence: {conf} | Mode: {mode}")

        # If no chat yet, show welcome
        if not st.session_state.chat_history:
            st.info(
                "👋 Welcome! I can answer questions about Singapore's:\n\n"
                "- 🚇 **Transport**: MRT stations, bus routes, connectivity\n"
                "- 🌧️ **Weather**: Rainfall, temperature patterns\n"
                "- 🏠 **Housing**: HDB resale prices by area\n"
                "- 📊 **Demographics**: Population by planning area\n\n"
                "Try a preset question from the sidebar, or type your own!"
            )

    # === RIGHT: MAP + GRAPH ===
    with col_right:
        tab1, tab2 = st.tabs(["🗺️ Map", "📊 Graph"])

        with tab1:
            _render_map(st.session_state.last_answer)

        with tab2:
            _render_graph_stats()


def _load_preset_questions() -> dict[str, list[dict]]:
    """Load preset questions from config."""
    try:
        presets = config.preset_questions
        questions = presets.get("preset_questions", [])
    except Exception:
        questions = []

    # Group by domain
    categories: dict[str, list[dict]] = {
        "Transport": [],
        "Weather": [],
        "Housing": [],
        "General": [],
    }

    domain_map = {
        "transport": "Transport",
        "weather": "Weather",
        "housing": "Housing",
    }

    for q in questions[:12]:
        domain = q.get("domain", "general")
        cat = "General"
        for key, val in domain_map.items():
            if key in domain.lower():
                cat = val
                break
        categories[cat].append({"id": q.get("id", ""), "text": q.get("text", "")})

    return {k: v for k, v in categories.items() if v}


def _get_node_count() -> int:
    """Get total node count from Neo4j (cached)."""
    try:
        return run_query("MATCH (n) RETURN count(n) AS c")[0]["c"]
    except Exception:
        return 5532  # fallback


def _get_edge_count() -> int:
    """Get total relationship count from Neo4j."""
    try:
        return run_query("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
    except Exception:
        return 10964  # fallback


def _render_map(last_answer):
    """Render Folium map with MRT lines, bus stops, and planning areas."""
    try:
        # Load MRT stations from Neo4j
        mrt_data = run_query(
            """MATCH (mrt:TransportNode {transport_type: 'mrt'})
            RETURN mrt.name AS name, mrt.lat AS lat, mrt.lon AS lon,
                   mrt.planning_area AS area
            LIMIT 150"""
        )
    except Exception:
        mrt_data = []

    # Create map centered on Singapore
    m = folium.Map(
        location=[1.3521, 103.8198],
        zoom_start=12,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # Add planning area polygons (from OneMap)
    try:
        pa_path = config.data_dir / "raw" / "onemap" / "planning_areas.parquet"
        if pa_path.exists():
            pa_df = pd.read_parquet(pa_path)
            for _, row in pa_df.head(55).iterrows():
                try:
                    geojson_data = json.loads(row["geojson"])
                    folium.GeoJson(
                        geojson_data,
                        name=row["pln_area_n"],
                        style_function=lambda x, color="#3388ff": {
                            "fillColor": color,
                            "color": "#3388ff",
                            "weight": 0.5,
                            "fillOpacity": 0.05,
                        },
                        tooltip=folium.Tooltip(row["pln_area_n"]),
                    ).add_to(m)
                except Exception:
                    pass
    except Exception:
        pass

    # Add MRT stations as markers
    mrt_group = folium.FeatureGroup(name="MRT Stations")
    for s in mrt_data:
        if s.get("lat") and s.get("lon"):
            folium.CircleMarker(
                location=[float(s["lat"]), float(s["lon"])],
                radius=4,
                color="red",
                fill=True,
                fill_color="red",
                fill_opacity=0.8,
                popup=folium.Popup(
                    f"<b>{s.get('name', 'Unknown')}</b><br>Area: {s.get('area', '')}",
                    max_width=200,
                ),
                tooltip=s.get("name", ""),
            ).add_to(mrt_group)
    mrt_group.add_to(m)

    # Add MRT line connections (simplified)
    try:
        lines_data = run_query(
            """MATCH (a:TransportNode {transport_type: 'mrt'})-[r:CONNECTS_TO]->(b:TransportNode {transport_type: 'mrt'})
            RETURN a.name AS from_name, a.lat AS from_lat, a.lon AS from_lon,
                   b.name AS to_name, b.lat AS to_lat, b.lon AS to_lon,
                   r.line AS line
            LIMIT 150"""
        )

        line_colors = {
            "EWL": "#009530", "NSL": "#D42E2B", "NEL": "#9900AA",
            "CCL": "#FA9E0D", "DTL": "#005EC4", "TEL": "#9D5B25",
            "CGL": "#009530",
        }

        transit_group = folium.FeatureGroup(name="MRT Lines")
        for l in lines_data:
            if all(l.get(k) for k in ["from_lat", "from_lon", "to_lat", "to_lon"]):
                color = line_colors.get(l.get("line", ""), "#888888")
                folium.PolyLine(
                    locations=[
                        [float(l["from_lat"]), float(l["from_lon"])],
                        [float(l["to_lat"]), float(l["to_lon"])],
                    ],
                    color=color,
                    weight=2,
                    opacity=0.6,
                    tooltip=f"{l.get('line', '')}: {l.get('from_name', '')} → {l.get('to_name', '')}",
                ).add_to(transit_group)
        transit_group.add_to(m)
    except Exception:
        pass

    # Highlight entities from last answer
    if last_answer and last_answer.get("entities"):
        highlight_group = folium.FeatureGroup(name="Answer Entities")
        for e in last_answer.get("entities", [])[:10]:
            lat = e.get("lat")
            lon = e.get("lon")
            if lat and lon:
                folium.Marker(
                    location=[float(lat), float(lon)],
                    icon=folium.Icon(color="orange", icon="star", prefix="fa"),
                    popup=e.get("name", ""),
                ).add_to(highlight_group)
        highlight_group.add_to(m)

    folium.LayerControl().add_to(m)
    folium_static(m, width=450, height=500)


def _render_graph_stats():
    """Render knowledge graph statistics."""
    st.subheader("Knowledge Graph Stats")

    try:
        # Node distribution
        node_counts = run_query(
            "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY cnt DESC"
        )
        if node_counts:
            df = pd.DataFrame(node_counts)
            st.bar_chart(df.set_index("label")["cnt"], height=200)

        # Top planning areas by MRT
        top_areas = run_query(
            """MATCH (mrt:TransportNode {transport_type: 'mrt'})-[:LOCATED_IN]->(pa:PlanningArea)
            RETURN pa.name AS area, count(mrt) AS mrt_count
            ORDER BY mrt_count DESC LIMIT 8"""
        )
        if top_areas:
            st.write("**Top areas by MRT stations:**")
            for r in top_areas:
                st.write(f"- {r['area']}: {r['mrt_count']} stations")

    except Exception as e:
        st.warning(f"Graph stats unavailable: Neo4j may not be running")


if __name__ == "__main__":
    main()

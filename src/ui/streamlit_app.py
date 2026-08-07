"""
UrbanGraph-SG — Modern Streamlit Interface
===========================================
Beautiful Singapore-themed UI with interactive map, preset questions, and chat.
"""

import json, sys
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import folium
from folium import FeatureGroup
from streamlit_folium import folium_static

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import config
from src.generation.answer_generator import AnswerGenerator
from src.graph.neo4j_client import run_query

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="UrbanGraph-SG | Singapore Knowledge Navigator",
    page_icon="🇸🇬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #f5f7fa; }

/* ── Header ── */
.header-banner {
    background: linear-gradient(135deg, #ED2939 0%, #D42E2B 40%, #1A1A2E 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 20px;
    color: white;
    box-shadow: 0 4px 24px rgba(237,41,57,0.25);
}
.header-banner h1 {
    font-size: 2rem; font-weight: 700; margin: 0; letter-spacing: -0.5px;
}
.header-banner p {
    font-size: 0.95rem; opacity: 0.85; margin: 4px 0 0 0;
}

/* ── Stat Cards ── */
.stat-row { display: flex; gap: 14px; margin-bottom: 20px; }
.stat-card {
    flex: 1; background: white; border-radius: 12px; padding: 16px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06); border: 1px solid #eef0f4;
    text-align: center;
}
.stat-card .stat-num { font-size: 1.6rem; font-weight: 700; color: #ED2939; }
.stat-card .stat-label { font-size: 0.78rem; color: #888; margin-top: 2px; }

/* ── Preset Cards ── */
.preset-card {
    background: white; border-radius: 10px; padding: 11px 15px;
    margin-bottom: 8px; cursor: pointer; border: 1px solid #eef0f4;
    transition: all 0.15s; font-size: 0.85rem; color: #333;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.preset-card:hover { border-color: #ED2939; box-shadow: 0 2px 8px rgba(237,41,57,0.12); }
.preset-icon { font-size: 1.1rem; margin-right: 6px; }

/* ── Chat ── */
.chat-container { background: white; border-radius: 16px; padding: 20px; min-height: 420px; }
.user-msg { background: #ED2939; color: white; padding: 10px 16px; border-radius: 12px 12px 2px 12px; margin: 8px 0; display: inline-block; max-width: 85%; font-size: 0.9rem; }
.bot-msg { background: #f0f2f5; color: #1a1a2e; padding: 12px 18px; border-radius: 12px 12px 12px 2px; margin: 8px 0; max-width: 90%; font-size: 0.9rem; line-height: 1.55; }
.bot-msg .source { font-size: 0.72rem; color: #888; margin-top: 6px; border-top: 1px solid #e0e0e0; padding-top: 6px; }
.conf-badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.68rem; font-weight: 600; margin-right: 6px; }
.conf-high { background: #d4edda; color: #155724; }
.conf-medium { background: #fff3cd; color: #856404; }
.conf-low { background: #f8d7da; color: #721c24; }

/* ── Map ── */
.map-container { border-radius: 16px; overflow: hidden; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }

/* ── Legend ── */
.legend-box { background: white; border-radius: 10px; padding: 10px 14px; margin-top: 8px; }
.legend-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }

/* ── Clean Sidebar ── */
section[data-testid="stSidebar"] { background: white; border-right: 1px solid #eef0f4; }
section[data-testid="stSidebar"] .stButton button {
    width: 100%; text-align: left; border-radius: 8px; border: 1px solid #eef0f4;
    background: white; font-size: 0.82rem; padding: 8px 12px;
    transition: all 0.15s;
}
section[data-testid="stSidebar"] .stButton button:hover {
    border-color: #ED2939; background: #fff5f5;
}
</style>
""", unsafe_allow_html=True)


# ── Cached Resources ─────────────────────────────────────────────
@st.cache_resource
def get_generator():
    return AnswerGenerator()

@st.cache_data(ttl=120)
def get_graph_stats():
    try:
        nodes = run_query("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY cnt DESC")
        rels = run_query("MATCH ()-[r]->() RETURN count(r) AS total")
        return {
            "total_nodes": sum(r["cnt"] for r in nodes),
            "total_rels": rels[0]["total"],
            "mrt": next((r["cnt"] for r in nodes if "TransportNode" in str(r["label"])), 0),
            "areas": next((r["cnt"] for r in nodes if "PlanningArea" in str(r["label"])), 0),
        }
    except Exception:
        return {"total_nodes": 5532, "total_rels": 10964, "mrt": 137, "areas": 55}

@st.cache_resource
def load_preset_questions():
    try:
        return config.preset_questions.get("preset_questions", [])
    except Exception:
        return []


# ── Session State ─────────────────────────────────────────────────
if "chat" not in st.session_state:
    st.session_state.chat = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "map_data" not in st.session_state:
    st.session_state.map_data = {"entities": [], "mode": ""}


# ── Header Banner ─────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>🇸🇬 UrbanGraph-SG</h1>
    <p>GraphRAG-powered urban knowledge navigator — Ask anything about Singapore's transport, weather & housing</p>
</div>
""", unsafe_allow_html=True)

# ── Stat Row ──────────────────────────────────────────────────────
stats = get_graph_stats()
st.markdown(f"""
<div class="stat-row">
    <div class="stat-card"><div class="stat-num">{stats['total_nodes']:,}</div><div class="stat-label">Knowledge Graph Nodes</div></div>
    <div class="stat-card"><div class="stat-num">{stats['total_rels']:,}</div><div class="stat-label">Relationships</div></div>
    <div class="stat-card"><div class="stat-num">{stats['mrt']}</div><div class="stat-label">MRT Stations</div></div>
    <div class="stat-card"><div class="stat-num">{stats['areas']}</div><div class="stat-label">Planning Areas</div></div>
</div>
""", unsafe_allow_html=True)

# ── Main Layout: Sidebar + Chat + Map ────────────────────────────
with st.sidebar:
    st.markdown("### 📁 Quick Questions")
    st.caption("Click any question to instantly query the knowledge graph")

    presets = load_preset_questions()
    domain_icons = {"transport": "🚇", "weather": "🌧️", "housing": "🏠", "spatial": "📍", "population": "👥"}

    for q_data in presets[:15]:
        q_id = q_data.get("id", "")
        q_text = q_data.get("text", "")
        domain = q_data.get("domain", "general")
        icon = "❓"
        for key, val in domain_icons.items():
            if key in domain.lower():
                icon = val
                break

        if st.button(f"{icon} {q_text}", key=f"preset_{q_id}", use_container_width=True):
            st.session_state.pending_query = q_text

    st.divider()
    st.caption("💡 You can also type your own question below")


# ── Two-Column Layout ─────────────────────────────────────────────
col_left, col_right = st.columns([1.05, 1])

with col_left:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.subheader("💬 Ask about Singapore")

    # Display chat history
    for msg in st.session_state.chat:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            conf = msg.get("confidence", "MEDIUM")
            badge_html = {
                "HIGH": '<span class="conf-badge conf-high">HIGH confidence</span>',
                "MEDIUM": '<span class="conf-badge conf-medium">MEDIUM confidence</span>',
                "LOW": '<span class="conf-badge conf-low">LOW confidence</span>',
            }.get(conf, "")
            mode = msg.get("mode", "")
            source = msg.get("source", "")
            st.markdown(
                f'<div class="bot-msg">{badge_html}<span class="mode-tag">{mode}</span><br>{msg["content"]}'
                f'<div class="source">{source}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('</div>', unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input("e.g. Which areas have the most MRT stations?")
    if user_input:
        st.session_state.pending_query = user_input

    # Process pending query
    if st.session_state.pending_query:
        query = st.session_state.pending_query
        st.session_state.pending_query = None

        # Add user message
        st.session_state.chat.append({"role": "user", "content": query})

        # Generate answer
        with st.spinner("🔍 Searching knowledge graph..."):
            gen = get_generator()
            result = gen.answer(query)

        # Add bot message
        source_text = " | ".join(s.get("Source", "") if isinstance(s, dict) else str(s)
                                  for s in result.get("sources_used", [])[:2])
        st.session_state.chat.append({
            "role": "assistant",
            "content": result["answer_text"],
            "confidence": result.get("confidence", "MEDIUM"),
            "mode": result.get("retrieval_mode", "?"),
            "source": f"📎 {source_text}" if source_text else "",
        })

        # Update map data
        st.session_state.map_data = {
            "entities": result.get("entities", []),
            "mode": result.get("retrieval_mode", ""),
            "query": query,
        }
        st.rerun()


# ── Right Column: Map + Legend ────────────────────────────────────
with col_right:
    st.markdown('<div class="map-container">', unsafe_allow_html=True)

    # Build Folium map
    m = folium.Map(
        location=[1.3521, 103.8198],
        zoom_start=12,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # ── Planning Area Polygons ──
    try:
        pa_path = config.data_dir / "raw" / "onemap" / "planning_areas.parquet"
        if pa_path.exists():
            pa_df = pd.read_parquet(pa_path)
            area_fg = FeatureGroup(name="Planning Areas", show=False)
            for _, row in pa_df.iterrows():
                try:
                    folium.GeoJson(
                        json.loads(row["geojson"]),
                        style_function=lambda x: {"fillColor": "#3388ff", "color": "#3388ff", "weight": 0.5, "fillOpacity": 0.03},
                        tooltip=row["pln_area_n"].title(),
                    ).add_to(area_fg)
                except Exception:
                    pass
            area_fg.add_to(m)
    except Exception:
        pass

    # ── MRT Lines (colored) ──
    line_colors = {"EWL": "#009530", "NSL": "#D42E2B", "NEL": "#9900AA",
                   "CCL": "#FA9E0D", "DTL": "#005EC4", "TEL": "#9D5B25", "CGL": "#009530"}
    line_names = {"EWL": "East-West Line", "NSL": "North-South Line", "NEL": "North East Line",
                  "CCL": "Circle Line", "DTL": "Downtown Line", "TEL": "Thomson-East Coast Line", "CGL": "Changi Airport Line"}

    try:
        lines_data = run_query("""
            MATCH (a:TransportNode {transport_type: 'mrt'})-[r:CONNECTS_TO]->(b:TransportNode {transport_type: 'mrt'})
            RETURN a.name AS from_n, a.lat AS f_lat, a.lon AS f_lon,
                   b.name AS to_n, b.lat AS t_lat, b.lon AS t_lon, r.line AS line
            LIMIT 150
        """)
        mrt_groups = {}
        for ld in lines_data:
            line = ld.get("line", "?")
            if line not in mrt_groups:
                mrt_groups[line] = FeatureGroup(name=f"{line} — {line_names.get(line, line)}")
            color = line_colors.get(line, "#888")
            try:
                folium.PolyLine(
                    [[float(ld["f_lat"]), float(ld["f_lon"])], [float(ld["t_lat"]), float(ld["t_lon"])]],
                    color=color, weight=3, opacity=0.75,
                    tooltip=f"{line}: {ld.get('from_n','')} → {ld.get('to_n','')}"
                ).add_to(mrt_groups[line])
            except Exception:
                pass
        for fg in mrt_groups.values():
            fg.add_to(m)
    except Exception:
        pass

    # ── MRT Stations ──
    try:
        mrt_data = run_query("MATCH (n:TransportNode {transport_type: 'mrt'}) RETURN n.name AS name, n.lat AS lat, n.lon AS lon, n.planning_area AS area LIMIT 200")
        station_fg = FeatureGroup(name="MRT Stations")
        for s in mrt_data:
            if s.get("lat") and s.get("lon"):
                folium.CircleMarker(
                    [float(s["lat"]), float(s["lon"])], radius=3.5,
                    color="#cc0000", fill=True, fill_color="#cc0000", fill_opacity=0.8,
                    popup=f"<b>{s['name']}</b><br><small>{s.get('area','')}</small>",
                    tooltip=s["name"],
                ).add_to(station_fg)
        station_fg.add_to(m)
    except Exception:
        pass

    # ── Highlight entities from last answer ──
    map_data = st.session_state.map_data
    if map_data.get("entities"):
        hl_fg = FeatureGroup(name="🔍 Answer Highlights")
        for e in map_data["entities"][:10]:
            lat = e.get("lat")
            lon = e.get("lon")
            if lat and lon:
                folium.Marker(
                    [float(lat), float(lon)],
                    icon=folium.Icon(color="orange", icon="star", prefix="fa"),
                    popup=e.get("name", ""),
                ).add_to(hl_fg)
        hl_fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    folium_static(m, height=520)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── MRT Line Legend ──
    st.markdown("""<div class="legend-box"><b>MRT Lines</b>""", unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (code, color) in enumerate(line_colors.items()):
        if code == "CGL": continue  # same color as EWL
        name = line_names.get(code, code)
        cols[i % 4].markdown(f'<span class="legend-dot" style="background:{color}"></span> {code}', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.caption(f"UrbanGraph-SG · GraphRAG on Neo4j · {datetime.now().strftime('%Y-%m-%d %H:%M')} · LLM: DeepSeek")

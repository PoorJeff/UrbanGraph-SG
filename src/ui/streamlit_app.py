"""
UrbanGraph-SG — Clean Streamlit Interface
Two-column: Presets+Chat on left, Map on right
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

# ── Page Config ──
st.set_page_config(
    page_title="UrbanGraph-SG | Singapore Knowledge Graph",
    page_icon="🇸🇬",
    layout="wide",
)

# ── Compact CSS ──
st.markdown("""
<style>
.main .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 100%; }
header { visibility: hidden; }
.stButton button { text-align: left; border-radius: 6px; border: 1px solid #e0e0e0;
    background: white; font-size: 0.8rem; padding: 6px 10px; margin: 0 0 3px 0; width: 100%; }
.stButton button:hover { border-color: #ED2939; background: #fff5f5; }
.preset-btn-row { display: flex; gap: 4px; flex-wrap: wrap; margin: 4px 0 8px 0; }
.preset-chip { font-size: 0.72rem; padding: 4px 10px; border-radius: 14px; border: 1px solid #ddd;
    background: #f8f9fa; cursor: pointer; white-space: nowrap; display: inline-block; }
.preset-chip:hover { background: #fee; border-color: #ED2939; }
h3 { font-size: 1.1rem !important; margin: 0 0 4px 0 !important; }
.chat-box { border: 1px solid #eee; border-radius: 8px; padding: 10px; min-height: 250px; max-height: 450px; overflow-y: auto; background: #fafafa; }
.chat-user { background: #ED2939; color: white; padding: 6px 12px; border-radius: 10px 10px 2px 10px; margin: 4px 0; display: inline-block; font-size: 0.82rem; max-width: 90%; }
.chat-bot { background: white; padding: 8px 14px; border-radius: 10px 10px 10px 2px; margin: 4px 0; border: 1px solid #eee; font-size: 0.82rem; max-width: 95%; }
.badge-green { background: #d4edda; color: #155724; padding: 1px 6px; border-radius: 8px; font-size: 0.65rem; }
.badge-yellow { background: #fff3cd; color: #856404; padding: 1px 6px; border-radius: 8px; font-size: 0.65rem; }
.badge-red { background: #f8d7da; color: #721c24; padding: 1px 6px; border-radius: 8px; font-size: 0.65rem; }
</style>
""", unsafe_allow_html=True)


# ── Cached ──
@st.cache_resource
def get_gen(): return AnswerGenerator()

@st.cache_data(ttl=60)
def get_stats():
    try:
        n = run_query("MATCH (mrt:TransportNode {transport_type:'mrt'}) RETURN count(mrt) AS c")[0]["c"]
        b = run_query("MATCH (bus:TransportNode {transport_type:'bus'}) RETURN count(bus) AS c")[0]["c"]
        a = run_query("MATCH (pa:PlanningArea) RETURN count(pa) AS c")[0]["c"]
        return n, b, a
    except: return 137, 5207, 55

@st.cache_data
def get_presets():
    try: return config.preset_questions.get("preset_questions", [])
    except: return []


# ── Session ──
if "chat" not in st.session_state: st.session_state.chat = []
if "pending" not in st.session_state: st.session_state.pending = None


# ══════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════

# ── Top Bar ──
st.markdown("### 🇸🇬 UrbanGraph-SG")
mrt_n, bus_n, area_n = get_stats()
st.caption(f"**{mrt_n}** MRT · **{bus_n}** Bus stops · **{area_n}** Planning Areas · Neo4j + DeepSeek")

# ── Preset Chips ──
presets = get_presets()
chips_html = '<div class="preset-btn-row">'
domain_icon = {"transport": "🚇", "population": "👥", "housing": "🏠", "spatial": "📍"}
for q in presets[:20]:
    d = q.get("domain","")
    icon = next((v for k,v in domain_icon.items() if k in d), "❓")
    t = q["text"][:60]
    chips_html += f'<span class="preset-chip">{icon} {t}</span>'
chips_html += '</div>'
st.markdown(chips_html, unsafe_allow_html=True)
st.caption("Click a question below or type your own —")

# ── Two Columns ──
left, right = st.columns([1, 1.15])

with left:
    # Preset buttons in a compact grid
    cols = st.columns(2)
    for i, q in enumerate(presets[:20]):
        icon = next((v for k,v in domain_icon.items() if k in q.get("domain","")), "❓")
        label = f"{icon} {q['text'][:55]}"
        with cols[i % 2]:
            if st.button(label, key=f"p{q['id']}", use_container_width=True):
                st.session_state.pending = q["text"]

    # Chat area
    st.divider()
    for msg in st.session_state.chat:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            conf = msg.get("confidence", "MEDIUM")
            badge = {"HIGH": "badge-green", "MEDIUM": "badge-yellow"}.get(conf, "badge-red")
            ans = msg["content"][:400].replace("\n","<br>")
            st.markdown(f'<div class="chat-bot"><span class="{badge}">{conf}</span> {ans}</div>', unsafe_allow_html=True)

    # Input
    q = st.chat_input("e.g. Which areas have the most MRT stations?")
    if q: st.session_state.pending = q

    # Process
    if st.session_state.pending:
        query = st.session_state.pending
        st.session_state.pending = None
        st.session_state.chat.append({"role": "user", "content": query})
        with st.spinner("Searching..."):
            r = get_gen().answer(query)
        st.session_state.chat.append({"role": "assistant", "content": r["answer_text"], "confidence": r.get("confidence","MEDIUM")})
        st.rerun()

# ══════════════════════════════════════════════════════════════════
# RIGHT: MAP
# ══════════════════════════════════════════════════════════════════
with right:
    m = folium.Map(location=[1.3521, 103.8198], zoom_start=12, tiles="CartoDB positron", control_scale=True)

    line_colors = {"EWL": "#009530", "NSL": "#D42E2B", "NEL": "#9900AA",
                   "CCL": "#FA9E0D", "DTL": "#005EC4", "TEL": "#9D5B25", "CGL": "#009530"}
    line_names = {"EWL": "East-West", "NSL": "North-South", "NEL": "North East",
                  "CCL": "Circle", "DTL": "Downtown", "TEL": "Thomson-East Coast", "CGL": "Changi Airport"}

    # MRT lines
    try:
        ld = run_query("MATCH (a:TransportNode {transport_type:'mrt'})-[r:CONNECTS_TO]->(b:TransportNode {transport_type:'mrt'}) RETURN a.lat AS alat, a.lon AS alon, b.lat AS blat, b.lon AS blon, r.line AS line LIMIT 150")
        groups = {}
        for row in ld:
            ln = row.get("line","?")
            if ln not in groups: groups[ln] = FeatureGroup(name=f"{ln} ({line_names.get(ln,'')})")
            try:
                folium.PolyLine([[float(row["alat"]),float(row["alon"])],[float(row["blat"]),float(row["blon"])]],
                    color=line_colors.get(ln,"#888"), weight=2.5, opacity=0.7).add_to(groups[ln])
            except: pass
        for g in groups.values(): g.add_to(m)
    except: pass

    # MRT stations
    try:
        mrts = run_query("MATCH (n:TransportNode {transport_type:'mrt'}) RETURN n.name AS name, n.lat AS lat, n.lon AS lon, n.planning_area AS pa LIMIT 200")
        fg = FeatureGroup(name="MRT Stations")
        for s in mrts:
            if s.get("lat") and s.get("lon"):
                folium.CircleMarker([float(s["lat"]),float(s["lon"])], radius=3,
                    color="#cc0000", fill=True, fill_color="#cc0000", fill_opacity=0.8,
                    tooltip=s["name"]).add_to(fg)
        fg.add_to(m)
    except: pass

    folium.LayerControl().add_to(m)
    folium_static(m, height=460)

    # Legend
    legend_html = '<small>'
    for i,(code,color) in enumerate(line_colors.items()):
        if code == "CGL": continue
        legend_html += f'<span style="color:{color}">●</span> {code} '
    legend_html += '</small>'
    st.markdown(legend_html, unsafe_allow_html=True)
    st.caption(f"7 MRT lines · 137 stations · Updated {datetime.now().strftime('%H:%M')}")

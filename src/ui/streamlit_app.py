"""UrbanGraph-SG — Compact Streamlit UI"""
import json, sys
from pathlib import Path

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from folium import FeatureGroup

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import config
from src.generation.answer_generator import AnswerGenerator
from src.graph.neo4j_client import run_query

st.set_page_config(page_title="UrbanGraph-SG", page_icon="🇸🇬", layout="wide")

# ── Clean CSS ──
st.markdown("""
<style>
.main .block-container { padding: 0.8rem 1.5rem; max-width: 100%; }
header { visibility: hidden; }
.stButton button { width: 100%; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_gen(): return AnswerGenerator()

@st.cache_data(ttl=60)
def get_stats():
    try:
        mrt = run_query("MATCH (mrt:TransportNode {transport_type:'mrt'}) RETURN count(mrt) AS c")[0]["c"]
        bus = run_query("MATCH (bus:TransportNode {transport_type:'bus'}) RETURN count(bus) AS c")[0]["c"]
        hdb = run_query("MATCH (pa:PlanningArea) WHERE pa.avg_resale_price IS NOT NULL RETURN count(pa) AS c")[0]["c"]
        return mrt, bus, hdb
    except: return 137, 5207, 24

@st.cache_data
def get_presets():
    try: return config.preset_questions.get("preset_questions", [])
    except: return []

if "chat" not in st.session_state: st.session_state.chat = []

presets = get_presets()
choices = {q["text"]: q for q in presets if "text" in q}

# ═══════════════════ HEADER ═══════════════════
m, b, h = get_stats()
cols_top = st.columns([1, 4])
with cols_top[0]: st.markdown("### 🇸🇬 UrbanGraph-SG")
with cols_top[1]:
    st.markdown(f"**{m}** MRT stations &nbsp;·&nbsp; **{b}** bus stops &nbsp;·&nbsp; **{h}** areas with HDB prices &nbsp;·&nbsp; Neo4j + DeepSeek")

st.divider()

# ═══════════════════ TWO COLUMNS ═══════════════════
left, right = st.columns([0.85, 1.15])

with left:
    # ── Preset selector + ask button ──
    pick = st.selectbox("Quick question:", list(choices.keys()), index=0, label_visibility="collapsed")
    c1, c2 = st.columns([3, 1])
    with c2:
        if st.button("Ask", type="primary", use_container_width=True):
            q = str(pick)
            st.session_state.chat.append({"role": "user", "content": q})
            with st.spinner("..."):
                r = get_gen().answer(q)
            st.session_state.chat.append({"role": "assistant", "content": r["answer_text"], "confidence": r.get("confidence", "MEDIUM")})
            st.rerun()

    # ── Or type ──
    with c1:
        manual = st.chat_input("Or type your own question...")
        if manual:
            st.session_state.chat.append({"role": "user", "content": manual})
            with st.spinner("..."):
                r = get_gen().answer(manual)
            st.session_state.chat.append({"role": "assistant", "content": r["answer_text"], "confidence": r.get("confidence", "MEDIUM")})
            st.rerun()

    # ── Chat ──
    for msg in st.session_state.chat:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                conf = msg.get("confidence", "MEDIUM")
                emoji = {"HIGH": "🟢", "MEDIUM": "🟡"}.get(conf, "⚪")
                st.write(f"{emoji} {msg['content']}")

# ═══════════════════ RIGHT: MAP ═══════════════════
with right:
    m = folium.Map(location=[1.3521, 103.8198], zoom_start=12, tiles="CartoDB positron", control_scale=True)

    line_colors = {"EWL": "#009530", "NSL": "#D42E2B", "NEL": "#9900AA",
                   "CCL": "#FA9E0D", "DTL": "#005EC4", "TEL": "#9D5B25"}
    line_names = {"EWL": "East-West", "NSL": "North-South", "NEL": "North East",
                  "CCL": "Circle", "DTL": "Downtown", "TEL": "Thomson-East Coast"}

    try:
        ld = run_query("MATCH (a:TransportNode {transport_type:'mrt'})-[r:CONNECTS_TO]->(b:TransportNode {transport_type:'mrt'}) RETURN a.lat AS alat, a.lon AS alon, b.lat AS blat, b.lon AS blon, r.line AS line LIMIT 150")
        groups = {}
        for row in ld:
            ln = row.get("line","?")
            if ln not in groups: groups[ln] = FeatureGroup(name=f"{ln} {line_names.get(ln,'')}")
            try:
                folium.PolyLine([[float(row["alat"]),float(row["alon"])],[float(row["blat"]),float(row["blon"])]],
                    color=line_colors.get(ln,"#888"), weight=2.5, opacity=0.7).add_to(groups[ln])
            except: pass
        for g in groups.values(): g.add_to(m)
    except: pass

    try:
        mrts = run_query("MATCH (n:TransportNode {transport_type:'mrt'}) RETURN n.name AS name, n.lat AS lat, n.lon AS lon LIMIT 200")
        fg = FeatureGroup(name="Stations")
        for s in mrts:
            if s.get("lat") and s.get("lon"):
                folium.CircleMarker([float(s["lat"]),float(s["lon"])], radius=3,
                    color="#cc0000", fill=True, fill_color="#cc0000", fill_opacity=0.8,
                    tooltip=s["name"]).add_to(fg)
        fg.add_to(m)
    except: pass

    folium.LayerControl().add_to(m)
    folium_static(m, height=480)

    # Legend
    parts = [f'<span style="color:{c}">●</span> {code}' for code, c in line_colors.items()]
    st.markdown(" &nbsp; ".join(parts), unsafe_allow_html=True)

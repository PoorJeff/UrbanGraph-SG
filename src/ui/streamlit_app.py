"""UrbanGraph-SG — Streamlit UI (no cache, hardcoded presets)"""
import sys
from pathlib import Path
import streamlit as st
import folium
from streamlit_folium import folium_static
from folium import FeatureGroup

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.generation.answer_generator import AnswerGenerator
from src.graph.neo4j_client import run_query

st.set_page_config(page_title="UrbanGraph-SG", page_icon="🇸🇬", layout="wide")

# ── Hardcoded verified presets (NO YAML, NO CACHE) ──
PRESETS = [
    "How many MRT stations are there in total?",
    "How many bus stops are there in Singapore?",
    "How many stations are on the Circle Line?",
    "How many MRT stations are in the CBD area?",
    "Which MRT lines pass through Bishan?",
    "Which MRT lines pass through Jurong East?",
    "Which MRT lines serve Woodlands?",
    "List all MRT stations in Orchard",
    "Which MRT stations are in Downtown Core?",
    "Which station has the most connections?",
    "Is Bishan station connected to Orchard?",
    "How many stations from Jurong East to City Hall?",
    "Which planning area is Bedok MRT in?",
    "List bus stops along Orchard Road",
    "How many bus stops are near Orchard MRT?",
    "What is the population of Bedok?",
    "What is the population of Tampines?",
    "What is the population of Punggol?",
    "Which planning area has the largest population?",
    "Which areas have the smallest population?",
    "Which area has the highest HDB resale prices?",
    "How many HDB transactions are in the database?",
]

def get_gen(): return AnswerGenerator()

def get_stats():
    try:
        mrt = run_query("MATCH (mrt:TransportNode {transport_type:'mrt'}) RETURN count(mrt) AS c")[0]["c"]
        bus = run_query("MATCH (bus:TransportNode {transport_type:'bus'}) RETURN count(bus) AS c")[0]["c"]
        hdb = run_query("MATCH (pa:PlanningArea) WHERE pa.avg_resale_price IS NOT NULL RETURN count(pa) AS c")[0]["c"]
        return mrt, bus, hdb
    except: return 137, 5207, 24

if "chat" not in st.session_state: st.session_state.chat = []
if "highlight" not in st.session_state: st.session_state.highlight = []

def find_entities_in_answer(answer_text):
    """Extract entity names from answer and get Neo4j coordinates."""
    import re
    # Look for proper nouns: capitalized words, MRT station names, area names
    # Also check for station names in the answer by querying Neo4j
    entities = []
    try:
        # Get all MRT station names and PlanningArea names that appear in the answer
        names = run_query("MATCH (n) WHERE n.name IS NOT NULL RETURN DISTINCT n.name AS name, n.lat AS lat, n.lon AS lon, labels(n)[0] AS label LIMIT 6000")
        for n in names:
            name = str(n.get("name", ""))
            if len(name) > 3 and name.lower() in answer_text.lower():
                lat = n.get("lat")
                lon = n.get("lon")
                if lat and lon and len(entities) < 10:
                    entities.append({"name": name, "lat": float(lat), "lon": float(lon), "label": n.get("label","")})
    except: pass
    return entities

m, b, h = get_stats()

# ═══════ HEADER ═══════
st.markdown(f"### 🇸🇬 UrbanGraph-SG")
st.caption(f"**{m}** MRT · **{b}** bus stops · **{h}** areas with HDB · Neo4j + DeepSeek | v2.1 | 23 verified questions")

# ═══════ TWO COLUMNS ═══════
left, right = st.columns([0.85, 1.15])

with left:
    pick = st.selectbox("Pick a question:", PRESETS, key="preset_select")
    c1, c2 = st.columns([3, 1])
    with c2:
        if st.button("Ask", type="primary", use_container_width=True):
            q = str(pick)
            st.session_state.chat.append({"role": "user", "content": q})
            with st.spinner("..."):
                r = get_gen().answer(q)
            st.session_state.chat.append({"role": "assistant", "content": r["answer_text"],
                "confidence": r.get("confidence","MEDIUM")})
            st.session_state.highlight = find_entities_in_answer(r["answer_text"])
            st.rerun()
    with c1:
        manual = st.chat_input("Or type your own...")
        if manual:
            st.session_state.chat.append({"role": "user", "content": manual})
            with st.spinner("..."):
                r = get_gen().answer(manual)
            st.session_state.chat.append({"role": "assistant", "content": r["answer_text"],
                "confidence": r.get("confidence","MEDIUM")})
            st.session_state.highlight = find_entities_in_answer(r["answer_text"])
            st.rerun()

    for msg in st.session_state.chat:
        if msg["role"] == "user":
            with st.chat_message("user"): st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                c = {"HIGH": "🟢", "MEDIUM": "🟡"}.get(msg.get("confidence",""), "")
                st.write(f"{c} {msg['content']}")

# ═══════ RIGHT: MAP ═══════
with right:
    mp = folium.Map(location=[1.3521, 103.8198], zoom_start=12, tiles="CartoDB positron", control_scale=True)
    colors = {"EWL": "#009530", "NSL": "#D42E2B", "NEL": "#9900AA", "CCL": "#FA9E0D", "DTL": "#005EC4", "TEL": "#9D5B25"}
    names = {"EWL":"East-West","NSL":"North-South","NEL":"North East","CCL":"Circle","DTL":"Downtown","TEL":"Thomson-East Coast"}

    try:
        ld = run_query("MATCH (a:TransportNode {transport_type:'mrt'})-[r:CONNECTS_TO]->(b:TransportNode {transport_type:'mrt'}) RETURN a.lat AS alat, a.lon AS alon, b.lat AS blat, b.lon AS blon, r.line AS line LIMIT 150")
        grps = {}
        for row in ld:
            ln = row.get("line","?")
            if ln not in grps: grps[ln] = FeatureGroup(name=f"{ln} {names.get(ln,'')}")
            try: folium.PolyLine([[float(row["alat"]),float(row["alon"])],[float(row["blat"]),float(row["blon"])]], color=colors.get(ln,"#888"), weight=2.5, opacity=0.7).add_to(grps[ln])
            except: pass
        for g in grps.values(): g.add_to(mp)
    except: pass

    try:
        mrts = run_query("MATCH (n:TransportNode {transport_type:'mrt'}) RETURN n.name AS name, n.lat AS lat, n.lon AS lon LIMIT 200")
        fg = FeatureGroup(name="Stations")
        for s in mrts:
            if s.get("lat") and s.get("lon"):
                folium.CircleMarker([float(s["lat"]),float(s["lon"])], radius=3, color="#cc0000", fill=True, fill_color="#cc0000", fill_opacity=0.8, tooltip=s["name"]).add_to(fg)
        fg.add_to(mp)
    except: pass

    # Highlight entities from last answer
    for e in st.session_state.get("highlight", []):
        try:
            folium.Marker([e["lat"], e["lon"]],
                icon=folium.Icon(color="orange", icon="star", prefix="fa"),
                popup=f"<b>{e['name']}</b><br><small>{e.get('label','')}</small>"
            ).add_to(mp)
        except: pass

    folium.LayerControl().add_to(mp)
    folium_static(mp, height=480)
    parts = [f'<span style="color:{c}">●</span> {code}' for code, c in colors.items()]
    st.markdown(" &nbsp; ".join(parts), unsafe_allow_html=True)

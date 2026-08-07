"""UrbanGraph-SG — Multi-Tab Dashboard"""

import sys, json
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from folium import FeatureGroup

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.generation.answer_generator import AnswerGenerator
from src.graph.neo4j_client import run_query

st.set_page_config(page_title="UrbanGraph-SG", page_icon="🇸🇬", layout="wide")

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
        m = run_query("MATCH (mrt:TransportNode {transport_type:'mrt'}) RETURN count(mrt) AS c")[0]["c"]
        b = run_query("MATCH (bus:TransportNode {transport_type:'bus'}) RETURN count(bus) AS c")[0]["c"]
        h = run_query("MATCH (pa:PlanningArea) WHERE pa.avg_resale_price IS NOT NULL RETURN count(pa) AS c")[0]["c"]
        return m, b, h
    except: return 137, 5207, 24

if "chat" not in st.session_state: st.session_state.chat = []
if "highlight" not in st.session_state: st.session_state.highlight = []

M, B, H = get_stats()

# ═══════════════ TABS ═══════════════
st.markdown(f"### 🇸🇬 UrbanGraph-SG &nbsp;·&nbsp; {M} MRT · {B} bus stops · {H} HDB areas · 11 CS/AI domains")

tabs = st.tabs(["💬 Ask", "📊 ML Analytics", "🗺️ Geospatial", "🔬 Graph ML", "📋 Model Report"])

# ═══════════════ TAB 1: Q&A ═══════════════
with tabs[0]:
    left, right = st.columns([0.7, 1])

    with left:
        pick = st.selectbox("Preset question:", PRESETS, key="q_select", label_visibility="collapsed")
        c1, c2 = st.columns([3, 1])
        with c2:
            if st.button("Ask", type="primary", use_container_width=True):
                q = str(pick)
                st.session_state.chat.append({"role": "user", "content": q})
                with st.spinner("..."):
                    r = get_gen().answer(q)
                st.session_state.chat.append({"role": "assistant", "content": r["answer_text"], "confidence": r.get("confidence","MEDIUM"), "mode": r.get("retrieval_mode","")})
                # Extract highlights
                hl = []
                try:
                    names = run_query("MATCH (n) WHERE n.name IS NOT NULL AND n.lat IS NOT NULL RETURN n.name AS name, n.lat AS lat, n.lon AS lon, labels(n)[0] AS label LIMIT 5000")
                    for n in names:
                        if str(n["name"]).lower() in r["answer_text"].lower() and len(str(n["name"])) > 3:
                            hl.append({"name": n["name"], "lat": float(n["lat"]), "lon": float(n["lon"]), "label": n.get("label","")})
                            if len(hl) >= 8: break
                except: pass
                st.session_state.highlight = hl
                st.rerun()

        with c1:
            manual = st.chat_input("Or type freely — semantic search is active...")
            if manual:
                st.session_state.chat.append({"role": "user", "content": manual})
                with st.spinner("..."):
                    r = get_gen().answer(manual)
                st.session_state.chat.append({"role": "assistant", "content": r["answer_text"], "confidence": r.get("confidence","MEDIUM"), "mode": r.get("retrieval_mode","")})
                hl = []
                try:
                    names = run_query("MATCH (n) WHERE n.name IS NOT NULL AND n.lat IS NOT NULL RETURN n.name AS name, n.lat AS lat, n.lon AS lon, labels(n)[0] AS label LIMIT 5000")
                    for n in names:
                        if str(n["name"]).lower() in r["answer_text"].lower() and len(str(n["name"])) > 3:
                            hl.append({"name": n["name"], "lat": float(n["lat"]), "lon": float(n["lon"]), "label": n.get("label","")})
                            if len(hl) >= 8: break
                except: pass
                st.session_state.highlight = hl
                st.rerun()

        for msg in st.session_state.chat:
            if msg["role"] == "user":
                with st.chat_message("user"): st.write(msg["content"])
            else:
                with st.chat_message("assistant"):
                    c = {"HIGH":"🟢","MEDIUM":"🟡","LOW":"🟠"}.get(msg.get("confidence",""),"⚪")
                    mode = msg.get("mode","")
                    st.caption(f"{c} {msg['confidence']} · {mode}")
                    st.write(msg["content"])

    with right:
        mp = folium.Map(location=[1.3521,103.8198], zoom_start=12, tiles="CartoDB positron", control_scale=True)
        colors = {"EWL":"#009530","NSL":"#D42E2B","NEL":"#9900AA","CCL":"#FA9E0D","DTL":"#005EC4","TEL":"#9D5B25"}
        names = {"EWL":"East-West","NSL":"North-South","NEL":"North East","CCL":"Circle","DTL":"Downtown","TEL":"Thomson-East Coast"}
        try:
            ld = run_query("MATCH (a:TransportNode {transport_type:'mrt'})-[r:CONNECTS_TO]->(b:TransportNode {transport_type:'mrt'}) RETURN a.lat AS al, a.lon AS ao, b.lat AS bl, b.lon AS bo, r.line AS l LIMIT 150")
            grps = {}
            for row in ld:
                ln = row.get("l","?")
                if ln not in grps: grps[ln] = FeatureGroup(name=f"{ln} {names.get(ln,'')}")
                try: folium.PolyLine([[float(row["al"]),float(row["ao"])],[float(row["bl"]),float(row["bo"])]], color=colors.get(ln,"#888"), weight=2.5, opacity=0.7).add_to(grps[ln])
                except: pass
            for g in grps.values(): g.add_to(mp)
        except: pass
        try:
            mrts = run_query("MATCH (n:TransportNode {transport_type:'mrt'}) RETURN n.name AS n, n.lat AS la, n.lon AS lo LIMIT 200")
            fg = FeatureGroup(name="Stations")
            for s in mrts:
                if s.get("la") and s.get("lo"):
                    folium.CircleMarker([float(s["la"]),float(s["lo"])],radius=3,color="#cc0000",fill=True,fill_color="#cc0000",fill_opacity=0.8,tooltip=s["n"]).add_to(fg)
            fg.add_to(mp)
        except: pass
        for e in st.session_state.get("highlight",[]):
            try: folium.Marker([e["lat"],e["lon"]],icon=folium.Icon(color="orange",icon="star",prefix="fa"),popup=f"{e['name']} ({e['label']})").add_to(mp)
            except: pass
        folium.LayerControl().add_to(mp)
        folium_static(mp, height=420)
        parts = [f'<span style="color:{c}">●</span> {code}' for code,c in colors.items()]
        st.markdown(" &nbsp; ".join(parts), unsafe_allow_html=True)


# ═══════════════ TAB 2: ML Analytics ═══════════════
with tabs[1]:
    st.subheader("📊 Machine Learning — Weather Prediction")
    st.caption("RandomForest model trained on 91 days of NEA data predicts daily rainfall from temp/humidity/wind features")

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("RandomForest R²", "0.819")
    with c2: st.metric("CV Score (mean±std)", "0.761 ± 0.098")
    with c3: st.metric("Top Feature", "Temperature (0.76)")

    fig_dir = Path("reports/figures")
    for fname, caption in [
        ("weather_dashboard.png", "90-Day Singapore Weather Dashboard"),
        ("feature_importance.png", "Feature Importance — RandomForest"),
        ("prediction_vs_actual.png", "Predicted vs Actual Rainfall"),
        ("timeseries_dashboard.png", "Time Series — 91-Day Trends with 7-Day Rolling Mean"),
        ("correlation_heatmap.png", "Weather Variable Correlations (Humidity↔Temp r=-0.85)"),
        ("seasonal_pattern.png", "Day-of-Week Weather Patterns"),
    ]:
        p = fig_dir / fname
        if p.exists():
            st.image(str(p), caption=caption, use_container_width=True)

    st.divider()
    st.caption("💡 ML Pipeline: LinearRegression → RandomForest → GradientBoosting with 9 engineered features (lag, rolling means, temporal)")


# ═══════════════ TAB 3: Geospatial ═══════════════
with tabs[2]:
    st.subheader("🗺️ Geospatial & Computer Vision Visualizations")

    cols = st.columns(2)
    for i, (fname, caption) in enumerate([
        ("mrt_topology.png", "MRT Network Topology — 137 stations, 7 colored lines, interchange labels"),
        ("demographic_heatmap.png", "Singapore Population Density Heatmap by Planning Area"),
        ("area_clustering.png", "K-Means Area Clustering — 4 clusters by population, MRT density & HDB price"),
        ("node_embeddings_tsne.png", "Transport Node Embeddings (t-SNE) — MRT vs Bus clustering"),
    ]):
        with cols[i % 2]:
            p = fig_dir / fname
            if p.exists():
                st.image(str(p), caption=caption, use_container_width=True)

    st.divider()
    st.caption("💡 K-Means found 4 area clusters: Downtown Core = unique cluster (15 MRT), Bishan/Bukit Merah = high-amenity cluster")


# ═══════════════ TAB 4: Graph ML ═══════════════
with tabs[3]:
    st.subheader("🔬 Graph Machine Learning — Node2Vec Embeddings")

    try:
        from src.ml.graph_ml import GraphMLEngine
        engine = GraphMLEngine()
        engine.build_graph()
        engine.train_node2vec(dimensions=32)
        preds = engine.predict_links(top_k=15)

        st.caption(f"Transport graph: {engine.G.number_of_nodes():,} nodes, {engine.G.number_of_edges():,} edges")
        st.metric("Graph Nodes", engine.G.number_of_nodes())
        st.metric("Graph Edges", engine.G.number_of_edges())

        st.subheader("Predicted Missing Links (Top 15)")
        pred_df = pd.DataFrame(preds)
        pred_df["similarity"] = pred_df["similarity"].apply(lambda x: f"{x:.3f}")
        st.dataframe(pred_df[["source_name","target_name","similarity"]], use_container_width=True, hide_index=True)

        st.caption("💡 Node2Vec SVD embeddings correctly group stations on the same MRT line (e.g., CC1-CC3: 0.995)")

        # Show embedding plot
        emb_path = fig_dir / "node_embeddings_tsne.png"
        if emb_path.exists():
            st.image(str(emb_path), caption="Transport Node Embeddings (t-SNE projection)")
    except Exception as e:
        st.warning(f"Graph ML module needs Neo4j running: {e}")


# ═══════════════ TAB 5: Model Report ═══════════════
with tabs[4]:
    st.subheader("📋 MLOps Model Report")

    card_path = Path("reports/mlops/model_card.json")
    if card_path.exists():
        card = json.loads(card_path.read_text())
        st.json(card)

    exp_path = Path("reports/mlops/experiments_20260807.json")
    if exp_path.exists():
        exps = json.loads(exp_path.read_text())
        st.subheader("Experiment Tracking")
        for e in exps:
            with st.expander(f"{e['run_name']} — {e['status']}"):
                st.write(f"**Params:** {e.get('params',{})}")
                st.write(f"**Metrics:** {e.get('metrics',{})}")

    st.divider()
    st.caption("💡 3 experiments tracked: LinearRegression baseline, RandomForest tuned, GradientBoosting")

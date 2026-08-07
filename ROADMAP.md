# UrbanGraph-SG Enhancement Roadmap

> Version 2.0 — Multi-domain AI expansion  
> Last updated: 2026-08-07

---

## Priority Tiers

### 🟢 Tier 1 — Foundation (DO FIRST)

These two modules are the foundation. Everything else builds on them. They also fix the biggest current weakness: free-text questions.

| # | Module | Domain | Why First | Risk |
|---|---|---|---|---|
| 1 | **NLP Semantic Search** | NLP, Embeddings | Replaces brittle keyword matching. Makes all free-text questions work. Foundation for everything below. | Low — adds parallel search path |
| 2 | **ChromaDB Vector Store** | Vector DB | Stores embeddings from #1. Enables hybrid graph+vector retrieval. | Low — ChromaDB is lightweight, local |

### 🟡 Tier 2 — Core AI/ML (HIGH RESUME IMPACT)

These are the resume-heavy modules. Each independently adds a new CS/AI domain.

| # | Module | Domain | Resume Line | Dependency |
|---|---|---|---|---|
| 3 | **ML: Weather→Ridership Prediction** | Machine Learning | "Trained RandomForest/GradientBoosting models to predict transit patterns from weather features with R² > 0.X" | NEA data (✅ ready) |
| 4 | **Time Series Analysis** | Time Series, Statistics | "Performed seasonal decomposition and trend analysis on 90-day Singapore weather data" | NEA data (✅ ready) |
| 5 | **Computer Vision Visualization** | CV, Data Viz | "Developed automated geospatial visualization pipeline: topology graphs, demographic heatmaps, cluster plots" | None |
| 6 | **Graph ML: Node2Vec + Link Prediction** | Graph ML, GNN | "Applied Node2Vec graph embeddings to discover latent urban connectivity patterns" | Neo4j graph (✅ ready) |

### 🔵 Tier 3 — Advanced Architecture (DEPTH)

These add architectural sophistication but have higher complexity/risk.

| # | Module | Domain | Resume Line | Dependency |
|---|---|---|---|---|
| 7 | **Multi-Agent System** | Agent Architecture | "Architected multi-agent reasoning system: Planner→Retriever(Cypher+Vector)→Reasoner(LLM)→Visualizer" | #1, #2 |
| 8 | **MLOps Pipeline** | MLOps, DevOps | "Implemented MLOps pipeline with MLflow experiment tracking and Evidently model monitoring" | #3, #4 |

---

## Quality Gates (MANDATORY after each module)

After completing each module, I MUST verify:

1. **`pytest tests/unit/ -v`** — all 37 tests still pass
2. **6 preset questions** — all still answer correctly via `python tests/test_answers.py`
3. **Streamlit UI** — preset questions + free-text both work
4. **No regression** — existing functionality unchanged

If any gate fails, STOP and fix before continuing.

---

## Implementation Log

| # | Module | Status | Date | pytest | 6-answers | Notes |
|---|---|---|---|---|---|---|
| 1 | NLP Semantic Search | ✅ Done | 2026-08-07 | 37/37 ✅ | 6/6 ✅ | Free-text now works via embeddings |
| 2 | ChromaDB Vector Store | ✅ Done | 2026-08-07 | 37/37 ✅ | 6/6 ✅ | 5,439 entities indexed, disk-persisted |
| 3 | ML Weather Prediction | ⏳ In Progress | - | - | - | - |
| 4 | Time Series Analysis | ⏳ Pending | - | - | - | - |
| 5 | CV Visualization | ⏳ Pending | - | - | - | - |
| 6 | Graph ML Node2Vec | ⏳ Pending | - | - | - | - |
| 7 | Multi-Agent System | ⏳ Pending | - | - | - | - |
| 8 | MLOps Pipeline | ⏳ Pending | - | - | - | - |

---

## Current Project State (Baseline)

```
pytest:  37/37 ✅
Answers: 6/6 presets ✅, free-text ⚠️ (keyword match only)
Data:    91 days NEA (2M+ records), 5,532 Neo4j nodes, 10,964 edges
UI:      Streamlit + Folium map with highlight sync
Tests:   entity_resolution, cypher_agent, spatial_validator, answer_generator
```

# =============================================================================
# UrbanGraph-SG Makefile
# =============================================================================

.PHONY: help install setup ingest-all process index load-graph up down clean test lint

# Default target
help:
	@echo "UrbanGraph-SG - Available commands:"
	@echo ""
	@echo "Environment:"
	@echo "  make install        Install all dependencies"
	@echo "  make setup          Full setup: install + create .env if missing"
	@echo ""
	@echo "Data Pipeline:"
	@echo "  make ingest-lta      Ingest LTA DataMall (MRT, bus, traffic)"
	@echo "  make ingest-nea      Ingest NEA weather data"
	@echo "  make ingest-singstat Ingest SingStat population data"
	@echo "  make ingest-geo      Ingest URA + OneMap spatial data"
	@echo "  make ingest-hdb      Ingest HDB resale + rental data"
	@echo "  make ingest-calendar Ingest holiday + school term data"
	@echo "  make ingest-all      Run all 6 ingestion agents"
	@echo "  make process         Run data processing + entity resolution"
	@echo ""
	@echo "GraphRAG:"
	@echo "  make index           Run full GraphRAG indexing pipeline"
	@echo "  make load-graph      Write knowledge graph to Neo4j"
	@echo ""
	@echo "Deployment:"
	@echo "  make up              Start Neo4j + Streamlit (Docker Compose)"
	@echo "  make down            Stop all services"
	@echo ""
	@echo "Quality:"
	@echo "  make test            Run pytest"
	@echo "  make lint            Run ruff linter"
	@echo "  make format          Run ruff formatter"
	@echo "  make evaluate        Run evaluation on preset questions"

# --- Environment ---

install:
	pip install -r requirements/base.txt -r requirements/graphrag.txt -r requirements/llm.txt -r requirements/dev.txt

setup: install
	@if [ ! -f .env ]; then cp .env.example .env && echo "Created .env — please edit it with your API keys"; fi

# --- Data Ingestion ---

ingest-lta:
	python -m src.ingestion.lta --data_types all

ingest-nea:
	python -m src.ingestion.nea

ingest-singstat:
	python -m src.ingestion.singstat

ingest-geo:
	python -m src.ingestion.onemap

ingest-hdb:
	python -m src.ingestion.hdb

ingest-calendar:
	python -m src.ingestion.calendar

ingest-all:
	python -m src.ingestion.lta --data_types all
	python -m src.ingestion.nea
	python -m src.ingestion.singstat
	python -m src.ingestion.onemap
	python -m src.ingestion.hdb
	python -m src.ingestion.calendar

# --- Processing ---

process:
	python -m src.processing.run_all

# --- GraphRAG ---

index:
	python -m src.graphrag.extract_entities
	python -m src.graphrag.extract_relationships
	python -m src.graphrag.detect_communities
	python -m src.graphrag.summarize_communities

load-graph:
	python -m src.graph.write_to_neo4j

# --- Deployment ---

up:
	docker compose up --build -d

down:
	docker compose down

# --- Quality ---

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

evaluate:
	python -m src.evaluation.run_eval

# Streamlit UI for UrbanGraph-SG
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements/base.txt requirements/base.txt
COPY requirements/graphrag.txt requirements/graphrag.txt
COPY requirements/llm.txt requirements/llm.txt
RUN pip install --no-cache-dir \
    -r requirements/base.txt \
    -r requirements/graphrag.txt \
    -r requirements/llm.txt

# Copy application code
COPY src/ src/
COPY configs/ configs/

# Expose Streamlit port
EXPOSE 8502

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8502/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "src/ui/streamlit_app.py", \
     "--server.port=8502", \
     "--server.address=0.0.0.0", \
     "--browser.serverAddress=localhost"]

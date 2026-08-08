"""UrbanGraph-SG — Streamlit Cloud Entry Point."""

import os
import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="UrbanGraph-SG", page_icon="🇸🇬", layout="wide")

SECRET_ENV_MAP = {
    "LLM_BASE_URL": "LLM_BASE_URL",
    "LLM_API_KEY": "LLM_API_KEY",
    "LLM_MODEL": "LLM_MODEL",
    "NEO4J_URI": "NEO4J_URI",
    "NEO4J_USER": "NEO4J_USER",
    "NEO4J_PASSWORD": "NEO4J_PASSWORD",
    "LOG_LEVEL": "LOG_LEVEL",
    "DATA_DIR": "DATA_DIR",
}

for secret_key, env_var in SECRET_ENV_MAP.items():
    if secret_key in st.secrets and env_var not in os.environ:
        os.environ[env_var] = str(st.secrets[secret_key])

sys.path.insert(0, str(Path(__file__).parent))
from src.ui.streamlit_app import *  # noqa: E402, F403

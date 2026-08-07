"""Configuration loader for UrbanGraph-SG.

Reads configuration from YAML files and environment variables.
All settings are available as a singleton Config object.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Environment variables loaded from .env file."""

    # LLM
    llm_base_url: str = "http://localhost:8000/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"

    # LTA
    lta_account_key: str = ""

    # OneMap
    onemap_api_token: str = ""

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "urbangraph-sg-dev"

    # Project
    log_level: str = "INFO"
    data_dir: str = "./data"
    singapore_timezone: str = "Asia/Singapore"

    # Data range
    data_start_year: int = 2024
    data_end_year: int = 2025

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


class Config:
    """Singleton configuration aggregating YAML configs + env settings."""

    _instance = None

    def __init__(self):
        self.settings = Settings()
        self.config_dir = Path(__file__).parent.parent / "configs"
        self.data_dir = Path(self.settings.data_dir).resolve()

        # Load YAML configs lazily
        self._cache: dict[str, dict[str, Any]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load_yaml(self, name: str) -> dict[str, Any]:
        """Load a YAML config file, with caching."""
        if name not in self._cache:
            path = self.config_dir / name
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")
            with open(path, "r", encoding="utf-8") as f:
                self._cache[name] = yaml.safe_load(f)
        return self._cache[name]

    @property
    def data_sources(self) -> dict[str, Any]:
        return self._load_yaml("data_sources.yaml")

    @property
    def schema_mapping(self) -> dict[str, Any]:
        return self._load_yaml("schema_mapping.yaml")

    @property
    def entity_types(self) -> dict[str, Any]:
        return self._load_yaml("entity_types.yaml")

    @property
    def relationship_types(self) -> dict[str, Any]:
        return self._load_yaml("relationship_types.yaml")

    @property
    def graphrag_config(self) -> dict[str, Any]:
        return self._load_yaml("graphrag_config.yaml")

    @property
    def search_config(self) -> dict[str, Any]:
        return self._load_yaml("search_config.yaml")

    @property
    def preset_questions(self) -> dict[str, Any]:
        return self._load_yaml("preset_questions.yaml")


# Global config singleton
config = Config()

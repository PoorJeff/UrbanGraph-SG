"""GraphRAG Entity Extraction Agent.

Uses LLM (DeepSeek) to extract Singapore urban entities from
community text descriptions. Follows §D.1 rules from UrbanGraph-SG-report.md.

The entity extraction runs on community_texts grouped by planning area.
Each planning area's text is processed independently to extract:
- Transport nodes (MRT stations, bus stops mentioned)
- Weather events referenced
- Points of interest
- Housing data mentioned

Note: Deterministic entities from Stage 2 are already loaded.
This module extracts ADDITIONAL entities from unstructured text.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import config
from src.graphrag.llm_client import LLMClient, load_prompt, load_few_shot_examples

logger = logging.getLogger(__name__)


class EntityExtractionAgent:
    """Extract entities from community text descriptions using LLM."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()
        self.prompt = load_prompt("entity_extraction")
        self.few_shots = load_few_shot_examples("entity_extraction")

    def extract_all(
        self,
        community_texts_dir: Path | None = None,
        output_path: Path | None = None,
        dry_run: bool = False,
    ) -> pd.DataFrame:
        """Extract entities from all community text files.

        Args:
            community_texts_dir: Directory with community .txt files
            output_path: Output parquet path
            dry_run: If True, only process the first file

        Returns:
            DataFrame with extracted entities
        """
        if community_texts_dir is None:
            community_texts_dir = config.data_dir / "processed" / "community_texts"
        if output_path is None:
            output_path = config.data_dir / "graphrag" / "output" / "extracted_entities.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_entities: list[dict[str, Any]] = []

        txt_files = sorted(community_texts_dir.glob("*.txt"))
        if dry_run:
            txt_files = txt_files[:1]
            logger.info("DRY RUN: processing only %s", txt_files[0].name if txt_files else "none")

        for i, txt_file in enumerate(txt_files):
            text = txt_file.read_text(encoding="utf-8")
            if len(text.strip()) < 50:
                continue

            area_name = txt_file.stem.replace("_", " ").title()
            logger.debug("[%d/%d] Extracting from: %s", i + 1, len(txt_files), area_name)

            try:
                entities = self._extract_from_text(text, area_name)
                all_entities.extend(entities)
            except Exception as e:
                logger.warning("Failed to extract from %s: %s", area_name, e)
                continue

            # Progress every 10 files
            if (i + 1) % 10 == 0:
                logger.info(
                    "Entity extraction: %d/%d files, %d entities so far",
                    i + 1, len(txt_files), len(all_entities),
                )

        df = pd.DataFrame(all_entities)
        if not df.empty:
            df.to_parquet(output_path, index=False)
            logger.info(
                "Entity extraction complete: %d entities from %d files → %s",
                len(df), len(txt_files), output_path,
            )
        else:
            logger.warning("No entities extracted from %d files", len(txt_files))

        return df

    def _extract_from_text(
        self, text: str, area_name: str,
    ) -> list[dict[str, Any]]:
        """Extract entities from a single community text block.

        Uses few-shot prompting to guide the LLM toward extracting
        Singapore urban domain entities with proper formatting.
        """
        # Build few-shot examples
        few_shot_text = ""
        if self.few_shots:
            few_shot_text = "\n\n### EXAMPLES:\n\n"
            for ex in self.few_shots[:3]:  # Use up to 3 examples
                few_shot_text += f"INPUT:\n{ex.get('input', '')}\n\n"
                few_shot_text += f"OUTPUT:\n{ex.get('output', '')}\n\n"

        user_prompt = self.prompt["user"].format(text=text) if "{text}" in self.prompt.get("user", "") else text

        system_prompt = f"""{self.prompt['system']}

Context: This is about the {area_name} planning area in Singapore.
{few_shot_text}
"""
        try:
            response = self.llm.chat_with_json_output(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                label=f"entity_extract_{area_name}",
                max_tokens=2000,
            )

            if "parse_error" in response:
                logger.warning("JSON parse error for %s", area_name)
                return []

            # Response should be a list of entity dicts
            if isinstance(response, dict) and "entities" in response:
                entities = response["entities"]
            elif isinstance(response, list):
                entities = response
            else:
                logger.debug("Unexpected response format for %s: %s", area_name, type(response))
                return []

            # Normalize and validate
            valid_entities = []
            for e in entities:
                if isinstance(e, dict) and "name" in e:
                    e.setdefault("planning_area", area_name)
                    e.setdefault("source", "llm_extraction")
                    valid_entities.append(e)

            return valid_entities

        except RuntimeError as e:
            logger.error("LLM call failed for %s: %s", area_name, e)
            return []


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.settings.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    agent = EntityExtractionAgent()
    df = agent.extract_all(dry_run=True)
    if not df.empty:
        print(f"Extracted {len(df)} entities")
        print(df.head())
    stats = agent.llm.get_stats()
    print(f"\nLLM Stats: {json.dumps(stats, indent=2)}")

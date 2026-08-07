"""Community Summarization Agent.

Generates natural language summaries for each detected community.
Summaries are used by Global Search for map-reduce style answers.

Follows §D.3 rules from UrbanGraph-SG-report.md:
- Each summary 200-500 tokens
- Must include community theme (one sentence)
- List ≤5 representative entities
- Include ≥1 numerical highlight
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import config
from src.graphrag.llm_client import LLMClient, load_prompt, load_few_shot_examples

logger = logging.getLogger(__name__)


class CommunitySummarizationAgent:
    """Generate natural language summaries for graph communities."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()
        self.prompt = load_prompt("summarization")
        self.few_shots = load_few_shot_examples("summarization")

    def summarize_all(
        self,
        communities_path: Path | None = None,
        entity_community_map_path: Path | None = None,
        entities_path: Path | None = None,
        relationships_path: Path | None = None,
        output_path: Path | None = None,
        dry_run: bool = False,
    ) -> pd.DataFrame:
        """Generate summaries for all communities.

        Args:
            communities_path: communities.parquet
            entity_community_map_path: entity_community_map.parquet
            entities_path: entities.parquet
            relationships_path: relationships.parquet
            output_path: output path for community_reports.parquet
            dry_run: Only summarize first 3 communities

        Returns:
            DataFrame with community_id, title, summary, entities, highlights
        """
        if output_path is None:
            output_path = config.data_dir / "graphrag" / "output" / "community_reports.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load data
        comms = self._load(communities_path, config.data_dir / "graphrag" / "output" / "communities.parquet")
        ent_map = self._load(entity_community_map_path, config.data_dir / "graphrag" / "output" / "entity_community_map.parquet")
        ents = self._load(entities_path, config.data_dir / "graphrag" / "input" / "entities.parquet")
        rels = self._load(relationships_path, config.data_dir / "graphrag" / "input" / "relationships.parquet")

        if comms.empty:
            logger.error("No communities to summarize")
            return pd.DataFrame()

        # Build lookup maps
        entity_lookup = {}
        if not ents.empty:
            entity_lookup = dict(zip(ents["id"], ents.to_dict("records")))

        community_entities: dict[str, list[str]] = {}
        if not ent_map.empty:
            for _, row in ent_map.iterrows():
                cid = row.get("community_id", "")
                eid = row.get("entity_id", "")
                if cid and eid:
                    community_entities.setdefault(cid, []).append(eid)

        reports: list[dict[str, Any]] = []
        if dry_run:
            comms_to_process = comms.nlargest(3, "member_count")
        else:
            comms_to_process = comms

        for i, (_, comm) in enumerate(comms_to_process.iterrows()):
            cid = comm.get("community_id", "")
            members = community_entities.get(cid, [])

            if len(members) < MIN_MEMBERS_FOR_SUMMARY:
                continue

            # Build context for the LLM
            member_info = self._build_member_context(members, entity_lookup)
            community_context = (
                f"Community: {comm.get('title', cid)}\n"
                f"Member count: {comm.get('member_count', 0)}\n"
                f"Type distribution: {comm.get('type_distribution', '')}\n\n"
                f"Member details:\n{member_info}"
            )

            try:
                summary = self._summarize_community(community_context, cid)
                if summary:
                    reports.append({
                        "community_id": cid,
                        "title": comm.get("title", ""),
                        "summary": summary.get("summary", ""),
                        "theme": summary.get("theme", ""),
                        "key_entities": summary.get("key_entities", []),
                        "numerical_highlight": summary.get("numerical_highlight", ""),
                        "member_count": comm.get("member_count", 0),
                        "level": comm.get("level", 0),
                    })
            except RuntimeError as e:
                logger.warning("Summarization failed for %s: %s", cid, e)
                continue

            if (i + 1) % 5 == 0:
                logger.info("Summarized %d/%d communities", i + 1, len(comms_to_process))

        df = pd.DataFrame(reports)
        if not df.empty:
            df.to_parquet(output_path, index=False)
            logger.info(
                "Summarization complete: %d community reports → %s",
                len(df), output_path,
            )

        return df

    def _summarize_community(
        self, community_context: str, community_id: str,
    ) -> dict[str, Any] | None:
        """Call LLM to generate a community summary. Accepts text or JSON."""
        system_prompt = self.prompt.get("system", "")

        few_shot_text = ""
        if self.few_shots:
            few_shot_text = "\n\n### EXAMPLE:\n"
            ex = self.few_shots[0]
            few_shot_text += f"INPUT:\n{ex.get('input', '')}\n\n"
            few_shot_text += f"OUTPUT:\n{ex.get('output', '')}\n"

        full_system = f"{system_prompt}{few_shot_text}"

        text = self.llm.chat(
            system_prompt=full_system,
            user_prompt=f"Summarize this community:\n\n{community_context}",
            label=f"summarize_{community_id}",
            max_tokens=1000,
        )

        if not text:
            return None

        # Try to parse as JSON first, accept text as fallback
        try:
            import json
            cleaned = text.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = "\n".join(lines)
            result = json.loads(cleaned)
            return result
        except (json.JSONDecodeError, ValueError):
            # Accept text output as summary
            summary = text.strip()
            # Try to extract theme from first line
            theme = ""
            lines = summary.split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("## Community:") or line.startswith("This community"):
                    theme = line.replace("## Community:", "").strip()
                    break

            return {
                "summary": summary,
                "theme": theme,
                "key_entities": [],
                "numerical_highlight": "",
            }

    def _build_member_context(
        self, member_ids: list[str], entity_lookup: dict[str, dict[str, Any]],
    ) -> str:
        """Build a text description of community members for the LLM."""
        lines = []
        seen_types: dict[str, int] = {}

        for eid in member_ids[:30]:  # cap at 30 for token budget
            entity = entity_lookup.get(eid, {})
            etype = entity.get("type", "unknown")
            count = seen_types.get(etype, 0)
            if count >= 10:  # max 10 per type
                continue
            seen_types[etype] = count + 1

            name = entity.get("name", eid)
            desc = entity.get("description", "")
            lines.append(f"- [{etype}] {name}: {desc[:150]}")

        return "\n".join(lines)

    def _load(self, path: Path | None, default: Path) -> pd.DataFrame:
        """Load parquet if exists, return empty DataFrame otherwise."""
        p = path or default
        if p.exists():
            return pd.read_parquet(p)
        return pd.DataFrame()


MIN_MEMBERS_FOR_SUMMARY = 3


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.settings.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    agent = CommunitySummarizationAgent()
    df = agent.summarize_all(dry_run=True)
    if not df.empty:
        print(f"Generated {len(df)} community summaries")
        print(df["summary"].iloc[0][:300] if len(df) > 0 else "")
    stats = agent.llm.get_stats()
    print(f"\nLLM Stats: {json.dumps(stats, indent=2)}")

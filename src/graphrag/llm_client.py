"""Shared LLM client for GraphRAG pipeline.

Wraps OpenAI-compatible API (DeepSeek) with:
- Prompt template rendering
- Token usage tracking
- Retry on transient errors
- Response logging for audit
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

from src.config import config

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI-compatible LLM client configured for DeepSeek."""

    def __init__(self):
        settings = config.settings
        self.client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
        self.model = settings.llm_model
        self.stats: dict[str, Any] = {
            "total_calls": 0,
            "total_tokens": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cost_estimate": 0.0,
            "start_time": time.time(),
        }
        logger.info(
            "LLM Client initialized: model=%s, base_url=%s",
            self.model, settings.llm_base_url,
        )

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.1,
        response_format: dict[str, Any] | None = None,
        label: str = "llm_call",
    ) -> str:
        """Send a chat completion request with retry.

        Args:
            system_prompt: System message
            user_prompt: User message
            max_tokens: Max completion tokens
            temperature: Sampling temperature (0.1 = nearly deterministic)
            response_format: Optional JSON schema for structured output
            label: Label for logging

        Returns:
            LLM response text
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(1, 4):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if response_format:
                    kwargs["response_format"] = response_format

                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""

                # Track usage
                usage = response.usage
                if usage:
                    self.stats["total_calls"] += 1
                    self.stats["total_tokens"] += usage.total_tokens
                    self.stats["total_prompt_tokens"] += usage.prompt_tokens
                    self.stats["total_completion_tokens"] += usage.completion_tokens
                    # DeepSeek pricing (approximate): $0.14/1M input, $0.28/1M output
                    cost = (
                        usage.prompt_tokens * 0.14 / 1_000_000
                        + usage.completion_tokens * 0.28 / 1_000_000
                    )
                    self.stats["total_cost_estimate"] += cost

                logger.debug(
                    "[%s] Call %d: %d tokens (prompt=%d, completion=%d), cost=$%.4f",
                    label, self.stats["total_calls"],
                    usage.total_tokens if usage else 0,
                    usage.prompt_tokens if usage else 0,
                    usage.completion_tokens if usage else 0,
                    cost if usage else 0,
                )
                return content

            except Exception as e:
                logger.warning(
                    "[%s] Attempt %d/3 failed: %s", label, attempt, e,
                )
                if attempt == 3:
                    raise RuntimeError(
                        f"LLM call [{label}] failed after 3 attempts: {e}"
                    ) from e
                time.sleep(2 ** attempt)  # 2s, 4s, 8s

        raise RuntimeError("Unreachable")

    def chat_with_json_output(
        self,
        system_prompt: str,
        user_prompt: str,
        label: str = "llm_json",
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        """Send a chat request and parse the response as JSON.

        Handles markdown code fences and partial JSON.
        """
        text = self.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=0.1,
            label=label,
        )

        # Strip markdown code fences
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:] if len(lines) > 1 else lines
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        # Try to parse as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON array/object boundaries
            text = text.strip()
            if text.startswith("[") or text.startswith("{"):
                try:
                    # Find matching bracket
                    return json.loads(text)
                except json.JSONDecodeError:
                    pass
            logger.warning("[%s] Failed to parse JSON response: %s...", label, text[:200])
            return {"raw_response": text, "parse_error": True}

    def get_stats(self) -> dict[str, Any]:
        """Return accumulated usage statistics."""
        elapsed = time.time() - self.stats["start_time"]
        return {
            **self.stats,
            "elapsed_seconds": round(elapsed, 1),
            "cost_per_hour": round(
                self.stats["total_cost_estimate"] / (elapsed / 3600), 4
            ) if elapsed > 0 else 0,
        }


def load_prompt(prompt_name: str) -> dict[str, str]:
    """Load a prompt template from YAML file.

    Looks in src/graphrag/prompts/ directory.
    Returns dict with 'system' and 'user' keys.
    """
    prompts_dir = Path(__file__).parent / "prompts"
    path = prompts_dir / f"{prompt_name}.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        prompt_data = yaml.safe_load(f)

    if not isinstance(prompt_data, dict):
        raise ValueError(f"Invalid prompt format in {path}")

    return {
        "system": prompt_data.get("system", ""),
        "user": prompt_data.get("user", ""),
    }


def load_few_shot_examples(prompt_name: str) -> list[dict[str, str]]:
    """Load few-shot examples from a prompt YAML file."""
    prompts_dir = Path(__file__).parent / "prompts"
    path = prompts_dir / f"{prompt_name}.yaml"

    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        prompt_data = yaml.safe_load(f)

    return prompt_data.get("few_shot_examples", [])

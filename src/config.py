from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig


@dataclass
class LabConfig:
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def _build_provider_config(prefix: str = "") -> ProviderConfig:
    provider = os.getenv(f"{prefix}LLM_PROVIDER", "openai").strip().lower()
    model_name = os.getenv(f"{prefix}LLM_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv(f"{prefix}LLM_TEMPERATURE", "0.7"))
    api_key = os.getenv(f"{prefix}API_KEY") or os.getenv(f"{provider.upper()}_API_KEY") or None
    base_url = os.getenv(f"{prefix}CUSTOM_BASE_URL") or os.getenv(f"{provider.upper()}_BASE_URL") or None
    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    compact_threshold = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "400"))
    compact_keep = int(os.getenv("COMPACT_KEEP_MESSAGES", "6"))

    model = _build_provider_config("")
    judge_model = _build_provider_config("JUDGE_")

    if judge_model.model_name == "gpt-4o-mini" and judge_model.provider == "openai":
        judge_model.model_name = os.getenv("JUDGE_LLM_MODEL", "gpt-4o-mini")

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold,
        compact_keep_messages=compact_keep,
        model=model,
        judge_model=judge_model,
    )

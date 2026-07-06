"""Configuration loading: limits + the LIVE/DRY_RUN switch."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .guardrails import Limits

load_dotenv()  # pick up .env (gitignored) so keys don't have to be exported


@dataclass(frozen=True)
class AgentConfig:
    model: str
    effort: str
    max_tokens: int


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    params: dict
    universe: list


@dataclass(frozen=True)
class Config:
    limits: Limits
    agent: AgentConfig
    dry_run: bool
    mcp_url: str
    mcp_token: str
    nvidia_api_key: str
    nvidia_base_url: str
    strategy: "StrategyConfig | None" = None


def is_live() -> bool:
    """Real orders require the literal string 'true'. Everything else is paper."""
    return os.environ.get("LIVE", "").strip().lower() == "true"


def load(path: str | Path = "config.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    limits = Limits(**raw["limits"])
    agent = AgentConfig(**raw["agent"])
    strategy = (
        StrategyConfig(**raw["strategy"]) if raw.get("strategy") else None
    )
    return Config(
        limits=limits,
        agent=agent,
        dry_run=not is_live(),
        mcp_url=os.environ.get("ROBINHOOD_MCP_URL", ""),
        mcp_token=os.environ.get("ROBINHOOD_MCP_TOKEN", ""),
        nvidia_api_key=os.environ.get("NVIDIA_API_KEY", ""),
        nvidia_base_url=os.environ.get(
            "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
        ),
        strategy=strategy,
    )

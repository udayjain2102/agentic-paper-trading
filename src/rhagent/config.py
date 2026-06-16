"""Configuration loading: limits + the LIVE/DRY_RUN switch."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from .guardrails import Limits


@dataclass(frozen=True)
class AgentConfig:
    model: str
    effort: str
    max_tokens: int


@dataclass(frozen=True)
class Config:
    limits: Limits
    agent: AgentConfig
    dry_run: bool
    mcp_url: str
    mcp_token: str


def is_live() -> bool:
    """Real orders require the literal string 'true'. Everything else is paper."""
    return os.environ.get("LIVE", "").strip().lower() == "true"


def load(path: str | Path = "config.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    limits = Limits(**raw["limits"])
    agent = AgentConfig(**raw["agent"])
    return Config(
        limits=limits,
        agent=agent,
        dry_run=not is_live(),
        mcp_url=os.environ.get("ROBINHOOD_MCP_URL", ""),
        mcp_token=os.environ.get("ROBINHOOD_MCP_TOKEN", ""),
    )

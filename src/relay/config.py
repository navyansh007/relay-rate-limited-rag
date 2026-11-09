"""Configuration loading and well-known project paths.

All tunables live in ``config/default.yaml`` so the system can be changed live
(limits, models, retrieval setting, retry policy) without code edits. Scenario
files can overlay any subset of keys.

Typical usage example:

    cfg = load_config("config/scenarios/burst.yaml")
    model = cfg["models"]["draft"]
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
RAW_DIR = ROOT / "raw"
RESULTS_DIR = ROOT / "results"


def _merge(base: dict, over: dict) -> dict:
    """Deep-merges ``over`` into a copy of ``base``.

    Args:
        base: The dictionary to start from; not modified.
        over: Values that take precedence; nested dicts are merged recursively.

    Returns:
        A new merged dictionary.
    """
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(*overlays: str | Path) -> dict[str, Any]:
    """Loads the default config and applies overlay files in order.

    Args:
        *overlays: Paths to YAML files whose keys override the defaults.

    Returns:
        The merged configuration dictionary.
    """
    with open(CONFIG_DIR / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    for path in overlays:
        with open(path) as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    return cfg


def groq_api_key() -> str:
    """Reads the Groq API key from the environment or the project ``.env``.

    Returns:
        The API key.

    Raises:
        RuntimeError: If no real key is configured.
    """
    load_dotenv(ROOT / ".env")
    key = os.environ.get("GROQ_API_KEY", "")
    if not key or key == "gsk_replace_me":
        raise RuntimeError("GROQ_API_KEY not set: copy .env.example to .env and add your key")
    return key

"""Load config/default.yaml (optionally overlaid with another YAML file)."""

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
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(*overlays: str | Path) -> dict[str, Any]:
    with open(CONFIG_DIR / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    for path in overlays:
        with open(path) as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    return cfg


def groq_api_key() -> str:
    load_dotenv(ROOT / ".env")
    key = os.environ.get("GROQ_API_KEY", "")
    if not key or key == "gsk_replace_me":
        raise RuntimeError("GROQ_API_KEY not set: copy .env.example to .env and add your key")
    return key

"""Persistent CLI configuration stored in ~/.ds-skills/config.json."""

from __future__ import annotations

import json
from pathlib import Path

CONFIG_DIR = Path.home() / ".ds-skills"
CONFIG_FILE = CONFIG_DIR / "config.json"

_DEFAULTS = {
    "username": "",
    "hub_url": "https://ds-skills.com",
}


def _load() -> dict:
    if CONFIG_FILE.exists():
        try:
            return {**_DEFAULTS, **json.loads(CONFIG_FILE.read_text(encoding="utf-8"))}
        except (json.JSONDecodeError, OSError):
            pass
    return dict(_DEFAULTS)


def _save(data: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def get(key: str) -> str:
    """Get a config value by key."""
    return _load().get(key, _DEFAULTS.get(key, ""))


def get_all() -> dict:
    """Return full config dict."""
    return _load()


def set_value(key: str, value: str) -> None:
    """Set a single config key."""
    data = _load()
    data[key] = value
    _save(data)


def get_hub_url() -> str:
    """Return the hub base URL (used by Client)."""
    return get("hub_url") or _DEFAULTS["hub_url"]

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    "nickname": {
        "min_score": 40.0,
        "min_top_margin_ratio": 0.05,
        "top_region_ratio": 0.2,
    },
    "dialog": {
        "min_bubble_count": 5,
    },
}


def get_default_config_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "config" / "analysis_config.yaml"


def _deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else get_default_config_path()
    if not path.exists():
        return dict(DEFAULT_CONFIG)
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return _deep_merge(DEFAULT_CONFIG, data)


def save_config(
    config: Dict[str, Any],
    config_path: Optional[str] = None,
    keep_history: bool = True,
) -> Path:
    path = Path(config_path) if config_path else get_default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    if keep_history and path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_dir = path.parent / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        history_path = history_dir / f"{path.stem}_{timestamp}{path.suffix}"
        shutil.copy2(path, history_path)

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)

    return path


def update_config(
    updates: Dict[str, Any],
    config_path: Optional[str] = None,
    keep_history: bool = True,
) -> Dict[str, Any]:
    config = load_config(config_path)
    config = _deep_merge(config, updates)
    save_config(config, config_path=config_path, keep_history=keep_history)
    return config

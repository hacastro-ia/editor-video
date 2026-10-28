"""Preferencias guardadas entre sesiones (cámara usada, proyector, etc.)."""
from __future__ import annotations

import json
from pathlib import Path

PATH = Path.home() / ".editor_video.json"


def load():
    try:
        return json.loads(PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save(data):
    try:
        PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

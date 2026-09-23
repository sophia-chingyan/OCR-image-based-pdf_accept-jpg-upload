"""
Engine Factory
==============
Returns the configured OCR engine instance.

Currently supported engines: gemini (default), poe, openrouter. Which one
actually runs a given job is resolved per job by Worker/worker.py — from
the Settings page (settings:ocr_engine in Redis) if set, else ocr.engine in
config.yaml — not fixed once at worker startup. See
PoeOCREngine/OpenRouterOCREngine/GeminiOCREngine docstrings for per-engine
credential handling.

To add a new engine in the future:
1. Implement OCREngine in a new file (e.g. claude_vision_engine.py)
2. Add a loader function below
3. Register it in the ENGINES dict
4. Set ocr.engine in config.yaml (or select it on the Settings page)
"""

from __future__ import annotations
import logging
from ocr_engine import OCREngine

logger = logging.getLogger(__name__)


def _load_gemini(config: dict) -> OCREngine:
    from gemini_engine import GeminiOCREngine
    return GeminiOCREngine(config)


def _load_poe(config: dict) -> OCREngine:
    from poe_engine import PoeOCREngine
    return PoeOCREngine(config)


def _load_openrouter(config: dict) -> OCREngine:
    from openrouter_engine import OpenRouterOCREngine
    return OpenRouterOCREngine(config)


# Exposed at module level (not just inside get_engine) so callers — e.g.
# worker.py resolving the Settings-page engine choice, or Api/main.py
# validating a POST body — can enumerate valid engine names without
# duplicating this list.
ENGINES = {
    "gemini":     _load_gemini,
    "poe":        _load_poe,
    "openrouter": _load_openrouter,
}


def get_engine(config: dict) -> OCREngine:
    """
    Instantiate and return the OCR engine named by config["engine"].

    config: the full 'ocr' section of config.yaml, with "engine" already
    resolved to whichever name should actually be used for this call (the
    caller — worker.py — decides that; this function just builds it).
    """
    engine_name = config.get("engine", "gemini").lower()

    factory = ENGINES.get(engine_name)
    if factory is None:
        raise ValueError(
            f"Unknown OCR engine: '{engine_name}'. "
            f"Valid options: {list(ENGINES.keys())}"
        )

    logger.info(f"Initialising OCR engine: {engine_name}")
    return factory(config)

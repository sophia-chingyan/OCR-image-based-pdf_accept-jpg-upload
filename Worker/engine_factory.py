"""
Engine Factory
==============
Returns the configured OCR engine instance.

Currently supported engines: gemini (default), poe.

To add a new engine in the future:
1. Implement OCREngine in a new file (e.g. claude_vision_engine.py)
2. Add a loader function below
3. Register it in the ENGINES dict
4. Set ocr.engine in config.yaml
"""

from __future__ import annotations
import logging
from ocr_engine import OCREngine

logger = logging.getLogger(__name__)


def get_engine(config: dict) -> OCREngine:
    """
    Instantiate and return the OCR engine specified in config.yaml.

    config: the full 'ocr' section of config.yaml
    """
    engine_name = config.get("engine", "gemini").lower()

    ENGINES = {
        "gemini": _load_gemini,
        "poe":    _load_poe,
    }

    factory = ENGINES.get(engine_name)
    if factory is None:
        raise ValueError(
            f"Unknown OCR engine: '{engine_name}'. "
            f"Valid options: {list(ENGINES.keys())}"
        )

    logger.info(f"Initialising OCR engine: {engine_name}")
    return factory(config)


def _load_gemini(config: dict) -> OCREngine:
    from gemini_engine import GeminiOCREngine
    return GeminiOCREngine(config)


def _load_poe(config: dict) -> OCREngine:
    from poe_engine import PoeOCREngine
    return PoeOCREngine(config)

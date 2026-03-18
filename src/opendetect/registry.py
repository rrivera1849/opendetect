"""Detector registry — a simple plugin system for machine-text detectors."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opendetect.detectors.base import BaseDetector

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type[BaseDetector]] = {}


def register_detector(name: str):
    """Class decorator that registers a detector under *name*."""

    def wrapper(cls: type[BaseDetector]):
        if name in _REGISTRY:
            logger.warning("Detector %r is being re-registered (overwriting).", name)
        _REGISTRY[name] = cls
        cls.name = name
        return cls

    return wrapper


def get_detector(name: str) -> type[BaseDetector]:
    """Look up a registered detector by name."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown detector {name!r}. "
            f"Available: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[name]


def list_detectors() -> list[str]:
    """Return sorted list of all registered detector names."""
    return sorted(_REGISTRY.keys())


def get_all_detectors() -> dict[str, type[BaseDetector]]:
    """Return a copy of the full registry."""
    return dict(_REGISTRY)

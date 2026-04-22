"""Abstract base class for all machine-text detectors."""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod


class BaseDetector(ABC):
    """Base class that every detector must subclass.

    Subclasses should:
    1. Be decorated with ``@register_detector("name")``.
    2. Implement :meth:`score`.
    3. Optionally override :meth:`add_arguments` for extra CLI flags.
    """

    name: str  # Set by @register_detector

    @abstractmethod
    def score(self, texts: list[str], **kwargs) -> list[float]:
        """Compute a detection score for each text.

        Parameters
        ----------
        texts:
            List of strings to score.
        **kwargs:
            Detector-specific options (e.g. ``background_texts``).

        Returns
        -------
        list[float]
            One score per input text.
        """

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add detector-specific arguments to the CLI parser.

        Override this in subclasses that need extra flags.
        """

    @property
    def requires_fewshot(self) -> bool:
        """Whether this detector needs few-shot background texts."""
        return False

    def teardown(self) -> None:
        """Release resources held on instance attributes.

        Called by the CLI after :meth:`score` finishes (success or
        failure) so the next detector starts with a clean GPU.  The
        default is a no-op; subclasses that persist heavy state on
        ``self`` (HF models, vLLM engines) should override this to
        drop those references explicitly.  Detectors whose heavy
        objects are local to :meth:`score` do not need to override —
        the CLI's ``del`` + ``gc.collect()`` + ``empty_cache()`` covers
        them.
        """

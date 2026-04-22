"""Revise-Detect (Zhu et al., EMNLP 2023).

Zero-shot detector that asks an LLM to revise each input and scores
``Sim(text, revised_text)`` via BARTScore-CNN.  LLM-generated text
requires fewer revisions than human text, so high similarity ⇒ machine.

Known limitation (paper §6): very short inputs yield uninformative
revisions.  We score every text regardless and let the user filter by
length if they care.
"""

from __future__ import annotations

import logging

from opendetect.detectors.base import BaseDetector
from opendetect.registry import register_detector
from opendetect.revisers import load_reviser
from opendetect.revisers.cache import RevisionCache
from opendetect.similarity import BARTScorer

logger = logging.getLogger(__name__)

DEFAULT_REVISER_BACKEND = "hf"


def _resolve_reviser_id(backend: str, model: str | None) -> str:
    """Return the reviser's identifier without loading the model.

    Used as the cache shard key.  Mirrors the defaults in
    :mod:`opendetect.revisers.hf_reviser` and
    :mod:`opendetect.revisers.openai_reviser`.
    """
    if model:
        return model
    if backend == "hf":
        from opendetect.revisers.hf_reviser import DEFAULT_HF_REVISER

        return DEFAULT_HF_REVISER
    if backend == "vllm":
        from opendetect.revisers.vllm_reviser import DEFAULT_VLLM_REVISER

        return DEFAULT_VLLM_REVISER
    if backend == "openai":
        from opendetect.revisers.openai_reviser import DEFAULT_OPENAI_REVISER

        return DEFAULT_OPENAI_REVISER
    raise ValueError(f"Unknown reviser backend: {backend!r}")


@register_detector("revise-detect")
class ReviseDetect(BaseDetector):
    """Revise-and-compare detector using BARTScore-CNN as similarity."""

    def score(
        self,
        texts: list[str],
        **kwargs,
    ) -> list[float]:
        """Compute Revise-Detect scores for each input text.

        Parameters
        ----------
        texts:
            Input texts to score.
        **kwargs:
            Optional keys:

            * ``reviser`` — backend name, ``"hf"`` (default) or ``"openai"``.
            * ``reviser_model`` — model identifier; defaults depend on
              backend (see :mod:`opendetect.revisers`).
            * ``batch_size`` — passed through to both reviser and
              BARTScore (default 16).

        Returns
        -------
        list[float]
            BARTScore similarity between each text and its revised
            version.  Higher ⇒ more likely LLM-generated.
        """
        backend: str = kwargs.get("reviser", DEFAULT_REVISER_BACKEND)
        reviser_model: str | None = kwargs.get("reviser_model")
        batch_size: int = kwargs.get("batch_size", 16)
        max_model_len: int | None = kwargs.get("vllm_max_model_len")
        gpu_memory_utilization: float | None = kwargs.get(
            "vllm_gpu_memory_utilization",
        )

        # Resolve the reviser's identity without loading the model yet —
        # if every text is already cached we never need to instantiate it.
        reviser_id = _resolve_reviser_id(backend, reviser_model)
        cache = RevisionCache(reviser_id=reviser_id)

        # 1. Cache lookup.
        revised: list[str | None] = [cache.get(t) for t in texts]
        missing_idx = [i for i, r in enumerate(revised) if r is None]
        logger.info(
            "Revise-Detect: %d/%d cached, %d to revise.",
            len(texts) - len(missing_idx),
            len(texts),
            len(missing_idx),
        )

        # 2. Revise the cache misses and persist.  Lazy-load the reviser
        #    so runs that fully hit the cache skip model download
        #    entirely.  Close it before BARTScore loads, otherwise
        #    vLLM's allocation keeps the GPU pinned during scoring.
        if missing_idx:
            reviser = load_reviser(
                backend=backend,
                model=reviser_model,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            try:
                missing_texts = [texts[i] for i in missing_idx]
                new_revisions = reviser.revise(
                    missing_texts,
                    batch_size=batch_size,
                )
                cache.put_many(list(zip(missing_texts, new_revisions)))
                for idx, rev in zip(missing_idx, new_revisions):
                    revised[idx] = rev
            finally:
                reviser.close()
                del reviser

        # 3. BARTScore(src=revised, tgt=original) — matches the reference
        #    driver's convention and gives higher scores to LLM text.
        scorer = BARTScorer()
        return scorer.score(
            srcs=[r for r in revised],  # type: ignore[misc]
            tgts=list(texts),
            batch_size=batch_size,
        )

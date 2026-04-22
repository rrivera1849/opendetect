"""DNA-GPT (Yang et al., 2023) — truncate-then-regenerate black-box detector.

Given a text ``x``, take the first γ fraction as prefix ``x'`` and ask a
regenerator LLM to continue ``x'`` ``K`` times.  Compare the weighted
n-gram overlap between each regenerated continuation and the true
remainder ``y_0``: machine-written text re-generates to highly-similar
continuations (high BScore), human-written text does not (low BScore).

We ship the paper's **BScore** (black-box) only.  WScore requires
target-model logits, which we rarely have in practice.

Known limitation (paper §"Truncation Ratio"): very short inputs yield
uninformative prefixes/remainders.  We score every text regardless.
"""

from __future__ import annotations

import logging
import math
import re

from tqdm import tqdm

from opendetect.detectors.base import BaseDetector
from opendetect.regenerators import load_regenerator
from opendetect.regenerators.cache import ContinuationCache
from opendetect.registry import register_detector

logger = logging.getLogger(__name__)

DEFAULT_REGENERATOR_BACKEND = "hf"
DEFAULT_K = 20
DEFAULT_TRUNCATE_RATIO = 0.5
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_NEW_TOKENS = 300
DEFAULT_N_MIN = 1
DEFAULT_N_MAX = 25


# --- Tokenization (ported from DNA-GPT-dist.py) -----------------------------

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_WHITESPACE = re.compile(r"\s+")
_TOKEN_OK = re.compile(r"^[a-z0-9]+$")


def _tokenize(text: str, stemmer, stopwords: frozenset[str]) -> list[str]:
    """Lowercase, regex-split, Porter-stem (len>3), drop stopwords.

    Matches ``DNA-GPT-dist.py::tokenize``.
    """
    text = text.lower()
    text = _NON_ALNUM.sub(" ", text)
    raw = _WHITESPACE.split(text)
    out: list[str] = []
    for tok in raw:
        if tok in stopwords:
            continue
        if len(tok) > 3:
            tok = stemmer.stem(tok)
        if _TOKEN_OK.match(tok):
            out.append(tok)
    return out


def _ngram_set(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    """Return the set of ``n``-grams (as tuples) in ``tokens``."""
    if n <= 0 or len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _bscore(
    remainder_tokens: list[str],
    continuations_tokens: list[list[str]],
    n_min: int,
    n_max: int,
) -> float:
    """Compute BScore between one remainder and its K continuations.

    Uses the paper's weighting ``f(n) = n·log(n)`` (so n=1 contributes
    zero) and normalizes by both ``|y_k|`` and ``|grams(y_0, n)|``, per
    eq. (BScore).  Returns 0.0 if the remainder has no content.
    """
    if not remainder_tokens:
        return 0.0

    K = len(continuations_tokens)
    if K == 0:
        return 0.0

    total = 0.0
    for n in range(n_min, n_max + 1):
        w = n * math.log(n) if n >= 2 else 0.0
        if w == 0.0:
            continue
        g0 = _ngram_set(remainder_tokens, n)
        if not g0:
            continue
        g0_count = len(g0)
        for y_k in continuations_tokens:
            if len(y_k) < n:
                continue
            gk = _ngram_set(y_k, n)
            if not gk:
                continue
            overlap = len(gk & g0)
            if overlap == 0:
                continue
            total += w * overlap / (len(y_k) * g0_count)
    return total / K


# --- Prefix / remainder split ----------------------------------------------


def _truncate_by_words(text: str, ratio: float) -> tuple[str, str]:
    """Split ``text`` at the ``ratio`` word boundary.

    Follows ``DNA-GPT-dist.py::truncate_string_by_words``: use
    whitespace-separated "words" (``str.split()``), then rejoin.
    Returns ``(prefix, remainder)``.  If the text has fewer than 2
    words, the remainder is empty.
    """
    words = text.split()
    if len(words) < 2:
        return text, ""
    cut = max(1, int(ratio * len(words)))
    cut = min(cut, len(words) - 1)
    prefix = " ".join(words[:cut])
    remainder = " ".join(words[cut:])
    return prefix, remainder


# --- Detector --------------------------------------------------------------


def _resolve_regenerator_id(backend: str, model: str | None) -> str:
    """Return the regenerator's identifier without loading it.

    Used as the cache shard key.
    """
    if model:
        return model
    if backend == "hf":
        from opendetect.regenerators.hf_regen import DEFAULT_HF_REGENERATOR

        return DEFAULT_HF_REGENERATOR
    if backend == "vllm":
        from opendetect.regenerators.vllm_regen import (
            DEFAULT_VLLM_REGENERATOR,
        )

        return DEFAULT_VLLM_REGENERATOR
    if backend == "openai":
        from opendetect.regenerators.openai_regen import (
            DEFAULT_OPENAI_REGENERATOR,
        )

        return DEFAULT_OPENAI_REGENERATOR
    raise ValueError(f"Unknown regenerator backend: {backend!r}")


@register_detector("dna-gpt")
class DnaGpt(BaseDetector):
    """Truncate-then-regenerate BScore detector (Yang et al., 2023)."""

    def score(
        self,
        texts: list[str],
        **kwargs,
    ) -> list[float]:
        """Compute DNA-GPT BScores.

        Parameters
        ----------
        texts:
            Input texts to score.
        **kwargs:
            Optional keys:

            * ``regenerator`` — ``"hf"`` (default) or ``"openai"``.
            * ``regenerator_model`` — model identifier; defaults
              depend on backend.
            * ``K`` — number of regenerations per text (default 20).
            * ``truncate_ratio`` — γ ∈ (0, 1) (default 0.5).
            * ``max_new_tokens`` — generation cap (default 300).
            * ``temperature`` — sampling T (default 0.7).
            * ``n_min``, ``n_max`` — n-gram range (default [1, 25]).
            * ``prompts`` — optional list of per-text prompt strings
              prepended to the prefix before regeneration.
            * ``batch_size`` — passed through to the regenerator
              (default 8).

        Returns
        -------
        list[float]
            Per-text BScore.  Higher ⇒ more likely machine-generated.
        """
        backend: str = kwargs.get("regenerator", DEFAULT_REGENERATOR_BACKEND)
        regenerator_model: str | None = kwargs.get("regenerator_model")
        K: int = int(kwargs.get("K", DEFAULT_K))
        truncate_ratio: float = float(
            kwargs.get("truncate_ratio", DEFAULT_TRUNCATE_RATIO),
        )
        max_new_tokens: int = int(
            kwargs.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
        )
        temperature: float = float(
            kwargs.get("temperature", DEFAULT_TEMPERATURE),
        )
        n_min: int = int(kwargs.get("n_min", DEFAULT_N_MIN))
        n_max: int = int(kwargs.get("n_max", DEFAULT_N_MAX))
        prompts: list[str] | None = kwargs.get("prompts")
        batch_size: int = int(kwargs.get("batch_size", 8))
        max_model_len: int | None = kwargs.get("vllm_max_model_len")
        gpu_memory_utilization: float | None = kwargs.get(
            "vllm_gpu_memory_utilization",
        )

        if prompts is not None and len(prompts) != len(texts):
            raise ValueError(
                f"prompts has length {len(prompts)} but texts has "
                f"length {len(texts)}.",
            )

        # Split each text into (prefix, remainder).
        prefixes: list[str] = []
        remainders: list[str] = []
        for i, t in enumerate(texts):
            prefix, remainder = _truncate_by_words(t, truncate_ratio)
            if prompts is not None:
                prompt = prompts[i]
                if prompt:
                    prefix = f"{prompt}\n{prefix}"
            prefixes.append(prefix)
            remainders.append(remainder)

        # Cache lookup.
        regenerator_id = _resolve_regenerator_id(backend, regenerator_model)
        cache = ContinuationCache(regenerator_id=regenerator_id)
        cached: list[list[str] | None] = [
            cache.get(p, truncate_ratio, K) for p in prefixes
        ]
        missing_idx = [i for i, c in enumerate(cached) if c is None]
        logger.info(
            "DNA-GPT: %d/%d prefixes cached, %d to regenerate.",
            len(texts) - len(missing_idx),
            len(texts),
            len(missing_idx),
        )

        # Regenerate cache misses.  Close the regenerator after use so
        # vLLM releases its GPU allocation before the scoring step
        # (and before the next detector in the CLI loop) runs.
        if missing_idx:
            regenerator = load_regenerator(
                backend=backend,
                model=regenerator_model,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            try:
                missing_prefixes = [prefixes[i] for i in missing_idx]
                new_continuations = regenerator.regenerate(
                    missing_prefixes,
                    K=K,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    batch_size=batch_size,
                )
                cache.put_many(
                    [
                        (prefixes[i], truncate_ratio, K, conts)
                        for i, conts in zip(missing_idx, new_continuations)
                    ],
                )
                for idx, conts in zip(missing_idx, new_continuations):
                    cached[idx] = conts
            finally:
                regenerator.close()
                del regenerator

        # Score.
        from nltk.stem.porter import PorterStemmer
        from spacy.lang.en import STOP_WORDS

        stemmer = PorterStemmer()
        stopwords = frozenset(STOP_WORDS)

        scores: list[float] = []
        for remainder, continuations in tqdm(
            list(zip(remainders, cached)),
            desc="DNA-GPT BScore",
        ):
            assert continuations is not None
            if not remainder:
                scores.append(0.0)
                continue
            rem_tokens = _tokenize(remainder, stemmer, stopwords)
            cont_tokens = [
                _tokenize(c, stemmer, stopwords) for c in continuations
            ]
            scores.append(
                _bscore(rem_tokens, cont_tokens, n_min=n_min, n_max=n_max),
            )

        return [float(s) for s in scores]

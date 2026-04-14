"""Regenerators: models that produce K continuations of a text prefix.

Used by :mod:`opendetect.detectors.dna_gpt` to implement the paper's
"truncate-then-regenerate" black-box detection strategy.

Two backends are provided:

* :class:`HFChatRegenerator` — any instruction-tuned HF causal LM.
  Uses vLLM when available for fast K-sample generation; falls back to
  ``transformers.generate(num_return_sequences=K)`` otherwise.  Default
  ``Qwen/Qwen2.5-7B-Instruct``.
* :class:`OpenAIRegenerator` — OpenAI's completions API.  Default
  ``gpt-3.5-turbo-instruct`` (the live successor to the paper's
  deprecated ``text-davinci-003``).
"""

from typing import Protocol


class Regenerator(Protocol):
    """Contract every regenerator must satisfy."""

    id: str  # Stable identifier used as a cache shard key.

    def regenerate(
        self,
        prefixes: list[str],
        K: int,
        max_new_tokens: int,
        temperature: float,
        batch_size: int = 8,
    ) -> list[list[str]]:
        """Return ``K`` continuations for each prefix.

        Parameters
        ----------
        prefixes:
            Text prefixes to continue.  Each prefix is fed to the
            regenerator as-is.
        K:
            Number of continuations to sample per prefix.
        max_new_tokens:
            Maximum number of tokens to generate per continuation.
        temperature:
            Sampling temperature.
        batch_size:
            Backend-specific batching hint.

        Returns
        -------
        list[list[str]]
            A list of length ``len(prefixes)``; each element is a list
            of ``K`` continuation strings.
        """
        ...


def load_regenerator(
    backend: str,
    model: str | None = None,
) -> Regenerator:
    """Instantiate a regenerator by backend name.

    Parameters
    ----------
    backend:
        ``"hf"`` for a local HuggingFace chat model, ``"openai"`` for
        the OpenAI API.
    model:
        Model identifier.  If ``None``, the backend default is used.

    Returns
    -------
    Regenerator
        A ready-to-use regenerator.
    """
    if backend == "hf":
        from opendetect.regenerators.hf_regen import HFChatRegenerator

        return (
            HFChatRegenerator(model_id=model) if model else HFChatRegenerator()
        )
    if backend == "openai":
        from opendetect.regenerators.openai_regen import OpenAIRegenerator

        return (
            OpenAIRegenerator(model_id=model) if model else OpenAIRegenerator()
        )
    raise ValueError(
        f"Unknown regenerator backend: {backend!r}. Expected 'hf' or 'openai'.",
    )


__all__ = ["Regenerator", "load_regenerator"]

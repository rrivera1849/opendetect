"""Revisers: models that rewrite text given the paper's prompt.

Two backends are provided:

* :class:`HFChatReviser` — any instruction-tuned HF chat model.  Default
  ``Qwen/Qwen2.5-7B-Instruct``.  Use this for reproducible, offline
  runs.
* :class:`OpenAIReviser` — OpenAI's chat API.  Default
  ``gpt-4o-mini``.  Use this to reproduce the paper's numbers at lower
  cost; the paper's original ``gpt-3.5-turbo-0301`` is deprecated.

Both share the prompt ``"Revise the following text: " + x`` from
Zhu et al. (2023).
"""

from typing import Protocol

REVISE_PROMPT_PREFIX = "Revise the following text: "


class Reviser(Protocol):
    """Contract every reviser must satisfy."""

    id: str  # Stable identifier used as a cache shard key.

    def revise(
        self,
        texts: list[str],
        batch_size: int = 8,
    ) -> list[str]:
        """Return the reviser's output for each input text."""
        ...

    def close(self) -> None:
        """Release any heavy resources (GPU memory, engine workers).

        Safe to call multiple times; subsequent calls are no-ops.
        """
        ...


def load_reviser(
    backend: str,
    model: str | None = None,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
) -> Reviser:
    """Factory: instantiate a reviser by backend name.

    Parameters
    ----------
    backend:
        ``"hf"`` for a local HuggingFace chat model, ``"vllm"`` for the
        same model served via vLLM, ``"openai"`` for the OpenAI API.
    model:
        Model identifier.  If ``None``, the backend's default is used.
    max_model_len:
        vLLM-only: cap on prompt + generated tokens.
    gpu_memory_utilization:
        vLLM-only: fraction of GPU memory reserved for the model.

    Returns
    -------
    Reviser
        A reviser ready to :meth:`revise` batches of texts.
    """
    if backend == "hf":
        from opendetect.revisers.hf_reviser import HFChatReviser

        return HFChatReviser(model_id=model) if model else HFChatReviser()
    if backend == "vllm":
        from opendetect.revisers.vllm_reviser import (
            DEFAULT_GPU_MEMORY_UTILIZATION,
            DEFAULT_MAX_MODEL_LEN,
            DEFAULT_VLLM_REVISER,
            VLLMReviser,
        )

        return VLLMReviser(
            model_id=model or DEFAULT_VLLM_REVISER,
            max_model_len=(
                max_model_len
                if max_model_len is not None
                else DEFAULT_MAX_MODEL_LEN
            ),
            gpu_memory_utilization=(
                gpu_memory_utilization
                if gpu_memory_utilization is not None
                else DEFAULT_GPU_MEMORY_UTILIZATION
            ),
        )
    if backend == "openai":
        from opendetect.revisers.openai_reviser import OpenAIReviser

        return OpenAIReviser(model_id=model) if model else OpenAIReviser()
    raise ValueError(
        f"Unknown reviser backend: {backend!r}. "
        "Expected 'hf', 'vllm', or 'openai'.",
    )


__all__ = ["Reviser", "REVISE_PROMPT_PREFIX", "load_reviser"]

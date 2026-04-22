"""Shared helper for releasing a vLLM ``LLM`` engine's GPU memory.

vLLM holds weights + KV cache in its own allocator; a plain ``del`` does
not reliably return that memory.  :func:`shutdown_vllm_engine` walks the
documented teardown path (drop executor, destroy model-parallel + the
distributed environment, GC, empty CUDA cache) so the next detector can
load its own model into the same device.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

logger = logging.getLogger(__name__)


def shutdown_vllm_engine(
    owner: Any,
    attr: str,
) -> None:
    """Tear down a vLLM ``LLM`` stored on ``owner.<attr>``.

    Parameters
    ----------
    owner:
        The object holding the engine reference (e.g. a
        ``VLLMRegenerator`` or ``VLLMReviser`` instance).
    attr:
        Name of the attribute holding the ``LLM`` instance.  After
        this call it is set to ``None`` so the function is idempotent.
    """
    llm = getattr(owner, attr, None)
    if llm is None:
        return

    engine = getattr(llm, "llm_engine", None)
    if engine is not None:
        executor = getattr(engine, "model_executor", None)
        if executor is not None and hasattr(executor, "shutdown"):
            try:
                executor.shutdown()
            except Exception:
                logger.exception(
                    "vLLM model_executor.shutdown raised; continuing.",
                )

    setattr(owner, attr, None)

    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        logger.exception("vLLM distributed teardown raised; continuing.")

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

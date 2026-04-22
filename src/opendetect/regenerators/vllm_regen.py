"""vLLM-backed regenerator for DNA-GPT.

Uses vLLM's continuous batching + paged attention to produce K
continuations per prefix much faster than ``transformers.generate``.
"""

from __future__ import annotations

import logging

from transformers import AutoTokenizer

from opendetect._vllm_shutdown import shutdown_vllm_engine

logger = logging.getLogger(__name__)

DEFAULT_VLLM_REGENERATOR = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MAX_MODEL_LEN = 4096
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9


class VLLMRegenerator:
    """Regenerate text continuations with vLLM."""

    def __init__(
        self,
        model_id: str = DEFAULT_VLLM_REGENERATOR,
        max_model_len: int = DEFAULT_MAX_MODEL_LEN,
        gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
        dtype: str = "bfloat16",
        trust_remote_code: bool = False,
    ) -> None:
        """Load the model under vLLM.

        Parameters
        ----------
        model_id:
            HuggingFace model identifier.  Must support a chat template.
        max_model_len:
            vLLM ``max_model_len``.  Caps prompt + generated tokens;
            lower values reduce KV-cache memory.
        gpu_memory_utilization:
            vLLM ``gpu_memory_utilization`` in ``(0, 1]``.  Fraction of
            GPU memory reserved for weights + KV cache.
        dtype:
            vLLM compute dtype.  ``"bfloat16"`` matches the HF backend.
        trust_remote_code:
            Forward to vLLM for custom model implementations.
        """
        from vllm import LLM

        self.id = model_id
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(
            "VLLMRegenerator: loading %s (max_model_len=%d, "
            "gpu_memory_utilization=%.2f)",
            model_id, max_model_len, gpu_memory_utilization,
        )
        self._llm = LLM(
            model=model_id,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
        )

    def _build_prompt(self, prefix: str) -> str:
        """Apply the chat template to a prefix."""
        messages = [{"role": "user", "content": prefix}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def regenerate(
        self,
        prefixes: list[str],
        K: int,
        max_new_tokens: int = 300,
        temperature: float = 0.7,
        batch_size: int = 8,
    ) -> list[list[str]]:
        """Return K continuations per prefix.

        ``batch_size`` is unused — vLLM's scheduler batches internally.
        """
        from vllm import SamplingParams

        prompts = [self._build_prompt(p) for p in prefixes]
        sampling = SamplingParams(
            n=K,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        outputs = self._llm.generate(prompts, sampling)
        return [
            [o.text.strip() for o in req_out.outputs]
            for req_out in outputs
        ]

    def close(self) -> None:
        """Shut down the vLLM engine and return GPU memory.

        vLLM grabs ``gpu_memory_utilization`` of the device at init
        and does not release it on a plain ``del``.
        """
        shutdown_vllm_engine(self, "_llm")

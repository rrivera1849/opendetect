"""vLLM-backed reviser for Revise-Detect.

Uses vLLM's continuous batching for fast greedy revision of many texts.
Greedy decoding (``temperature=0``) matches the HF backend's
``do_sample=False`` for reproducibility.
"""

from __future__ import annotations

import logging

from transformers import AutoTokenizer

from opendetect._vllm_shutdown import shutdown_vllm_engine
from opendetect.revisers import REVISE_PROMPT_PREFIX

logger = logging.getLogger(__name__)

DEFAULT_VLLM_REVISER = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MAX_MODEL_LEN = 4096
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
DEFAULT_MAX_NEW_TOKENS_CEILING = 1024


class VLLMReviser:
    """Revise texts with a local vLLM model."""

    def __init__(
        self,
        model_id: str = DEFAULT_VLLM_REVISER,
        max_model_len: int = DEFAULT_MAX_MODEL_LEN,
        gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
        max_new_tokens_ceiling: int = DEFAULT_MAX_NEW_TOKENS_CEILING,
        dtype: str = "bfloat16",
        trust_remote_code: bool = False,
    ) -> None:
        """Load the model under vLLM.

        Parameters
        ----------
        model_id:
            HuggingFace model identifier.  Must support a chat template.
        max_model_len:
            vLLM ``max_model_len`` cap on prompt + generated tokens.
        gpu_memory_utilization:
            vLLM ``gpu_memory_utilization`` in ``(0, 1]``.
        max_new_tokens_ceiling:
            Upper bound on tokens generated per sample.  The effective
            budget is ``min(2 * prompt_tokens, ceiling)``.
        dtype:
            vLLM compute dtype.
        trust_remote_code:
            Forward to vLLM for custom model implementations.
        """
        from vllm import LLM

        self.id = model_id
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_new_tokens_ceiling = max_new_tokens_ceiling

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(
            "VLLMReviser: loading %s (max_model_len=%d, "
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

    def _build_prompt(self, text: str) -> str:
        """Apply the chat template around the paper's prompt."""
        messages = [
            {"role": "user", "content": REVISE_PROMPT_PREFIX + text},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def revise(
        self,
        texts: list[str],
        batch_size: int = 8,
    ) -> list[str]:
        """Return one revised text per input.

        ``batch_size`` is unused — vLLM's scheduler batches internally.
        """
        from vllm import SamplingParams

        prompts = [self._build_prompt(t) for t in texts]
        prompt_token_ids = [
            self.tokenizer(p, add_special_tokens=False)["input_ids"]
            for p in prompts
        ]

        # Per-prompt max_tokens = min(2 * prompt_len, ceiling), further
        # capped by (max_model_len - prompt_len) to avoid vLLM errors.
        sampling_params: list[SamplingParams] = []
        for ids in prompt_token_ids:
            prompt_len = len(ids)
            budget = min(2 * prompt_len, self.max_new_tokens_ceiling)
            headroom = max(1, self.max_model_len - prompt_len)
            budget = max(1, min(budget, headroom))
            sampling_params.append(
                SamplingParams(
                    n=1,
                    temperature=0.0,
                    max_tokens=budget,
                ),
            )

        outputs = self._llm.generate(prompts, sampling_params)

        revisions: list[str] = []
        for req_out, original in zip(outputs, texts):
            raw = req_out.outputs[0].text
            revised = raw.strip()
            if not revised:
                logger.warning(
                    "Reviser returned empty output; "
                    "falling back to original text.",
                )
                revised = original
            revisions.append(revised)
        return revisions

    def close(self) -> None:
        """Shut down the vLLM engine and return GPU memory.

        vLLM grabs ``gpu_memory_utilization`` of the device at init
        and does not release it on a plain ``del``.
        """
        shutdown_vllm_engine(self, "_llm")

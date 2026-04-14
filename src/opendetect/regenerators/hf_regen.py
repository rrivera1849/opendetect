"""Local HuggingFace chat-model regenerator.

Produces K continuations of a text prefix via sampled generation.
Uses vLLM when available for fast K-in-parallel sampling; falls back to
``transformers.generate(num_return_sequences=K)`` otherwise.
"""

from __future__ import annotations

import logging

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from opendetect.config import get_device

logger = logging.getLogger(__name__)

DEFAULT_HF_REGENERATOR = "Qwen/Qwen2.5-7B-Instruct"


def _try_import_vllm():
    """Return the ``vllm`` module if importable, else ``None``."""
    try:
        import vllm  # type: ignore[import-not-found]

        return vllm
    except ImportError:
        return None


class HFChatRegenerator:
    """Regenerate text continuations with a local HF chat model.

    The DNA-GPT paper's premise is that the regenerator *is* the target
    model.  In practice OpenDetect datasets mix sources, so we default
    to a generic instruction-tuned 7B; this gives a proxy signal rather
    than the paper's strictly-faithful one.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_HF_REGENERATOR,
        device: torch.device | None = None,
        prefer_vllm: bool = True,
    ) -> None:
        """Load the chat model + tokenizer.

        Parameters
        ----------
        model_id:
            HuggingFace model identifier.  Must support a chat template.
        device:
            Torch device (only used by the transformers fallback).
        prefer_vllm:
            If ``True`` and vLLM is importable, use vLLM for generation.
        """
        self.id = model_id
        self.device = device if device is not None else get_device()

        self._vllm = _try_import_vllm() if prefer_vllm else None
        self._llm = None
        self._hf_model = None

        # Tokenizer is always needed (for the chat template).
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if self._vllm is not None:
            logger.info("HFChatRegenerator: using vLLM backend for %s", model_id)
            self._llm = self._vllm.LLM(
                model=model_id,
                dtype="bfloat16",
                trust_remote_code=False,
            )
        else:
            logger.info(
                "HFChatRegenerator: vLLM not available; "
                "falling back to transformers.generate for %s",
                model_id,
            )
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
            ).eval().to(self.device)

    def _build_prompt(self, prefix: str) -> str:
        """Apply the chat template.  The user message *is* the prefix
        to continue (no task preamble); DNA-GPT's signal relies on the
        model treating the prefix as its own past output.
        """
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
        """Return K continuations per prefix."""
        prompts = [self._build_prompt(p) for p in prefixes]

        if self._llm is not None:
            return self._regen_vllm(prompts, K, max_new_tokens, temperature)
        return self._regen_hf(
            prompts, K, max_new_tokens, temperature, batch_size,
        )

    def _regen_vllm(
        self,
        prompts: list[str],
        K: int,
        max_new_tokens: int,
        temperature: float,
    ) -> list[list[str]]:
        """Generate via vLLM with ``n=K`` per prompt."""
        from vllm import SamplingParams  # type: ignore[import-not-found]

        sampling = SamplingParams(
            n=K,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        outputs = self._llm.generate(prompts, sampling)
        # Preserve prompt order via the ``outputs`` list index.
        return [
            [o.text.strip() for o in req_out.outputs]
            for req_out in outputs
        ]

    @torch.inference_mode()
    def _regen_hf(
        self,
        prompts: list[str],
        K: int,
        max_new_tokens: int,
        temperature: float,
        batch_size: int,
    ) -> list[list[str]]:
        """Fallback generation via ``transformers.generate``."""
        assert self._hf_model is not None
        all_continuations: list[list[str]] = []
        for start in tqdm(
            range(0, len(prompts), batch_size),
            desc=f"Regenerating ({self.id})",
        ):
            batch = prompts[start : start + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(self.device)

            gen = self._hf_model.generate(
                **enc,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                num_return_sequences=K,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            # ``gen`` has shape (batch * K, seq_len); strip each prompt.
            prompt_len = enc["input_ids"].shape[1]
            new_tokens = gen[:, prompt_len:]
            decoded = self.tokenizer.batch_decode(
                new_tokens,
                skip_special_tokens=True,
            )
            # Regroup into K per prompt.
            for i in range(len(batch)):
                group = [
                    decoded[i * K + j].strip() for j in range(K)
                ]
                all_continuations.append(group)
        return all_continuations

"""Local HuggingFace chat-model regenerator (transformers backend).

Produces K continuations of a text prefix via sampled generation.
For faster batched generation, use the ``vllm`` backend instead.
"""

from __future__ import annotations

import logging

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from opendetect.config import get_device

logger = logging.getLogger(__name__)

DEFAULT_HF_REGENERATOR = "Qwen/Qwen2.5-7B-Instruct"


class HFChatRegenerator:
    """Regenerate text continuations with ``transformers.generate``.

    The DNA-GPT paper's premise is that the regenerator *is* the target
    model.  In practice OpenDetect datasets mix sources, so we default
    to a generic instruction-tuned 7B; this gives a proxy signal rather
    than the paper's strictly-faithful one.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_HF_REGENERATOR,
        device: torch.device | None = None,
    ) -> None:
        """Load the chat model + tokenizer.

        Parameters
        ----------
        model_id:
            HuggingFace model identifier.  Must support a chat template.
        device:
            Torch device.  Defaults to :func:`opendetect.config.get_device`.
        """
        self.id = model_id
        self.device = device if device is not None else get_device()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

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

    @torch.inference_mode()
    def regenerate(
        self,
        prefixes: list[str],
        K: int,
        max_new_tokens: int = 300,
        temperature: float = 0.7,
        batch_size: int = 8,
    ) -> list[list[str]]:
        """Return K continuations per prefix via ``transformers.generate``."""
        prompts = [self._build_prompt(p) for p in prefixes]
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

    def close(self) -> None:
        """No-op: the HF model is dropped when this object is released."""

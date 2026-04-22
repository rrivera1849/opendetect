"""Local HuggingFace chat-model reviser.

Uses any chat-template-equipped causal LM to rewrite text given the
paper's prompt.  Greedy decoding (``do_sample=False``) for
reproducibility.
"""

from __future__ import annotations

import logging

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from opendetect.config import get_device
from opendetect.revisers import REVISE_PROMPT_PREFIX

logger = logging.getLogger(__name__)

DEFAULT_HF_REVISER = "Qwen/Qwen2.5-7B-Instruct"


class HFChatReviser:
    """Revise texts with a local instruction-tuned HF chat model."""

    def __init__(
        self,
        model_id: str = DEFAULT_HF_REVISER,
        device: torch.device | None = None,
        max_new_tokens_ceiling: int = 1024,
    ) -> None:
        """Load the chat model + tokenizer.

        Parameters
        ----------
        model_id:
            HuggingFace model identifier.  Must support a chat template.
        device:
            Torch device.  Defaults to :func:`opendetect.config.get_device`.
        max_new_tokens_ceiling:
            Upper bound on generated tokens per sample; the effective
            budget is ``min(2 * input_len, ceiling)``.
        """
        self.id = model_id
        self.device = device if device is not None else get_device()
        self.max_new_tokens_ceiling = max_new_tokens_ceiling

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        ).eval().to(self.device)

    def _build_prompt(self, text: str) -> str:
        """Apply the chat template around the paper's prompt."""
        messages = [{"role": "user", "content": REVISE_PROMPT_PREFIX + text}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def revise(
        self,
        texts: list[str],
        batch_size: int = 8,
    ) -> list[str]:
        """Return one revised text per input.

        Greedy decoding is used so calls are deterministic per
        (model, text) pair.
        """
        outputs: list[str] = []
        for start in tqdm(
            range(0, len(texts), batch_size),
            desc=f"Revising ({self.id})",
        ):
            batch = texts[start : start + batch_size]
            prompts = [self._build_prompt(t) for t in batch]
            enc = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(self.device)

            input_len = enc["input_ids"].shape[1]
            max_new_tokens = min(
                2 * input_len,
                self.max_new_tokens_ceiling,
            )

            gen = self.model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            # Strip the prompt off each sample.
            new_tokens = gen[:, enc["input_ids"].shape[1] :]
            decoded = self.tokenizer.batch_decode(
                new_tokens,
                skip_special_tokens=True,
            )
            for raw, original in zip(decoded, batch):
                revised = raw.strip()
                if not revised:
                    logger.warning(
                        "Reviser returned empty output; "
                        "falling back to original text.",
                    )
                    revised = original
                outputs.append(revised)
        return outputs

    def close(self) -> None:
        """No-op: the HF model is dropped when this object is released."""

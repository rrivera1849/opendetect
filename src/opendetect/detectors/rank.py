"""Rank and LogRank detectors — token-rank based methods."""

from __future__ import annotations

import logging

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from opendetect.config import get_device
from opendetect.detectors.base import BaseDetector
from opendetect.registry import register_detector

logger = logging.getLogger(__name__)


def _get_rank(
    text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    log: bool = False,
) -> float:
    """Compute the average (log-)rank of label tokens.

    Adapted from: https://github.com/eric-mitchell/detect-gpt
    """
    tokenized = tokenizer(
        text, max_length=1024, truncation=True, return_tensors="pt"
    ).to(model.device)
    logits = model(**tokenized).logits[:, :-1]
    labels = tokenized["input_ids"][:, 1:]

    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3
    ranks = matches[:, -1].float() + 1  # 1-indexed

    if log:
        ranks = torch.log(ranks)

    return ranks.mean().item()


@register_detector("rank")
class RankDetector(BaseDetector):
    """Average token-rank detector (GPT-2 XL)."""

    DEFAULT_MODEL = "gpt2-xl"

    @torch.no_grad()
    def score(self, texts: list[str], **kwargs) -> list[float]:
        model_name = kwargs.get("model_name", self.DEFAULT_MODEL)
        device = get_device()

        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()

        return [
            _get_rank(t, model, tokenizer, log=False)
            for t in tqdm(texts, desc="Rank")
        ]


@register_detector("log-rank")
class LogRankDetector(BaseDetector):
    """Average log-token-rank detector (GPT-2 XL)."""

    DEFAULT_MODEL = "gpt2-xl"

    @torch.no_grad()
    def score(self, texts: list[str], **kwargs) -> list[float]:
        model_name = kwargs.get("model_name", self.DEFAULT_MODEL)
        device = get_device()

        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()

        return [
            _get_rank(t, model, tokenizer, log=True)
            for t in tqdm(texts, desc="LogRank")
        ]

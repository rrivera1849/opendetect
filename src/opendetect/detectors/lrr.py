"""LRR (Log-Likelihood Ratio) detector."""

from __future__ import annotations

import logging

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from opendetect.config import get_device
from opendetect.detectors.base import BaseDetector
from opendetect.registry import register_detector

logger = logging.getLogger(__name__)


def _get_ll(text: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> float:
    """Compute log-likelihood of a text."""
    tokenized = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=1024,
        return_tensors="pt",
    ).to(model.device)
    labels = tokenized.input_ids
    return -model(**tokenized, labels=labels).loss.item()


def _get_rank(text: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> float:
    """Compute average log-rank of a text."""
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=1024,
        return_tensors="pt",
    ).to(model.device)
    logits = model(**tokenized).logits[:, :-1]
    labels = tokenized.input_ids[:, 1:]

    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3
    ranks = matches[:, -1].float() + 1
    ranks = torch.log(ranks)

    return ranks.float().mean().item()


@register_detector("lrr")
class LRRDetector(BaseDetector):
    """LRR (Log-Likelihood Ratio) detector.

    Computes the ratio of log-likelihood to log-rank (ll / logrank).
    Adapted from MGTBench.
    """

    DEFAULT_MODEL = "gpt2-xl"

    @torch.no_grad()
    def score(self, texts: list[str], **kwargs) -> list[float]:
        model_name = kwargs.get("model_name", self.DEFAULT_MODEL)
        device = get_device()

        logger.info("Loading model %s for LRR...", model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Ensure pad token is set for LL computation
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model.eval()

        scores: list[float] = []
        for text in tqdm(texts, desc="LRR"):
            ll = _get_ll(text, model, tokenizer)
            logrank = _get_rank(text, model, tokenizer)
            
            if logrank == 0:
                logger.warning("Log-rank is 0, setting LRR score to NaN")
                scores.append(float("nan"))
            else:
                scores.append(ll / logrank)

        return scores

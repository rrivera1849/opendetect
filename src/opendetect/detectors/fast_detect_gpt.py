"""FastDetectGPT detector — sampling discrepancy analytic method."""

from __future__ import annotations

import logging

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from opendetect.config import get_device
from opendetect.detectors.base import BaseDetector
from opendetect.registry import register_detector

logger = logging.getLogger(__name__)


def get_sampling_discrepancy_analytic(
    logits_ref: torch.Tensor,
    logits_score: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Compute the sampling discrepancy analytic statistic.

    Adapted from the FastDetectGPT codebase.
    """
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1

    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    discrepancy = discrepancy.mean()
    return discrepancy.item()


@register_detector("fast-detect-gpt")
class FastDetectGPTDetector(BaseDetector):
    """Machine-text detector based on the FastDetectGPT method.

    Uses sampling discrepancy between the model's own predictions
    and its reference logits.
    """

    DEFAULT_MODEL = "EleutherAI/gpt-neo-2.7B"

    @torch.no_grad()
    def score(self, texts: list[str], **kwargs) -> list[float]:
        model_name = kwargs.get("model_name", self.DEFAULT_MODEL)
        device = get_device()

        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()

        scores: list[float] = []
        for sample in tqdm(texts, desc="FastDetectGPT"):
            tok = tokenizer(
                sample,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            base_logits = model(**tok).logits[:, :-1]
            reference_logits = base_logits
            labels = tok["input_ids"][:, 1:]
            discrepancy = get_sampling_discrepancy_analytic(
                reference_logits, base_logits, labels
            )
            scores.append(discrepancy)

        return scores

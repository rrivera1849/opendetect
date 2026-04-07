"""Supervised detector — generic wrapper for HF sequence-classification models."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

from opendetect.config import get_device
from opendetect.detectors.base import BaseDetector
from opendetect.detectors.mage_utils import preprocess_mage
from opendetect.registry import register_detector

logger = logging.getLogger(__name__)

# @register_detector("lusr")
class LUSRDetector(BaseDetector):
    """LUSR detector (rrivera1849/LUSR)."""

    @torch.no_grad()
    def score(self, texts: list[str], **kwargs) -> list[float]:
        batch_size = kwargs.get("batch_size", 128)
        device = get_device()

        texts = [preprocess_mage(t) for t in texts]

        detector = AutoModelForSequenceClassification.from_pretrained(
            "./LUSR_MLP_new",
            trust_remote_code=True,
        ).to(device)
        detector.eval()

        probabilities: list[float] = []
        pbar = tqdm(
            total=(len(texts) + batch_size - 1) // batch_size,
            desc=self.name,
        )
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            output_probs = (
                F.log_softmax(detector(texts=batch).logits, dim=-1)[:, 0]
                .exp()
                .tolist()
            )
            probabilities.extend(output_probs)
            pbar.update(1)

        pbar.close()
        return probabilities

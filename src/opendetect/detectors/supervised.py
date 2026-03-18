"""Supervised detector — generic wrapper for HF sequence-classification models."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from opendetect.config import get_device
from opendetect.detectors.base import BaseDetector
from opendetect.registry import register_detector
from opendetect.detectors.mage_utils import preprocess_mage

logger = logging.getLogger(__name__)


class _SupervisedDetector(BaseDetector):
    """Shared logic for any HuggingFace sequence classifier detector."""

    MODEL_NAME: str  # subclasses must set this
    USE_RAW_LOGITS: bool = False
    MAX_LENGTH: int = 512

    def preprocess(self, texts: list[str]) -> list[str]:
        """Optional dataset preprocessing hook."""
        return texts

    @torch.no_grad()
    def score(self, texts: list[str], **kwargs) -> list[float]:
        batch_size = kwargs.get("batch_size", 128)
        device = get_device()

        detector = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_NAME
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        detector.eval()

        texts = self.preprocess(texts)

        probabilities: list[float] = []
        pbar = tqdm(
            total=(len(texts) + batch_size - 1) // batch_size,
            desc=self.name,
        )
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = tokenizer(
                batch,
                max_length=self.MAX_LENGTH,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            if self.USE_RAW_LOGITS:
                output_probs = detector(**encoded).logits[:, 0].tolist()
            else:
                output_probs = (
                    F.log_softmax(detector(**encoded).logits, dim=-1)[:, 0]
                    .exp()
                    .tolist()
                )
            probabilities.extend(output_probs)
            pbar.update(1)

        pbar.close()
        return probabilities


@register_detector("radar")
class RADARDetector(_SupervisedDetector):
    """RADAR detector (TrustSafeAI/RADAR-Vicuna-7B)."""

    MODEL_NAME = "TrustSafeAI/RADAR-Vicuna-7B"


@register_detector("remodetect")
class ReMoDetectDetector(_SupervisedDetector):
    """ReMoDetect detector (hyunseoki/ReMoDetect-deberta)."""

    MODEL_NAME = "hyunseoki/ReMoDetect-deberta"
    USE_RAW_LOGITS = True


@register_detector("mage")
class MAGEDetector(_SupervisedDetector):
    """MAGE detector. Longformer-based sequence classifier.
    Higher scores indicate higher likelihood of being machine-generated.
    """

    MODEL_NAME = "yaful/MAGE"
    USE_RAW_LOGITS = True
    MAX_LENGTH = 4096

    def preprocess(self, texts: list[str]) -> list[str]:
        logger.info("Preprocessing texts for MAGE...")
        return [preprocess_mage(t) for t in tqdm(texts, desc="MAGE preprocess")]

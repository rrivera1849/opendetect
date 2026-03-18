"""StyleDetect family — style-embedding similarity detectors."""

from __future__ import annotations

import logging
from typing import Union

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from opendetect.config import get_device
from opendetect.detectors.base import BaseDetector
from opendetect.registry import register_detector

logger = logging.getLogger(__name__)


@torch.inference_mode()
def get_luar_embeddings(
    text: Union[list[str], list[list[str]]],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    batch_size: int = 32,
    single: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute LUAR embeddings for a list of texts.

    Adapted from ``legacy/create_preference_data.py``.

    Parameters
    ----------
    text:
        Either a flat list of strings, or a list of lists of strings
        (one list per author).
    model:
        Pre-loaded LUAR model.
    tokenizer:
        Corresponding tokenizer.
    batch_size:
        Batch size for non-single mode.
    single:
        If *True*, treat the entire list as a single author's texts
        and produce one embedding.
    normalize:
        L2-normalize the output embeddings.
    """
    if isinstance(text[0], list):
        outputs = torch.cat(
            [get_luar_embeddings(t, model, tokenizer, single=True) for t in text],
            dim=0,
        )
        return outputs

    device = model.device

    inputs = tokenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    if single:
        inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)
        inputs["attention_mask"] = inputs["attention_mask"].unsqueeze(0)
        inputs.to(device)
        outputs = model(**inputs)
    else:
        outputs = []
        for batch_idx in tqdm(range(0, len(text), batch_size), desc="LUAR embeddings"):
            batch_inputs = {
                k: v[batch_idx : batch_idx + batch_size].unsqueeze(1).to(device)
                for k, v in inputs.items()
            }
            outputs.append(model(**batch_inputs))
        outputs = torch.cat(outputs, dim=0)

    if normalize:
        outputs = F.normalize(outputs, dim=-1, p=2)
    return outputs


class _StyleDetectorBase(BaseDetector):
    """Base for style-embedding similarity detectors.

    All variants compute cosine similarity between each text embedding and
    a mean embedding of background (few-shot) machine texts.
    """

    MODEL_NAME: str  # subclasses must set
    USE_LUAR: bool = False

    @property
    def requires_fewshot(self) -> bool:
        return True

    @torch.no_grad()
    def score(self, texts: list[str], **kwargs) -> list[float]:
        background_texts: list[str] = kwargs.get("background_texts", [])
        if not background_texts:
            logger.warning(
                "%s requires few-shot background texts (--num-fewshot > 0). "
                "Returning empty scores.",
                self.name,
            )
            return [float("nan")] * len(texts)

        device = get_device()

        if self.USE_LUAR:
            return self._score_luar(texts, background_texts, device)
        else:
            return self._score_sentence_transformer(texts, background_texts, device)

    def _score_luar(
        self,
        texts: list[str],
        background: list[str],
        device: torch.device,
    ) -> list[float]:
        model = (
            AutoModel.from_pretrained(self.MODEL_NAME, trust_remote_code=True)
            .eval()
            .to(device)
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME, trust_remote_code=True
        )

        background_emb = get_luar_embeddings(
            background, model, tokenizer, single=True
        )
        emb = get_luar_embeddings(texts, model, tokenizer)

        scores = F.cosine_similarity(
            background_emb.repeat(emb.size(0), 1), emb
        )
        return scores.cpu().tolist()

    def _score_sentence_transformer(
        self,
        texts: list[str],
        background: list[str],
        device: torch.device,
    ) -> list[float]:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.MODEL_NAME).eval().to(device)

        background_emb = model.encode(
            background,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_tensor=True,
        )
        background_emb = background_emb.mean(dim=0, keepdim=True)
        background_emb = F.normalize(background_emb, p=2, dim=-1)

        emb = model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_tensor=True,
        )

        scores = F.cosine_similarity(
            background_emb.repeat(emb.size(0), 1), emb
        )
        return scores.cpu().tolist()


# @register_detector("styledetect")
class StyleDetectLUAR(_StyleDetectorBase):
    """StyleDetect using LUAR-MUD embeddings."""

    MODEL_NAME = "rrivera1849/LUAR-MUD"
    USE_LUAR = True


# @register_detector("styledetect-cisr")
class StyleDetectCISR(_StyleDetectorBase):
    """StyleDetect using CISR style embeddings."""

    MODEL_NAME = "AnnaWegmann/Style-Embedding"


# @register_detector("styledetect-sd")
class StyleDetectSD(_StyleDetectorBase):
    """StyleDetect using StyleDistance embeddings."""

    MODEL_NAME = "StyleDistance/styledistance"


# @register_detector("semdetect")
class SemDetect(_StyleDetectorBase):
    """Semantic similarity detector using all-mpnet-base-v2."""

    MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

"""BARTScore (Yuan et al., 2021) similarity metric.

Adapted from the reference implementation shipped with the Revise-Detect
paper (``baselines_to_implement/Revise-Detect/bart_score.py``), with
small changes: device selection via :func:`get_device`, ``tqdm``
progress bar, no hard-coded CUDA device index.

``BARTScorer.score(srcs, tgts)`` returns per-sample
``-Σ log p(tgt_t | src) / |tgt|``.  Higher (less negative) ⇒ higher
similarity between the two texts.
"""

from __future__ import annotations

import logging
from typing import List

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer

from opendetect.config import get_device

logger = logging.getLogger(__name__)

DEFAULT_BART_CHECKPOINT = "facebook/bart-large-cnn"


class BARTScorer:
    """Compute BARTScore between paired source / target texts."""

    def __init__(
        self,
        checkpoint: str = DEFAULT_BART_CHECKPOINT,
        max_length: int = 1024,
        device: torch.device | None = None,
    ) -> None:
        """Load the BART model for scoring.

        Parameters
        ----------
        checkpoint:
            HuggingFace model identifier.  Default
            ``facebook/bart-large-cnn`` matches the paper.
        max_length:
            Max tokenizer length for both source and target.
        device:
            Torch device.  Defaults to :func:`opendetect.config.get_device`.
        """
        self.device = device if device is not None else get_device()
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(self.device)

        self.loss_fct = nn.NLLLoss(
            reduction="none",
            ignore_index=self.model.config.pad_token_id,
        )
        self.lsm = nn.LogSoftmax(dim=1)

    @torch.inference_mode()
    def score(
        self,
        srcs: List[str],
        tgts: List[str],
        batch_size: int = 16,
    ) -> list[float]:
        """Score paired (src, tgt) texts.

        Parameters
        ----------
        srcs:
            Source texts (fed as encoder input).
        tgts:
            Target texts (fed as decoder labels).  Must be the same
            length as ``srcs``.
        batch_size:
            Batch size for scoring.

        Returns
        -------
        list[float]
            Per-pair scores ``-Σ log p(tgt | src) / |tgt|``.  Higher is
            more similar.
        """
        if len(srcs) != len(tgts):
            raise ValueError(
                f"srcs ({len(srcs)}) and tgts ({len(tgts)}) "
                "must have the same length.",
            )

        score_list: list[float] = []
        for i in tqdm(
            range(0, len(srcs), batch_size),
            desc="BARTScore",
        ):
            src_list = srcs[i : i + batch_size]
            tgt_list = tgts[i : i + batch_size]

            encoded_src = self.tokenizer(
                src_list,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            encoded_tgt = self.tokenizer(
                tgt_list,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            src_tokens = encoded_src["input_ids"].to(self.device)
            src_mask = encoded_src["attention_mask"].to(self.device)
            tgt_tokens = encoded_tgt["input_ids"].to(self.device)
            tgt_mask = encoded_tgt["attention_mask"]
            tgt_len = tgt_mask.sum(dim=1).to(self.device)

            output = self.model(
                input_ids=src_tokens,
                attention_mask=src_mask,
                labels=tgt_tokens,
            )
            logits = output.logits.view(-1, self.model.config.vocab_size)
            loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
            loss = loss.view(tgt_tokens.shape[0], -1)
            loss = loss.sum(dim=1) / tgt_len.clamp(min=1)
            score_list.extend((-loss).float().cpu().tolist())

        return score_list

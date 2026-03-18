"""Binoculars detector — uses observer/performer model pair."""

from __future__ import annotations

import logging

import numpy as np
import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from opendetect.detectors.base import BaseDetector
from opendetect.registry import register_detector

logger = logging.getLogger(__name__)

ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)

def _perplexity(encoding: transformers.BatchEncoding,
                logits: torch.Tensor,
                median: bool = False,
                temperature: float = 1.0):
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    if median:
        ce_nan = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels).
                  masked_fill(~shifted_attention_mask.bool(), float("nan")))
        ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        ppl = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) *
               shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
        ppl = ppl.to("cpu").float().numpy()

    return ppl


def _entropy(p_logits: torch.Tensor,
             q_logits: torch.Tensor,
             encoding: transformers.BatchEncoding,
             pad_token_id: int,
             median: bool = False,
             sample_p: bool = False,
             temperature: float = 1.0):
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]
    p_scores, q_scores = p_logits / temperature, q_logits / temperature

    p_proba = softmax_fn(p_scores).view(-1, vocab_size)

    if sample_p:
        p_proba = torch.multinomial(p_proba.view(-1, vocab_size), replacement=True, num_samples=1).view(-1)

    q_scores = q_scores.view(-1, vocab_size)

    ce = ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

    if median:
        ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
        agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        agg_ce = (((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy())

    return agg_ce


class _BinocularsCore:
    def __init__(self,
                 observer_name_or_path: str = "tiiuae/falcon-7b",
                 performer_name_or_path: str = "tiiuae/falcon-7b-instruct",
                 use_bfloat16: bool = True,
                 max_token_observed: int = 512,
                 ) -> None:
        
        self.device_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device_2 = "cuda:1" if torch.cuda.device_count() > 1 else self.device_1

        logger.info(f"Loading observer model ({observer_name_or_path}) to {self.device_1}")
        self.observer_model = AutoModelForCausalLM.from_pretrained(
            observer_name_or_path,
            device_map={"": self.device_1},
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
        )
        logger.info(f"Loading performer model ({performer_name_or_path}) to {self.device_2}")
        self.performer_model = AutoModelForCausalLM.from_pretrained(
            performer_name_or_path,
            device_map={"": self.device_2},
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
        )
        self.observer_model.eval()
        self.performer_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False
        ).to(self.observer_model.device)
        return encodings

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> tuple[torch.Tensor, torch.Tensor]:
        observer_logits = self.observer_model(**encodings.to(self.device_1)).logits
        performer_logits = self.performer_model(**encodings.to(self.device_2)).logits
        if self.device_1 != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    def compute_score(self, batch: list[str]) -> list[float]:
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl = _perplexity(encodings, performer_logits)
        x_ppl = _entropy(observer_logits.to(self.device_1), performer_logits.to(self.device_1),
                         encodings.to(self.device_1), self.tokenizer.pad_token_id)
        binoculars_scores = ppl / x_ppl
        return binoculars_scores.tolist()


@register_detector("binoculars")
class BinocularsDetector(BaseDetector):
    """Machine-text detector based on the Binoculars method.

    Uses a pair of language models (observer + performer) to compute
    a cross-perplexity ratio score.
    """

    def score(self, texts: list[str], **kwargs) -> list[float]:
        bino = _BinocularsCore()

        batch_size = kwargs.get("batch_size", 16)
        scores: list[float] = []
        pbar = tqdm(total=len(texts), desc="Binoculars")
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            scores += bino.compute_score(batch)
            pbar.update(len(batch))
        pbar.close()
        scores = [-s for s in scores]
        return scores

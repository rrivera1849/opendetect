"""NPR (Normalized Perturbation Rank) detector."""

from __future__ import annotations

import logging
import re
import random
from tqdm import tqdm

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM

from opendetect.config import get_device
from opendetect.detectors.base import BaseDetector
from opendetect.registry import register_detector

logger = logging.getLogger(__name__)

# define regex to match all <extra_id_*> tokens, where * is an integer
PATTERN = re.compile(r"<extra_id_\d+>")


def tokenize_and_mask(text: str, span_length: int, buffer_size: int, pct: float, ceil_pct: bool = False) -> str:
    tokens = text.split(" ")
    if len(tokens) > 1024:
        tokens = tokens[:1024]
    mask_string = "<<<mask>>>"

    n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, max(1, len(tokens) - span_length))
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1

    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f"<extra_id_{num_filled}>"
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    return " ".join(tokens)


def count_masks(texts: list[str]) -> list[int]:
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]


def replace_masks(texts: list[str], mask_model: AutoModelForSeq2SeqLM, mask_tokenizer: AutoTokenizer, mask_top_p: float, device: torch.device) -> list[str]:
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0] if n_expected else None

    # T5 requires generating up to the stop_id
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(device)
    
    kwargs = {
        "max_length": 150,
        "do_sample": True,
        "top_p": mask_top_p,
        "num_return_sequences": 1,
    }
    if stop_id is not None:
        kwargs["eos_token_id"] = stop_id
        
    outputs = mask_model.generate(**tokens, **kwargs)
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)


def extract_fills(texts: list[str]) -> list[list[str]]:
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]
    extracted_fills = [PATTERN.split(x)[1:-1] for x in texts]
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]
    return extracted_fills


def apply_extracted_fills(masked_texts: list[str], extracted_fills: list[list[str]]) -> list[str]:
    tokens = [x.split(" ") for x in masked_texts]
    n_expected = count_masks(masked_texts)

    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                mask_str = f"<extra_id_{fill_idx}>"
                if mask_str in text:
                    text[text.index(mask_str)] = fills[fill_idx]

    return [" ".join(x) for x in tokens]


def perturb_texts(
    texts: list[str],
    mask_model: AutoModelForSeq2SeqLM,
    mask_tokenizer: AutoTokenizer,
    device: torch.device,
    span_length: int = 2,
    buffer_size: int = 1,
    mask_top_p: float = 1.0,
    pct: float = 0.3,
    ceil_pct: bool = False,
    n_perturbations: int = 1,
    batch_size: int = 16,
) -> list[list[str]]:
    """Perturb texts by masking and filling using T5.
    Returns a list of lists: for each input text, a list of perturbed texts.
    """
    all_perturbed = []
    
    repeated_texts = [x for x in texts for _ in range(n_perturbations)]
    perturbed_repeated = []

    for i in tqdm(range(0, len(repeated_texts), batch_size), desc="Perturbing"):
        batch_texts = repeated_texts[i : i + batch_size]
        
        masked_texts = [tokenize_and_mask(x, span_length, buffer_size, pct, ceil_pct) for x in batch_texts]
        raw_fills = replace_masks(masked_texts, mask_model, mask_tokenizer, mask_top_p, device)
        extracted_fills = extract_fills(raw_fills)
        new_perturbed = apply_extracted_fills(masked_texts, extracted_fills)
        
        # Handle failed fills
        attempts = 1
        while "" in new_perturbed and attempts < 5:
            idxs = [idx for idx, x in enumerate(new_perturbed) if x == ""]
            subset_masked = [tokenize_and_mask(batch_texts[idx], span_length, buffer_size, pct, ceil_pct) for idx in idxs]
            subset_raw = replace_masks(subset_masked, mask_model, mask_tokenizer, mask_top_p, device)
            subset_fills = extract_fills(subset_raw)
            subset_perturbed = apply_extracted_fills(subset_masked, subset_fills)
            for idx, x in zip(idxs, subset_perturbed):
                new_perturbed[idx] = x
            attempts += 1
            
        perturbed_repeated.extend(new_perturbed)

    for i in range(len(texts)):
        all_perturbed.append(perturbed_repeated[i * n_perturbations : (i + 1) * n_perturbations])
        
    return all_perturbed


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
    # If the text is empty or too short, matches.shape[1] might not be 3.
    if matches.numel() == 0:
        return float("nan")

    ranks = matches[:, -1].float() + 1
    ranks = torch.log(ranks)

    return ranks.float().mean().item()


@register_detector("npr")
class NPRDetector(BaseDetector):
    """NPR (Normalized Perturbation Rank) detector.

    Computes the ratio of the average log-rank of perturbed versions of the text
    to the log-rank of the original text.
    Adapted from MGTBench.
    """

    DEFAULT_BASE_MODEL = "gpt2-xl"
    DEFAULT_MASK_MODEL = "t5-base"

    @torch.no_grad()
    def score(self, texts: list[str], **kwargs) -> list[float]:
        base_model_name = kwargs.get("model_name", self.DEFAULT_BASE_MODEL)
        mask_model_name = kwargs.get("mask_model_name", self.DEFAULT_MASK_MODEL)
        n_perturbations = kwargs.get("n_perturbations", 10)
        batch_size = kwargs.get("batch_size", 16)
        
        device = get_device()

        logger.info("Loading mask model %s for NPR perturbation...", mask_model_name)
        mask_model = AutoModelForSeq2SeqLM.from_pretrained(mask_model_name).to(device)
        mask_tokenizer = AutoTokenizer.from_pretrained(mask_model_name)
        mask_model.eval()

        logger.info("Loading base model %s for NPR ranking...", base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)
        base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if base_tokenizer.pad_token is None:
            base_tokenizer.pad_token = base_tokenizer.eos_token
        base_model.eval()

        # Perturb texts
        perturbed_texts_list = perturb_texts(
            texts=texts,
            mask_model=mask_model,
            mask_tokenizer=mask_tokenizer,
            device=device,
            span_length=kwargs.get("span_length", 2),
            buffer_size=kwargs.get("buffer_size", 1),
            mask_top_p=kwargs.get("mask_top_p", 1.0),
            pct=kwargs.get("pct", 0.3),
            n_perturbations=n_perturbations,
            batch_size=batch_size,
        )

        scores: list[float] = []
        for orig_text, p_texts in tqdm(zip(texts, perturbed_texts_list), total=len(texts), desc="NPR Scoring"):
            orig_logrank = _get_rank(orig_text, base_model, base_tokenizer)
            
            p_logranks = []
            for p_text in p_texts:
                if not p_text.strip():
                    continue
                lr = _get_rank(p_text, base_model, base_tokenizer)
                if not np.isnan(lr):
                    p_logranks.append(lr)
                    
            if not p_logranks or orig_logrank == 0 or np.isnan(orig_logrank):
                scores.append(float("nan"))
            else:
                p_logrank_mean = np.mean(p_logranks)
                scores.append(p_logrank_mean / orig_logrank)

        return scores

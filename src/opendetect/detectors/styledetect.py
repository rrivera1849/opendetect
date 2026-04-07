"""StyleDetect family — style-embedding similarity detectors.

Supports multiple random trials to account for support-set randomness,
and an optional multi-target setting with per-source-class support sets.
Scores are saved as an NPZ matrix of shape ``(num_trials, num_samples)``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from opendetect.config import get_device
from opendetect.detectors.base import BaseDetector
from opendetect.registry import register_detector

logger = logging.getLogger(__name__)

DEFAULT_NUM_TRIALS = 100


# ---------------------------------------------------------------------------
# LUAR embedding helper
# ---------------------------------------------------------------------------

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
        for batch_idx in tqdm(
            range(0, len(text), batch_size),
            desc="LUAR embeddings",
        ):
            batch_inputs = {
                k: v[batch_idx : batch_idx + batch_size].unsqueeze(1).to(device)
                for k, v in inputs.items()
            }
            outputs.append(model(**batch_inputs))
        outputs = torch.cat(outputs, dim=0)

    if normalize:
        outputs = F.normalize(outputs, dim=-1, p=2)
    return outputs


# ---------------------------------------------------------------------------
# StyleDetect base with multi-trial support
# ---------------------------------------------------------------------------

class _StyleDetectorBase(BaseDetector):
    """Base for style-embedding similarity detectors.

    All variants compute cosine similarity between each text embedding and
    a mean embedding of a randomly-sampled support set of machine texts.
    Multiple trials are run to account for support-set randomness, producing
    a score matrix of shape ``(num_trials, num_samples)``.
    """

    MODEL_NAME: str  # subclasses must set
    USE_LUAR: bool = False

    @property
    def requires_fewshot(self) -> bool:
        """StyleDetect needs few-shot examples."""
        return True

    # ------------------------------------------------------------------
    # Main scoring entry point
    # ------------------------------------------------------------------

    def score(
        self,
        texts: list[str],
        **kwargs,
    ) -> list[float]:
        """Score texts using multi-trial style similarity.

        Parameters
        ----------
        texts:
            All texts to score.
        **kwargs:
            Must include ``machine_indices``, ``num_fewshot``, and
            ``num_trials``.  Optionally ``source_labels`` for multi-target
            mode, ``output_path`` for saving the score matrix, and
            ``batch_size``.

        Returns
        -------
        list[float]
            Mean score across trials (nanmean) for each text.
        """
        machine_indices: list[int] = kwargs["machine_indices"]
        num_fewshot: int = kwargs["num_fewshot"]
        num_trials: int = kwargs.get("num_trials", DEFAULT_NUM_TRIALS)
        output_path: Path | None = kwargs.get("output_path")
        source_labels: list[str] | None = kwargs.get("source_labels")
        batch_size: int = kwargs.get("batch_size", 128)

        if not machine_indices:
            logger.warning(
                "%s requires machine texts (--num-fewshot > 0). "
                "Returning empty scores.",
                self.name,
            )
            return [float("nan")] * len(texts)

        device = get_device()

        # 1. Load model and embed all texts once (for per-sample scoring)
        model_bundle = self._load_model(device)
        embeddings = self._embed_all(texts, batch_size, device, model_bundle)

        # 2. Build aggregate function — LUAR uses native attention-based
        #    aggregation (single=True), others use mean pooling.
        aggregate_fn = self._make_aggregate_fn(
            texts,
            embeddings,
            device,
            model_bundle,
        )

        # 3. Sample support sets
        machine_indices_arr = np.array(machine_indices)
        support_sets = _sample_support_sets(
            machine_indices_arr,
            num_trials,
            num_fewshot,
            source_labels=source_labels,
        )

        # 4. Compute score matrix
        if source_labels is not None:
            score_matrix = _compute_scores_multitarget(
                embeddings,
                support_sets,
                source_labels,
                machine_indices_arr,
                aggregate_fn,
            )
        else:
            score_matrix = _compute_scores(
                embeddings,
                support_sets,
                aggregate_fn,
            )

        # 5. Save NPZ
        if output_path is not None:
            _save_npz(
                scores=score_matrix,
                support_sets=support_sets,
                output_path=output_path,
                metadata={
                    "detector": self.name,
                    "model": self.MODEL_NAME,
                    "num_trials": num_trials,
                    "num_fewshot": num_fewshot,
                    "num_samples": len(texts),
                    "multitarget": source_labels is not None,
                },
            )

        # 6. Return nanmean across trials
        return np.nanmean(score_matrix, axis=0).tolist()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(
        self,
        device: torch.device,
    ) -> object:
        """Load the embedding model once for reuse.

        Parameters
        ----------
        device:
            Torch device.

        Returns
        -------
        object
            For LUAR: tuple of ``(model, tokenizer)``.
            For SentenceTransformer: the ``SentenceTransformer`` instance.
        """
        if self.USE_LUAR:
            model = (
                AutoModel.from_pretrained(
                    self.MODEL_NAME,
                    trust_remote_code=True,
                )
                .eval()
                .to(device)
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_NAME,
                trust_remote_code=True,
            )
            return (model, tokenizer)

        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.MODEL_NAME).eval().to(device)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _make_aggregate_fn(
        self,
        texts: list[str],
        embeddings: torch.Tensor,
        device: torch.device,
        model_bundle: object,
    ) -> Callable[[np.ndarray], torch.Tensor]:
        """Return a function that aggregates support-set indices into one embedding.

        For LUAR, this re-embeds the support texts using the model's native
        attention-based aggregation (``single=True``).  For SentenceTransformer
        models, this mean-pools the pre-computed embeddings.

        Parameters
        ----------
        texts:
            All texts (needed for LUAR re-embedding).
        embeddings:
            Pre-computed per-text embeddings of shape ``(N, d)``.
        device:
            Torch device.
        model_bundle:
            Pre-loaded model from ``_load_model``.

        Returns
        -------
        Callable[[np.ndarray], torch.Tensor]
            Function mapping support-set indices to a single normalized
            embedding of shape ``(1, d)``.
        """
        if self.USE_LUAR:
            model, tokenizer = model_bundle

            def luar_aggregate(indices: np.ndarray) -> torch.Tensor:
                support_texts = [texts[i] for i in indices]
                return get_luar_embeddings(
                    support_texts,
                    model,
                    tokenizer,
                    single=True,
                )

            return luar_aggregate

        def mean_aggregate(indices: np.ndarray) -> torch.Tensor:
            return F.normalize(
                embeddings[indices].mean(dim=0, keepdim=True),
                p=2,
                dim=-1,
            )

        return mean_aggregate

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed_all(
        self,
        texts: list[str],
        batch_size: int,
        device: torch.device,
        model_bundle: object,
    ) -> torch.Tensor:
        """Embed all texts in a single pass.

        Parameters
        ----------
        texts:
            Texts to embed.
        batch_size:
            Batch size for encoding.
        device:
            Torch device.
        model_bundle:
            Pre-loaded model from ``_load_model``.

        Returns
        -------
        torch.Tensor
            Normalized embeddings of shape ``(N, d)``.
        """
        if self.USE_LUAR:
            model, tokenizer = model_bundle
            return get_luar_embeddings(
                texts,
                model,
                tokenizer,
                batch_size=batch_size,
            )

        emb = model_bundle.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_tensor=True,
        )
        return emb


# ---------------------------------------------------------------------------
# Support-set sampling
# ---------------------------------------------------------------------------

def _sample_support_sets(
    machine_indices: np.ndarray,
    num_trials: int,
    num_fewshot: int,
    source_labels: list[str] | None = None,
) -> list[np.ndarray]:
    """Randomly sample support sets from machine-text indices.

    Parameters
    ----------
    machine_indices:
        Array of indices into the full dataset that are machine-labeled.
    num_trials:
        Number of random trials.
    num_fewshot:
        Number of samples per trial (simple) or per class per trial
        (multi-target).
    source_labels:
        Per-machine-index source labels for multi-target mode.  Length
        must match ``machine_indices``.

    Returns
    -------
    list[np.ndarray]
        One array of selected dataset indices per trial.

    Raises
    ------
    ValueError
        If there are fewer machine texts (or per-class texts) than
        ``num_fewshot``.
    """
    rng = np.random.default_rng()

    if source_labels is None:
        if len(machine_indices) < num_fewshot:
            raise ValueError(
                f"Not enough machine texts ({len(machine_indices)}) "
                f"to sample {num_fewshot} few-shot examples."
            )
        return [
            rng.choice(machine_indices, size=num_fewshot, replace=False)
            for _ in range(num_trials)
        ]

    # Multi-target: sample num_fewshot per class
    labels_arr = np.array(source_labels)
    classes = np.unique(labels_arr)
    for cls in classes:
        cls_count = int(np.sum(labels_arr == cls))
        if cls_count < num_fewshot:
            raise ValueError(
                f"Not enough machine texts for class {cls!r} "
                f"({cls_count}) to sample {num_fewshot} few-shot examples."
            )

    support_sets = []
    for _ in range(num_trials):
        trial_indices = []
        for cls in classes:
            cls_indices = machine_indices[labels_arr == cls]
            trial_indices.append(
                rng.choice(cls_indices, size=num_fewshot, replace=False),
            )
        support_sets.append(np.concatenate(trial_indices))
    return support_sets


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def _compute_scores(
    embeddings: torch.Tensor,
    support_sets: list[np.ndarray],
    aggregate_fn: Callable[[np.ndarray], torch.Tensor],
) -> np.ndarray:
    """Compute score matrix for the simple (single-target) setting.

    Parameters
    ----------
    embeddings:
        Normalized embeddings of shape ``(N, d)``.
    support_sets:
        List of arrays, each containing dataset indices for a trial's
        support set.
    aggregate_fn:
        Function mapping support-set indices to a single normalized
        embedding of shape ``(1, d)``.

    Returns
    -------
    np.ndarray
        Score matrix of shape ``(num_trials, N)`` with NaN for self-excluded
        entries.
    """
    num_trials = len(support_sets)
    num_samples = embeddings.size(0)
    scores = np.empty((num_trials, num_samples), dtype=np.float32)

    for t, support_idx in enumerate(tqdm(
        support_sets,
        desc="StyleDetect trials",
    )):
        aggregate = aggregate_fn(support_idx)
        trial_scores = F.cosine_similarity(
            aggregate.expand(num_samples, -1),
            embeddings,
        )
        scores[t] = trial_scores.cpu().numpy()
        # Self-exclusion: NaN for samples in this trial's support set
        scores[t, support_idx] = np.nan

    return scores


def _compute_scores_multitarget(
    embeddings: torch.Tensor,
    support_sets: list[np.ndarray],
    source_labels: list[str],
    machine_indices: np.ndarray,
    aggregate_fn: Callable[[np.ndarray], torch.Tensor],
) -> np.ndarray:
    """Compute score matrix for the multi-target setting.

    For each trial, computes per-class aggregate embeddings and scores each
    sample by its maximum cosine similarity to any class embedding.

    Parameters
    ----------
    embeddings:
        Normalized embeddings of shape ``(N, d)``.
    support_sets:
        List of arrays, each containing dataset indices for a trial's
        support set (all classes concatenated).
    source_labels:
        Per-machine-index source labels.  Length matches
        ``machine_indices``.
    machine_indices:
        Array of machine-text indices into the full dataset.
    aggregate_fn:
        Function mapping support-set indices to a single normalized
        embedding of shape ``(1, d)``.

    Returns
    -------
    np.ndarray
        Score matrix of shape ``(num_trials, N)`` with NaN for self-excluded
        entries.
    """
    num_trials = len(support_sets)
    num_samples = embeddings.size(0)
    idx_to_label = dict(zip(machine_indices.tolist(), source_labels))
    classes = sorted(set(source_labels))

    scores = np.empty((num_trials, num_samples), dtype=np.float32)

    for t, support_idx in enumerate(tqdm(
        support_sets,
        desc="StyleDetect trials (multi-target)",
    )):
        # Group support indices by class and aggregate each
        class_aggregates = []
        for cls in classes:
            cls_support = np.array([
                idx for idx in support_idx if idx_to_label[int(idx)] == cls
            ])
            if len(cls_support) == 0:
                raise ValueError(
                    f"Trial {t} has no support samples for class {cls!r}. "
                    f"This should not happen — check sampling logic."
                )
            class_aggregates.append(aggregate_fn(cls_support))

        # (num_classes, d)
        agg_stack = torch.cat(class_aggregates, dim=0)

        # Each row i has cosine similarity of sample i to each class.
        # Shape: (N, num_classes) — since embeddings and agg_stack are both
        # L2-normalized, matmul gives cosine similarity directly.
        sim_matrix = torch.mm(embeddings, agg_stack.t())

        # Score = max similarity across classes — shape: (N,)
        trial_scores = sim_matrix.max(dim=1).values
        scores[t] = trial_scores.cpu().numpy()

        # Self-exclusion
        scores[t, support_idx] = np.nan

    return scores


# ---------------------------------------------------------------------------
# NPZ I/O
# ---------------------------------------------------------------------------

def _save_npz(
    scores: np.ndarray,
    support_sets: list[np.ndarray],
    output_path: Path,
    metadata: dict,
) -> None:
    """Save the score matrix and support-set indices as an NPZ file.

    Parameters
    ----------
    scores:
        Score matrix of shape ``(num_trials, num_samples)``.
    support_sets:
        List of support-set index arrays (one per trial).
    output_path:
        Path to write the ``.npz`` file.
    metadata:
        Dictionary of metadata to store alongside the arrays.
    """
    # TODO: Consider migrating all detector score saving to NPZ format.

    # In multi-target mode with unequal class sizes, support sets can have
    # different lengths across trials, so we pad to a uniform length.
    max_len = max(len(s) for s in support_sets)
    padded = np.full((len(support_sets), max_len), -1, dtype=np.int64)
    for i, s in enumerate(support_sets):
        padded[i, : len(s)] = s

    np.savez_compressed(
        output_path,
        scores=scores,
        support_indices=padded,
        metadata=np.array(json.dumps(metadata)),
    )
    logger.info("Score matrix saved to %s", output_path)


# ---------------------------------------------------------------------------
# Detector subclasses
# ---------------------------------------------------------------------------

# @register_detector("styledetect")
class StyleDetectLUAR(_StyleDetectorBase):
    """StyleDetect using LUAR-MUD embeddings."""

    MODEL_NAME = "rrivera1849/LUAR-MUD"
    USE_LUAR = True


@register_detector("styledetect-cisr")
class StyleDetectCISR(_StyleDetectorBase):
    """StyleDetect using CISR style embeddings."""

    MODEL_NAME = "AnnaWegmann/Style-Embedding"


@register_detector("styledetect-sd")
class StyleDetectSD(_StyleDetectorBase):
    """StyleDetect using StyleDistance embeddings."""

    MODEL_NAME = "StyleDistance/styledistance"


@register_detector("semdetect")
class SemDetect(_StyleDetectorBase):
    """Semantic similarity detector using all-mpnet-base-v2."""

    MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

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
            mode, ``multitarget_mode`` (``"max"``, ``"single"``, or
            ``"domain"``, default ``"max"``) controlling how the stratified
            support set is scored, ``domain_labels`` (required for
            ``"domain"`` mode) providing per-text domain labels for all
            texts, ``output_path`` for saving the score matrix, and
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
        multitarget_mode: str = kwargs.get("multitarget_mode", "max")
        domain_labels: list[str] | None = kwargs.get("domain_labels")
        batch_size: int = kwargs.get("batch_size", 128)
        group_size: int | None = kwargs.get("group_size")

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
        machine_domain_labels = None
        if domain_labels is not None and multitarget_mode in (
            "domain",
            "domain-single",
        ):
            machine_domain_labels = [
                domain_labels[i] for i in machine_indices
            ]

        support_sets = _sample_support_sets(
            machine_indices_arr,
            num_trials,
            num_fewshot,
            source_labels=source_labels,
            machine_domain_labels=machine_domain_labels,
        )

        # 3b. Form per-trial query groups (if --group-size is requested).
        #
        # Groups are bucketed for homogeneity and ordered deterministically
        # so that the j-th group in every trial carries the same binary
        # label.  This keeps the score matrix rectangular at
        # (num_trials, num_groups) and lets evaluate_score_matrix consume
        # a single per-column label vector.
        query_groups: list[list[np.ndarray]] | None = None
        group_labels: np.ndarray | None = None
        if group_size is not None:
            is_machine = np.zeros(len(texts), dtype=bool)
            is_machine[machine_indices_arr] = True
            source_labels_full: list[object] | None = None
            if source_labels is not None:
                source_labels_full = [None] * len(texts)
                for idx, lbl in zip(machine_indices, source_labels):
                    source_labels_full[idx] = lbl
            domain_restricted = multitarget_mode in ("domain", "domain-single")

            rng = np.random.default_rng()
            query_groups = []
            reference_labels: list[int] | None = None
            for support_idx in support_sets:
                trial_groups, trial_labels = _form_query_groups(
                    support_idx=support_idx,
                    num_samples=len(texts),
                    group_size=group_size,
                    is_machine=is_machine,
                    source_labels_full=source_labels_full,
                    domain_labels=domain_labels,
                    domain_restricted=domain_restricted,
                    rng=rng,
                )
                query_groups.append(trial_groups)
                if reference_labels is None:
                    reference_labels = trial_labels
                elif trial_labels != reference_labels:
                    raise AssertionError(
                        "Per-column label vector is not stable across "
                        "trials — check bucket-size determinism.",
                    )
            group_labels = np.array(reference_labels, dtype=np.int64)
            logger.info(
                "Grouped scoring: %d groups per trial (group_size=%d)",
                len(group_labels),
                group_size,
            )

        # 4. Compute score matrix
        #
        # When source_labels is provided, _sample_support_sets has already
        # stratified the support set per class.  ``multitarget_mode`` then
        # decides how those stratified indices are scored:
        #   * "max"    — per-class aggregates, score = max cosine similarity
        #                across class centroids (the original multi-target
        #                behavior).
        #   * "single" — pool all stratified samples into one mean centroid,
        #                exactly like the no-source-labels path but with a
        #                class-balanced support set.
        #   * "domain" — domain-restricted multi-target.  Support sets are
        #                sampled per (domain, class) pair.  Each query is
        #                scored only against class centroids built from its
        #                own domain's support samples.
        #   * "domain-single" — domain-restricted single-centroid.  Support
        #                sets are sampled per (domain, class) pair, but each
        #                domain's samples are pooled into one mean centroid.
        #                Each query is scored against its own domain's single
        #                centroid.
        if source_labels is not None and multitarget_mode == "max":
            score_matrix = _compute_scores_multitarget(
                embeddings,
                support_sets,
                source_labels,
                machine_indices_arr,
                aggregate_fn,
                query_groups=query_groups,
            )
        elif source_labels is not None and multitarget_mode == "domain":
            score_matrix = _compute_scores_domain_multitarget(
                embeddings,
                support_sets,
                source_labels,
                machine_indices_arr,
                domain_labels,
                aggregate_fn,
                query_groups=query_groups,
            )
        elif source_labels is not None and multitarget_mode == "domain-single":
            score_matrix = _compute_scores_domain_single(
                embeddings,
                support_sets,
                domain_labels,
                aggregate_fn,
                query_groups=query_groups,
            )
        else:
            score_matrix = _compute_scores(
                embeddings,
                support_sets,
                aggregate_fn,
                query_groups=query_groups,
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
                    "multitarget_mode": (
                        multitarget_mode if source_labels is not None else None
                    ),
                    "group_size": group_size,
                },
                group_labels=group_labels,
            )

        # 6. Return per-sample score vector for the JSONL column.
        #    In grouped mode the score matrix is per-group, not per-sample,
        #    so there is no meaningful per-sample scalar — use NaN.  Per-group
        #    scores live in the NPZ.
        if query_groups is not None:
            return [float("nan")] * len(texts)
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
    machine_domain_labels: list[str] | None = None,
) -> list[np.ndarray]:
    """Randomly sample support sets from machine-text indices.

    Parameters
    ----------
    machine_indices:
        Array of indices into the full dataset that are machine-labeled.
    num_trials:
        Number of random trials.
    num_fewshot:
        Number of samples per trial (simple), per class per trial
        (multi-target), or per (domain, class) pair per trial
        (domain-aware multi-target).
    source_labels:
        Per-machine-index source labels for multi-target mode.  Length
        must match ``machine_indices``.
    machine_domain_labels:
        Per-machine-index domain labels for domain-aware multi-target
        mode.  Length must match ``machine_indices``.  When provided
        alongside ``source_labels``, sampling is stratified per
        (domain, class) pair.

    Returns
    -------
    list[np.ndarray]
        One array of selected dataset indices per trial.

    Raises
    ------
    ValueError
        If there are fewer machine texts (or per-class / per-pair texts)
        than ``num_fewshot``.
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

    labels_arr = np.array(source_labels)
    classes = np.unique(labels_arr)
    
    # Domain-aware multi-target: sample num_fewshot per (domain, class) pair
    if machine_domain_labels is not None:
        domain_arr = np.array(machine_domain_labels)
        domains = np.unique(domain_arr)

        for dom in domains:
            for cls in classes:
                mask = (domain_arr == dom) & (labels_arr == cls)
                pair_count = int(np.sum(mask))
                if pair_count < num_fewshot:
                    raise ValueError(
                        f"Not enough machine texts for "
                        f"(domain={dom!r}, class={cls!r}) "
                        f"({pair_count}) to sample {num_fewshot} "
                        f"few-shot examples."
                    )

        support_sets = []
        for _ in range(num_trials):
            trial_indices = []
            for dom in domains:
                for cls in classes:
                    pair_mask = (domain_arr == dom) & (labels_arr == cls)
                    pair_indices = machine_indices[pair_mask]
                    trial_indices.append(
                        rng.choice(
                            pair_indices,
                            size=num_fewshot,
                            replace=False,
                        ),
                    )
            support_sets.append(np.concatenate(trial_indices))
        return support_sets

    # Standard multi-target: sample num_fewshot per class
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
# Query grouping
# ---------------------------------------------------------------------------

def _form_query_groups(
    support_idx: np.ndarray,
    num_samples: int,
    group_size: int,
    is_machine: np.ndarray,
    source_labels_full: list[object] | None,
    domain_labels: list[str] | None,
    domain_restricted: bool,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[int]]:
    """Form homogeneous query groups of ``group_size`` from non-support indices.

    Groups are bucketed by ``(is_machine, [source_class], [domain])``:

    * human-vs-machine is always a bucket dimension;
    * source class is added for machine texts when ``source_labels_full`` is
      supplied;
    * domain is added for all texts when ``domain_restricted`` is true
      (i.e. in ``domain`` / ``domain-single`` modes).

    Buckets are sorted deterministically so that — given stable bucket
    sizes across trials — the j-th group always carries the same label.
    Within each bucket, members are shuffled and consumed in chunks of
    ``group_size``; any remainder smaller than ``group_size`` is dropped.

    Parameters
    ----------
    support_idx:
        Trial support-set indices to exclude from the group pool.
    num_samples:
        Total number of samples ``N``.
    group_size:
        Number of texts per group.
    is_machine:
        Boolean array of shape ``(N,)``; ``True`` iff sample is machine.
    source_labels_full:
        Per-sample source label (``None`` for non-machines), length ``N``,
        or ``None`` if source stratification is not in use.
    domain_labels:
        Per-sample domain label, length ``N``, or ``None``.
    domain_restricted:
        If ``True``, bucket groups by domain in addition to class.
    rng:
        NumPy random generator.

    Returns
    -------
    groups:
        List of group index arrays, ordered by bucket key.
    labels:
        Binary group label (``1`` = machine, ``0`` = human), same order.
    """
    support_set = set(int(i) for i in support_idx)
    buckets: dict[tuple, list[int]] = {}
    for i in range(num_samples):
        if i in support_set:
            continue
        mach = bool(is_machine[i])
        cls_key = (
            source_labels_full[i]
            if source_labels_full is not None and mach
            else None
        )
        dom_key = (
            domain_labels[i]
            if domain_restricted and domain_labels is not None
            else None
        )
        key = (mach, cls_key, dom_key)
        buckets.setdefault(key, []).append(i)

    sorted_keys = sorted(
        buckets.keys(),
        key=lambda k: (k[0], str(k[1]), str(k[2])),
    )

    groups: list[np.ndarray] = []
    labels: list[int] = []
    for key in sorted_keys:
        bucket = np.array(buckets[key])
        rng.shuffle(bucket)
        n_full = len(bucket) // group_size
        for k in range(n_full):
            groups.append(bucket[k * group_size : (k + 1) * group_size])
            labels.append(1 if key[0] else 0)
    return groups, labels


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def _compute_scores(
    embeddings: torch.Tensor,
    support_sets: list[np.ndarray],
    aggregate_fn: Callable[[np.ndarray], torch.Tensor],
    query_groups: list[list[np.ndarray]] | None = None,
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
    query_groups:
        If provided, per-trial list of query-group index arrays; each
        group's aggregate embedding is scored instead of each raw text.
        Output shape becomes ``(num_trials, num_groups)``.

    Returns
    -------
    np.ndarray
        Score matrix of shape ``(num_trials, N)`` (or
        ``(num_trials, num_groups)`` in grouped mode) with NaN for
        self-excluded entries in the ungrouped path.
    """
    num_trials = len(support_sets)

    if query_groups is None:
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
            # Cast to float32 — some HF models load in bfloat16, which numpy
            # cannot convert directly.
            scores[t] = trial_scores.float().cpu().numpy()
            # Self-exclusion: NaN for samples in this trial's support set
            scores[t, support_idx] = np.nan

        return scores

    num_groups = len(query_groups[0])
    scores = np.empty((num_trials, num_groups), dtype=np.float32)
    for t, support_idx in enumerate(tqdm(
        support_sets,
        desc="StyleDetect trials (grouped)",
    )):
        aggregate = aggregate_fn(support_idx)
        groups = query_groups[t]
        group_embs = torch.cat(
            [aggregate_fn(g) for g in groups],
            dim=0,
        )
        group_scores = F.cosine_similarity(
            aggregate.expand(len(groups), -1),
            group_embs,
        )
        scores[t] = group_scores.float().cpu().numpy()

    return scores


def _compute_scores_multitarget(
    embeddings: torch.Tensor,
    support_sets: list[np.ndarray],
    source_labels: list[str],
    machine_indices: np.ndarray,
    aggregate_fn: Callable[[np.ndarray], torch.Tensor],
    query_groups: list[list[np.ndarray]] | None = None,
) -> np.ndarray:
    """Compute score matrix for the multi-target setting.

    For each trial, computes per-class aggregate embeddings and scores each
    sample (or each query group) by its maximum cosine similarity to any
    class embedding.

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
    query_groups:
        If provided, per-trial list of query-group index arrays; each
        group's aggregate embedding is scored instead of each raw text.

    Returns
    -------
    np.ndarray
        Score matrix of shape ``(num_trials, N)`` (or
        ``(num_trials, num_groups)`` in grouped mode).
    """
    num_trials = len(support_sets)
    num_samples = embeddings.size(0)
    idx_to_label = dict(zip(machine_indices.tolist(), source_labels))
    classes = sorted(set(source_labels))

    grouped = query_groups is not None
    num_cols = len(query_groups[0]) if grouped else num_samples
    scores = np.empty((num_trials, num_cols), dtype=np.float32)

    desc = (
        "StyleDetect trials (multi-target, grouped)"
        if grouped
        else "StyleDetect trials (multi-target)"
    )
    for t, support_idx in enumerate(tqdm(support_sets, desc=desc)):
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

        if grouped:
            groups = query_groups[t]
            group_embs = torch.cat(
                [aggregate_fn(g) for g in groups],
                dim=0,
            )
            sim_matrix = torch.mm(group_embs, agg_stack.t())
            trial_scores = sim_matrix.max(dim=1).values
            scores[t] = trial_scores.float().cpu().numpy()
            continue

        # Each row i has cosine similarity of sample i to each class.
        # Shape: (N, num_classes) — since embeddings and agg_stack are both
        # L2-normalized, matmul gives cosine similarity directly.
        sim_matrix = torch.mm(embeddings, agg_stack.t())

        # Score = max similarity across classes — shape: (N,)
        trial_scores = sim_matrix.max(dim=1).values
        # Cast to float32 — some HF models load in bfloat16, which numpy
        # cannot convert directly.
        scores[t] = trial_scores.float().cpu().numpy()

        # Self-exclusion
        scores[t, support_idx] = np.nan

    return scores


def _compute_scores_domain_multitarget(
    embeddings: torch.Tensor,
    support_sets: list[np.ndarray],
    source_labels: list[str],
    machine_indices: np.ndarray,
    domain_labels: list[str],
    aggregate_fn: Callable[[np.ndarray], torch.Tensor],
    query_groups: list[list[np.ndarray]] | None = None,
) -> np.ndarray:
    """Compute score matrix for the domain-restricted multi-target setting.

    For each trial and each domain, computes per-class (model) aggregate
    embeddings using only support samples from that domain, then scores
    each text against its own domain's class centroids via max cosine
    similarity.

    Parameters
    ----------
    embeddings:
        Normalized embeddings of shape ``(N, d)``.
    support_sets:
        List of arrays, each containing dataset indices for a trial's
        support set, sampled per (domain, class) pair.
    source_labels:
        Per-machine-index source labels (e.g. model names).  Length
        matches ``machine_indices``.
    machine_indices:
        Array of machine-text indices into the full dataset.
    domain_labels:
        Domain label for every text in the full dataset (human and
        machine).  Length equals ``N``.
    aggregate_fn:
        Function mapping support-set indices to a single normalized
        embedding of shape ``(1, d)``.

    Returns
    -------
    np.ndarray
        Score matrix of shape ``(num_trials, N)``.  NaN for
        self-excluded entries and for texts in domains without
        support coverage in a given trial.
    """
    num_trials = len(support_sets)
    num_samples = embeddings.size(0)
    idx_to_label = dict(zip(machine_indices.tolist(), source_labels))
    grouped = query_groups is not None

    # Precompute: for each domain, the indices of ALL texts in that domain
    all_domains = sorted(set(domain_labels))
    domain_text_indices: dict[str, np.ndarray] = {}
    for dom in all_domains:
        domain_text_indices[dom] = np.array(
            [i for i, d in enumerate(domain_labels) if d == dom],
        )

    if grouped:
        num_cols = len(query_groups[0])
        scores = np.empty((num_trials, num_cols), dtype=np.float32)
    else:
        scores = np.full(
            (num_trials, num_samples),
            np.nan,
            dtype=np.float32,
        )

    desc = (
        "StyleDetect trials (domain multi-target, grouped)"
        if grouped
        else "StyleDetect trials (domain multi-target)"
    )
    for t, support_idx in enumerate(tqdm(support_sets, desc=desc)):
        # Group support indices by domain, then by class within domain
        domain_class_support: dict[str, dict[str, list[int]]] = {}
        for idx in support_idx:
            dom = domain_labels[idx]
            cls = idx_to_label[int(idx)]
            (
                domain_class_support
                .setdefault(dom, {})
                .setdefault(cls, [])
                .append(idx)
            )

        # Bucket groups by their domain (all members share the same domain).
        if grouped:
            trial_group_col: dict[str, list[int]] = {}
            for col, g in enumerate(query_groups[t]):
                dom = domain_labels[int(g[0])]
                trial_group_col.setdefault(dom, []).append(col)

        for dom in all_domains:
            text_idxs = domain_text_indices[dom]
            if len(text_idxs) == 0:
                continue

            class_support = domain_class_support.get(dom)
            if not class_support:
                continue

            # Build per-class centroids for this domain
            class_aggregates = []
            for cls in sorted(class_support.keys()):
                cls_indices = np.array(class_support[cls])
                class_aggregates.append(aggregate_fn(cls_indices))

            # (num_classes_in_domain, d)
            agg_stack = torch.cat(class_aggregates, dim=0)

            if grouped:
                cols = trial_group_col.get(dom, [])
                if not cols:
                    continue
                dom_groups = [query_groups[t][c] for c in cols]
                group_embs = torch.cat(
                    [aggregate_fn(g) for g in dom_groups],
                    dim=0,
                )
                sim_matrix = torch.mm(group_embs, agg_stack.t())
                trial_scores = sim_matrix.max(dim=1).values
                scores[t, cols] = trial_scores.float().cpu().numpy()
                continue

            # Score this domain's texts against this domain's centroids
            dom_embeddings = embeddings[text_idxs]
            sim_matrix = torch.mm(dom_embeddings, agg_stack.t())
            trial_scores = sim_matrix.max(dim=1).values
            scores[t, text_idxs] = trial_scores.float().cpu().numpy()

        if not grouped:
            # Self-exclusion (groups are disjoint from support by construction)
            scores[t, support_idx] = np.nan

    return scores


def _compute_scores_domain_single(
    embeddings: torch.Tensor,
    support_sets: list[np.ndarray],
    domain_labels: list[str],
    aggregate_fn: Callable[[np.ndarray], torch.Tensor],
    query_groups: list[list[np.ndarray]] | None = None,
) -> np.ndarray:
    """Compute score matrix for the domain-restricted single-centroid setting.

    For each trial and each domain, pools all support samples from that
    domain into one mean centroid (ignoring class labels at aggregation
    time).  Each text is scored against its own domain's single centroid
    via cosine similarity.  Support sets are assumed to be sampled per
    (domain, class) pair so the pool is class-balanced by construction.

    Parameters
    ----------
    embeddings:
        Normalized embeddings of shape ``(N, d)``.
    support_sets:
        List of arrays, each containing dataset indices for a trial's
        support set, sampled per (domain, class) pair.
    domain_labels:
        Domain label for every text in the full dataset (human and
        machine).  Length equals ``N``.
    aggregate_fn:
        Function mapping support-set indices to a single normalized
        embedding of shape ``(1, d)``.

    Returns
    -------
    np.ndarray
        Score matrix of shape ``(num_trials, N)``.  NaN for
        self-excluded entries and for texts in domains without
        support coverage in a given trial.
    """
    num_trials = len(support_sets)
    num_samples = embeddings.size(0)
    grouped = query_groups is not None

    # Precompute: for each domain, the indices of ALL texts in that domain
    all_domains = sorted(set(domain_labels))
    domain_text_indices: dict[str, np.ndarray] = {}
    for dom in all_domains:
        domain_text_indices[dom] = np.array(
            [i for i, d in enumerate(domain_labels) if d == dom],
        )

    if grouped:
        num_cols = len(query_groups[0])
        scores = np.empty((num_trials, num_cols), dtype=np.float32)
    else:
        scores = np.full(
            (num_trials, num_samples),
            np.nan,
            dtype=np.float32,
        )

    desc = (
        "StyleDetect trials (domain single-centroid, grouped)"
        if grouped
        else "StyleDetect trials (domain single-centroid)"
    )
    for t, support_idx in enumerate(tqdm(support_sets, desc=desc)):
        # Group support indices by domain
        domain_support: dict[str, list[int]] = {}
        for idx in support_idx:
            dom = domain_labels[idx]
            domain_support.setdefault(dom, []).append(idx)

        if grouped:
            trial_group_col: dict[str, list[int]] = {}
            for col, g in enumerate(query_groups[t]):
                dom = domain_labels[int(g[0])]
                trial_group_col.setdefault(dom, []).append(col)

        for dom in all_domains:
            text_idxs = domain_text_indices[dom]
            if len(text_idxs) == 0:
                continue

            dom_support = domain_support.get(dom)
            if not dom_support:
                continue

            # Single centroid for this domain
            aggregate = aggregate_fn(np.array(dom_support))

            if grouped:
                cols = trial_group_col.get(dom, [])
                if not cols:
                    continue
                dom_groups = [query_groups[t][c] for c in cols]
                group_embs = torch.cat(
                    [aggregate_fn(g) for g in dom_groups],
                    dim=0,
                )
                trial_scores = F.cosine_similarity(
                    aggregate.expand(group_embs.size(0), -1),
                    group_embs,
                )
                scores[t, cols] = trial_scores.float().cpu().numpy()
                continue

            # Score this domain's texts against its single centroid
            dom_embeddings = embeddings[text_idxs]
            trial_scores = F.cosine_similarity(
                aggregate.expand(dom_embeddings.size(0), -1),
                dom_embeddings,
            )
            scores[t, text_idxs] = trial_scores.float().cpu().numpy()

        if not grouped:
            # Self-exclusion (groups are disjoint from support by construction)
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
    group_labels: np.ndarray | None = None,
) -> None:
    """Save the score matrix and support-set indices as an NPZ file.

    Parameters
    ----------
    scores:
        Score matrix of shape ``(num_trials, num_samples)`` or
        ``(num_trials, num_groups)`` in grouped mode.
    support_sets:
        List of support-set index arrays (one per trial).
    output_path:
        Path to write the ``.npz`` file.
    metadata:
        Dictionary of metadata to store alongside the arrays.
    group_labels:
        Optional per-column binary labels (length ``num_groups``) for
        grouped scoring.  When provided, stored under the ``group_labels``
        key and consumed by the evaluate path.
    """
    # TODO: Consider migrating all detector score saving to NPZ format.

    # In multi-target mode with unequal class sizes, support sets can have
    # different lengths across trials, so we pad to a uniform length.
    max_len = max(len(s) for s in support_sets)
    padded = np.full((len(support_sets), max_len), -1, dtype=np.int64)
    for i, s in enumerate(support_sets):
        padded[i, : len(s)] = s

    save_kwargs: dict = {
        "scores": scores,
        "support_indices": padded,
        "metadata": np.array(json.dumps(metadata)),
    }
    if group_labels is not None:
        save_kwargs["group_labels"] = group_labels

    np.savez_compressed(output_path, **save_kwargs)
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

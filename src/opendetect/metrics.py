"""Evaluation metrics for machine-text detection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def get_tpr_target(
    fpr: np.ndarray,
    tpr: np.ndarray,
    target_fpr: float,
) -> float:
    """Interpolate TPR at a given FPR threshold.

    Parameters
    ----------
    fpr:
        False-positive-rate array from ``roc_curve``.
    tpr:
        True-positive-rate array from ``roc_curve``.
    target_fpr:
        The FPR threshold to interpolate at.

    Returns
    -------
    float
        TPR as a percentage (0-100).
    """
    indices = None
    for i in range(len(fpr)):
        if fpr[i] >= target_fpr:
            if i == 0:
                indices = [i]
            else:
                indices = [i - 1, i]
            break

    if indices is None:
        return tpr[-1] * 100
    else:
        tpr_values = [tpr[i] for i in indices]
        return np.mean(tpr_values) * 100


def evaluate_score_matrix(
    score_matrix: np.ndarray,
    labels: np.ndarray,
    target_fprs: list[float] | None = None,
) -> dict:
    """Compute macro-averaged metrics across trials of a score matrix.

    For each trial (row), computes ROC AUC and TPR at target FPR thresholds
    using only the non-NaN entries, then returns the macro-average across
    trials.

    Parameters
    ----------
    score_matrix:
        Array of shape ``(num_trials, num_samples)``.  NaN entries are
        excluded per trial.
    labels:
        Binary labels for each sample (0 = human, 1 = machine).
    target_fprs:
        FPR thresholds for TPR and partial AUROC computation.
        Defaults to ``[0.001, 0.01, 0.1]``.

    Returns
    -------
    dict
        Keys are metric names (e.g. ``"ROC AUC"``, ``"TPR(FPR=1%)"``),
        values are the macro-averaged metric across trials.
    """
    if target_fprs is None:
        target_fprs = [0.001, 0.01, 0.1]

    num_trials = score_matrix.shape[0]
    trial_metrics: dict[str, list[float]] = {}

    for t in range(num_trials):
        valid = ~np.isnan(score_matrix[t])
        t_scores = score_matrix[t, valid]
        t_labels = labels[valid]

        if len(np.unique(t_labels)) < 2:
            continue

        roc_auc = float(roc_auc_score(t_labels, t_scores))
        trial_metrics.setdefault("ROC AUC", []).append(roc_auc)

        fpr, tpr, _ = roc_curve(t_labels, t_scores)

        for target_fpr in target_fprs:
            auroc_key = f"AUROC({target_fpr * 100})"
            auroc_val = float(
                roc_auc_score(t_labels, t_scores, max_fpr=target_fpr),
            )
            trial_metrics.setdefault(auroc_key, []).append(auroc_val)

            tpr_key = f"TPR(FPR={target_fpr * 100}%)"
            tpr_val = float(get_tpr_target(fpr, tpr, target_fpr))
            trial_metrics.setdefault(tpr_key, []).append(tpr_val)

    # Macro-average across trials
    return {k: float(np.mean(v)) for k, v in trial_metrics.items()}
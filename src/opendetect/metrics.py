"""Evaluation metrics for machine-text detection."""

from __future__ import annotations

import numpy as np


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

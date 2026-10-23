"""Shared evaluation metrics for binary fraud classification."""

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def binary_metrics(probs, y_true, threshold=0.5):
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)

    # Validate inputs
    if probs.ndim != 1 or y_true.ndim != 1:
        raise ValueError("Probabilities and labels must be 1D arrays")

    if len(probs) == 0:
        raise ValueError("Cannot evaluate empty arrays")

    if len(probs) != len(y_true):
        raise ValueError("Probabilities and labels must have equal lengths")

    if not np.isfinite(probs).all():
        raise ValueError("Probabilities contain NaN or infinity")

    if np.any((probs < 0) | (probs > 1)):
        raise ValueError("Probabilities must be between 0 and 1")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1")

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")

    # Convert probabilities to predictions
    y_pred = (probs >= threshold).astype(int)

    # False-positive rate
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    return {
        "precision_illicit": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall_illicit": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "f1_illicit": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "roc_auc": (
            float(roc_auc_score(y_true, probs))
            if len(np.unique(y_true)) > 1
            else float("nan")
        ),
        "false_positive_rate": fp / max(fp + tn, 1),
        "n_eval": int(len(y_true)),
    }
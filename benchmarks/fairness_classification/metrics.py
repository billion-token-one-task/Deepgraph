"""Metrics for fairness classification benchmarks."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score


def demographic_parity_gap(y_pred: np.ndarray, sensitive: np.ndarray) -> float:
    rates = []
    for group in np.unique(sensitive):
        mask = sensitive == group
        rates.append(float(np.mean(y_pred[mask])) if np.any(mask) else 0.0)
    if len(rates) < 2:
        return 0.0
    return float(max(rates) - min(rates))


def _rates_for_label(y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray,
                     label: int) -> list[float]:
    rates = []
    for group in np.unique(sensitive):
        mask = (sensitive == group) & (y_true == label)
        rates.append(float(np.mean(y_pred[mask])) if np.any(mask) else 0.0)
    return rates


def equalized_odds_gap(y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray) -> float:
    gaps = []
    for label in (0, 1):
        rates = _rates_for_label(y_true, y_pred, sensitive, label)
        if len(rates) >= 2:
            gaps.append(max(rates) - min(rates))
    return float(max(gaps)) if gaps else 0.0


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    fairness_penalty: float = 0.45,
) -> dict:
    accuracy = float(accuracy_score(y_true, y_pred))
    dp_gap = demographic_parity_gap(y_pred, sensitive)
    eo_gap = equalized_odds_gap(y_true, y_pred, sensitive)
    metrics = {
        "accuracy": accuracy,
        "demographic_parity_gap": dp_gap,
        "equalized_odds_gap": eo_gap,
        "fairness_score": accuracy - float(fairness_penalty) * dp_gap,
        "fairness_penalty": float(fairness_penalty),
    }
    for group in (0, 1):
        mask = sensitive == group
        y_group = y_true[mask]
        pred_group = y_pred[mask]
        positives = y_group == 1
        negatives = y_group == 0
        metrics[f"group_{group}_n"] = int(mask.sum())
        metrics[f"group_{group}_base_rate"] = float(np.mean(y_group)) if y_group.size else 0.0
        metrics[f"group_{group}_selection_rate"] = float(np.mean(pred_group)) if pred_group.size else 0.0
        metrics[f"group_{group}_false_positive_rate"] = (
            float(np.mean(pred_group[negatives] == 1)) if np.any(negatives) else 0.0
        )
        metrics[f"group_{group}_false_negative_rate"] = (
            float(np.mean(pred_group[positives] == 0)) if np.any(positives) else 0.0
        )
    return metrics

"""Reusable evaluation utilities — implemented with numpy only.

The notebooks favour first-principles implementations to demonstrate ML
fundamentals. These helpers keep the notebooks clean: each notebook focuses on
the modelling narrative, not on metric boilerplate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Regression metrics                                                          #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RegressionScores:
    """RMSE, MAE, and R² in a single container."""

    rmse: float
    mae: float
    r2: float

    def as_series(self, name: str) -> pd.Series:
        return pd.Series({"RMSE": self.rmse, "MAE": self.mae, "R2": self.r2}, name=name)


def regression_scores(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionScores:
    """Compute RMSE, MAE and R² without any external dependencies."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return RegressionScores(rmse=rmse, mae=mae, r2=r2)


# --------------------------------------------------------------------------- #
# Classification metrics                                                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ClassificationScores:
    """Standard panel of binary classification metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float

    def as_series(self, name: str) -> pd.Series:
        return pd.Series(
            {
                "Accuracy": self.accuracy,
                "Precision": self.precision,
                "Recall": self.recall,
                "F1": self.f1,
                "ROC-AUC": self.roc_auc,
                "PR-AUC": self.pr_auc,
            },
            name=name,
        )


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """Return (TN, FP, FN, TP)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tn, fp, fn, tp


def roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the ROC curve points (FPR, TPR, thresholds)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    distinct_idx = np.where(np.diff(y_score_sorted))[0]
    thresholds_idx = np.r_[distinct_idx, y_true.size - 1]

    tps = np.cumsum(y_true_sorted)[thresholds_idx]
    fps = 1 + thresholds_idx - tps

    total_pos = y_true.sum()
    total_neg = y_true.size - total_pos

    tpr = np.r_[0.0, tps / total_pos] if total_pos > 0 else np.zeros_like(tps, dtype=float)
    fpr = np.r_[0.0, fps / total_neg] if total_neg > 0 else np.zeros_like(fps, dtype=float)
    thresholds = np.r_[np.inf, y_score_sorted[thresholds_idx]]
    return fpr, tpr, thresholds


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """Area under the curve via the trapezoidal rule (assumes x is sorted)."""
    order = np.argsort(x)
    return float(np.trapz(np.asarray(y)[order], np.asarray(x)[order]))


def pr_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision-Recall curve points (precision, recall, thresholds)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    precision = tps / np.maximum(tps + fps, 1)
    total_pos = y_true.sum()
    recall = tps / max(total_pos, 1)

    # Keep one point per distinct threshold + sentinel (recall=0, precision=1).
    distinct_idx = np.where(np.diff(y_score_sorted))[0]
    keep = np.r_[distinct_idx, y_true.size - 1]
    return np.r_[1.0, precision[keep]], np.r_[0.0, recall[keep]], y_score_sorted[keep]


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision = sum_i (R_i - R_{i-1}) * P_i. Matches sklearn convention."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    precision = tps / np.maximum(tps + fps, 1)
    total_pos = y_true.sum()
    if total_pos == 0:
        return 0.0
    recall = tps / total_pos
    recall_step = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_step * precision))


def classification_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> ClassificationScores:
    """Compute the standard panel of metrics for a binary classifier."""
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12) if (precision + recall) > 0 else 0.0

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    pr_auc = average_precision(y_true, y_proba)

    return ClassificationScores(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=float(roc_auc),
        pr_auc=float(pr_auc),
    )


# --------------------------------------------------------------------------- #
# Cross-validation                                                            #
# --------------------------------------------------------------------------- #


def kfold_indices(n: int, k: int, shuffle: bool = True, random_state: int = 42) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, valid_idx) for K folds."""
    rng = np.random.default_rng(random_state)
    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)
    folds = np.array_split(indices, k)
    for i in range(k):
        valid_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train_idx, valid_idx


def stratified_kfold_indices(
    y: np.ndarray, k: int, shuffle: bool = True, random_state: int = 42
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Yield stratified (train_idx, valid_idx) for K folds — preserves class ratios."""
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(random_state)
    per_class_folds: dict[int, list[np.ndarray]] = {}
    for cls in np.unique(y):
        idx_cls = np.where(y == cls)[0]
        if shuffle:
            rng.shuffle(idx_cls)
        per_class_folds[int(cls)] = np.array_split(idx_cls, k)
    for i in range(k):
        valid_idx = np.concatenate([per_class_folds[c][i] for c in per_class_folds])
        train_idx = np.concatenate(
            [
                per_class_folds[c][j]
                for c in per_class_folds
                for j in range(k)
                if j != i
            ]
        )
        yield train_idx, valid_idx


def cv_score(
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    scorer: Callable[[np.ndarray, np.ndarray], float],
    k: int = 5,
    stratified: bool = False,
    random_state: int = 42,
) -> tuple[float, float, np.ndarray]:
    """Generic CV runner.

    ``fit_predict(X_train, y_train, X_valid)`` must return predictions for the
    validation set. ``scorer(y_true, y_pred)`` returns a scalar to maximise (or
    minimise — we just report the mean/std).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    fold_iter = (
        stratified_kfold_indices(y, k=k, random_state=random_state)
        if stratified
        else kfold_indices(len(y), k=k, random_state=random_state)
    )
    scores: list[float] = []
    for train_idx, valid_idx in fold_iter:
        y_pred = fit_predict(X[train_idx], y[train_idx], X[valid_idx])
        scores.append(scorer(y[valid_idx], y_pred))
    arr = np.array(scores)
    return float(arr.mean()), float(arr.std()), arr


def compare_metrics(rows: Iterable[pd.Series]) -> pd.DataFrame:
    """Stack metric series into a comparison table."""
    return pd.DataFrame(list(rows)).round(4)

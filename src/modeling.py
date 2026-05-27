"""From-scratch ML models, implemented with numpy only.

The notebooks use these implementations to demonstrate understanding of the
underlying linear algebra and optimisation, paired with `scikit-learn`
equivalents shown in markdown for the production pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _add_intercept(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((X.shape[0], 1)), X])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid.
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


# --------------------------------------------------------------------------- #
# Linear regression — Ordinary Least Squares & Ridge via closed form          #
# --------------------------------------------------------------------------- #


@dataclass
class LinearRegression:
    """Multivariate OLS via the normal equations with numerical safeguards.

    Uses `numpy.linalg.lstsq` so it stays stable even with multicollinearity.
    """

    fit_intercept: bool = True
    coef_: np.ndarray = field(default_factory=lambda: np.array([]))
    intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        Xb = _add_intercept(X) if self.fit_intercept else X
        beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        if self.fit_intercept:
            self.intercept_ = float(beta[0])
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


@dataclass
class RidgeRegression:
    """L2-regularised linear regression. Closed-form solution; does not penalise the intercept."""

    alpha: float = 1.0
    fit_intercept: bool = True
    coef_: np.ndarray = field(default_factory=lambda: np.array([]))
    intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if self.fit_intercept:
            x_mean = X.mean(axis=0)
            y_mean = y.mean()
            Xc = X - x_mean
            yc = y - y_mean
        else:
            Xc, yc = X, y
            x_mean = np.zeros(X.shape[1])
            y_mean = 0.0

        n_features = Xc.shape[1]
        A = Xc.T @ Xc + self.alpha * np.eye(n_features)
        b = Xc.T @ yc
        beta = np.linalg.solve(A, b)

        self.coef_ = beta
        self.intercept_ = float(y_mean - x_mean @ beta) if self.fit_intercept else 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


# --------------------------------------------------------------------------- #
# K-Nearest Neighbors                                                         #
# --------------------------------------------------------------------------- #


@dataclass
class KNNRegressor:
    """K-Nearest Neighbors regressor using Euclidean distance."""

    k: int = 5
    _X: np.ndarray = field(default_factory=lambda: np.array([]))
    _y: np.ndarray = field(default_factory=lambda: np.array([]))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNRegressor":
        self._X = np.asarray(X, dtype=float)
        self._y = np.asarray(y, dtype=float).ravel()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xq = np.asarray(X, dtype=float)
        # Pairwise squared distances via broadcasting.
        diff = Xq[:, None, :] - self._X[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        nearest = np.argpartition(d2, kth=self.k - 1, axis=1)[:, : self.k]
        return self._y[nearest].mean(axis=1)


# --------------------------------------------------------------------------- #
# Logistic regression — gradient descent with optional L2 regularisation     #
# --------------------------------------------------------------------------- #


@dataclass
class LogisticRegression:
    """Binary logistic regression trained with full-batch gradient descent.

    Parameters
    ----------
    learning_rate : float
        Step size for gradient descent.
    n_iter : int
        Maximum number of iterations.
    l2 : float
        L2 regularisation strength (`0` = no regularisation).
    class_weight : Literal["balanced"] | None
        If ``"balanced"`` weights are inversely proportional to class frequency.
    tol : float
        Stop early when the gradient norm drops below this.
    """

    learning_rate: float = 0.1
    n_iter: int = 2_000
    l2: float = 0.0
    class_weight: Literal["balanced"] | None = None
    tol: float = 1e-6
    coef_: np.ndarray = field(default_factory=lambda: np.array([]))
    intercept_: float = 0.0
    loss_history_: list[float] = field(default_factory=list)

    def _weights(self, y: np.ndarray) -> np.ndarray:
        if self.class_weight is None:
            return np.ones_like(y, dtype=float)
        n = len(y)
        n_pos = max(int(y.sum()), 1)
        n_neg = max(n - n_pos, 1)
        w_pos = n / (2 * n_pos)
        w_neg = n / (2 * n_neg)
        w = np.where(y == 1, w_pos, w_neg).astype(float)
        return w

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, d = X.shape
        Xb = _add_intercept(X)
        beta = np.zeros(d + 1)
        w = self._weights(y)
        self.loss_history_ = []

        for _ in range(self.n_iter):
            z = Xb @ beta
            p = _sigmoid(z)
            # Weighted negative log-likelihood + L2 (intercept excluded).
            eps = 1e-12
            loss = -np.mean(w * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
            loss += 0.5 * self.l2 * np.dot(beta[1:], beta[1:]) / n
            self.loss_history_.append(float(loss))

            grad = (Xb.T @ (w * (p - y))) / n
            grad[1:] += self.l2 * beta[1:] / n

            beta -= self.learning_rate * grad

            if np.linalg.norm(grad) < self.tol:
                break

        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = np.asarray(X, dtype=float) @ self.coef_ + self.intercept_
        return _sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


# --------------------------------------------------------------------------- #
# Standardisation (StandardScaler-style)                                      #
# --------------------------------------------------------------------------- #


@dataclass
class StandardScaler:
    """Centre and scale features to zero mean and unit variance."""

    mean_: np.ndarray = field(default_factory=lambda: np.array([]))
    scale_: np.ndarray = field(default_factory=lambda: np.array([]))

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)
        self.scale_ = np.where(std > 0, std, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lightweight stratified-aware train/test split."""
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    n = len(y)
    if stratify is None:
        idx = np.arange(n)
        rng.shuffle(idx)
        n_test = int(round(n * test_size))
        test_idx, train_idx = idx[:n_test], idx[n_test:]
    else:
        stratify = np.asarray(stratify)
        test_idx_parts = []
        train_idx_parts = []
        for cls in np.unique(stratify):
            cls_idx = np.where(stratify == cls)[0]
            rng.shuffle(cls_idx)
            n_test_cls = int(round(len(cls_idx) * test_size))
            test_idx_parts.append(cls_idx[:n_test_cls])
            train_idx_parts.append(cls_idx[n_test_cls:])
        test_idx = np.concatenate(test_idx_parts)
        train_idx = np.concatenate(train_idx_parts)
        rng.shuffle(test_idx)
        rng.shuffle(train_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

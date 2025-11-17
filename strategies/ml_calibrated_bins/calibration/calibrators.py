#!/usr/bin/env python3
"""
Probability calibration utilities.

Provides Platt (logistic regression) and Isotonic calibration for binary predictions.
Adapted from trading_cex/analysis/calval_utils.py for production use.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _to_numpy(a: Iterable) -> np.ndarray:
    """Convert iterable to numpy array."""
    if hasattr(a, 'to_numpy'):
        return a.to_numpy()
    return np.asarray(a)


def logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute logit transform: log(p / (1-p))."""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def inv_logit(x: np.ndarray) -> np.ndarray:
    """Compute inverse logit: 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-x))


class PlattCalibrator:
    """Platt scaling calibrator using logistic regression on logit-transformed predictions.

    Fits a logistic regression model: P(y=1 | p) = sigmoid(a * logit(p) + b)

    Attributes:
        C: Regularization parameter for LogisticRegression (default 1.0)
        use_logit: If True, transform predictions to logit space before fitting
        max_iter: Maximum iterations for solver
        random_state: Random seed for reproducibility
    """

    def __init__(
        self,
        C: float = 1.0,
        use_logit: bool = True,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.C = C
        self.use_logit = use_logit
        self.max_iter = max_iter
        self.random_state = random_state
        self._model = LogisticRegression(
            C=C, solver="lbfgs", max_iter=max_iter, random_state=random_state
        )

    def fit(
        self,
        y_pred: Iterable,
        y_true: Iterable,
        *,
        sample_weight: Optional[Iterable] = None,
    ) -> PlattCalibrator:
        """Fit the calibrator on predictions and true labels.

        Args:
            y_pred: Predicted probabilities (0-1)
            y_true: True binary labels (0/1)
            sample_weight: Optional sample weights

        Returns:
            Self
        """
        p = _to_numpy(y_pred).astype(float)
        y = _to_numpy(y_true).astype(float)
        x = logit(p) if self.use_logit else np.clip(p, 1e-12, 1 - 1e-12)
        X = x.reshape(-1, 1)
        sw = None if sample_weight is None else _to_numpy(sample_weight).astype(float)
        self._model.fit(X, y, sample_weight=sw)
        return self

    def predict_proba(self, y_pred: Iterable) -> np.ndarray:
        """Transform predictions to calibrated probabilities.

        Args:
            y_pred: Predicted probabilities (0-1)

        Returns:
            Calibrated probabilities
        """
        p = _to_numpy(y_pred).astype(float)
        x = logit(p) if self.use_logit else np.clip(p, 1e-12, 1 - 1e-12)
        X = x.reshape(-1, 1)
        prob = self._model.predict_proba(X)[:, 1]
        return np.clip(prob, 1e-12, 1 - 1e-12)


class IsotonicCalibrator:
    """Isotonic regression calibrator for monotonic calibration.

    Fits a non-parametric isotonic regression: P(y=1 | p) is non-decreasing in p.

    Attributes:
        y_min: Minimum output value
        y_max: Maximum output value
        out_of_bounds: How to handle out-of-bounds predictions ('clip' or 'nan')
    """

    def __init__(
        self, y_min: float = 0.0, y_max: float = 1.0, out_of_bounds: str = "clip"
    ) -> None:
        self._iso = IsotonicRegression(
            y_min=y_min, y_max=y_max, out_of_bounds=out_of_bounds
        )

    def fit(
        self,
        y_pred: Iterable,
        y_true: Iterable,
        *,
        sample_weight: Optional[Iterable] = None,
    ) -> IsotonicCalibrator:
        """Fit the calibrator on predictions and true labels.

        Args:
            y_pred: Predicted probabilities (0-1)
            y_true: True binary labels (0/1)
            sample_weight: Optional sample weights

        Returns:
            Self
        """
        p = _to_numpy(y_pred).astype(float)
        y = _to_numpy(y_true).astype(float)
        sw = None if sample_weight is None else _to_numpy(sample_weight).astype(float)
        self._iso.fit(p, y, sample_weight=sw)
        return self

    def predict_proba(self, y_pred: Iterable) -> np.ndarray:
        """Transform predictions to calibrated probabilities.

        Args:
            y_pred: Predicted probabilities (0-1)

        Returns:
            Calibrated probabilities
        """
        p = _to_numpy(y_pred).astype(float)
        prob = self._iso.transform(p)
        return np.clip(prob, 1e-12, 1 - 1e-12)


def fit_calibrator(
    y_pred: Iterable, y_true: Iterable, method: str = "isotonic"
) -> PlattCalibrator | IsotonicCalibrator:
    """Fit a binary probability calibrator.

    Args:
        y_pred: Predicted probabilities (0-1)
        y_true: True binary labels (0/1)
        method: 'platt' or 'isotonic'

    Returns:
        Fitted calibrator object
    """
    method = str(method or "isotonic").lower()
    if method == "platt":
        return PlattCalibrator(C=1.0, use_logit=True).fit(y_pred, y_true)
    return IsotonicCalibrator().fit(y_pred, y_true)


def quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Compute quantile bin edges for n_bins.

    Args:
        values: Calibrated probability values
        n_bins: Number of bins

    Returns:
        Array of length (n_bins-1) containing inner quantile edges
    """
    values = np.asarray(values, dtype=float)
    qs = np.linspace(0, 1, int(n_bins) + 1)[1:-1]
    edges = np.quantile(values, qs)
    return np.unique(edges)


def assign_bins(values: np.ndarray, edges: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign values to bins using quantile edges.

    Args:
        values: Calibrated probability values
        edges: Bin edges array (length n_bins-1)
        n_bins: Number of bins

    Returns:
        Array of bin indices (1 to n_bins)
    """
    values = np.asarray(values, dtype=float)
    if edges.size == 0:
        return np.ones_like(values, dtype=int)
    bins = np.concatenate(([-np.inf], np.asarray(edges, dtype=float), [np.inf]))
    return np.digitize(values, bins)


__all__ = [
    "PlattCalibrator",
    "IsotonicCalibrator",
    "fit_calibrator",
    "quantile_edges",
    "assign_bins",
    "logit",
    "inv_logit",
]

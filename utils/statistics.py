"""
RLShield Statistics Utilities
Rolling stats, EMA, Z-score, anomaly scoring.
"""

import numpy as np
from collections import deque
from typing import Optional


class RollingStats:
    """Efficient rolling mean and std over a fixed window."""

    def __init__(self, window: int = 1000):
        self.window = window
        self.data = deque(maxlen=window)

    def update(self, value: float):
        self.data.append(value)

    def mean(self) -> float:
        if not self.data:
            return 0.0
        return float(np.mean(self.data))

    def std(self) -> float:
        if len(self.data) < 2:
            return 1.0
        return float(np.std(self.data)) + 1e-8

    def z_score(self, value: float) -> float:
        return abs((value - self.mean()) / self.std())

    def clip_to_bounds(self, value: float, n_std: float = 3.0) -> float:
        mean = self.mean()
        std = self.std()
        return float(np.clip(value, mean - n_std * std, mean + n_std * std))

    @property
    def count(self) -> int:
        return len(self.data)

    def is_warm(self, min_samples: int = 30) -> bool:
        return len(self.data) >= min_samples


class EMA:
    """Exponential Moving Average filter."""

    def __init__(self, alpha: float = 0.99):
        self.alpha = alpha
        self._value: Optional[float] = None

    def update(self, value: float) -> float:
        if self._value is None:
            self._value = value
        else:
            self._value = self.alpha * self._value + (1 - self.alpha) * value
        return self._value

    @property
    def value(self) -> Optional[float]:
        return self._value

    def reset(self):
        self._value = None


class TrendDetector:
    """Detects monotonic trends in a time series using linear regression."""

    def __init__(self, window: int = 20):
        self.window = window
        self.history = deque(maxlen=window)

    def update(self, value: float):
        self.history.append(value)

    def trend_slope(self) -> float:
        """Returns slope of linear fit. Positive = increasing trend."""
        if len(self.history) < self.window:
            return 0.0
        x = np.arange(len(self.history))
        y = np.array(self.history)
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    def is_trending_up(self, threshold: float = 0.001) -> bool:
        return self.trend_slope() > threshold

    def is_trending_down(self, threshold: float = 0.001) -> bool:
        return self.trend_slope() < -threshold


def compute_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL divergence KL(p || q), numerically stable."""
    p = np.asarray(p, dtype=np.float64) + 1e-8
    q = np.asarray(q, dtype=np.float64) + 1e-8
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))
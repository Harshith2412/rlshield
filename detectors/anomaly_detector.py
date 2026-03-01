"""
RLShield Anomaly Detector
General-purpose statistical anomaly detector.
Can be applied to rewards, observations, actions, Q-values, losses, etc.
"""

import numpy as np
from typing import Any, Optional

from ..core.base_defender import BaseDefender
from ..core.alert_system import AlertSystem
from ..core.threat_model import AttackType, Severity
from ..utils.config import RLShieldConfig
from ..utils.statistics import RollingStats, TrendDetector


class AnomalyDetector(BaseDefender):
    """
    General statistical anomaly detector using Z-score + trend analysis.
    Can be attached to any scalar signal in the RL pipeline.

    Usage:
        detector = AnomalyDetector(config, alert_system, name="loss")
        detector.update(loss_value)  # returns True if anomaly
    """

    def __init__(
        self,
        config: RLShieldConfig,
        alert_system: AlertSystem,
        name: str = "signal",
        attack_type: AttackType = AttackType.UNKNOWN,
        z_threshold: Optional[float] = None,
    ):
        super().__init__(config, alert_system)
        self.name = name
        self.attack_type = attack_type
        self.z_threshold = z_threshold or config.anomaly_z_thresh

        self._stats = RollingStats(window=config.anomaly_window)
        self._trend = TrendDetector(window=30)
        self._anomaly_count = 0
        self._total = 0

    def update(self, value: float) -> bool:
        """
        Update with new value. Returns True if anomaly detected.
        """
        self.tick()
        self._total += 1
        value = float(value)

        self._trend.update(value)

        if not self._stats.is_warm(30):
            self._stats.update(value)
            return False

        z = self._stats.z_score(value)
        anomaly = z > self.z_threshold

        self._stats.update(value)

        if anomaly:
            self._anomaly_count += 1
            self._alert(
                self.attack_type,
                Severity.MEDIUM,
                {
                    "signal": self.name,
                    "value": round(value, 6),
                    "z_score": round(z, 4),
                    "threshold": self.z_threshold,
                    "rolling_mean": round(self._stats.mean(), 6),
                    "rolling_std": round(self._stats.std(), 6),
                },
            )
            return True

        return False

    def defend(self, data: Any) -> Any:
        return data

    def detect(self, value: Any = None) -> bool:
        if value is not None:
            return self.update(float(value))
        return False

    def get_cleaned_value(self, value: float) -> float:
        """Return z-score clipped version of value."""
        if not self._stats.is_warm(30):
            return value
        return self._stats.clip_to_bounds(float(value), n_std=self.z_threshold)

    @property
    def anomaly_rate(self) -> float:
        return self._anomaly_count / self._total if self._total > 0 else 0.0

    @property
    def stats(self) -> dict:
        base = super().stats
        base.update({
            "signal_name": self.name,
            "anomaly_count": self._anomaly_count,
            "anomaly_rate": round(self.anomaly_rate, 4),
            "rolling_mean": round(self._stats.mean(), 6),
            "rolling_std": round(self._stats.std(), 6),
        })
        return base


class GradientMonitor(BaseDefender):
    """
    Monitors gradient norms across training.
    Framework-agnostic: feed it gradient norm values directly.
    """

    def __init__(self, config: RLShieldConfig, alert_system: AlertSystem):
        super().__init__(config, alert_system)
        self._norm_stats = RollingStats(window=200)
        self._anomalous_steps = 0

    def update(self, grad_norm: float) -> bool:
        """
        Call with the total gradient norm at each step.
        Returns True if anomaly detected.
        """
        self.tick()
        grad_norm = float(grad_norm)
        threshold = self.config.max_grad_norm * self.config.grad_norm_multiplier

        self._norm_stats.update(grad_norm)

        if not self._norm_stats.is_warm(20):
            return False

        z = self._norm_stats.z_score(grad_norm)
        hard_violation = grad_norm > threshold

        if hard_violation or z > self.config.z_threshold * 2:
            self._anomalous_steps += 1
            self._alert(
                AttackType.GRADIENT_EXPLOSION,
                Severity.CRITICAL if hard_violation else Severity.HIGH,
                {
                    "grad_norm": round(grad_norm, 4),
                    "z_score": round(z, 4),
                    "threshold": round(threshold, 4),
                    "rolling_mean": round(self._norm_stats.mean(), 4),
                },
            )
            return True
        return False

    def defend(self, data: Any) -> Any:
        return data

    def detect(self, data: Any = None) -> bool:
        return False

    @property
    def stats(self) -> dict:
        base = super().stats
        base.update({"anomalous_steps": self._anomalous_steps})
        return base
"""
RLShield Policy Defender
General policy protection for DQN, SAC, TD3, DDPG, A2C, REINFORCE, TRPO, DreamerV3.
Handles gradient monitoring, entropy collapse, action anomalies.
"""

import numpy as np
from typing import Any, Optional

from ..core.base_defender import BaseDefender
from ..core.alert_system import AlertSystem
from ..core.threat_model import AttackType, Severity
from ..utils.config import RLShieldConfig
from ..utils.statistics import RollingStats, TrendDetector


class PolicyDefender(BaseDefender):
    """
    General-purpose policy defender for all non-PPO algorithms.

    Covers:
    - Gradient explosion / anomaly detection
    - Entropy collapse detection (SAC specific)
    - Action distribution monitoring
    - Q-value anomaly detection (DQN, SAC, TD3, DDPG)
    - Target network drift (TD3)
    """

    def __init__(self, config: RLShieldConfig, alert_system: AlertSystem):
        super().__init__(config, alert_system)
        self._grad_history = RollingStats(window=100)
        self._q_value_history = RollingStats(window=500)
        self._entropy_history = RollingStats(window=200)
        self._action_history = RollingStats(window=500)
        self._entropy_trend = TrendDetector(window=30)

    def defend(self, data: Any) -> Any:
        return data 
    def detect(self, data: Any = None) -> bool:
        return False


    def defend_gradients_torch(self, model, max_grad_norm: Optional[float] = None) -> bool:
        """
        Gradient protection for any PyTorch policy.
        Returns True if step should proceed.
        """
        try:
            import torch

            max_norm = max_grad_norm or self.config.max_grad_norm
            threshold = max_norm * self.config.grad_norm_multiplier

            norms = [
                p.grad.norm().item()
                for p in model.parameters()
                if p.grad is not None
            ]
            if not norms:
                return True

            total_norm = float(np.mean(norms))
            self._grad_history.update(total_norm)

            if self._grad_history.is_warm() and total_norm > threshold:
                self._alert(
                    AttackType.GRADIENT_EXPLOSION,
                    Severity.CRITICAL,
                    {
                        "grad_norm": round(total_norm, 4),
                        "threshold": round(threshold, 4),
                        "rolling_mean": round(self._grad_history.mean(), 4),
                    },
                )
                if self.config.grad_zero_on_alert:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.zero_()
                    return False

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            return True

        except ImportError:
            raise RuntimeError("PyTorch required for defend_gradients_torch()")


    def monitor_q_values(self, q_values: np.ndarray) -> bool:
        """
        Track Q-value distribution. Alerts if Q-values explode or collapse.
        Returns True if anomaly detected.
        """
        q_mean = float(np.mean(q_values))
        self._q_value_history.update(q_mean)

        if not self._q_value_history.is_warm(50):
            return False

        z = self._q_value_history.z_score(q_mean)
        if z > self.config.z_threshold * 1.5:
            self._alert(
                AttackType.GRADIENT_ANOMALY,
                Severity.HIGH,
                {
                    "q_value_mean": round(q_mean, 4),
                    "z_score": round(z, 4),
                    "rolling_mean": round(self._q_value_history.mean(), 4),
                },
            )
            return True
        return False


    def monitor_entropy(self, entropy: float) -> bool:
        """
        Detect entropy collapse — a sign of temperature/alpha manipulation.
        Low entropy = deterministic policy = more vulnerable.
        Returns True if collapse detected.
        """
        self._entropy_history.update(entropy)
        self._entropy_trend.update(entropy)

        if not self._entropy_history.is_warm(30):
            return False

        if entropy < 1e-4:
            self._alert(
                AttackType.ENTROPY_COLLAPSE,
                Severity.HIGH,
                {"entropy": entropy, "reason": "near_zero_entropy"},
            )
            return True

        if self._entropy_trend.is_trending_down(threshold=0.01):
            self._alert(
                AttackType.ENTROPY_COLLAPSE,
                Severity.MEDIUM,
                {
                    "entropy_slope": round(self._entropy_trend.trend_slope(), 6),
                    "current_entropy": round(entropy, 6),
                    "reason": "entropy_decreasing_trend",
                },
            )
            return True

        return False


    def monitor_actions(self, action: np.ndarray) -> bool:
        """
        Track action norms over time. Flags if actions suddenly shift.
        Returns True if anomaly detected.
        """
        action = np.asarray(action, dtype=np.float32)
        norm = float(np.linalg.norm(action))
        self._action_history.update(norm)

        if not self._action_history.is_warm(50):
            return False

        z = self._action_history.z_score(norm)
        if z > self.config.z_threshold:
            self._alert(
                AttackType.POLICY_DRIFT,
                Severity.MEDIUM,
                {
                    "action_norm": round(norm, 4),
                    "z_score": round(z, 4),
                    "rolling_mean": round(self._action_history.mean(), 4),
                },
            )
            return True
        return False
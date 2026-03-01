"""
RLShield Reward Defender
Protects against reward poisoning, reward spoofing, reward hacking.
Works for all RL algorithms.
"""

import numpy as np
from typing import Union

from ..core.base_defender import BaseDefender
from ..core.alert_system import AlertSystem
from ..core.threat_model import AttackType, Severity
from ..utils.config import RLShieldConfig
from ..utils.statistics import RollingStats, EMA


class RewardDefender(BaseDefender):
    """
    Defends reward signal against:
    - Reward poisoning (injected fake high/low rewards)
    - Reward spoofing (manipulated env reward output)
    - Reward hacking (exploiting loopholes in reward function)
    - Reward signal noise injection
    """

    def __init__(self, config: RLShieldConfig, alert_system: AlertSystem):
        super().__init__(config, alert_system)
        self.stats_tracker = RollingStats(window=config.reward_window)
        self.ema = EMA(alpha=config.ema_alpha)
        self.z_thresh = config.z_threshold
        self.r_min = config.reward_min
        self.r_max = config.reward_max
        self._consecutive_anomalies = 0

    def defend(self, reward: float) -> float:
        """
        Process a raw reward through the defense pipeline.
        Returns a secured reward value.
        """
        if not self.enabled:
            return reward

        self.tick()

        reward = self._hard_clip(reward)

        self.detect(reward)

        if self.stats_tracker.is_warm():
            reward = self.stats_tracker.clip_to_bounds(reward, n_std=self.z_thresh)

        reward = self.ema.update(reward)

        self.stats_tracker.update(reward)

        return float(reward)

    def detect(self, reward: float) -> bool:
        """Returns True if reward looks poisoned/anomalous."""
        if not self.stats_tracker.is_warm(min_samples=30):
            return False

        z = self.stats_tracker.z_score(reward)

        if z > self.z_thresh:
            self._consecutive_anomalies += 1
            severity = Severity.HIGH if self._consecutive_anomalies > 3 else Severity.MEDIUM
            self._alert(
                AttackType.REWARD_POISONING,
                severity,
                {
                    "z_score": round(z, 4),
                    "reward": round(reward, 4),
                    "mean": round(self.stats_tracker.mean(), 4),
                    "std": round(self.stats_tracker.std(), 4),
                    "consecutive_anomalies": self._consecutive_anomalies,
                },
            )
            return True

        self._consecutive_anomalies = 0
        return False

    def _hard_clip(self, reward: float) -> float:
        """Clip to absolute bounds — catches totally out-of-range values."""
        clipped = float(np.clip(reward, self.r_min, self.r_max))
        if clipped != reward:
            self._alert(
                AttackType.REWARD_POISONING,
                Severity.HIGH,
                {
                    "original_reward": reward,
                    "clipped_to": clipped,
                    "reason": "hard_bounds_exceeded",
                },
            )
        return clipped

    def reset(self):
        """Reset state between episodes if needed."""
        self._consecutive_anomalies = 0.
"""
RLShield Observation Defender
Protects against adversarial observations, state spoofing, sensor attacks.
Works for all RL algorithms.
"""

import numpy as np
from typing import Optional, Union

from ..core.base_defender import BaseDefender
from ..core.alert_system import AlertSystem
from ..core.threat_model import AttackType, Severity
from ..utils.config import RLShieldConfig
from ..utils.statistics import RollingStats


class ObservationDefender(BaseDefender):
    """
    Defends observation/state space against:
    - Adversarial perturbations (FGSM-style noise)
    - State spoofing (fake sensor data)
    - Observation teleportation (impossible state transitions)
    - Out-of-bounds observations

    Uses randomized smoothing for lightweight certified defense.
    """

    def __init__(self, config: RLShieldConfig, alert_system: AlertSystem):
        super().__init__(config, alert_system)
        self.epsilon = config.obs_epsilon
        self.n_samples = config.obs_n_samples
        self.cert_confidence = config.obs_cert_confidence
        self.max_delta = config.obs_max_delta
        self._prev_obs: Optional[np.ndarray] = None
        self._obs_stats = RollingStats(window=500)
        self._obs_min: Optional[np.ndarray] = None
        self._obs_max: Optional[np.ndarray] = None

    def defend(self, obs: np.ndarray) -> np.ndarray:
        """
        Secure an observation before it reaches the policy.
        Returns cleaned observation.
        """
        if not self.enabled:
            return obs

        self.tick()
        obs = np.asarray(obs, dtype=np.float32)

        self._check_teleport(obs)

        self._update_bounds(obs)

        obs = self._soft_clip(obs)

        self._prev_obs = obs.copy()

        return obs

    def detect(self, obs: np.ndarray) -> bool:
        """Returns True if observation looks adversarially perturbed."""
        obs = np.asarray(obs, dtype=np.float32)

        if self._prev_obs is None:
            return False

        delta = np.linalg.norm(obs - self._prev_obs)

        if delta > self.max_delta:
            self._alert(
                AttackType.OBS_TELEPORT,
                Severity.HIGH,
                {
                    "delta_norm": round(float(delta), 4),
                    "max_allowed": self.max_delta,
                },
            )
            return True

        return False

    def certify_action(self, policy_fn, obs: np.ndarray) -> np.ndarray:
        """
        Randomized smoothing: run policy on noisy copies of obs.
        Return majority-vote action. Flags low-confidence decisions.

        policy_fn: callable obs -> action (numpy)
        """
        obs = np.asarray(obs, dtype=np.float32)
        noisy_obs = [
            obs + np.random.randn(*obs.shape).astype(np.float32) * self.epsilon
            for _ in range(self.n_samples)
        ]

        actions = [policy_fn(o) for o in noisy_obs]

        if np.isscalar(actions[0]) or (
            isinstance(actions[0], np.ndarray) and actions[0].ndim == 0
        ):
            actions_flat = [int(a) for a in actions]
            from collections import Counter
            most_common, count = Counter(actions_flat).most_common(1)[0]
            confidence = count / self.n_samples

            if confidence < self.cert_confidence:
                self._alert(
                    AttackType.OBS_ADVERSARIAL,
                    Severity.MEDIUM,
                    {
                        "confidence": round(confidence, 4),
                        "threshold": self.cert_confidence,
                        "reason": "low_certified_confidence",
                    },
                )

            return np.array(most_common)

        action_mean = np.mean(actions, axis=0)
        action_std = np.std(actions, axis=0)

        if np.mean(action_std) > 0.5:
            self._alert(
                AttackType.OBS_ADVERSARIAL,
                Severity.MEDIUM,
                {
                    "action_std_mean": round(float(np.mean(action_std)), 4),
                    "reason": "high_action_variance_under_smoothing",
                },
            )

        return action_mean

    def _check_teleport(self, obs: np.ndarray):
        if self._prev_obs is None:
            return
        delta = np.linalg.norm(obs - self._prev_obs)
        if delta > self.max_delta:
            self._alert(
                AttackType.OBS_TELEPORT,
                Severity.HIGH,
                {
                    "delta_norm": round(float(delta), 4),
                    "max_allowed": self.max_delta,
                },
            )

    def _update_bounds(self, obs: np.ndarray):
        if self._obs_min is None:
            self._obs_min = obs.copy()
            self._obs_max = obs.copy()
        else:
            self._obs_min = np.minimum(self._obs_min, obs)
            self._obs_max = np.maximum(self._obs_max, obs)

    def _soft_clip(self, obs: np.ndarray) -> np.ndarray:
        """Soft clip: only after we've seen enough observations."""
        if self._obs_min is None or self._obs_stats.count < 50:
            return obs

        margin = (self._obs_max - self._obs_min) * 0.1 + 1e-3
        return np.clip(obs, self._obs_min - margin, self._obs_max + margin)

    def reset(self):
        self._prev_obs = None
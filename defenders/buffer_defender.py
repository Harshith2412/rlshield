import numpy as np
from typing import Any, Optional, Tuple

from ..core.base_defender import BaseDefender
from ..core.alert_system import AlertSystem
from ..core.threat_model import AttackType, Severity
from ..utils.config import RLShieldConfig
from ..utils.statistics import RollingStats


class BufferDefender(BaseDefender):
    """
    Validates transitions before they are stored in replay / rollout buffers.

    Detects:
    - Impossible state transitions (teleportation)
    - Out-of-bound rewards
    - Duplicate / cloned transitions (replay injection)
    - Action space violations
    """

    def __init__(
        self,
        config: RLShieldConfig,
        alert_system: AlertSystem,
        action_space=None,      
        obs_space=None,         
    ):
        super().__init__(config, alert_system)
        self.action_space = action_space
        self.obs_space = obs_space

        self._reward_stats = RollingStats(window=2000)
        self._transition_hashes = set()   
        self._rejected = 0
        self._accepted = 0

    def defend(self, transition: Tuple) -> Optional[Tuple]:
        """
        Validate a (s, a, r, s', done) transition.
        Returns the transition if valid, None if rejected.
        """
        if not self.enabled:
            return transition

        self.tick()

        try:
            s, a, r, s_next, done = transition
        except (ValueError, TypeError):
            self._rejected += 1
            self._alert(
                AttackType.BUFFER_INJECTION,
                Severity.LOW,
                {"reason": "malformed_transition"},
            )
            return None

        if not self._check_reward(r):
            self._rejected += 1
            return None

        if not self._check_transition(s, s_next):
            self._rejected += 1
            return None

        if not self._check_action(a):
            self._rejected += 1
            return None

        if not self._check_duplicate(s, a, r):
            self._rejected += 1
            return None

        self._reward_stats.update(float(r))
        self._accepted += 1

        return transition

    def detect(self, data: Any = None) -> bool:
        return False

    def _check_reward(self, r: float) -> bool:
        r = float(r)
        if not (self.config.buffer_reward_min <= r <= self.config.buffer_reward_max):
            self._alert(
                AttackType.BUFFER_INJECTION,
                Severity.HIGH,
                {
                    "reward": r,
                    "min": self.config.buffer_reward_min,
                    "max": self.config.buffer_reward_max,
                    "reason": "reward_out_of_bounds",
                },
            )
            return False

        if self._reward_stats.is_warm(30):
            z = self._reward_stats.z_score(r)
            if z > self.config.z_threshold * 1.5:
                self._alert(
                    AttackType.REPLAY_POISONING,
                    Severity.HIGH,
                    {"reward": round(r, 4), "z_score": round(z, 4), "reason": "reward_zscore_violation"},
                )
                return False

        return True

    def _check_transition(self, s: np.ndarray, s_next: np.ndarray) -> bool:
        try:
            s = np.asarray(s, dtype=np.float32)
            s_next = np.asarray(s_next, dtype=np.float32)
        except Exception:
            return True 

        if s.shape != s_next.shape:
            self._alert(
                AttackType.BUFFER_INJECTION,
                Severity.HIGH,
                {"reason": "state_shape_mismatch", "s_shape": str(s.shape), "s_next_shape": str(s_next.shape)},
            )
            return False

        delta = float(np.linalg.norm(s_next - s))
        if delta > self.config.buffer_max_state_delta:
            self._alert(
                AttackType.BUFFER_INJECTION,
                Severity.HIGH,
                {
                    "delta_norm": round(delta, 4),
                    "max_allowed": self.config.buffer_max_state_delta,
                    "reason": "impossible_state_transition",
                },
            )
            return False

        return True

    def _check_action(self, a) -> bool:
        if not self.config.buffer_action_check:
            return True
        if self.action_space is None:
            return True

        try:
            if hasattr(self.action_space, "contains"):
                if not self.action_space.contains(np.asarray(a)):
                    self._alert(
                        AttackType.BUFFER_INJECTION,
                        Severity.MEDIUM,
                        {"action": str(a), "reason": "action_outside_space"},
                    )
                    return False
        except Exception:
            pass

        return True

    def _check_duplicate(self, s, a, r: float) -> bool:
        """Simple hash-based duplicate detection."""
        try:
            s_arr = np.asarray(s, dtype=np.float32)
            key = hash((s_arr.tobytes(), str(a), round(r, 4)))
        except Exception:
            return True

        if key in self._transition_hashes:
            self._alert(
                AttackType.REPLAY_POISONING,
                Severity.MEDIUM,
                {"reason": "duplicate_transition_detected"},
            )
            return True

        if len(self._transition_hashes) > 10000:
            self._transition_hashes = set(list(self._transition_hashes)[-5000:])

        self._transition_hashes.add(key)
        return True

    @property
    def rejection_rate(self) -> float:
        total = self._accepted + self._rejected
        return self._rejected / total if total > 0 else 0.0

    @property
    def stats(self) -> dict:
        base = super().stats
        base.update({
            "accepted": self._accepted,
            "rejected": self._rejected,
            "rejection_rate": round(self.rejection_rate, 4),
        })
        return base
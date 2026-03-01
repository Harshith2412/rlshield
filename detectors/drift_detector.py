"""
RLShield Drift Detector
Monitors policy behavioral drift over training by snapshotting
action distributions on fixed probe states.
Triggers rollback on significant drift.
"""

import numpy as np
from typing import Any, Callable, List, Optional

from ..core.base_defender import BaseDefender
from ..core.alert_system import AlertSystem
from ..core.threat_model import AttackType, Severity
from ..utils.config import RLShieldConfig
from ..utils.statistics import compute_kl_divergence
from ..utils.snapshot import SnapshotManager


class DriftDetector(BaseDefender):
    """
    Takes snapshots of a policy's action distribution on fixed probe states.
    Detects gradual policy drift (which can be invisible to PPO clip alone).

    Usage:
        detector = DriftDetector(config, alert_system, probe_states)
        # In training loop:
        rollback_needed = detector.update(policy_fn, policy_model, step)
    """

    def __init__(
        self,
        config: RLShieldConfig,
        alert_system: AlertSystem,
        probe_states: Optional[np.ndarray] = None,
    ):
        super().__init__(config, alert_system)
        self.probe_states = probe_states
        self.snapshot_interval = config.snapshot_interval
        self.drift_threshold = config.drift_threshold
        self.snapshot_manager = SnapshotManager(max_snapshots=config.max_snapshots)

        self._action_snapshots: List[np.ndarray] = []
        self._drift_history: List[float] = []
        self._rollbacks = 0

    def set_probe_states(self, probe_states: np.ndarray):
        """Set probe states after initialization."""
        self.probe_states = probe_states

    def update(
        self,
        policy_fn: Callable,
        policy_model: Any,
        step: int,
    ) -> bool:
        """
        Call this in your training loop at each step.

        Args:
            policy_fn:    callable (obs -> action), used to get action distributions
            policy_model: the policy object with state_dict() for snapshot saving
            step:         current training step

        Returns:
            True if rollback was triggered, False otherwise.
        """
        if not self.enabled or step % self.snapshot_interval != 0:
            return False

        if self.probe_states is None:
            return False

        self.tick()

        # Get current action distribution on probe states
        try:
            current_actions = self._get_action_distribution(policy_fn)
        except Exception:
            return False

        # If we have prior snapshots, check drift
        if self._action_snapshots:
            drift = self._measure_drift(self._action_snapshots[-1], current_actions)
            self._drift_history.append(drift)

            if drift > self.drift_threshold:
                self._alert(
                    AttackType.POLICY_DRIFT,
                    Severity.HIGH,
                    {
                        "drift_kl": round(drift, 6),
                        "threshold": self.drift_threshold,
                        "step": step,
                    },
                )

                if self.config.auto_rollback and self.snapshot_manager.count() >= 2:
                    success = self.snapshot_manager.rollback(policy_model, steps_back=2)
                    if success:
                        self._rollbacks += 1
                        self._alert(
                            AttackType.POLICY_DRIFT,
                            Severity.CRITICAL,
                            {
                                "action": "rollback_executed",
                                "rollback_count": self._rollbacks,
                                "step": step,
                            },
                        )
                        self._action_snapshots = self._action_snapshots[:-1]
                        return True

        self._action_snapshots.append(current_actions)
        if len(self._action_snapshots) > self.config.max_snapshots:
            self._action_snapshots.pop(0)

        self.snapshot_manager.save(policy_model, step)
        return False

    def defend(self, data: Any) -> Any:
        return data

    def detect(self, data: Any = None) -> bool:
        if len(self._drift_history) < 2:
            return False
        recent_drift = np.mean(self._drift_history[-5:])
        return recent_drift > self.drift_threshold

    def _get_action_distribution(self, policy_fn: Callable) -> np.ndarray:
        """Get policy's output for all probe states."""
        results = []
        for s in self.probe_states:
            try:
                action = policy_fn(s)
                if hasattr(action, "detach"):
                    action = action.detach().cpu().numpy()
                results.append(np.asarray(action, dtype=np.float32).flatten())
            except Exception:
                results.append(np.zeros(1, dtype=np.float32))
        return np.array(results)

    def _measure_drift(self, old_dist: np.ndarray, new_dist: np.ndarray) -> float:
        """
        Measure behavioral drift between two action distributions.
        Uses normalized L2 distance (works for both discrete and continuous).
        """
        old_flat = old_dist.flatten().astype(np.float64)
        new_flat = new_dist.flatten().astype(np.float64)

        # Normalize to same scale
        old_norm = np.linalg.norm(old_flat)
        new_norm = np.linalg.norm(new_flat)

        if old_norm < 1e-8 or new_norm < 1e-8:
            return 0.0

        old_normalized = old_flat / old_norm
        new_normalized = new_flat / new_norm

        return float(np.linalg.norm(old_normalized - new_normalized))

    @property
    def stats(self) -> dict:
        base = super().stats
        base.update({
            "rollbacks": self._rollbacks,
            "snapshots_saved": self.snapshot_manager.count(),
            "avg_drift": round(float(np.mean(self._drift_history)) if self._drift_history else 0.0, 6),
            "max_drift": round(float(np.max(self._drift_history)) if self._drift_history else 0.0, 6),
        })
        return base
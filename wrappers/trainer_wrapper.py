"""
RLShield Trainer Wrapper
Hooks into any RL training loop to secure the update process.
Provides before/after update hooks for PPO, DQN, SAC, TD3, etc.
"""

import numpy as np
from typing import Any, Callable, Optional

from ..defenders.ppo_defender import PPODefender
from ..defenders.policy_defender import PolicyDefender
from ..defenders.buffer_defender import BufferDefender
from ..detectors.drift_detector import DriftDetector
from ..detectors.anomaly_detector import AnomalyDetector, GradientMonitor, AttackType
from ..core.alert_system import AlertSystem
from ..utils.config import RLShieldConfig


class SecureTrainerWrapper:
    """
    Wraps a training loop / trainer object.
    Provides secure hooks for:
    - PPO: before update, gradient defense
    - DQN/SAC/TD3: Q-value monitoring, gradient defense
    - All: loss anomaly detection, drift detection
    """

    def __init__(
        self,
        trainer: Any,
        config: RLShieldConfig,
        alert_system: AlertSystem,
        ppo_defender: Optional[PPODefender] = None,
        policy_defender: Optional[PolicyDefender] = None,
        drift_detector: Optional[DriftDetector] = None,
        buffer_defender: Optional[BufferDefender] = None,
    ):
        self.trainer = trainer
        self.config = config
        self.alert_system = alert_system

        self.ppo_defender = ppo_defender
        self.policy_defender = policy_defender
        self.drift_detector = drift_detector
        self.buffer_defender = buffer_defender

        self._loss_monitor = AnomalyDetector(
            config, alert_system,
            name="loss",
            attack_type=AttackType.GRADIENT_ANOMALY,
            z_threshold=4.0,
        )
        self._grad_monitor = GradientMonitor(config, alert_system)
        self._step = 0

    # ── PPO Hooks ─────────────────────────────────────────────────

    def ppo_before_update(self, old_log_probs, new_log_probs, advantages):
        """
        Call before each PPO gradient update.
        Returns secure loss or None (skip update).
        """
        if self.ppo_defender is None:
            return None
        return self.ppo_defender.defend_update_numpy(
            np.asarray(old_log_probs),
            np.asarray(new_log_probs),
            np.asarray(advantages),
        )

    def ppo_before_update_torch(self, old_log_probs, new_log_probs, advantages):
        """PyTorch version of PPO update hook."""
        if self.ppo_defender is None:
            return None
        return self.ppo_defender.defend_update_torch(old_log_probs, new_log_probs, advantages)

    def after_backward(self, model) -> bool:
        """
        Call after loss.backward(), before optimizer.step().
        Returns True if optimizer.step() should proceed.
        """
        self._step += 1

        # PPO-specific gradient defense
        if self.config.algo == "PPO" and self.ppo_defender:
            return self.ppo_defender.defend_gradients_torch(model)

        # General gradient defense
        if self.policy_defender:
            return self.policy_defender.defend_gradients_torch(model)

        return True

    # ── General Hooks ─────────────────────────────────────────────

    def on_loss(self, loss_value: float) -> bool:
        """Monitor training loss for anomalies. Returns True if anomaly."""
        return self._loss_monitor.update(loss_value)

    def on_grad_norm(self, grad_norm: float) -> bool:
        """Monitor gradient norm. Returns True if anomaly."""
        return self._grad_monitor.update(grad_norm)

    def on_q_values(self, q_values) -> bool:
        """Monitor Q-values (DQN, SAC, TD3, DDPG). Returns True if anomaly."""
        if self.policy_defender:
            return self.policy_defender.monitor_q_values(np.asarray(q_values))
        return False

    def on_entropy(self, entropy: float) -> bool:
        """Monitor policy entropy (SAC). Returns True if anomaly."""
        if self.policy_defender:
            return self.policy_defender.monitor_entropy(entropy)
        return False

    def on_transition(self, s, a, r, s_next, done) -> Optional[tuple]:
        """Validate a transition before adding to buffer."""
        if self.buffer_defender:
            return self.buffer_defender.defend((s, a, r, s_next, done))
        return (s, a, r, s_next, done)

    def on_step(self, policy_fn, policy_model, step: int) -> bool:
        """
        Call once per training step for drift detection.
        Returns True if rollback was triggered.
        """
        if self.drift_detector and self.drift_detector.probe_states is not None:
            return self.drift_detector.update(policy_fn, policy_model, step)
        return False

    def check_ppo_drift(self) -> bool:
        """Periodic PPO KL trend check. Returns True if drift detected."""
        if self.ppo_defender:
            return self.ppo_defender.detect()
        return False

    # ── Passthrough ───────────────────────────────────────────────

    def __getattr__(self, name: str):
        return getattr(self.trainer, name)
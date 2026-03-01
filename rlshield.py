"""
RLShield — Reinforcement Learning Security Library
===================================================
Plug-and-play security hardening for RL algorithms.
Supports: PPO, DQN, SAC, TD3, DDPG, A2C, A3C, REINFORCE, TRPO, DreamerV3

Quick Start:
    import rlshield

    env     = rlshield.protect_env(env)
    policy  = rlshield.protect_policy(policy, algo="PPO")
    trainer = rlshield.protect_trainer(trainer, algo="PPO")

Or with full control:
    from rlshield import RLShield

    shield = RLShield(algo="PPO", threat_level="high")
    env    = shield.protect_env(env)
    policy = shield.protect_policy(policy)
    report = shield.get_threat_report()
"""

import numpy as np
from typing import Any, Callable, Optional

from .utils.config import RLShieldConfig
from .utils.snapshot import SnapshotManager
from .core.alert_system import AlertSystem, SecurityAlertException
from .core.threat_model import AttackType, Severity, ThreatEvent
from .defenders.reward_defender import RewardDefender
from .defenders.observation_defender import ObservationDefender
from .defenders.ppo_defender import PPODefender
from .defenders.policy_defender import PolicyDefender
from .defenders.buffer_defender import BufferDefender
from .detectors.drift_detector import DriftDetector
from .detectors.anomaly_detector import AnomalyDetector, GradientMonitor
from .wrappers.env_wrapper import SecureEnvWrapper
from .wrappers.policy_wrapper import SecurePolicyWrapper
from .wrappers.trainer_wrapper import SecureTrainerWrapper


class RLShield:
    """
    Master security controller for RL pipelines.

    Args:
        algo:          RL algorithm name. PPO | DQN | SAC | TD3 | DDPG | A2C | A3C | REINFORCE | TRPO | DreamerV3
        threat_level:  Sensitivity preset. low | medium | high
        alert_mode:    How to handle threats. log | raise | callback | silent
        auto_rollback: Automatically roll back policy on critical drift.
        probe_states:  Fixed states for drift detection (numpy array).
        callback:      Function called with ThreatEvent when alert fires (alert_mode='callback').
        config:        Override with a custom RLShieldConfig (ignores other params).
    """

    SUPPORTED_ALGOS = {
        "PPO", "DQN", "SAC", "TD3", "DDPG",
        "A2C", "A3C", "REINFORCE", "TRPO", "DreamerV3"
    }

    def __init__(
        self,
        algo: str = "PPO",
        threat_level: str = "medium",
        alert_mode: str = "log",
        auto_rollback: bool = True,
        probe_states: Optional[np.ndarray] = None,
        callback: Optional[Callable] = None,
        config: Optional[RLShieldConfig] = None,
    ):
        if algo not in self.SUPPORTED_ALGOS:
            raise ValueError(f"Unknown algo '{algo}'. Supported: {self.SUPPORTED_ALGOS}")


        if config is not None:
            self.config = config
        else:
            self.config = RLShieldConfig.from_threat_level(threat_level, algo=algo)
            self.config.alert_mode = alert_mode
            self.config.auto_rollback = auto_rollback
            self.config.alert_callback = callback


        self.alert_system = AlertSystem(
            mode=self.config.alert_mode,
            callback=self.config.alert_callback,
        )


        self.reward_defender = RewardDefender(self.config, self.alert_system)
        self.obs_defender = ObservationDefender(self.config, self.alert_system)
        self.buffer_defender = BufferDefender(self.config, self.alert_system)

        if algo == "PPO":
            self.policy_defender = PPODefender(self.config, self.alert_system)
        else:
            self.policy_defender = PolicyDefender(self.config, self.alert_system)

        self.drift_detector = DriftDetector(
            self.config, self.alert_system, probe_states=probe_states
        )

        self.loss_monitor = AnomalyDetector(
            self.config, self.alert_system,
            name="training_loss",
            attack_type=AttackType.GRADIENT_ANOMALY,
        )
        self.grad_monitor = GradientMonitor(self.config, self.alert_system)

        self.algo = algo
        self._protected_env = None
        self._protected_policy = None
        self._protected_trainer = None


    def protect_env(self, env: Any) -> SecureEnvWrapper:
        """
        Wrap a Gym/Gymnasium environment with security layer.
        Returns a drop-in replacement for your env.
        """
        wrapped = SecureEnvWrapper(env, self.config, self.alert_system)
        wrapped.reward_defender = self.reward_defender
        wrapped.obs_defender = self.obs_defender
        self._protected_env = wrapped
        return wrapped

    def protect_policy(
        self,
        policy: Any,
        use_certification: bool = False,
    ) -> SecurePolicyWrapper:
        """
        Wrap a policy with transparent security.

        Args:
            policy:            Any callable policy (PyTorch, SB3, etc.)
            use_certification: Enable randomized smoothing (slight overhead).
        """
        wrapped = SecurePolicyWrapper(
            policy=policy,
            config=self.config,
            alert_system=self.alert_system,
            obs_defender=self.obs_defender,
            policy_defender=self.policy_defender if isinstance(self.policy_defender, PolicyDefender) else None,
            drift_detector=self.drift_detector,
            use_certification=use_certification,
        )
        self._protected_policy = wrapped
        return wrapped

    def protect_trainer(self, trainer: Any) -> SecureTrainerWrapper:
        """
        Wrap a trainer with security hooks.
        Use the returned wrapper's hooks in your training loop.
        """
        ppo_def = self.policy_defender if isinstance(self.policy_defender, PPODefender) else None
        gen_def = self.policy_defender if isinstance(self.policy_defender, PolicyDefender) else None

        wrapped = SecureTrainerWrapper(
            trainer=trainer,
            config=self.config,
            alert_system=self.alert_system,
            ppo_defender=ppo_def,
            policy_defender=gen_def,
            drift_detector=self.drift_detector,
            buffer_defender=self.buffer_defender,
        )
        self._protected_trainer = wrapped
        return wrapped

    def set_probe_states(self, probe_states: np.ndarray):
        """Set probe states for drift detection after init."""
        self.drift_detector.set_probe_states(probe_states)


    def defend_reward(self, reward: float) -> float:
        """Directly secure a reward value."""
        return self.reward_defender.defend(reward)

    def defend_observation(self, obs: np.ndarray) -> np.ndarray:
        """Directly secure an observation."""
        return self.obs_defender.defend(obs)

    def defend_transition(self, s, a, r, s_next, done):
        """Validate and secure a buffer transition."""
        return self.buffer_defender.defend((s, a, r, s_next, done))


    def ppo_secure_update(self, old_log_probs, new_log_probs, advantages):
        """Directly call PPO update defense (numpy)."""
        if not isinstance(self.policy_defender, PPODefender):
            raise RuntimeError("ppo_secure_update only available when algo='PPO'")
        return self.policy_defender.defend_update_numpy(
            np.asarray(old_log_probs),
            np.asarray(new_log_probs),
            np.asarray(advantages),
        )

    def ppo_secure_update_torch(self, old_log_probs, new_log_probs, advantages):
        """Directly call PPO update defense (PyTorch)."""
        if not isinstance(self.policy_defender, PPODefender):
            raise RuntimeError("ppo_secure_update_torch only available when algo='PPO'")
        return self.policy_defender.defend_update_torch(old_log_probs, new_log_probs, advantages)

    def ppo_secure_gradients(self, model) -> bool:
        """Directly call PPO gradient defense."""
        if not isinstance(self.policy_defender, PPODefender):
            raise RuntimeError("ppo_secure_gradients only available when algo='PPO'")
        return self.policy_defender.defend_gradients_torch(model)


    def get_threat_report(self) -> dict:
        """Full threat report from the current session."""
        return self.alert_system.get_report()

    def get_component_stats(self) -> dict:
        """Stats from all active defenders."""
        return {
            "reward_defender": self.reward_defender.stats,
            "obs_defender": self.obs_defender.stats,
            "buffer_defender": self.buffer_defender.stats,
            "policy_defender": self.policy_defender.stats,
            "drift_detector": self.drift_detector.stats,
            "loss_monitor": self.loss_monitor.stats,
            "grad_monitor": self.grad_monitor.stats,
        }

    def print_summary(self):
        """Print a human-readable security summary."""
        report = self.get_threat_report()
        stats = self.get_component_stats()

        print("\n" + "=" * 60)
        print("         RLShield Security Summary")
        print("=" * 60)
        print(f"  Algorithm     : {self.algo}")
        print(f"  Threat Level  : {self.config.threat_level}")
        print(f"  Total Alerts  : {report['total']}")

        if report["total"] > 0:
            print("\n  Alerts by Type:")
            for attack_type, count in report["summary"].items():
                print(f"    {attack_type:<35} {count}")

        print("\n  Component Stats:")
        for component, s in stats.items():
            alerts = s.get("alerts_fired", 0)
            if alerts > 0:
                print(f"    {component:<30} {alerts} alerts fired")

        rollbacks = stats["drift_detector"].get("rollbacks", 0)
        if rollbacks > 0:
            print(f"\n  Policy Rollbacks Triggered: {rollbacks}")



    def disable_reward_defense(self):
        self.reward_defender.disable()

    def disable_obs_defense(self):
        self.obs_defender.disable()

    def disable_drift_detection(self):
        self.drift_detector.disable()

    def enable_all(self):
        for d in [self.reward_defender, self.obs_defender,
                  self.buffer_defender, self.policy_defender,
                  self.drift_detector]:
            d.enable()


def protect_env(env: Any, algo: str = "PPO", threat_level: str = "medium") -> SecureEnvWrapper:
    """Wrap an environment. Minimal setup."""
    shield = RLShield(algo=algo, threat_level=threat_level)
    return shield.protect_env(env)


def protect_policy(
    policy: Any,
    algo: str = "PPO",
    threat_level: str = "medium",
    use_certification: bool = False,
) -> SecurePolicyWrapper:
    """Wrap a policy. Minimal setup."""
    shield = RLShield(algo=algo, threat_level=threat_level)
    return shield.protect_policy(policy, use_certification=use_certification)


def protect_trainer(
    trainer: Any,
    algo: str = "PPO",
    threat_level: str = "medium",
) -> SecureTrainerWrapper:
    """Wrap a trainer. Minimal setup."""
    shield = RLShield(algo=algo, threat_level=threat_level)
    return shield.protect_trainer(trainer)
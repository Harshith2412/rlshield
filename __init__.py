"""
RLShield — Reinforcement Learning Security Library
===================================================
Plug-and-play security hardening for RL training pipelines.

Supported Algorithms: PPO, DQN, SAC, TD3, DDPG, A2C, A3C, REINFORCE, TRPO, DreamerV3

Quick Start:
    import rlshield

    # 3-line API
    env     = rlshield.protect_env(env, algo="PPO")
    policy  = rlshield.protect_policy(policy, algo="PPO")
    trainer = rlshield.protect_trainer(trainer, algo="PPO")

    # Full API
    from rlshield import RLShield
    shield  = RLShield(algo="PPO", threat_level="high")
    env     = shield.protect_env(env)
    policy  = shield.protect_policy(policy)
    report  = shield.get_threat_report()
"""

from .rlshield import RLShield, protect_env, protect_policy, protect_trainer

from .utils.config import RLShieldConfig
from .core.threat_model import AttackType, Severity, ThreatEvent
from .core.alert_system import AlertSystem, SecurityAlertException

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

__version__ = "0.1.0"
__author__ = "Harshith Madhavaram"

__all__ = [
    "RLShield",
    "protect_env",
    "protect_policy",
    "protect_trainer",
    "RLShieldConfig",
    "AttackType",
    "Severity",
    "ThreatEvent",
    "AlertSystem",
    "SecurityAlertException",
    "RewardDefender",
    "ObservationDefender",
    "PPODefender",
    "PolicyDefender",
    "BufferDefender",
    "DriftDetector",
    "AnomalyDetector",
    "GradientMonitor",
    "SecureEnvWrapper",
    "SecurePolicyWrapper",
    "SecureTrainerWrapper",
]
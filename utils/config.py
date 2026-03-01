"""
RLShield Configuration Management
Centralized config for all defenders and detectors.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable


@dataclass
class RLShieldConfig:
    """
    Master configuration for RLShield.
    Can be created manually or via preset threat levels.
    """

  
    algo: str = "PPO"               # PPO | DQN | SAC | TD3 | DDPG | A2C | A3C | TRPO | REINFORCE | DreamerV3
    threat_level: str = "medium"    # low | medium | high
    alert_mode: str = "log"         # log | raise | callback | silent
    auto_rollback: bool = True
    alert_callback: Optional[Callable] = None   

    reward_window: int = 1000
    z_threshold: float = 3.0
    ema_alpha: float = 0.99
    reward_min: float = -1e6
    reward_max: float = 1e6

    obs_epsilon: float = 0.01           # smoothing radius for certification
    obs_n_samples: int = 50             # samples for randomized smoothing
    obs_cert_confidence: float = 0.7    # min confidence to trust action
    obs_max_delta: float = 1e4          # max norm change between consecutive obs

    kl_hard_limit: float = 0.05
    clip_eps: float = 0.2
    max_grad_norm: float = 0.5
    clip_fraction_limit: float = 0.5    # alert if >50% ratios clipped
    kl_trend_threshold: float = 0.001   # alert if KL trending upward

    buffer_max_state_delta: float = 1e3
    buffer_reward_min: float = -1e6
    buffer_reward_max: float = 1e6
    buffer_action_check: bool = True

    snapshot_interval: int = 1000
    drift_threshold: float = 0.1
    max_snapshots: int = 5

    grad_norm_multiplier: float = 10.0  # alert if norm > multiplier * max_grad_norm
    grad_zero_on_alert: bool = True     # zero out gradients on anomaly

    anomaly_window: int = 200
    anomaly_z_thresh: float = 3.5

    @classmethod
    def from_threat_level(cls, level: str, algo: str = "PPO") -> "RLShieldConfig":
        """Create config from a named threat level preset."""
        presets = {
            "low": cls(
                algo=algo,
                threat_level="low",
                z_threshold=4.0,
                kl_hard_limit=0.10,
                drift_threshold=0.2,
                clip_fraction_limit=0.7,
                obs_cert_confidence=0.6,
            ),
            "medium": cls(
                algo=algo,
                threat_level="medium",
                z_threshold=3.0,
                kl_hard_limit=0.05,
                drift_threshold=0.1,
                clip_fraction_limit=0.5,
                obs_cert_confidence=0.7,
            ),
            "high": cls(
                algo=algo,
                threat_level="high",
                z_threshold=2.0,
                kl_hard_limit=0.02,
                drift_threshold=0.05,
                clip_fraction_limit=0.3,
                obs_cert_confidence=0.8,
            ),
        }
        if level not in presets:
            raise ValueError(f"Unknown threat level '{level}'. Choose: low | medium | high")
        return presets[level]

    def to_dict(self) -> dict:
        import dataclasses
        d = dataclasses.asdict(self)
        d.pop("alert_callback", None)
        return d
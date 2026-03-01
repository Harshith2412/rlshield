"""
RLShield Policy Wrapper
Wraps any policy (PyTorch Module, callable, SB3 policy, etc.)
Adds transparent observation certification and action monitoring.
"""

import numpy as np
from typing import Any, Callable, Optional

from ..defenders.observation_defender import ObservationDefender
from ..defenders.policy_defender import PolicyDefender
from ..detectors.drift_detector import DriftDetector
from ..core.alert_system import AlertSystem
from ..utils.config import RLShieldConfig


class SecurePolicyWrapper:
    """
    Wraps a policy with transparent security checks.

    Provides:
    - Observation certification before policy call
    - Action monitoring after policy call
    - Drift detection integration

    Works with:
    - PyTorch nn.Module
    - Any callable: fn(obs) -> action
    - Stable-Baselines3 policies
    """

    def __init__(
        self,
        policy: Any,
        config: RLShieldConfig,
        alert_system: AlertSystem,
        obs_defender: Optional[ObservationDefender] = None,
        policy_defender: Optional[PolicyDefender] = None,
        drift_detector: Optional[DriftDetector] = None,
        use_certification: bool = False,  # randomized smoothing (adds overhead)
    ):
        self.policy = policy
        self.config = config
        self.alert_system = alert_system
        self.use_certification = use_certification

        self.obs_defender = obs_defender or ObservationDefender(config, alert_system)
        self.policy_defender = policy_defender or PolicyDefender(config, alert_system)
        self.drift_detector = drift_detector
        self._step = 0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.predict(obs)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """
        Secure forward pass through policy.
        Obs is defended before the call, action is monitored after.
        """
        obs = np.asarray(obs, dtype=np.float32)

        # 1. Defend observation
        obs = self.obs_defender.defend(obs)

        # 2. Get action (with optional certification)
        if self.use_certification:
            action = self.obs_defender.certify_action(self._raw_predict, obs)
        else:
            action = self._raw_predict(obs)

        # 3. Monitor output action for distribution anomalies
        if action is not None:
            self.policy_defender.monitor_actions(np.asarray(action).flatten())

        self._step += 1
        return action

    def _raw_predict(self, obs: np.ndarray) -> np.ndarray:
        """Call the underlying policy."""
        try:
            # PyTorch Module
            import torch
            if isinstance(self.policy, torch.nn.Module):
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0) if obs.ndim == 1 else torch.FloatTensor(obs)
                    action = self.policy(obs_t)
                    return action.squeeze(0).cpu().numpy()
        except ImportError:
            pass

        # SB3 style: has .predict() method
        if hasattr(self.policy, "predict"):
            action, _ = self.policy.predict(obs, deterministic=True)
            return np.asarray(action)

        # Plain callable
        if callable(self.policy):
            return np.asarray(self.policy(obs))

        raise ValueError("Policy must be a PyTorch Module, have predict(), or be callable.")

    # ── PyTorch passthrough ───────────────────────────────────────

    def parameters(self):
        if hasattr(self.policy, "parameters"):
            return self.policy.parameters()
        return iter([])

    def state_dict(self):
        if hasattr(self.policy, "state_dict"):
            return self.policy.state_dict()
        return {}

    def load_state_dict(self, state_dict):
        if hasattr(self.policy, "load_state_dict"):
            self.policy.load_state_dict(state_dict)

    def train(self, mode: bool = True):
        if hasattr(self.policy, "train"):
            self.policy.train(mode)
        return self

    def eval(self):
        if hasattr(self.policy, "eval"):
            self.policy.eval()
        return self

    def __getattr__(self, name: str):
        return getattr(self.policy, name)
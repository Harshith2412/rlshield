"""
RLShield Environment Wrapper
Transparent Gymnasium/Gym wrapper.
User replaces env with SecureEnvWrapper — no other code changes.
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ..defenders.reward_defender import RewardDefender
from ..defenders.observation_defender import ObservationDefender
from ..defenders.buffer_defender import BufferDefender
from ..core.alert_system import AlertSystem
from ..utils.config import RLShieldConfig


class SecureEnvWrapper:
    """
    Drop-in Gym/Gymnasium environment wrapper.
    Secures observations and rewards transparently.

    Works with:
    - gymnasium environments
    - gym (legacy) environments
    - Any env with step() and reset() methods
    """

    def __init__(
        self,
        env: Any,
        config: RLShieldConfig,
        alert_system: AlertSystem,
    ):
        self.env = env
        self.config = config
        self.alert_system = alert_system

        # Defenders
        self.reward_defender = RewardDefender(config, alert_system)
        self.obs_defender = ObservationDefender(config, alert_system)

        # Mirror env attributes
        self._step_count = 0

    # ── Core Interface ────────────────────────────────────────────

    def step(self, action) -> Tuple:
        result = self.env.step(action)

        # Handle both gym (4-tuple) and gymnasium (5-tuple)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        elif len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = done, False
        else:
            raise ValueError(f"Unexpected step() return length: {len(result)}")

        self._step_count += 1

        # Secure observation and reward
        obs = self.obs_defender.defend(np.asarray(obs, dtype=np.float32))
        reward = self.reward_defender.defend(float(reward))

        if len(result) == 5:
            return obs, reward, terminated, truncated, info
        return obs, reward, done, info

    def reset(self, **kwargs) -> Union[np.ndarray, Tuple]:
        result = self.env.reset(**kwargs)

        self.obs_defender.reset()
        self.reward_defender.reset()

        # Handle gymnasium (obs, info) vs gym (obs,)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            obs = self.obs_defender.defend(np.asarray(obs, dtype=np.float32))
            return obs, info
        else:
            obs = result
            obs = self.obs_defender.defend(np.asarray(obs, dtype=np.float32))
            return obs

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    # ── Passthrough attributes ─────────────────────────────────────

    def __getattr__(self, name: str):
        """Forward unknown attributes to wrapped env."""
        return getattr(self.env, name)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def reward_range(self):
        return getattr(self.env, "reward_range", (-float("inf"), float("inf")))

    @property
    def spec(self):
        return getattr(self.env, "spec", None)

    @property
    def step_count(self) -> int:
        return self._step_count
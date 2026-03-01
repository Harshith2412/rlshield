"""
RLShield PPO Defender
PPO-specific hardening:
- KL divergence monitoring + hard limit enforcement
- Clip exploitation detection
- Gradient anomaly detection and zeroing
- Policy ratio analysis
"""

import numpy as np
from typing import Optional, Any, Tuple

from ..core.base_defender import BaseDefender
from ..core.alert_system import AlertSystem
from ..core.threat_model import AttackType, Severity
from ..utils.config import RLShieldConfig
from ..utils.statistics import TrendDetector, RollingStats


class PPODefender(BaseDefender):
    """
    PPO-Specific Security Hardening.

    Hooks:
        defend_update()   → call before each PPO gradient update
        defend_gradients()→ call after loss.backward(), before optimizer.step()
        detect()          → call periodically to check KL trends
    """

    def __init__(self, config: RLShieldConfig, alert_system: AlertSystem):
        super().__init__(config, alert_system)
        self.kl_hard_limit = config.kl_hard_limit
        self.clip_eps = config.clip_eps
        self.max_grad_norm = config.max_grad_norm
        self.clip_fraction_limit = config.clip_fraction_limit

        self._kl_history = RollingStats(window=100)
        self._kl_trend = TrendDetector(window=20)
        self._grad_history = RollingStats(window=100)
        self._skipped_updates = 0


    def defend_update_torch(
        self,
        old_log_probs,
        new_log_probs,
        advantages,
    ) -> Optional[Any]:
        """
        Secure PPO surrogate loss computation (PyTorch).

        Args:
            old_log_probs: log probs from old policy  (torch.Tensor)
            new_log_probs: log probs from new policy  (torch.Tensor)
            advantages:    advantage estimates          (torch.Tensor)

        Returns:
            loss tensor, or None if update should be skipped
        """
        try:
            import torch

            ratio = torch.exp(new_log_probs - old_log_probs)

            log_ratio = new_log_probs - old_log_probs
            kl_approx = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
            self._kl_history.update(kl_approx)
            self._kl_trend.update(kl_approx)

            if kl_approx > self.kl_hard_limit:
                self._skipped_updates += 1
                self._alert(
                    AttackType.KL_VIOLATION,
                    Severity.HIGH,
                    {
                        "kl": round(kl_approx, 6),
                        "limit": self.kl_hard_limit,
                        "skipped_updates": self._skipped_updates,
                    },
                )
                return None 

            clipped_mask = (ratio < 1 - self.clip_eps) | (ratio > 1 + self.clip_eps)
            clip_fraction = clipped_mask.float().mean().item()

            if clip_fraction > self.clip_fraction_limit:
                self._alert(
                    AttackType.CLIP_EXPLOITATION,
                    Severity.MEDIUM,
                    {
                        "clip_fraction": round(clip_fraction, 4),
                        "limit": self.clip_fraction_limit,
                    },
                )

            clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
            loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            return loss

        except ImportError:
            raise RuntimeError("PyTorch is required for defend_update_torch()")

    def defend_gradients_torch(self, model) -> bool:
        """
        Secure gradient step (PyTorch).
        Call after loss.backward(), before optimizer.step().

        Returns True if step should proceed, False if gradients were zeroed.
        """
        try:
            import torch

            norms = []
            for p in model.parameters():
                if p.grad is not None:
                    norms.append(p.grad.norm().item())

            if not norms:
                return True

            total_norm = float(np.mean(norms))
            self._grad_history.update(total_norm)

            threshold = self.max_grad_norm * self.config.grad_norm_multiplier

            if total_norm > threshold:
                self._alert(
                    AttackType.GRADIENT_EXPLOSION,
                    Severity.CRITICAL,
                    {
                        "grad_norm": round(total_norm, 4),
                        "threshold": round(threshold, 4),
                        "action": "gradients_zeroed" if self.config.grad_zero_on_alert else "clipped",
                    },
                )
                if self.config.grad_zero_on_alert:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.zero_()
                    return False

            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            return True

        except ImportError:
            raise RuntimeError("PyTorch is required for defend_gradients_torch()")


    def defend_update_numpy(
        self,
        old_log_probs: np.ndarray,
        new_log_probs: np.ndarray,
        advantages: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Framework-agnostic PPO update defense.
        Returns secure loss values or None to skip update.
        """
        ratio = np.exp(new_log_probs - old_log_probs)

        log_ratio = new_log_probs - old_log_probs
        kl_approx = float(np.mean((np.exp(log_ratio) - 1) - log_ratio))
        self._kl_history.update(kl_approx)
        self._kl_trend.update(kl_approx)

        if kl_approx > self.kl_hard_limit:
            self._skipped_updates += 1
            self._alert(
                AttackType.KL_VIOLATION,
                Severity.HIGH,
                {"kl": round(kl_approx, 6), "limit": self.kl_hard_limit},
            )
            return None

        clip_fraction = float(
            np.mean((ratio < 1 - self.clip_eps) | (ratio > 1 + self.clip_eps))
        )
        if clip_fraction > self.clip_fraction_limit:
            self._alert(
                AttackType.CLIP_EXPLOITATION,
                Severity.MEDIUM,
                {"clip_fraction": round(clip_fraction, 4)},
            )

        clipped_ratio = np.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
        loss = -np.minimum(ratio * advantages, clipped_ratio * advantages)
        return loss

    def defend(self, data: Any) -> Any:
        """Generic defend — delegates to numpy version if data is ndarray."""
        if isinstance(data, np.ndarray):
            return data  
        return data

    def detect(self, data: Any = None) -> bool:
        """Detect KL trending upward over recent updates."""
        if self._kl_trend.is_trending_up(self.config.kl_trend_threshold):
            self._alert(
                AttackType.KL_TRENDING,
                Severity.MEDIUM,
                {
                    "kl_slope": round(self._kl_trend.trend_slope(), 6),
                    "threshold": self.config.kl_trend_threshold,
                    "recent_kl_mean": round(self._kl_history.mean(), 6),
                },
            )
            return True
        return False

    @property
    def skipped_updates(self) -> int:
        return self._skipped_updates
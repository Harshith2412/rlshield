from abc import ABC, abstractmethod
from typing import Any

from .alert_system import AlertSystem
from .threat_model import AttackType, Severity
from ..utils.config import RLShieldConfig


class BaseDefender(ABC):

    def __init__(self, config: RLShieldConfig, alert_system: AlertSystem):
        self.config = config
        self.alert_system = alert_system
        self.enabled = True
        self._step = 0
        self._alerts_fired = 0

    @abstractmethod
    def defend(self, data: Any) -> Any:
        pass

    @abstractmethod
    def detect(self, data: Any) -> bool:
        pass

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def tick(self):
        """Call once per environment step."""
        self._step += 1

    def _alert(self, attack_type: AttackType, severity: Severity, details: dict):
        self._alerts_fired += 1
        self.alert_system.fire(
            attack_type=attack_type,
            severity=severity,
            details=details,
            step=self._step,
            source=self.__class__.__name__,
        )

    @property
    def stats(self) -> dict:
        return {
            "defender": self.__class__.__name__,
            "enabled": self.enabled,
            "steps": self._step,
            "alerts_fired": self._alerts_fired,
        }
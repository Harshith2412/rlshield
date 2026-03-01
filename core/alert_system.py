import logging
import time
from typing import Callable, List, Optional

from .threat_model import ThreatEvent, AttackType, Severity

logger = logging.getLogger("rlshield")


class AlertSystem:

    def __init__(self, mode: str = "log", callback: Optional[Callable] = None):
        self.mode = mode
        self.callback = callback
        self._events: List[ThreatEvent] = []
        self._setup_logger()

    def _setup_logger(self):
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s  [RLShield]  %(levelname)s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)

    def fire(
        self,
        attack_type: AttackType,
        severity: Severity,
        details: dict,
        step: int = -1,
        source: str = "unknown",
    ) -> ThreatEvent:
        event = ThreatEvent(
            attack_type=attack_type,
            severity=severity,
            details=details,
            timestamp=time.time(),
            step=step,
            source=source,
        )
        self._events.append(event)
        self._dispatch(event)
        return event

    def _dispatch(self, event: ThreatEvent):
        msg = str(event)

        if self.mode == "log":
            level_map = {
                Severity.LOW: logger.info,
                Severity.MEDIUM: logger.warning,
                Severity.HIGH: logger.error,
                Severity.CRITICAL: logger.critical,
            }
            level_map.get(event.severity, logger.warning)(msg)

        elif self.mode == "raise":
            if event.severity.value >= Severity.HIGH.value:
                raise SecurityAlertException(event)
            else:
                logger.warning(msg)

        elif self.mode == "callback":
            if self.callback:
                self.callback(event)

        elif self.mode == "silent":
            pass  

    def get_report(self) -> dict:
        """Full threat report summary."""
        if not self._events:
            return {"total": 0, "events": [], "summary": {}}

        summary = {}
        for e in self._events:
            key = e.attack_type.value
            summary[key] = summary.get(key, 0) + 1

        return {
            "total": len(self._events),
            "summary": summary,
            "events": [e.to_dict() for e in self._events],
            "highest_severity": max(e.severity.value for e in self._events),
        }

    def get_events_by_type(self, attack_type: AttackType) -> List[ThreatEvent]:
        return [e for e in self._events if e.attack_type == attack_type]

    def clear(self):
        self._events.clear()

    @property
    def total_alerts(self) -> int:
        return len(self._events)


class SecurityAlertException(Exception):
    """Raised when alert_mode='raise' and a HIGH/CRITICAL threat is detected."""

    def __init__(self, event: ThreatEvent):
        self.event = event
        super().__init__(str(event))
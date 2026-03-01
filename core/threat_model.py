import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict


class AttackType(Enum):
    REWARD_POISONING     = "reward_poisoning"
    OBS_ADVERSARIAL      = "observation_adversarial"
    OBS_TELEPORT         = "observation_teleport"
    POLICY_DRIFT         = "policy_drift"
    GRADIENT_EXPLOSION   = "gradient_explosion"
    GRADIENT_ANOMALY     = "gradient_anomaly"
    BUFFER_INJECTION     = "buffer_injection"
    KL_VIOLATION         = "kl_violation"
    KL_TRENDING          = "kl_trending"
    CLIP_EXPLOITATION    = "clip_exploitation"
    REPLAY_POISONING     = "replay_poisoning"
    ENTROPY_COLLAPSE     = "entropy_collapse"
    WORLD_MODEL_DRIFT    = "world_model_drift"
    UNKNOWN              = "unknown"


class Severity(Enum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4

    def label(self) -> str:
        labels = {1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "CRITICAL"}
        return labels[self.value]


@dataclass
class ThreatEvent:
    attack_type: AttackType
    severity: Severity
    details: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    step: int = -1
    source: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "attack_type": self.attack_type.value,
            "severity": self.severity.label(),
            "details": self.details,
            "timestamp": self.timestamp,
            "step": self.step,
            "source": self.source,
        }

    def __str__(self) -> str:
        return (
            f"[{self.severity.label()}] {self.attack_type.value} "
            f"from {self.source} | {self.details}"
        )
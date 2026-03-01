"""
RLShield Snapshot Manager
Stores policy snapshots for rollback on detected attacks.
"""

import copy
import time
from collections import deque
from typing import Any, Optional


class SnapshotManager:
    """
    Maintains a circular buffer of policy snapshots.
    Enables rollback to a known-good policy state.
    """

    def __init__(self, max_snapshots: int = 5):
        self.max_snapshots = max_snapshots
        self._snapshots: deque = deque(maxlen=max_snapshots)

    def save(self, policy: Any, step: int):
        """Deep copy and store the policy state."""
        try:
            snapshot = {
                "step": step,
                "timestamp": time.time(),
                "state": copy.deepcopy(policy.state_dict()) if hasattr(policy, "state_dict") else copy.deepcopy(policy),
            }
            self._snapshots.append(snapshot)
        except Exception as e:
            pass  

    def rollback(self, policy: Any, steps_back: int = 1) -> bool:
        """
        Restore policy to a previous snapshot.
        Returns True if rollback succeeded.
        """
        if len(self._snapshots) < steps_back:
            return False

        target_idx = -(steps_back)
        snapshot = list(self._snapshots)[target_idx]

        try:
            if hasattr(policy, "load_state_dict"):
                policy.load_state_dict(snapshot["state"])
            else:
                policy.__dict__.update(snapshot["state"].__dict__)
            return True
        except Exception:
            return False

    def get_latest(self) -> Optional[dict]:
        if not self._snapshots:
            return None
        return self._snapshots[-1]

    def count(self) -> int:
        return len(self._snapshots)

    def clear(self):
        self._snapshots.clear()
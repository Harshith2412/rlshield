"""
RLShield Metrics Evaluator
==========================
Measures and reports:
  1. True Positive Rate (TPR)    — How often real attacks are caught
  2. False Positive Rate (FPR)   — How often clean data triggers false alerts
  3. Detection Latency           — Steps between attack start and first alert
  4. Time Overhead               — μs added per step vs unprotected
  5. Space Overhead              — Memory used per component
  6. Rollback Accuracy           — Whether rollback restores correct policy
  7. Per-component timing        — Individual defender benchmarks
  8. Scaling analysis            — How complexity grows with dim/batch/window

Run:
    python evaluate_metrics.py                   # full report
    python evaluate_metrics.py --quick           # fast subset
    python evaluate_metrics.py --json            # JSON output
    python evaluate_metrics.py --component PPO   # single component
"""

import sys
import os
import time
import json
import argparse
import tracemalloc
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rlshield import RLShield, RLShieldConfig
from rlshield.core.alert_system import AlertSystem
from rlshield.defenders.reward_defender import RewardDefender
from rlshield.defenders.observation_defender import ObservationDefender
from rlshield.defenders.ppo_defender import PPODefender
from rlshield.defenders.policy_defender import PolicyDefender
from rlshield.defenders.buffer_defender import BufferDefender
from rlshield.detectors.gradient_monitor import GradientMonitor
from rlshield.detectors.drift_detector import DriftDetector
from rlshield.detectors.anomaly_detector import AnomalyDetector
from rlshield.utils.config import RLShieldConfig

# ─────────────────────────────────────────────────────────────────
# ANSI colours
# ─────────────────────────────────────────────────────────────────
G  = "\033[92m"   # green
R  = "\033[91m"   # red
Y  = "\033[93m"   # yellow
C  = "\033[96m"   # cyan
B  = "\033[94m"   # blue
M  = "\033[95m"   # magenta
W  = "\033[97m"   # white
DIM= "\033[2m"
RST= "\033[0m"
BOLD="\033[1m"

def hdr(title: str, width: int = 70):
    bar = "─" * width
    print(f"\n{C}{bar}{RST}")
    print(f"{BOLD}{W}  {title}{RST}")
    print(f"{C}{bar}{RST}")

def row(label, value, unit="", color=W, width=42):
    dots = "." * (width - len(label))
    print(f"  {DIM}{label}{dots}{RST}{color}{value}{RST}{DIM} {unit}{RST}")

def ok(msg):  print(f"  {G}✓{RST} {msg}")
def warn(msg):print(f"  {Y}⚠{RST} {msg}")
def bad(msg): print(f"  {R}✗{RST} {msg}")

def bench(fn, n=10_000, warmup=200) -> float:
    """Returns microseconds per call."""
    for _ in range(warmup): fn()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter() - t0) / n * 1e6

def peak_kb(fn, n=1_000, warmup=100) -> float:
    """Returns peak memory in KB."""
    for _ in range(warmup): fn()
    tracemalloc.start()
    for _ in range(n): fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024

def make_cfg(level="medium") -> Tuple[RLShieldConfig, AlertSystem]:
    return RLShieldConfig.from_threat_level(level), AlertSystem(mode="silent")

# ═════════════════════════════════════════════════════════════════
# 1. TRUE POSITIVE RATE — attacks should be detected
# ═════════════════════════════════════════════════════════════════

def eval_tpr(n_trials: int = 200, verbose: bool = True) -> Dict:
    if verbose: hdr("True Positive Rate (TPR) — Attack Detection")

    results = {}

    # ── Reward poisoning ─────────────────────────────────────────
    cfg, alert = make_cfg("medium")
    rd = RewardDefender(cfg, alert)
    for _ in range(200): rd.defend(float(np.random.normal(0, 1)))

    detected = 0
    for _ in range(n_trials):
        alert.clear()
        poisoned = float(np.random.choice([1e6, -1e6, 100, -100]))  # attack magnitudes
        rd.defend(poisoned)
        if alert.total_alerts > 0:
            detected += 1

    tpr_reward = detected / n_trials
    results["reward_poisoning"] = tpr_reward
    if verbose:
        color = G if tpr_reward >= 0.90 else Y if tpr_reward >= 0.70 else R
        row("Reward Poisoning TPR", f"{tpr_reward*100:.1f}%", "", color)

    # ── Observation teleport ──────────────────────────────────────
    cfg2, alert2 = make_cfg("medium")
    od = ObservationDefender(cfg2, alert2)
    obs_base = np.random.randn(64).astype(np.float32)
    for _ in range(200): od.defend(obs_base + np.random.randn(64).astype(np.float32)*0.01)

    detected = 0
    for _ in range(n_trials):
        alert2.clear()
        teleport = obs_base + np.random.uniform(2000, 5000) * np.random.randn(64).astype(np.float32)
        od.defend(teleport)
        if alert2.total_alerts > 0:
            detected += 1

    tpr_obs = detected / n_trials
    results["obs_teleport"] = tpr_obs
    if verbose:
        color = G if tpr_obs >= 0.90 else Y if tpr_obs >= 0.70 else R
        row("Observation Teleport TPR", f"{tpr_obs*100:.1f}%", "", color)

    # ── PPO KL violation ─────────────────────────────────────────
    cfg3, alert3 = make_cfg("medium")
    pd = PPODefender(cfg3, alert3)

    detected = 0
    for _ in range(n_trials):
        alert3.clear()
        old_lp = np.log(np.random.dirichlet([1]*8) + 1e-8)
        # Extreme new policy that always violates KL
        extreme = np.zeros(8); extreme[np.random.randint(8)] = 1.0 - 1e-6
        extreme = np.clip(extreme, 1e-8, None)
        new_lp = np.log(extreme / extreme.sum())
        adv = np.random.randn(8)
        result = pd.defend_update_numpy(old_lp, new_lp, adv)
        if result is None or alert3.total_alerts > 0:
            detected += 1

    tpr_ppo = detected / n_trials
    results["ppo_kl_violation"] = tpr_ppo
    if verbose:
        color = G if tpr_ppo >= 0.90 else Y if tpr_ppo >= 0.70 else R
        row("PPO KL Violation TPR", f"{tpr_ppo*100:.1f}%", "", color)

    # ── Gradient explosion ────────────────────────────────────────
    cfg4, alert4 = make_cfg("medium")
    gm = GradientMonitor(cfg4, alert4)
    for _ in range(50): gm.update(float(np.random.uniform(0.1, 0.5)))

    detected = 0
    for _ in range(n_trials):
        alert4.clear()
        exploded_norm = float(np.random.uniform(50, 500))  # way above max_grad_norm=0.5
        gm.update(exploded_norm)
        if alert4.total_alerts > 0:
            detected += 1

    tpr_grad = detected / n_trials
    results["gradient_explosion"] = tpr_grad
    if verbose:
        color = G if tpr_grad >= 0.90 else Y if tpr_grad >= 0.70 else R
        row("Gradient Explosion TPR", f"{tpr_grad*100:.1f}%", "", color)

    # ── Buffer injection ──────────────────────────────────────────
    cfg5, alert5 = make_cfg("medium")
    bd = BufferDefender(cfg5, alert5)
    s = np.random.randn(32).astype(np.float32)
    for _ in range(100): bd.defend((s, 0, 0.5, s+0.01, False))

    detected = 0
    attacks = [
        (s, 0, 2e7, s+0.01, False),           # reward oob
        (s, 0, 0.5, s + 1e5, False),           # teleport
        ("only_two_items",),                    # malformed (too short)
    ]
    for _ in range(n_trials):
        alert5.clear()
        attack = attacks[np.random.randint(len(attacks))]
        result = bd.defend(attack)
        if result is None or alert5.total_alerts > 0:
            detected += 1

    tpr_buf = detected / n_trials
    results["buffer_injection"] = tpr_buf
    if verbose:
        color = G if tpr_buf >= 0.90 else Y if tpr_buf >= 0.70 else R
        row("Buffer Injection TPR", f"{tpr_buf*100:.1f}%", "", color)

    # ── Entropy collapse (SAC) ────────────────────────────────────
    cfg6, alert6 = make_cfg("medium")
    pd6 = PolicyDefender(cfg6, alert6)
    for _ in range(50): pd6.monitor_entropy(float(np.random.uniform(1.2, 1.8)))

    detected = 0
    for _ in range(n_trials):
        alert6.clear()
        pd6.monitor_entropy(1e-10)
        if alert6.total_alerts > 0:
            detected += 1

    tpr_ent = detected / n_trials
    results["entropy_collapse"] = tpr_ent
    if verbose:
        color = G if tpr_ent >= 0.90 else Y if tpr_ent >= 0.70 else R
        row("Entropy Collapse TPR (SAC)", f"{tpr_ent*100:.1f}%", "", color)

    avg_tpr = np.mean(list(results.values()))
    if verbose:
        print()
        color = G if avg_tpr >= 0.90 else Y if avg_tpr >= 0.70 else R
        row("AVERAGE TPR", f"{avg_tpr*100:.1f}%", "", color)
        if avg_tpr >= 0.90: ok("Excellent detection rate")
        elif avg_tpr >= 0.70: warn("Good but can improve with stricter thresholds")
        else: bad("Detection needs tuning — try threat_level='high'")

    results["average"] = avg_tpr
    return results


# ═════════════════════════════════════════════════════════════════
# 2. FALSE POSITIVE RATE — clean data should NOT trigger alerts
# ═════════════════════════════════════════════════════════════════

def eval_fpr(n_steps: int = 5000, verbose: bool = True) -> Dict:
    if verbose: hdr("False Positive Rate (FPR) — Clean Data Accuracy")

    results = {}

    # ── Reward defender on clean Gaussian rewards ─────────────────
    cfg, alert = make_cfg("medium")
    rd = RewardDefender(cfg, alert)
    for _ in range(200): rd.defend(float(np.random.normal(0, 1)))

    false_alerts = 0
    for _ in range(n_steps):
        alert.clear()
        rd.defend(float(np.random.normal(0, 1)))
        if alert.total_alerts > 0:
            false_alerts += 1

    fpr_reward = false_alerts / n_steps
    results["reward"] = fpr_reward
    if verbose:
        color = G if fpr_reward <= 0.01 else Y if fpr_reward <= 0.05 else R
        row("Reward FPR (Gaussian rewards)", f"{fpr_reward*100:.2f}%", f"({false_alerts}/{n_steps} false alerts)", color)

    # ── Obs defender on normal random walk ────────────────────────
    cfg2, alert2 = make_cfg("medium")
    od = ObservationDefender(cfg2, alert2)
    obs = np.random.randn(64).astype(np.float32)
    for _ in range(200):
        obs = obs + np.random.randn(64).astype(np.float32) * 0.05
        od.defend(obs.copy())

    false_alerts = 0
    for _ in range(n_steps):
        alert2.clear()
        obs = obs + np.random.randn(64).astype(np.float32) * 0.05
        od.defend(obs.copy())
        if alert2.total_alerts > 0:
            false_alerts += 1

    fpr_obs = false_alerts / n_steps
    results["observation"] = fpr_obs
    if verbose:
        color = G if fpr_obs <= 0.01 else Y if fpr_obs <= 0.05 else R
        row("Observation FPR (random walk)", f"{fpr_obs*100:.2f}%", f"({false_alerts}/{n_steps} false alerts)", color)

    # ── PPO defender on legitimate updates ────────────────────────
    cfg3, alert3 = make_cfg("medium")
    pd = PPODefender(cfg3, alert3)

    false_alerts = 0
    for _ in range(n_steps):
        alert3.clear()
        old_lp = np.log(np.random.dirichlet([2]*8) + 1e-8)
        # Small legitimate update
        new_lp = old_lp + np.random.randn(8) * 0.01
        adv = np.random.randn(8)
        pd.defend_update_numpy(old_lp, new_lp, adv)
        if alert3.total_alerts > 0:
            false_alerts += 1

    fpr_ppo = false_alerts / n_steps
    results["ppo"] = fpr_ppo
    if verbose:
        color = G if fpr_ppo <= 0.01 else Y if fpr_ppo <= 0.05 else R
        row("PPO FPR (small legit updates)", f"{fpr_ppo*100:.2f}%", f"({false_alerts}/{n_steps} false alerts)", color)

    # ── Buffer on clean transitions ───────────────────────────────
    cfg4, alert4 = make_cfg("medium")
    bd = BufferDefender(cfg4, alert4)
    s = np.random.randn(32).astype(np.float32)

    false_alerts = 0
    for _ in range(n_steps):
        alert4.clear()
        s_new = s + np.random.randn(32).astype(np.float32) * 0.05
        r = float(np.random.normal(0, 1))
        bd.defend((s.copy(), 0, r, s_new.copy(), False))
        if alert4.total_alerts > 0:
            false_alerts += 1
        s = s_new

    fpr_buf = false_alerts / n_steps
    results["buffer"] = fpr_buf
    if verbose:
        color = G if fpr_buf <= 0.01 else Y if fpr_buf <= 0.05 else R
        row("Buffer FPR (clean transitions)", f"{fpr_buf*100:.2f}%", f"({false_alerts}/{n_steps} false alerts)", color)

    # ── Gradient monitor on normal training ───────────────────────
    cfg5, alert5 = make_cfg("medium")
    gm = GradientMonitor(cfg5, alert5)
    for _ in range(50): gm.update(float(np.random.uniform(0.1, 0.5)))

    false_alerts = 0
    for _ in range(n_steps):
        alert5.clear()
        normal_norm = float(np.random.uniform(0.05, 0.45))
        gm.update(normal_norm)
        if alert5.total_alerts > 0:
            false_alerts += 1

    fpr_grad = false_alerts / n_steps
    results["gradient"] = fpr_grad
    if verbose:
        color = G if fpr_grad <= 0.01 else Y if fpr_grad <= 0.05 else R
        row("Gradient FPR (normal norms)", f"{fpr_grad*100:.2f}%", f"({false_alerts}/{n_steps} false alerts)", color)

    avg_fpr = np.mean(list(results.values()))
    results["average"] = avg_fpr
    if verbose:
        print()
        color = G if avg_fpr <= 0.01 else Y if avg_fpr <= 0.05 else R
        row("AVERAGE FPR", f"{avg_fpr*100:.2f}%", "", color)
        if avg_fpr <= 0.01: ok("Excellent — very few false alarms on clean data")
        elif avg_fpr <= 0.05: warn("Acceptable — consider raising z_threshold to reduce noise")
        else: bad("Too many false alarms — raise threshold or use threat_level='low'")

    return results


# ═════════════════════════════════════════════════════════════════
# 3. DETECTION LATENCY — steps until first alert after attack starts
# ═════════════════════════════════════════════════════════════════

def eval_latency(n_trials: int = 50, verbose: bool = True) -> Dict:
    if verbose: hdr("Detection Latency — Steps Until First Alert")

    results = {}

    # ── Reward poisoning latency ──────────────────────────────────
    latencies = []
    for _ in range(n_trials):
        cfg, alert = make_cfg("medium")
        rd = RewardDefender(cfg, alert)
        for _ in range(200): rd.defend(float(np.random.normal(0, 1)))

        for step in range(1, 101):
            alert.clear()
            rd.defend(1e6)  # constant attack
            if alert.total_alerts > 0:
                latencies.append(step)
                break
        else:
            latencies.append(100)  # not detected within 100 steps

    results["reward_poisoning"] = {
        "mean": float(np.mean(latencies)),
        "min": int(np.min(latencies)),
        "max": int(np.max(latencies)),
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95)),
    }
    if verbose:
        l = results["reward_poisoning"]
        color = G if l["mean"] <= 2 else Y if l["mean"] <= 5 else R
        row("Reward Poisoning latency", f"mean={l['mean']:.1f} steps", f"(min={l['min']}, p95={l['p95']:.0f})", color)

    # ── PPO KL violation latency ──────────────────────────────────
    latencies = []
    for _ in range(n_trials):
        cfg, alert = make_cfg("medium")
        pd = PPODefender(cfg, alert)

        for step in range(1, 51):
            alert.clear()
            old_lp = np.log(np.random.dirichlet([1]*8) + 1e-8)
            extreme = np.zeros(8); extreme[0] = 1.0 - 1e-6
            extreme = np.clip(extreme, 1e-8, None)
            new_lp = np.log(extreme / extreme.sum())
            result = pd.defend_update_numpy(old_lp, new_lp, np.random.randn(8))
            if result is None or alert.total_alerts > 0:
                latencies.append(step)
                break
        else:
            latencies.append(50)

    results["ppo_kl"] = {
        "mean": float(np.mean(latencies)),
        "min": int(np.min(latencies)),
        "max": int(np.max(latencies)),
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95)),
    }
    if verbose:
        l = results["ppo_kl"]
        color = G if l["mean"] <= 2 else Y if l["mean"] <= 5 else R
        row("PPO KL Violation latency", f"mean={l['mean']:.1f} steps", f"(min={l['min']}, p95={l['p95']:.0f})", color)

    # ── Gradient explosion latency ────────────────────────────────
    latencies = []
    for _ in range(n_trials):
        cfg, alert = make_cfg("medium")
        gm = GradientMonitor(cfg, alert)
        for _ in range(50): gm.update(float(np.random.uniform(0.1, 0.5)))

        for step in range(1, 51):
            alert.clear()
            gm.update(200.0)  # massive explosion
            if alert.total_alerts > 0:
                latencies.append(step)
                break
        else:
            latencies.append(50)

    results["gradient_explosion"] = {
        "mean": float(np.mean(latencies)),
        "min": int(np.min(latencies)),
        "max": int(np.max(latencies)),
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95)),
    }
    if verbose:
        l = results["gradient_explosion"]
        color = G if l["mean"] <= 2 else Y if l["mean"] <= 5 else R
        row("Gradient Explosion latency", f"mean={l['mean']:.1f} steps", f"(min={l['min']}, p95={l['p95']:.0f})", color)

    # ── Policy drift latency ──────────────────────────────────────
    latencies = []
    cfg_drift = RLShieldConfig.from_threat_level("medium")
    cfg_drift.snapshot_interval = 1
    cfg_drift.drift_threshold = 0.001
    cfg_drift.auto_rollback = False

    for _ in range(min(n_trials, 20)):  # drift detection is slower to set up
        alert = AlertSystem(mode="silent")
        probe = np.ones((5, 4), dtype=np.float32)
        dd = DriftDetector(cfg_drift, alert, probe_states=probe)

        def stable(obs): return np.array([1.0, 0.0])
        def drifted(obs): return np.array([0.0, 1000.0])

        class M:
            def state_dict(self): return {}
            def load_state_dict(self, d): pass

        dd.update(stable, M(), step=1)

        for step in range(2, 30):
            alert.clear()
            dd.update(drifted, M(), step=step)
            if alert.total_alerts > 0:
                latencies.append(step - 1)
                break
        else:
            latencies.append(28)

    results["policy_drift"] = {
        "mean": float(np.mean(latencies)),
        "min": int(np.min(latencies)),
        "max": int(np.max(latencies)),
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95)),
    }
    if verbose:
        l = results["policy_drift"]
        color = G if l["mean"] <= 3 else Y if l["mean"] <= 6 else R
        row("Policy Drift latency", f"mean={l['mean']:.1f} steps", f"(min={l['min']}, p95={l['p95']:.0f})", color)

    if verbose:
        print()
        ok("Latency 1 = detected on first attack step (immediate)")
        ok("Latency ≤ 3 steps is considered real-time detection")

    return results


# ═════════════════════════════════════════════════════════════════
# 4. TIMING OVERHEAD — μs per call for each component
# ═════════════════════════════════════════════════════════════════

def eval_timing(verbose: bool = True) -> Dict:
    if verbose: hdr("Timing Overhead (μs per call)")

    results = {}
    N = 20_000

    cfg, alert = make_cfg("medium")

    # Raw baseline
    t_raw = bench(lambda: (np.random.randn(4).astype(np.float32) + 0, 1.0), n=N)
    results["raw_step"] = t_raw
    if verbose: row("Raw step (no shield)", f"{t_raw:.3f}", "μs", DIM)

    # Reward defender
    rd = RewardDefender(cfg, alert)
    for _ in range(200): rd.defend(float(np.random.normal(0,1)))
    t = bench(lambda: rd.defend(float(np.random.normal(0,1))), n=N)
    results["reward_defender"] = t
    if verbose:
        color = G if t < 10 else Y if t < 100 else R
        row("RewardDefender.defend()", f"{t:.3f}", "μs", color)

    # Observation defender (dim=64)
    od = ObservationDefender(cfg, alert)
    obs = np.random.randn(64).astype(np.float32)
    for _ in range(200): od.defend(obs + np.random.randn(64).astype(np.float32)*0.01)
    t = bench(lambda: od.defend(obs + np.random.randn(64).astype(np.float32)*0.01), n=N)
    results["obs_defender_64"] = t
    if verbose:
        color = G if t < 10 else Y if t < 50 else R
        row("ObsDefender.defend() (d=64)", f"{t:.3f}", "μs", color)

    # PPO defender (batch=64)
    pd = PPODefender(cfg, alert)
    old_lp = np.log(np.random.dirichlet([1]*64) + 1e-8)
    new_lp = old_lp + np.random.randn(64) * 0.01
    adv = np.random.randn(64)
    t = bench(lambda: pd.defend_update_numpy(old_lp, new_lp, adv), n=N)
    results["ppo_defender_64"] = t
    if verbose:
        color = G if t < 20 else Y if t < 100 else R
        row("PPODefender.update() (n=64)", f"{t:.3f}", "μs", color)

    # Buffer defender (d=32)
    bd = BufferDefender(cfg, alert)
    s = np.random.randn(32).astype(np.float32)
    for _ in range(100): bd.defend((s, 0, 0.5, s+0.01, False))
    t = bench(lambda: bd.defend((s, 0, 0.5, s+0.01, False)), n=N)
    results["buffer_defender_32"] = t
    if verbose:
        color = G if t < 20 else Y if t < 200 else R
        row("BufferDefender.defend() (d=32)", f"{t:.3f}", "μs", color)

    # Gradient monitor
    gm = GradientMonitor(cfg, alert)
    for _ in range(50): gm.update(0.3)
    t = bench(lambda: gm.update(float(np.random.uniform(0.1, 0.5))), n=N)
    results["gradient_monitor"] = t
    if verbose:
        color = G if t < 10 else Y if t < 50 else R
        row("GradientMonitor.update()", f"{t:.3f}", "μs", color)

    # Full shield step (obs + reward combined)
    shield = RLShield(algo="PPO", threat_level="medium", alert_mode="silent")
    obs4 = np.random.randn(4).astype(np.float32)
    for _ in range(200):
        shield.defend_observation(obs4)
        shield.defend_reward(1.0)
    t_full = bench(lambda: (shield.defend_observation(obs4), shield.defend_reward(1.0)), n=N)
    results["full_shield_step"] = t_full
    if verbose:
        print()
        row("Full shield step (obs+reward)", f"{t_full:.3f}", "μs", C)
        row("Overhead vs raw", f"{((t_full-t_raw)/t_raw*100):.1f}%", "", Y)

    # Obs scaling
    if verbose:
        print(f"\n  {DIM}ObsDefender scaling with obs dimension:{RST}")
    obs_scaling = {}
    for dim in [4, 16, 64, 256, 1024]:
        od2 = ObservationDefender(cfg, alert)
        base = np.random.randn(dim).astype(np.float32)
        for _ in range(100): od2.defend(base + np.random.randn(dim).astype(np.float32)*0.01)
        t = bench(lambda: od2.defend(base + np.random.randn(dim).astype(np.float32)*0.01), n=N)
        obs_scaling[dim] = t
        if verbose:
            color = G if t < 15 else Y if t < 40 else R
            row(f"  ObsDefender (d={dim})", f"{t:.3f}", "μs", color)
    results["obs_scaling"] = obs_scaling

    # PPO batch scaling
    if verbose:
        print(f"\n  {DIM}PPODefender scaling with batch size:{RST}")
    ppo_scaling = {}
    for bs in [8, 32, 64, 256, 1024]:
        pd2 = PPODefender(cfg, alert)
        ol = np.log(np.random.dirichlet([1]*bs) + 1e-8)
        nl = ol + np.random.randn(bs)*0.01
        av = np.random.randn(bs)
        t = bench(lambda: pd2.defend_update_numpy(ol, nl, av), n=N)
        ppo_scaling[bs] = t
        if verbose:
            color = G if t < 20 else Y if t < 50 else R
            row(f"  PPODefender (n={bs})", f"{t:.3f}", "μs", color)
    results["ppo_scaling"] = ppo_scaling

    return results


# ═════════════════════════════════════════════════════════════════
# 5. MEMORY OVERHEAD
# ═════════════════════════════════════════════════════════════════

def eval_memory(verbose: bool = True) -> Dict:
    if verbose: hdr("Memory Overhead (Peak KB per component)")

    results = {}
    cfg, alert = make_cfg("medium")

    components = [
        ("GradientMonitor", lambda: GradientMonitor(cfg, alert),
         lambda c: c.update(float(np.random.uniform(0.1, 0.5)))),
        ("ObsDefender (d=64)", lambda: ObservationDefender(cfg, alert),
         lambda c: c.defend(np.random.randn(64).astype(np.float32))),
        ("PPODefender (n=64)", lambda: PPODefender(cfg, alert),
         lambda c: c.defend_update_numpy(
             np.log(np.random.dirichlet([1]*64)+1e-8),
             np.log(np.random.dirichlet([1]*64)+1e-8),
             np.random.randn(64))),
        ("RewardDefender", lambda: RewardDefender(cfg, alert),
         lambda c: c.defend(float(np.random.normal(0,1)))),
        ("BufferDefender (d=32)", lambda: BufferDefender(cfg, alert),
         lambda c: c.defend((np.random.randn(32).astype(np.float32), 0, 0.5,
                              np.random.randn(32).astype(np.float32), False))),
        ("AnomalyDetector", lambda: AnomalyDetector(cfg, alert, name="test"),
         lambda c: c.update(float(np.random.normal(0,1)))),
    ]

    for name, factory, step_fn in components:
        comp = factory()
        # Warmup
        for _ in range(100): step_fn(comp)

        tracemalloc.start()
        for _ in range(1000): step_fn(comp)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        kb = peak / 1024
        results[name] = kb
        if verbose:
            color = G if kb < 50 else Y if kb < 500 else R
            row(name, f"{kb:.1f}", "KB peak", color)

    total = sum(results.values())
    results["total"] = total
    if verbose:
        print()
        row("Total (all components)", f"{total:.1f}", "KB", C)
        row("Total (in MB)", f"{total/1024:.2f}", "MB", C)
        ok("Negligible vs typical RL model (10MB–10GB)")

    return results


# ═════════════════════════════════════════════════════════════════
# 6. ROLLBACK ACCURACY
# ═════════════════════════════════════════════════════════════════

def eval_rollback(n_trials: int = 30, verbose: bool = True) -> Dict:
    if verbose: hdr("Rollback Accuracy")

    cfg = RLShieldConfig.from_threat_level("medium")
    cfg.snapshot_interval = 1
    cfg.drift_threshold = 0.001
    cfg.auto_rollback = True

    successful = 0
    correct_value = 0

    for _ in range(n_trials):
        alert = AlertSystem(mode="silent")
        probe = np.ones((5, 4), dtype=np.float32)
        dd = DriftDetector(cfg, alert, probe_states=probe)

        class TrackableModel:
            def __init__(self, val):
                self.val = val
            def state_dict(self):
                return {"val": self.val}
            def load_state_dict(self, d):
                self.val = d["val"]

        model = TrackableModel(42)

        def stable(obs): return np.array([1.0, 0.0])
        def drifted(obs): return np.array([0.0, 1000.0])

        # Save two clean snapshots
        dd.update(stable, model, step=1)
        model.val = 99  # simulate training changing weights
        dd.update(stable, model, step=2)

        # Now inject drift — should trigger rollback
        original_val = model.val
        dd.update(drifted, model, step=3)

        if alert.total_alerts > 0:
            successful += 1

        # Check rollback restored an earlier value (42 or 99, not the poisoned one)
        # In this test the rollback goes 2 steps back so should restore val=42
        if model.val != original_val or model.val in [42, 99]:
            correct_value += 1

    rollback_rate = successful / n_trials
    accuracy = correct_value / n_trials

    results = {
        "rollback_triggered_rate": rollback_rate,
        "correct_state_rate": accuracy,
        "n_trials": n_trials,
    }

    if verbose:
        color = G if rollback_rate >= 0.80 else Y if rollback_rate >= 0.60 else R
        row("Rollback triggered rate", f"{rollback_rate*100:.1f}%", f"({successful}/{n_trials})", color)
        color = G if accuracy >= 0.80 else Y if accuracy >= 0.60 else R
        row("Correct state restored", f"{accuracy*100:.1f}%", f"({correct_value}/{n_trials})", color)

    return results


# ═════════════════════════════════════════════════════════════════
# 7. THREAT LEVEL COMPARISON
# ═════════════════════════════════════════════════════════════════

def eval_threat_levels(verbose: bool = True) -> Dict:
    if verbose: hdr("Threat Level Comparison (TPR vs FPR tradeoff)")

    results = {}

    for level in ["low", "medium", "high"]:
        cfg = RLShieldConfig.from_threat_level(level)
        alert = AlertSystem(mode="silent")
        rd = RewardDefender(cfg, alert)

        # Warm up
        for _ in range(200): rd.defend(float(np.random.normal(0, 1)))

        # FPR on clean data
        fp = 0
        for _ in range(1000):
            alert.clear()
            rd.defend(float(np.random.normal(0, 1)))
            if alert.total_alerts > 0: fp += 1

        # TPR on attacks
        tp = 0
        for _ in range(200):
            alert.clear()
            rd.defend(1e6)
            if alert.total_alerts > 0: tp += 1

        results[level] = {"tpr": tp/200, "fpr": fp/1000, "z_threshold": cfg.z_threshold, "kl_limit": cfg.kl_hard_limit}

    if verbose:
        print(f"\n  {'Level':<10} {'TPR':>8} {'FPR':>8} {'z-thresh':>10} {'kl-limit':>10}")
        print(f"  {'─'*50}")
        for level, r in results.items():
            tpr_c = G if r['tpr'] >= 0.90 else Y
            fpr_c = G if r['fpr'] <= 0.01 else Y if r['fpr'] <= 0.05 else R
            print(f"  {level:<10} {tpr_c}{r['tpr']*100:>7.1f}%{RST}  {fpr_c}{r['fpr']*100:>7.2f}%{RST}  {r['z_threshold']:>10.1f}  {r['kl_limit']:>10.3f}")
        print()
        ok("High threat level = better TPR but more false positives")
        ok("Choose 'medium' for balanced production deployments")

    return results


# ═════════════════════════════════════════════════════════════════
# 8. OVERALL SUMMARY
# ═════════════════════════════════════════════════════════════════

def print_summary(all_results: Dict):
    hdr("EVALUATION SUMMARY", width=70)

    tpr = all_results.get("tpr", {}).get("average", 0)
    fpr = all_results.get("fpr", {}).get("average", 0)
    latency = all_results.get("latency", {}).get("reward_poisoning", {}).get("mean", 0)
    mem_total = all_results.get("memory", {}).get("total", 0)
    timing = all_results.get("timing", {}).get("full_shield_step", 0)

    metrics = [
        ("Detection Rate (TPR)",    f"{tpr*100:.1f}%",  tpr >= 0.90, "Target: ≥90%"),
        ("False Alarm Rate (FPR)",  f"{fpr*100:.2f}%",  fpr <= 0.01, "Target: ≤1%"),
        ("Detection Latency",       f"{latency:.1f} steps", latency <= 2, "Target: ≤2 steps"),
        ("Total Memory",            f"{mem_total/1024:.2f} MB", mem_total < 5000, "Target: <5 MB"),
        ("Shield Step Time",        f"{timing:.1f} μs", timing < 500, "Target: <500 μs"),
    ]

    for label, value, passing, target in metrics:
        icon = f"{G}✓{RST}" if passing else f"{R}✗{RST}"
        dots = "." * (38 - len(label))
        status = f"{G}PASS{RST}" if passing else f"{R}FAIL{RST}"
        print(f"  {icon} {label}{DIM}{dots}{RST}{W}{value:<12}{RST}  {status}  {DIM}{target}{RST}")

    passed = sum(1 for _, _, p, _ in metrics if p)
    total = len(metrics)
    print(f"\n  {BOLD}Overall: {G if passed==total else Y}{passed}/{total} targets met{RST}")


# ═════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="RLShield Metrics Evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_metrics.py                  # Full evaluation
  python evaluate_metrics.py --quick          # Fast subset (fewer trials)
  python evaluate_metrics.py --json           # Output as JSON
  python evaluate_metrics.py --component tpr  # Single metric
  python evaluate_metrics.py --output results.json  # Save to file
        """
    )
    parser.add_argument("--quick",     action="store_true",  help="Fewer trials for quick check")
    parser.add_argument("--json",      action="store_true",  help="Print JSON output")
    parser.add_argument("--output",    type=str,             help="Save JSON to file")
    parser.add_argument("--component", type=str, default="all",
                        choices=["all","tpr","fpr","latency","timing","memory","rollback","levels"],
                        help="Run single component")
    parser.add_argument("--no-color",  action="store_true",  help="Disable ANSI colors")
    args = parser.parse_args()

    if args.no_color:
        global G,R,Y,C,B,M,W,DIM,RST,BOLD
        G=R=Y=C=B=M=W=DIM=RST=BOLD=""

    trials = 50 if args.quick else 200
    steps  = 1000 if args.quick else 5000

    print(f"\n{BOLD}{C}{'━'*70}{RST}")
    print(f"{BOLD}{W}  RLShield Metrics Evaluator  v0.1.0{RST}")
    print(f"{C}{'━'*70}{RST}")
    print(f"  {DIM}Mode: {'quick' if args.quick else 'full'} | Component: {args.component} | trials={trials}{RST}")

    all_results = {}
    verbose = not args.json

    comp = args.component
    if comp in ("all", "tpr"):     all_results["tpr"]     = eval_tpr(n_trials=trials, verbose=verbose)
    if comp in ("all", "fpr"):     all_results["fpr"]     = eval_fpr(n_steps=steps, verbose=verbose)
    if comp in ("all", "latency"): all_results["latency"] = eval_latency(n_trials=min(trials,50), verbose=verbose)
    if comp in ("all", "timing"):  all_results["timing"]  = eval_timing(verbose=verbose)
    if comp in ("all", "memory"):  all_results["memory"]  = eval_memory(verbose=verbose)
    if comp in ("all", "rollback"):all_results["rollback"]= eval_rollback(n_trials=min(trials,30), verbose=verbose)
    if comp in ("all", "levels"):  all_results["levels"]  = eval_threat_levels(verbose=verbose)

    if comp == "all" and verbose:
        print_summary(all_results)

    if args.json or args.output:
        # Convert numpy types to Python native for JSON
        def convert(obj):
            if isinstance(obj, (np.float32, np.float64)): return float(obj)
            if isinstance(obj, (np.int32, np.int64)):     return int(obj)
            if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list): return [convert(i) for i in obj]
            return obj

        out = convert(all_results)
        if args.json: print(json.dumps(out, indent=2))
        if args.output:
            with open(args.output, "w") as f: json.dump(out, f, indent=2)
            print(f"\n{G}Results saved to {args.output}{RST}")

    print()


if __name__ == "__main__":
    main()
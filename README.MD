# RLShield

**Plug-and-play security hardening for Reinforcement Learning pipelines.**

> Transparent, zero-architecture-change security layer for RL training.  
> Wraps your existing environment, policy, and trainer in 3 lines of code.

---

## Table of Contents

1. [What Is This?](#what-is-this)
2. [Why Does This Exist?](#why-does-this-exist)
3. [Who Is This For?](#who-is-this-for)
4. [What Problems Does It Solve?](#what-problems-does-it-solve)
5. [How It Works](#how-it-works)
6. [System Architecture](#system-architecture)
7. [Data Flow](#data-flow)
8. [Component Deep-Dive](#component-deep-dive)
9. [Attack Coverage](#attack-coverage)
10. [Quick Start](#quick-start)
11. [Integration Guide](#integration-guide)
12. [Configuration](#configuration)
13. [Metrics and Performance](#metrics-and-performance)
14. [Project Structure](#project-structure)
15. [Running Tests](#running-tests)
16. [Roadmap](#roadmap)

---

## What Is This?

**RLShield** is a Python security library that wraps any Reinforcement Learning training pipeline with real-time attack detection and defense — without touching a single line of your existing RL code.

It supports **10 major RL algorithms**: PPO, DQN, SAC, TD3, DDPG, A2C, A3C, REINFORCE, TRPO, and DreamerV3.

### In one sentence

RLShield sits between your RL environment, policy, and training loop, silently monitoring every data point flowing through the pipeline, and fires alerts (or blocks updates) when it detects signs of tampering.

### What it looks like in code

```python
# Before RLShield — your normal code
obs, reward, done, info = env.step(action)
loss = compute_ppo_loss(old_log_probs, new_log_probs, advantages)

# After RLShield — 3 added lines, everything else unchanged
import rlshield
env    = rlshield.protect_env(env, algo="PPO")
policy = rlshield.protect_policy(policy, algo="PPO")

obs, reward, done, info = env.step(action)              # auto-secured
loss = compute_ppo_loss(old_log_probs, new_log_probs, advantages)  # same as before
```

---

## Why Does This Exist?

Reinforcement Learning is increasingly deployed in high-stakes, real-world systems — autonomous vehicles, robotics, financial trading, medical devices, and the RLHF pipelines inside large language models like GPT and Claude.

These systems are uniquely vulnerable to a class of attacks that most security tools completely ignore.

### The Problem Nobody Is Talking About

Traditional cybersecurity focuses on network intrusion, malware, and data breaches. But RL systems have a completely different attack surface:

| Traditional System | RL System |
|--------------------|-----------|
| Attack the network | Attack the reward signal |
| Steal data | Poison the replay buffer |
| Inject malware | Manipulate the policy gradient |
| DDoS the server | Cause entropy collapse in SAC |
| Phishing | Adversarial observations |

An attacker who can manipulate the reward signal by even a few points — or inject a handful of fake transitions into a replay buffer — can gradually steer a trained RL agent toward completely different behavior, all without ever touching the model weights directly.

### Real-World Examples of This Problem

**Autonomous Vehicles:** A compromised CAN bus sensor feeding fake observations to an RL-based lane-keeping system could gradually cause the agent to learn unsafe behaviors over thousands of miles of operation.

**RLHF (LLM Training):** Poisoned human feedback data during PPO fine-tuning can embed backdoor behaviors into language models that only activate on specific triggers — and this happens silently, within the normal KL bounds of PPO's clip.

**Robotics:** A manufacturing robot trained with RL can be made to gradually increase actuator stress through reward poisoning, causing "accidental" mechanical failure months after the attack.

**Financial RL:** A replay buffer poisoning attack in a trading RL agent can inject fake high-reward transitions for specific market conditions, creating hidden vulnerabilities that are only exploited when the attacker chooses.

### Why Existing Tools Do Not Help

- Intrusion detection systems do not understand RL training pipelines
- ML safety tools focus on inference-time adversarial examples, not training-time attacks
- Monitoring dashboards log metrics but do not detect anomalous patterns within them
- There was no dedicated RL security library before RLShield

RLShield was built specifically to fill this gap.

---

## Who Is This For?

### Primary Users

**RL Researchers** who want to study attack/defense dynamics in controlled experiments without building infrastructure from scratch.

**ML Engineers** deploying RL in production who need a security layer that does not require rewriting their existing training code.

**Automotive and Robotics Engineers** building safety-critical RL systems in ISO 26262 or IEC 61508 compliance contexts.

**AI Safety Researchers** studying the security properties of RLHF pipelines for LLM training.

### Secondary Users

**Penetration Testers** who want a reference implementation of what RL attacks look like in order to test defenses.

**Academic Researchers** who need reproducible metrics for TPR, FPR, and detection latency in RL security papers.

---

## What Problems Does It Solve?

RLShield defends against 17 distinct attack scenarios across the full RL training lifecycle:

### Training-Time Attacks

| Attack | What Happens | Impact |
|--------|-------------|--------|
| Reward Poisoning | Attacker injects artificially high or low rewards | Policy learns wrong behavior |
| Reward Spoofing | Environment reward function is manipulated | Agent optimizes fake objective |
| Replay Buffer Injection | Fake (s, a, r, s') transitions inserted into memory | Corrupts Q-value estimates |
| Duplicate Replay | Same malicious transition injected repeatedly | Amplifies poisoning effect |
| Impossible Transitions | Teleporting state jumps injected into buffer | Destabilizes value function |

### Policy Update Attacks (PPO-Specific)

| Attack | What Happens | Impact |
|--------|-------------|--------|
| KL Violation | Policy update pushes far outside trust region | Catastrophic policy collapse |
| Clip Exploitation | Attacker crafts updates that stay within clip bounds but drift policy | Slow undetected manipulation |
| Gradient Explosion | Poisoned gradients sent to optimizer | Weight corruption |
| Gradient Injection | Targeted layer-specific gradient manipulation | Precise behavioral modification |

### Observation and Inference Attacks

| Attack | What Happens | Impact |
|--------|-------------|--------|
| Adversarial Observations | FGSM or PGD noise added to observations | Wrong actions at inference |
| State Teleportation | Observation jumps impossibly between frames | Policy confusion |
| Sensor Spoofing | Physical sensor data manipulated in robotics | Wrong state estimate |

### Behavioral Attacks

| Attack | What Happens | Impact |
|--------|-------------|--------|
| Policy Drift / Backdoor | Gradual policy manipulation over many steps | Hidden unsafe behaviors |
| Entropy Collapse (SAC) | Temperature parameter manipulated | Policy becomes dangerously deterministic |
| Q-Value Explosion | Q-value estimates inflated | Overconfident, unstable policy |

---

## How It Works

RLShield uses a **multi-layer defense architecture** where each layer independently monitors a different component of the RL pipeline. An attacker must defeat all active layers simultaneously to succeed.

### The Core Principle: Wrap, Don't Modify

RLShield never modifies your RL algorithm. Instead, it wraps each interface point with a transparent security layer that:

1. Inspects every value flowing through that interface
2. Validates it against statistical expectations built from clean training history
3. Cleans or blocks anomalous values before they reach the algorithm
4. Alerts through your chosen channel (log, exception, callback)

```
                ┌─────────────────────────────────────────┐
                │           Your RL Algorithm              │
                │   (PPO / DQN / SAC / TD3 / any algo)    │
                └─────────────────────────────────────────┘
                      ↑              ↑              ↑
           Policy Update         Actions        Gradients
                      │              │              │
          ┌───────────┼──────────────┼──────────────┼────────┐
          │    PPODefender     PolicyWrapper   GradientMonitor│
          │    KL watchdog     Obs defense     Norm monitoring│
          └───────────┼──────────────┼──────────────┼────────┘
                      │              │              │
                Transitions      Observations     Rewards
                      │              │              │
          ┌───────────┼──────────────┼──────────────┼────────┐
          │  BufferDefender  ObsDefender      RewardDefender  │
          │  Transition      Adversarial      Z-score + EMA   │
          │  validation      detection        reward cleaning  │
          └────────────────────────────────────────────────────┘
                                    ↑
                          Your Gym Environment
```

### Five Defense Mechanisms

#### 1. Statistical Anomaly Detection

Every numeric value entering the pipeline is scored against a rolling statistical model of what "normal" looks like in your specific training run. An anomaly score (Z-score) is computed in O(1) time using a circular buffer with a running sum.

```
z_score = |value - rolling_mean| / rolling_std

if z_score > threshold:
    ALERT and clip or block
```

The threshold is configurable by `threat_level`:

- `low` → Z=4.0 (very permissive, few false positives)
- `medium` → Z=3.0 (balanced, default)
- `high` → Z=2.0 (strict, catches subtle attacks)

#### 2. Structural Validation

Some attacks leave structural signatures that statistical methods alone cannot catch:

- Transition validator checks that the L2 norm of state change is physically plausible
- Shape validator ensures s.shape equals s_next.shape — injected transitions often have wrong dimensions
- Bounds validator hard-clips rewards to absolute bounds regardless of statistics
- Hash deduplication detects identical transitions being injected repeatedly

#### 3. PPO-Specific Trust Region Enforcement

PPO's clip mechanism is designed to keep policy updates bounded, but an attacker can craft gradual updates that stay within the clip while shifting policy behavior. RLShield adds three additional layers:

- KL hard limit: computes KL divergence and blocks the entire update if it exceeds the threshold
- KL trend monitoring: detects if KL is consistently trending upward across updates
- Clip fraction monitoring: alerts if more than 50% of probability ratios are being clipped

#### 4. Behavioral Drift Detection

Policy drift attacks are the hardest to detect because each individual update looks legitimate. RLShield detects them by:

- Taking snapshots of the policy's action distribution on a fixed set of probe states every N steps
- Computing normalized L2 distance between consecutive snapshots
- If drift exceeds threshold, firing an alert and optionally rolling back the policy to a saved snapshot

#### 5. Gradient Monitoring

Three types of gradient anomalies are detected:

- Explosion: hard threshold violation and Z-score violation — stops the optimizer step entirely
- Vanishing: gradient norm below 1e-7 — policy has stopped learning
- Trend: linear regression slope of gradient norms over last 30 steps above threshold — sustained destabilization

---

## System Architecture

```
rlshield/
│
├── rlshield.py                  Main API entry point (RLShield class)
├── __init__.py                  Public exports
│
├── core/                        Foundation layer
│   ├── base_defender.py         Abstract base class all defenders inherit
│   ├── threat_model.py          AttackType enum, Severity enum, ThreatEvent
│   └── alert_system.py          Central event bus (log/raise/callback/silent)
│
├── defenders/                   Per-component security modules
│   ├── reward_defender.py       Reward signal protection
│   ├── observation_defender.py  State space adversarial defense
│   ├── ppo_defender.py          PPO-specific: KL, clip, gradient hooks
│   ├── policy_defender.py       General: Q-values, entropy, actions
│   └── buffer_defender.py       Replay/rollout buffer transition validation
│
├── detectors/                   Monitoring modules
│   ├── drift_detector.py        Behavioral drift + automatic rollback
│   ├── anomaly_detector.py      General-purpose statistical anomaly detection
│   └── gradient_monitor.py      Gradient norm explosion/vanishing/trend
│
├── wrappers/                    Transparent interface wrappers
│   ├── env_wrapper.py           Gym/Gymnasium drop-in replacement
│   ├── policy_wrapper.py        Policy forward-pass security wrapper
│   └── trainer_wrapper.py       Training loop hook interface
│
├── utils/                       Shared utilities
│   ├── config.py                RLShieldConfig dataclass with presets
│   ├── statistics.py            RollingStats, EMA, TrendDetector
│   └── snapshot.py              Policy snapshot and rollback manager
│
├── tests/
│   └── test_rlshield.py         93 unit tests (all passing)
│
├── evaluate_metrics.py          TPR/FPR/Latency/Timing/Memory benchmarks
└── examples.py                  6 complete working examples
```

### Design Principles

**Zero Architecture Change.** Every defender is a wrapper. Your RL code never changes. If you remove RLShield, your code runs exactly as before.

**Independent Defense Layers.** Each defender works independently. Disabling one does not affect others. An attacker must defeat all active layers simultaneously.

**Fail Safe.** When uncertain, RLShield clips and cleans rather than blocking, preferring to preserve training progress over maximum security. Hard blocks (like KL violations) are reserved for clear safety violations.

**Honest Reporting.** RLShield reports FPR honestly. The evaluation script measures real metrics, not cherry-picked ones.

---

## Data Flow

This section traces exactly what happens to a single data point at each step of a shielded training loop.

### Step 1: Environment Observation

```
Raw env.step(action)
        |
        v
SecureEnvWrapper.step()
        |
        |-- ObservationDefender.defend(obs)
        |       |-- Check: norm(obs - prev_obs) < max_delta    [teleport detection]
        |       |-- Update running obs min/max bounds
        |       +-- Soft-clip to [min - 10%, max + 10%]
        |
        |-- RewardDefender.defend(reward)
        |       |-- Hard clip: reward in [reward_min, reward_max]
        |       |-- Z-score check against rolling history
        |       |-- EMA smoothing (alpha=0.99)
        |       +-- Update rolling stats
        |
        +-- Return (secured_obs, secured_reward, done, info)
```

### Step 2: Buffer Storage

```
transition = (s, a, r, s_next, done)
        |
        v
SecureTrainerWrapper.on_transition()
        |
        +-- BufferDefender.defend(transition)
                |-- Validate tuple structure (length, types)
                |-- Check: r in [reward_min, reward_max]
                |-- Check: Z-score of r vs rolling reward history
                |-- Check: norm(s_next - s) < max_state_delta
                |-- Check: s.shape == s_next.shape
                |-- Check: action in action_space (if provided)
                |-- Hash-based duplicate detection
                +-- Return transition if valid, None if rejected
```

### Step 3: PPO Policy Update

```
old_log_probs, new_log_probs, advantages
        |
        v
SecureTrainerWrapper.ppo_before_update_torch()
        |
        +-- PPODefender.defend_update_torch()
                |-- Compute KL approximation:
                |       log_ratio = new_lp - old_lp
                |       kl ~ mean((exp(log_ratio) - 1) - log_ratio)
                |
                |-- If kl > kl_hard_limit:
                |       Alert(KL_VIOLATION, HIGH)
                |       Return None  [UPDATE IS BLOCKED]
                |
                |-- Compute clip_fraction:
                |       ratio = exp(new_lp - old_lp)
                |       clip_fraction = mean(ratio not in [1-e, 1+e])
                |       If > clip_fraction_limit: Alert(CLIP_EXPLOITATION)
                |
                +-- Compute and return secure PPO loss
```

### Step 4: Gradient Step

```
loss.backward()
        |
        v
SecureTrainerWrapper.after_backward(model)
        |
        +-- PPODefender.defend_gradients_torch(model)
                |-- Compute per-parameter gradient norms
                |-- total_norm = mean of all norms
                |
                |-- If total_norm > 10 x max_grad_norm:
                |       Alert(GRADIENT_EXPLOSION, CRITICAL)
                |       Zero out ALL gradients
                |       Return False  [OPTIMIZER STEP SKIPPED]
                |
                |-- torch.nn.utils.clip_grad_norm_(model, max_grad_norm)
                +-- Return True  [optimizer.step() proceeds]
```

### Step 5: Drift Detection (Periodic, every snapshot_interval steps)

```
DriftDetector.update(policy_fn, policy_model, step)
        |
        |-- Call policy_fn on all probe_states
        |       -> current_action_distribution
        |
        |-- If snapshots exist:
        |       drift = norm(normalize(prev_dist) - normalize(curr_dist))
        |
        |       If drift > drift_threshold:
        |               Alert(POLICY_DRIFT, HIGH)
        |               If auto_rollback:
        |                       Restore policy from snapshot[-2]
        |                       Alert(POLICY_DRIFT, CRITICAL, "rollback_executed")
        |
        +-- Save current_action_distribution as new snapshot
```

### Step 6: Alert Dispatch

```
Any Alert(attack_type, severity, details)
        |
        v
AlertSystem.fire()
        |
        |-- Create ThreatEvent(attack_type, severity, details, timestamp, step)
        |-- Append to event log
        |
        +-- Dispatch based on alert_mode:
                |-- "log"      -> logging.warning/error/critical to console
                |-- "raise"    -> raise SecurityAlertException (if severity >= HIGH)
                |-- "callback" -> call user_callback(ThreatEvent)
                +-- "silent"   -> store only, no output
```

---

## Component Deep-Dive

### RewardDefender

Protects the reward signal — the most critical channel in any RL system. An attacker who controls rewards controls what the agent learns.

Maintains a rolling window of recent rewards. On each new reward, computes a Z-score, applies EMA smoothing, and hard-clips to absolute bounds. Consecutive anomalies escalate severity from MEDIUM to HIGH. All filtering is in-place: the reward is cleaned and returned, never silently dropped.

Key parameters: `reward_window` (default 1000), `z_threshold` (default 3.0), `ema_alpha` (default 0.99), `reward_min` and `reward_max` (default ±1e6).

---

### ObservationDefender

Protects the state space from adversarial perturbations and sensor spoofing.

Stores the previous observation on each step and computes the L2 norm of the state change. If the norm exceeds `obs_max_delta`, a teleport alert fires. Also tracks running observation bounds and soft-clips new observations to prevent impossible values from entering the policy.

Optional randomized smoothing: runs the policy on 50 noisy copies of the observation and returns the majority action, alerting on low confidence.

Key parameters: `obs_max_delta` (default 1e4), `obs_epsilon` (default 0.01), `obs_cert_confidence` (default 0.7).

---

### PPODefender

The most critical defender for RLHF and production PPO deployments.

PPO's clipping mechanism prevents large single updates, but an attacker can craft a sequence of small updates — each individually within the clip bounds — that collectively shift policy behavior significantly. This is called clip exploitation.

Before every gradient update: computes approximate KL divergence. If above the limit, blocks the update entirely. Also tracks clip fraction (alerts if above 50%) and gradient norms (zeros gradients and skips optimizer step on explosion). Periodically detects if KL is trending upward across updates using linear regression.

Key parameters: `kl_hard_limit` (default 0.05), `clip_eps` (default 0.2), `max_grad_norm` (default 0.5).

---

### PolicyDefender (DQN, SAC, TD3, DDPG)

General defense for non-PPO algorithms. Monitors Q-values, entropy, and action distributions.

For DQN, TD3, and DDPG: tracks Q-value rolling statistics and alerts on Z-score violations from replay poisoning.

For SAC specifically: monitors policy entropy and detects two failure modes — near-zero entropy (absolute collapse, policy has become dangerously deterministic) and declining entropy trend (temperature alpha being manipulated).

---

### BufferDefender

Protects replay buffers (DQN, SAC, TD3, DDPG) and rollout buffers (PPO, A2C) from transition injection.

Validates every (s, a, r, s', done) tuple before it enters the buffer. Checks reward bounds, state delta bounds, shape consistency, and action space validity. Uses hash-based duplicate detection with a bounded 10,000-entry hash set. Returns None for rejected transitions so the caller knows not to add them.

Buffer rejection rate is tracked and reported — a sudden spike in rejection rate can itself be a signal of an ongoing attack.

---

### DriftDetector

Catches gradual policy manipulation that individual-update defenders miss.

An attacker injecting a single poisoned gradient will be caught by PPODefender. But an attacker who carefully stays within bounds across thousands of steps cannot be caught per-step. DriftDetector monitors the cumulative behavioral effect by comparing the policy's action distribution on a fixed set of probe states over time.

When drift exceeds the threshold, it alerts and (if `auto_rollback=True`) restores the policy weights from a snapshot taken before the drift was detected. Maximum 5 snapshots are kept in a circular buffer.

---

### GradientMonitor

Dedicated, standalone gradient norm monitoring with per-layer anomaly detection.

Detects three distinct failure modes: explosion (hard threshold + Z-score violation), vanishing (norm below 1e-7, policy has stopped learning), and trend (linear regression slope over last 30 steps above threshold, indicating sustained destabilization).

When called with `update_from_model(model)` instead of a scalar norm, also performs per-layer analysis to detect single-layer gradient spikes — a signature of targeted layer attacks.

---

## Attack Coverage

| Attack | Defender | Detection Method | Action |
|--------|----------|-----------------|--------|
| Reward Poisoning | RewardDefender | Z-score above threshold | Clip to bounds |
| Reward Spoofing | RewardDefender | Hard bounds check | Clip to absolute range |
| Obs Adversarial | ObservationDefender | Randomized smoothing | Certified action |
| Obs Teleport | ObservationDefender | L2 state delta | Alert |
| Sensor Spoofing | ObservationDefender | Bounds tracking | Soft clip |
| KL Violation | PPODefender | KL approximation | Block update |
| Clip Exploitation | PPODefender | Clip fraction above 50% | Alert |
| Gradient Explosion | PPODefender / GradientMonitor | Norm threshold | Zero gradients |
| Gradient Vanishing | GradientMonitor | Norm below 1e-7 | Alert |
| Buffer Injection | BufferDefender | Structural and statistical | Reject transition |
| Duplicate Replay | BufferDefender | Hash comparison | Alert and reject |
| Impossible Transition | BufferDefender | State delta check | Reject |
| Q-Value Explosion | PolicyDefender | Z-score on Q means | Alert |
| Entropy Collapse | PolicyDefender | Absolute and trend check | Alert |
| Action Anomaly | PolicyDefender | Z-score on action norm | Alert |
| Policy Drift / Backdoor | DriftDetector | Behavioral snapshot diff | Alert and rollback |
| RLHF Poisoning | PPODefender and DriftDetector | KL and behavioral drift | Block and rollback |

---

## Quick Start

### Installation

```bash
# No pip package yet — install from source
git clone https://github.com/yourname/rlshield.git
cd rlshield
pip install numpy  # only hard dependency
```

### 3-Line Setup

```python
import rlshield

env     = rlshield.protect_env(env, algo="PPO")
policy  = rlshield.protect_policy(policy, algo="PPO")
trainer = rlshield.protect_trainer(trainer, algo="PPO")
```

### Full Setup with Options

```python
from rlshield import RLShield

shield = RLShield(
    algo="PPO",              # PPO, DQN, SAC, TD3, DDPG, A2C, A3C, REINFORCE, TRPO, DreamerV3
    threat_level="high",     # low | medium | high
    alert_mode="log",        # log | raise | callback | silent
    auto_rollback=True,
    probe_states=fixed_states,
)

env    = shield.protect_env(env)
policy = shield.protect_policy(policy)

shield.print_summary()
report = shield.get_threat_report()
```

---

## Integration Guide

### With Stable-Baselines3

```python
import gymnasium as gym
from stable_baselines3 import PPO
from rlshield import RLShield

env = gym.make("CartPole-v1")
shield = RLShield(algo="PPO", threat_level="medium", alert_mode="log")
env = shield.protect_env(env)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

shield.print_summary()
```

### Manual PPO Training Loop

```python
shield = RLShield(algo="PPO", threat_level="high")
trainer = shield.protect_trainer(my_trainer)

for epoch in range(num_epochs):
    for step in range(rollout_steps):
        obs, reward, done, _, info = env.step(action)

        t = trainer.on_transition(s, action, reward, obs, done)
        if t:
            rollout_buffer.add(*t)

    for minibatch in rollout_buffer.get_minibatches():
        loss = shield.ppo_secure_update_torch(old_log_probs, new_log_probs, advantages)

        if loss is None:
            continue  # KL violation — skip this batch

        loss.backward()

        if shield.ppo_secure_gradients(model):
            optimizer.step()

        trainer.on_step(policy_fn=lambda o: model.predict(o), policy_model=model, step=global_step)
```

### SAC Training Loop

```python
shield = RLShield(algo="SAC", threat_level="medium")
trainer = shield.protect_trainer(my_trainer)

for step in range(total_steps):
    obs, reward, done, _, _ = env.step(action)

    t = trainer.on_transition(state, action, reward, obs, done)
    if t:
        replay_buffer.add(*t)

    if len(replay_buffer) > batch_size:
        batch = replay_buffer.sample(batch_size)

        q_values = critic(batch.obs, batch.actions)
        trainer.on_q_values(q_values.detach().numpy())

        entropy = dist.entropy().mean().item()
        trainer.on_entropy(entropy)

        loss = compute_sac_loss(batch)
        trainer.on_loss(loss.item())

        loss.backward()
        trainer.after_backward(actor)
        optimizer.step()
```

### Alert Callback Mode

```python
def my_security_handler(event):
    print(f"ATTACK: {event.attack_type.value} | Severity: {event.severity.label()}")
    if event.severity.value >= 3:
        send_to_siem(event.to_dict())

shield = RLShield(
    algo="PPO",
    alert_mode="callback",
    callback=my_security_handler,
)
```

---

## Configuration

### Threat Level Presets

| Parameter | low | medium | high |
|-----------|-----|--------|------|
| z_threshold | 4.0 | 3.0 | 2.0 |
| kl_hard_limit | 0.10 | 0.05 | 0.02 |
| drift_threshold | 0.20 | 0.10 | 0.05 |
| clip_fraction_limit | 0.70 | 0.50 | 0.30 |
| Expected FPR | ~0.003% | ~0.27% | ~4.6% |
| Use case | Research | Production | Safety-critical |

### Custom Configuration

```python
from rlshield import RLShieldConfig, RLShield

config = RLShieldConfig(
    algo="SAC",
    threat_level="high",
    alert_mode="callback",
    reward_window=500,
    z_threshold=2.5,
    ema_alpha=0.995,
    reward_min=-10.0,
    reward_max=10.0,
    obs_max_delta=50.0,
    kl_hard_limit=0.03,
    snapshot_interval=500,
    drift_threshold=0.08,
    auto_rollback=True,
    max_snapshots=10,
    grad_norm_multiplier=8.0,
    grad_zero_on_alert=True,
)

shield = RLShield(config=config)
```




### Running the Evaluation Script

```bash
python evaluate_metrics.py             # full evaluation
python evaluate_metrics.py --quick     # fast mode (~15 seconds)
python evaluate_metrics.py --json      # JSON output
python evaluate_metrics.py --component tpr
python evaluate_metrics.py --component timing
python evaluate_metrics.py --output results.json
```

---

## Project Structure

```
rlshield/
├── rlshield.py                Main API and convenience functions
├── __init__.py                Package exports
├── core/
│   ├── base_defender.py       Abstract base: enabled, tick, _alert
│   ├── threat_model.py        AttackType, Severity, ThreatEvent
│   └── alert_system.py        Central dispatch (log/raise/callback/silent)
├── defenders/
│   ├── reward_defender.py     Z-score + EMA + hard bounds
│   ├── observation_defender.py  Teleport + soft clip + certification
│   ├── ppo_defender.py        KL watchdog + clip fraction + gradient zero
│   ├── policy_defender.py     Q-values + entropy + actions (non-PPO)
│   └── buffer_defender.py     Structural + statistical + hash dedup
├── detectors/
│   ├── drift_detector.py      Probe-state snapshots + rollback
│   ├── anomaly_detector.py    General Z-score detector
│   └── gradient_monitor.py    Explosion + vanishing + trend + per-layer
├── wrappers/
│   ├── env_wrapper.py         Gym/Gymnasium drop-in
│   ├── policy_wrapper.py      Forward-pass wrapper
│   └── trainer_wrapper.py     Training loop hooks
├── utils/
│   ├── config.py              RLShieldConfig + presets
│   ├── statistics.py          RollingStats, EMA, TrendDetector
│   └── snapshot.py            SnapshotManager (save/rollback)
├── tests/
│   └── test_rlshield.py       93 unit tests
├── evaluate_metrics.py        Benchmark and metrics script
└── examples.py                6 complete usage examples
```

---

## Running Tests

```bash
# From the parent directory containing rlshield/
python -m rlshield.tests.test_rlshield

# Expected result:
# Results: 93/93 passed | 0 failed
```

---

## Roadmap

**v0.2 — Performance Optimization**
Fix O(w) to O(1) in RollingStats using circular buffer (140× speedup). Rolling hash set for BufferDefender (3× speedup). defend_batch() API for vectorized episode-level defense.

**v0.3 — Extended Algorithm Support**
RLHF-specific pipeline wrapper for LLM fine-tuning. Ensemble reward model defense. Rater anomaly detection for human preference data.

**v0.4 — Advanced Detection**
Counterfactual drift detection. Adaptive threshold tuning via online calibration. Per-layer gradient attack attribution.

**v0.5 — PyPI Release**
`pip install rlshield`. SB3, RLlib, and CleanRL integrations. Full documentation site.

---

## Citation

If you use RLShield in academic work, please cite:

```
@software{rlshield2026,
  title   = {RLShield: Plug-and-Play Security Hardening for Reinforcement Learning},
  author  = {Harshith Madhavaram},
  year    = {2026},
  url     = {https://github.com/Harshith2412/rlshield},
  version = {0.1.0}
}
```

---

*RLShield v0.1.0 —  March 2026.*  
*93/93 tests passing — PPO · DQN · SAC · TD3 · DDPG · A2C · A3C · REINFORCE · TRPO · DreamerV3*

## Security

This project uses [Snyk](https://snyk.io/) to help identify and remediate vulnerabilities through the Snyk Secure Developer Program.

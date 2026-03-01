"""
RLShield Usage Examples
=======================
Complete working examples for all supported algorithms.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rlshield
from rlshield import RLShield, RLShieldConfig



def example_minimal():
    print("\n" + "=" * 60)
    print("  Example 1: Minimal 3-Line Setup")
    print("=" * 60)

    class FakeEnv:
        observation_space = None
        action_space = None
        def step(self, a): return np.random.randn(4).astype(np.float32), 1.0, False, False, {}
        def reset(self, **kw): return np.random.randn(4).astype(np.float32), {}

    class FakeTrainer:
        pass

    env     = rlshield.protect_env(FakeEnv(), algo="PPO")
    policy  = rlshield.protect_policy(lambda obs: np.array([0.5]), algo="PPO")
    trainer = rlshield.protect_trainer(FakeTrainer(), algo="PPO")

    obs, info = env.reset()
    obs, reward, term, trunc, info = env.step(np.array([0.5]))
    action = policy.predict(obs)

    print(f"  Obs shape  : {obs.shape}")
    print(f"  Reward     : {reward:.4f}")
    print(f"  Action     : {action}")
    print("  Minimal setup complete.")



def example_ppo_full():
    print("\n" + "=" * 60)
    print("  Example 2: Full PPO Hardening")
    print("=" * 60)

    shield = RLShield(
        algo="PPO",
        threat_level="high",
        alert_mode="log",
        auto_rollback=True,
    )

    for step in range(50):

        old_lp = np.log(np.array([0.5, 0.3, 0.2]) + 1e-8)
        new_lp = np.log(np.array([0.48, 0.31, 0.21]) + 1e-8)
        advantages = np.array([1.0, -0.5, 0.3])

        loss = shield.ppo_secure_update(old_lp, new_lp, advantages)

        if loss is None:
            print(f"  Step {step}: Update skipped (KL violation)")

        raw_reward = np.random.normal(0, 1)
        secured_reward = shield.defend_reward(raw_reward)

        raw_obs = np.random.randn(4).astype(np.float32)
        secured_obs = shield.defend_observation(raw_obs)

    print("\n  [Attack] Injecting reward poisoning...")
    raw_reward_attack = 1e9
    secured = shield.defend_reward(raw_reward_attack)
    print(f"  Poisoned reward {raw_reward_attack} → secured to {secured:.4f}")

    print("\n  [Attack] Injecting KL violation in PPO update...")
    old_lp = np.log(np.array([0.5, 0.3, 0.2]) + 1e-8)
    bad_lp  = np.log(np.array([0.99, 0.005, 0.005]) + 1e-8)
    result = shield.ppo_secure_update(old_lp, bad_lp, np.array([1.0, -1.0, 0.5]))
    print(f"  KL violation result: {'Update Blocked ' if result is None else 'Allowed ⚠'}")

    shield.print_summary()



def example_general_algo():
    print("\n" + "=" * 60)
    print("  Example 3: DQN / SAC / TD3 Setup")
    print("=" * 60)

    for algo in ["DQN", "SAC", "TD3"]:
        shield = RLShield(algo=algo, threat_level="medium", alert_mode="silent")

        class FakeTrainer:
            pass

        trainer = shield.protect_trainer(FakeTrainer())

        for _ in range(50):
            trainer.on_q_values(np.random.normal(10, 1, size=4))
            trainer.on_loss(np.random.uniform(0.1, 0.5))

        anomaly = trainer.on_q_values(np.full(4, 1e8))
        loss_anomaly = trainer.on_loss(1e6)

        s = np.array([1.0, 2.0])
        good = trainer.on_transition(s, 0, 1.0, s + 0.1, False)
        bad  = trainer.on_transition(s, 0, 2e6, s + 1e6, False)

        report = shield.get_threat_report()
        print(f"  {algo:10} — Alerts: {report['total']}, "
              f"Q-anomaly: {'Yes' if anomaly else 'No'}, "
              f"Buffer injection blocked: {'Yes' if bad is None else 'No'}")

    if algo == "SAC":
        for _ in range(40):
            trainer.on_entropy(1.5 + np.random.randn() * 0.1)
        entropy_alert = trainer.on_entropy(1e-10)
        print(f"  SAC entropy collapse detected: {'Yes' if entropy_alert else 'No'}")



def example_custom_config():
    print("\n" + "=" * 60)
    print("  Example 4: Custom Config")
    print("=" * 60)

    config = RLShieldConfig(
        algo="PPO",
        threat_level="high",
        alert_mode="silent",
        z_threshold=2.0,
        kl_hard_limit=0.02,
        max_grad_norm=0.3,
        clip_eps=0.15,
        reward_window=500,
        snapshot_interval=100,
        drift_threshold=0.05,
        auto_rollback=False,   
    )

    shield = RLShield(config=config)
    print(f"  Custom config — algo={config.algo}, z_thresh={config.z_threshold}, kl_limit={config.kl_hard_limit}")

    # Show config as dict
    d = config.to_dict()
    print(f"  Config keys: {list(d.keys())[:6]}...")
    print("  Custom config applied.")



def example_callback():
    print("\n" + "=" * 60)
    print("  Example 5: Callback Alert Mode")
    print("=" * 60)

    alerts_received = []

    def my_security_callback(event):
        alerts_received.append(event)
        print(f"  🚨 ALERT [{event.severity.label()}] {event.attack_type.value}: {event.details}")

    shield = RLShield(
        algo="PPO",
        alert_mode="callback",
        callback=my_security_callback,
    )

    for _ in range(50):
        shield.defend_reward(np.random.normal(0, 1))

    # Trigger attack
    shield.defend_reward(1e9)
    print(f"\n  Total alerts received via callback: {len(alerts_received)}")



def example_drift_detection():
    print("\n" + "=" * 60)
    print("  Example 6: Policy Drift Detection")
    print("=" * 60)

    probe_states = np.random.randn(10, 4).astype(np.float32)

    config = RLShieldConfig.from_threat_level("high", algo="PPO")
    config.snapshot_interval = 5
    config.drift_threshold = 0.05
    config.auto_rollback = False
    config.alert_mode = "silent"

    shield = RLShield(config=config)
    shield.set_probe_states(probe_states)

    class FakeModel:
        def state_dict(self): return {"weights": np.ones(10)}
        def load_state_dict(self, d): pass

    def stable_policy(obs):
        return np.array([0.5, -0.5])

    def drifted_policy(obs):
        return np.array([50.0, -50.0])

    model = FakeModel()


    for step in range(1, 20):
        shield.drift_detector.update(stable_policy, model, step)

    before = shield.alert_system.total_alerts
    shield.drift_detector.update(drifted_policy, model, step=20)
    after = shield.alert_system.total_alerts

    print(f"  Alerts before drift injection : {before}")
    print(f"  Alerts after drift injection  : {after}")
    print(f"  Drift detected: {'Yes ' if after > before else 'No '}")

    stats = shield.drift_detector.stats
    print(f"  Snapshots saved : {stats['snapshots_saved']}")
    print(f"  Max drift       : {stats['max_drift']:.6f}")


if __name__ == "__main__":
    example_minimal()
    example_ppo_full()
    example_general_algo()
    example_custom_config()
    example_callback()
    example_drift_detection()
    print("  All examples completed.")
  
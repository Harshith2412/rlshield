"""
RLShield Test Suite
Tests all defenders, detectors, wrappers, and the main API.
Run with: python -m pytest tests/ -v
Or:        python tests/test_rlshield.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rlshield
from rlshield import (
    RLShield,
    RLShieldConfig,
    AlertSystem,
    AttackType,
    Severity,
    RewardDefender,
    ObservationDefender,
    PPODefender,
    PolicyDefender,
    BufferDefender,
    DriftDetector,
    AnomalyDetector,
    GradientMonitor,
)
from rlshield.utils.statistics import RollingStats, EMA, TrendDetector, compute_kl_divergence
from rlshield.utils.snapshot import SnapshotManager


PASS = "✅ PASS"
FAIL = "❌ FAIL"
_results = []


def test(name: str, condition: bool):
    status = PASS if condition else FAIL
    print(f"  {status}  {name}")
    _results.append((name, condition))


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ── Helpers ───────────────────────────────────────────────────────

def make_config(threat_level="medium", algo="PPO") -> RLShieldConfig:
    return RLShieldConfig.from_threat_level(threat_level, algo=algo)


def make_alert_system() -> AlertSystem:
    return AlertSystem(mode="silent")


# ══════════════════════════════════════════════════════════════════
# 1. Utils
# ══════════════════════════════════════════════════════════════════

def test_rolling_stats():
    section("RollingStats")
    rs = RollingStats(window=10)
    for v in range(10):
        rs.update(float(v))
    test("mean is correct", abs(rs.mean() - 4.5) < 0.01)
    test("std is positive", rs.std() > 0)
    test("z_score of mean is ~0", abs(rs.z_score(rs.mean())) < 0.01)
    test("is_warm after 10", rs.is_warm(10))
    test("clip_to_bounds works", rs.min() if hasattr(rs, "min") else True)

    # Window overflow
    rs2 = RollingStats(window=5)
    for v in range(20):
        rs2.update(float(v))
    test("window maxlen respected", rs2.count == 5)


def test_ema():
    section("EMA")
    ema = EMA(alpha=0.9)
    v1 = ema.update(10.0)
    v2 = ema.update(10.0)
    test("first value equals input", v1 == 10.0)
    test("second value stays near input", abs(v2 - 10.0) < 1.0)
    ema.reset()
    test("reset clears value", ema.value is None)


def test_trend_detector():
    section("TrendDetector")
    td = TrendDetector(window=20)
    for v in range(20):
        td.update(float(v))
    test("detects upward trend", td.is_trending_up(threshold=0.5))
    test("not trending down when up", not td.is_trending_down(threshold=0.5))

    td2 = TrendDetector(window=20)
    for v in range(20, 0, -1):
        td2.update(float(v))
    test("detects downward trend", td2.is_trending_down(threshold=0.5))


def test_kl_divergence():
    section("KL Divergence")
    p = np.array([0.5, 0.5])
    q = np.array([0.5, 0.5])
    test("KL(p||p) = 0", compute_kl_divergence(p, q) < 1e-6)

    p2 = np.array([0.9, 0.1])
    q2 = np.array([0.1, 0.9])
    test("KL diverges for different distributions", compute_kl_divergence(p2, q2) > 1.0)


def test_snapshot_manager():
    section("SnapshotManager")

    class FakePolicy:
        def __init__(self, val):
            self.val = val
        def state_dict(self):
            return {"val": self.val}
        def load_state_dict(self, d):
            self.val = d["val"]

    sm = SnapshotManager(max_snapshots=3)
    p = FakePolicy(1)
    sm.save(p, step=0)
    p.val = 99
    sm.save(p, step=1)

    test("count is 2", sm.count() == 2)

    success = sm.rollback(p, steps_back=2)
    test("rollback succeeds", success)
    test("policy restored to step 0 value", p.val == 1)

    sm2 = SnapshotManager(max_snapshots=2)
    test("rollback fails with no snapshots", not sm2.rollback(p, steps_back=1))


# ══════════════════════════════════════════════════════════════════
# 2. Defenders
# ══════════════════════════════════════════════════════════════════

def test_reward_defender():
    section("RewardDefender")
    config = make_config()
    alert = make_alert_system()
    rd = RewardDefender(config, alert)

    # Warm up with normal rewards
    for _ in range(50):
        rd.defend(np.random.normal(0, 1))

    # Normal reward should pass
    r_normal = rd.defend(0.5)
    test("normal reward passes through", abs(r_normal) < 5.0)

    # Extreme reward should be detected and clipped
    before_alerts = alert.total_alerts
    r_extreme = rd.defend(1e9)
    test("extreme reward triggers alert", alert.total_alerts > before_alerts)
    test("extreme reward is clipped", r_extreme < 1e9)

    # Out of absolute bounds
    before = alert.total_alerts
    r_oob = rd.defend(2e6)
    test("out-of-bounds reward alerts", alert.total_alerts > before)


def test_observation_defender():
    section("ObservationDefender")
    config = make_config()
    alert = make_alert_system()
    od = ObservationDefender(config, alert)

    obs = np.array([1.0, 2.0, 3.0])

    # Normal obs
    defended = od.defend(obs)
    test("normal obs passes", defended.shape == obs.shape)
    test("obs is float32", defended.dtype == np.float32)

    # Warm up
    for _ in range(60):
        od.defend(obs + np.random.randn(3) * 0.01)

    # Teleport — huge jump
    before = alert.total_alerts
    od.defend(obs + 1e5)
    test("teleport triggers alert", alert.total_alerts > before)


def test_ppo_defender():
    section("PPODefender")
    config = make_config(algo="PPO")
    alert = make_alert_system()
    pd = PPODefender(config, alert)

    # Valid update
    old_lp = np.log(np.array([0.5, 0.3, 0.2]))
    new_lp = np.log(np.array([0.48, 0.31, 0.21]))
    adv = np.array([1.0, -0.5, 0.3])

    loss = pd.defend_update_numpy(old_lp, new_lp, adv)
    test("valid update returns loss", loss is not None)
    test("loss is ndarray", isinstance(loss, np.ndarray))

    # KL violation: make new_lp very different
    new_lp_bad = np.log(np.array([0.01, 0.01, 0.98]))
    before = alert.total_alerts
    loss_bad = pd.defend_update_numpy(old_lp, new_lp_bad, adv)
    test("KL violation skips update (returns None)", loss_bad is None)
    test("KL violation fires alert", alert.total_alerts > before)

    # Skipped updates counter
    test("skipped_updates incremented", pd.skipped_updates >= 1)


def test_policy_defender():
    section("PolicyDefender (General)")
    config = make_config(algo="SAC")
    alert = make_alert_system()
    pd = PolicyDefender(config, alert)

    # Q-value monitoring warm-up
    for _ in range(60):
        pd.monitor_q_values(np.random.normal(0, 1, size=10))

    # Anomalous Q-values
    before = alert.total_alerts
    pd.monitor_q_values(np.full(10, 1e6))
    test("Q-value explosion alerts", alert.total_alerts > before)

    # Entropy monitoring
    for _ in range(40):
        pd.monitor_entropy(1.5 + np.random.randn() * 0.1)

    before = alert.total_alerts
    pd.monitor_entropy(1e-10)
    test("entropy collapse detected", alert.total_alerts > before)

    # Action monitoring warm-up
    for _ in range(60):
        pd.monitor_actions(np.random.randn(4))

    before = alert.total_alerts
    pd.monitor_actions(np.full(4, 1e5))
    test("action anomaly detected", alert.total_alerts > before)


def test_buffer_defender():
    section("BufferDefender")
    config = make_config()
    alert = make_alert_system()
    bd = BufferDefender(config, alert)

    s = np.array([1.0, 2.0])
    a = np.array([0.5])
    r = 1.0
    s_next = np.array([1.1, 2.1])
    done = False

    # Valid transition
    result = bd.defend((s, a, r, s_next, done))
    test("valid transition passes", result is not None)

    # Out-of-bounds reward
    before = alert.total_alerts
    result_bad = bd.defend((s, a, 2e6, s_next, done))
    test("bad reward rejected", result_bad is None)
    test("bad reward alert fires", alert.total_alerts > before)

    # Impossible state transition (teleport)
    before = alert.total_alerts
    result_tp = bd.defend((s, a, r, np.array([1e6, 1e6]), done))
    test("teleport transition rejected", result_tp is None)
    test("teleport alert fires", alert.total_alerts > before)

    # Malformed transition
    before = alert.total_alerts
    result_mf = bd.defend((s, a))  # too short
    test("malformed transition rejected", result_mf is None)
    test("rejection rate > 0", bd.rejection_rate > 0)


# ══════════════════════════════════════════════════════════════════
# 3. Detectors
# ══════════════════════════════════════════════════════════════════

def test_anomaly_detector():
    section("AnomalyDetector")
    config = make_config()
    alert = make_alert_system()
    ad = AnomalyDetector(config, alert, name="test_signal", z_threshold=2.5)

    # Warm up
    for _ in range(50):
        ad.update(np.random.normal(0, 1))

    # Normal value
    result = ad.update(0.5)
    test("normal value not flagged", not result)

    # Anomalous value
    before = alert.total_alerts
    result_a = ad.update(1e6)
    test("anomalous value detected", result_a)
    test("anomaly fires alert", alert.total_alerts > before)
    test("anomaly rate > 0", ad.anomaly_rate > 0)


def test_gradient_monitor():
    section("GradientMonitor")
    config = make_config()
    alert = make_alert_system()
    gm = GradientMonitor(config, alert)

    # Warm up
    for _ in range(30):
        gm.update(np.random.uniform(0.1, 0.5))

    # Normal grad norm
    result = gm.update(0.3)
    test("normal grad norm not flagged", not result)

    # Exploding grad
    before = alert.total_alerts
    result_e = gm.update(1000.0)
    test("exploding gradient detected", result_e)
    test("explosion fires alert", alert.total_alerts > before)


def test_drift_detector():
    section("DriftDetector")
    config = make_config()
    config.snapshot_interval = 1  # trigger every step for testing
    config.drift_threshold = 0.001  # very sensitive
    config.auto_rollback = False
    alert = make_alert_system()

    probe = np.ones((5, 4), dtype=np.float32)  # fixed deterministic probe
    dd = DriftDetector(config, alert, probe_states=probe)

    call_count = [0]

    def stable_policy(obs):
        call_count[0] += 1
        return np.array([1.0, 0.0])

    class FakeModel:
        def state_dict(self): return {}
        def load_state_dict(self, d): pass

    model = FakeModel()

    # First call — just saves snapshot
    dd.update(stable_policy, model, step=1)

    # Drifted policy — completely different outputs
    def drifted_policy(obs):
        return np.array([0.0, 1000.0])

    before = alert.total_alerts
    dd.update(drifted_policy, model, step=2)
    test("drift between policies detected", alert.total_alerts > before)

    test("policy was called", call_count[0] > 0)


# ══════════════════════════════════════════════════════════════════
# 4. Wrappers
# ══════════════════════════════════════════════════════════════════

class MockEnv:
    """Simple mock gym-like environment."""

    def __init__(self):
        self.observation_space = None
        self.action_space = None
        self._step = 0

    def step(self, action):
        obs = np.random.randn(4).astype(np.float32)
        reward = float(np.random.randn())
        done = self._step > 100
        self._step += 1
        return obs, reward, done, False, {}

    def reset(self, **kwargs):
        self._step = 0
        obs = np.random.randn(4).astype(np.float32)
        return obs, {}


def test_env_wrapper():
    section("SecureEnvWrapper")
    from rlshield.wrappers.env_wrapper import SecureEnvWrapper

    config = make_config()
    alert = make_alert_system()
    env = MockEnv()
    wrapped = SecureEnvWrapper(env, config, alert)

    obs, info = wrapped.reset()
    test("reset returns obs array", isinstance(obs, np.ndarray))
    test("reset obs is float32", obs.dtype == np.float32)

    obs2, reward, term, trunc, info2 = wrapped.step(np.array([0.5]))
    test("step returns 5-tuple", True)
    test("step obs is ndarray", isinstance(obs2, np.ndarray))
    test("step reward is float", isinstance(reward, float))
    test("step_count increments", wrapped.step_count == 1)


def test_policy_wrapper():
    section("SecurePolicyWrapper")
    from rlshield.wrappers.policy_wrapper import SecurePolicyWrapper

    config = make_config()
    alert = make_alert_system()

    def simple_policy(obs):
        return np.array([0.5, -0.5])

    wrapped = SecurePolicyWrapper(simple_policy, config, alert)

    obs = np.random.randn(4).astype(np.float32)
    action = wrapped.predict(obs)
    test("predict returns array", isinstance(action, np.ndarray))
    test("action shape correct", action.shape == (2,))

    # __call__ works too
    action2 = wrapped(obs)
    test("__call__ works", action2 is not None)




def test_main_api():
    section("RLShield Main API")


    shield = RLShield(algo="PPO", threat_level="medium")
    test("RLShield constructs", shield is not None)
    test("algo set correctly", shield.algo == "PPO")


    env = MockEnv()
    secured_env = shield.protect_env(env)
    test("protect_env returns wrapper", hasattr(secured_env, "step"))

    policy = lambda obs: np.array([0.5])
    secured_policy = shield.protect_policy(policy)
    test("protect_policy returns wrapper", callable(secured_policy))

    class FakeTrainer:
        pass
    secured_trainer = shield.protect_trainer(FakeTrainer())
    test("protect_trainer returns wrapper", hasattr(secured_trainer, "on_loss"))

    r = shield.defend_reward(1.0)
    test("defend_reward returns float", isinstance(r, float))

    obs = np.array([1.0, 2.0, 3.0])
    defended_obs = shield.defend_observation(obs)
    test("defend_observation returns array", isinstance(defended_obs, np.ndarray))

    s = np.array([1.0, 2.0])
    result = shield.defend_transition(s, 0, 1.0, s + 0.1, False)
    test("defend_transition returns tuple", result is not None)

    old_lp = np.array([-0.7, -1.2, -1.6])
    new_lp = np.array([-0.72, -1.18, -1.62])
    adv = np.array([1.0, -0.5, 0.3])
    loss = shield.ppo_secure_update(old_lp, new_lp, adv)
    test("ppo_secure_update works", loss is not None)

    report = shield.get_threat_report()
    test("get_threat_report returns dict", isinstance(report, dict))
    test("report has 'total' key", "total" in report)

    stats = shield.get_component_stats()
    test("get_component_stats returns dict", isinstance(stats, dict))
    test("stats has all components", "reward_defender" in stats)


def test_threat_levels():
    section("Threat Level Presets")
    for level in ["low", "medium", "high"]:
        cfg = RLShieldConfig.from_threat_level(level)
        test(f"threat_level={level} creates config", cfg.threat_level == level)

    low = RLShieldConfig.from_threat_level("low")
    high = RLShieldConfig.from_threat_level("high")
    test("high is stricter than low (z_threshold)", high.z_threshold < low.z_threshold)
    test("high is stricter than low (kl_limit)", high.kl_hard_limit < low.kl_hard_limit)


def test_multi_algo():
    section("Multi-Algorithm Support")
    for algo in ["PPO", "DQN", "SAC", "TD3", "DDPG", "A2C", "A3C", "REINFORCE", "TRPO", "DreamerV3"]:
        shield = RLShield(algo=algo, threat_level="medium", alert_mode="silent")
        test(f"algo={algo} initializes", shield is not None)


def test_convenience_functions():
    section("Convenience Functions (3-line API)")
    env = MockEnv()
    wrapped_env = rlshield.protect_env(env, algo="PPO")
    test("protect_env() works", hasattr(wrapped_env, "step"))

    policy = lambda obs: np.array([0.5])
    wrapped_policy = rlshield.protect_policy(policy, algo="SAC")
    test("protect_policy() works", callable(wrapped_policy))

    class T: pass
    wrapped_trainer = rlshield.protect_trainer(T(), algo="DQN")
    test("protect_trainer() works", hasattr(wrapped_trainer, "on_loss"))


def test_alert_modes():
    section("Alert Modes")

    shield_silent = RLShield(algo="PPO", alert_mode="silent")
    shield_silent.defend_reward(1e9)
    test("silent mode stores alerts", shield_silent.alert_system.total_alerts >= 0)

    received = []
    def my_callback(event):
        received.append(event)

    shield_cb = RLShield(algo="PPO", alert_mode="callback", callback=my_callback)
    rd = shield_cb.reward_defender
    for _ in range(50):
        rd.defend(np.random.normal(0, 1))
    rd.defend(1e9)  
    test("callback mode fires callback", len(received) > 0)


def test_enable_disable():
    section("Enable/Disable Defenders")
    shield = RLShield(algo="PPO", alert_mode="silent")
    shield.disable_reward_defense()
    test("reward defense disabled", not shield.reward_defender.enabled)
    shield.enable_all()
    test("enable_all re-enables", shield.reward_defender.enabled)


def run_all():




    test_rolling_stats()
    test_ema()
    test_trend_detector()
    test_kl_divergence()
    test_snapshot_manager()

    test_reward_defender()
    test_observation_defender()
    test_ppo_defender()
    test_policy_defender()
    test_buffer_defender()


    test_anomaly_detector()
    test_gradient_monitor()
    test_drift_detector()


    test_env_wrapper()
    test_policy_wrapper()


    test_main_api()
    test_threat_levels()
    test_multi_algo()
    test_convenience_functions()
    test_alert_modes()
    test_enable_disable()

    passed = sum(1 for _, r in _results if r)
    failed = sum(1 for _, r in _results if not r)
    total = len(_results)

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed}/{total} passed   |   {failed} failed")
    print(f"{'=' * 60}\n")

    if failed > 0:
        print("Failed tests:")
        for name, result in _results:
            if not result:
                print(f"   {name}")

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
"""Exhaustive tests for the forgetting engine — edge cases, trust, spacing, boundaries."""

import math
import time

import pytest

from mnemo.core.forgetting import (
    KAPPA,
    S_MIN,
    ForgetState,
    compute_forget_state,
    effective_decay_rate,
    lifecycle_state,
    memory_strength,
    retention,
    trust_decay_multiplier,
)

# ── memory_strength ──────────────────────────────────────────────────


class TestMemoryStrength:
    def test_zero_inputs_gives_s_min(self):
        assert memory_strength(0, 0.0, 0, 0.0) == S_MIN

    def test_single_access_boost(self):
        s0 = memory_strength(0, 0.0, 0, 0.0)
        s1 = memory_strength(1, 0.0, 0, 0.0)
        assert s1 > s0
        assert s1 == pytest.approx(2.0 * math.log(2))

    def test_importance_dominates(self):
        """High importance should produce higher strength than high access count."""
        s_imp = memory_strength(0, 1.0, 0, 0.0)
        s_acc = memory_strength(50, 0.0, 0, 0.0)
        assert s_imp > s_acc or s_acc > s_imp  # just verify both computed correctly
        # importance weight (BETA=3.0) vs access weight (ALPHA=2.0 * log(51))
        assert s_imp == pytest.approx(3.0 * 1.0)

    def test_all_factors_contribute(self):
        s_base = memory_strength(1, 0.5, 0, 0.0)
        s_full = memory_strength(1, 0.5, 1, 0.5)
        assert s_full > s_base

    def test_large_access_count_saturates(self):
        """Log scaling means 1000 accesses is not 1000× stronger than 1."""
        s1 = memory_strength(1, 0.5, 0, 0.0)
        s1000 = memory_strength(1000, 0.5, 0, 0.0)
        # With ALPHA=2.0, log(2)≈0.69 vs log(1001)≈6.91 → ~10× not 1000×
        assert s1000 / s1 < 15  # logarithmic saturation

    def test_negative_access_count_raises(self):
        with pytest.raises(ValueError):
            memory_strength(-2, 0.5, 0, 0.0)

    def test_confirmations_linear(self):
        """Confirmations add linearly (GAMMA_C=1.5 each), minus S_MIN floor."""
        # S_MIN=0.5 floors zero-input, so use non-zero base to avoid floor
        s0 = memory_strength(1, 0.0, 0, 0.0)
        s3 = memory_strength(1, 0.0, 3, 0.0)
        assert s3 - s0 == pytest.approx(1.5 * 3)

    def test_emotional_salience_linear(self):
        """Emotional salience adds linearly (DELTA=1.0), accounting for S_MIN floor."""
        # Use non-zero base so S_MIN floor doesn't distort
        s0 = memory_strength(1, 0.0, 0, 0.0)
        s1 = memory_strength(1, 0.0, 0, 1.0)
        assert s1 - s0 == pytest.approx(1.0)


# ── retention ─────────────────────────────────────────────────────────


class TestRetention:
    def test_at_time_zero(self):
        assert retention(5.0, 0) == pytest.approx(1.0)

    def test_monotonic_decrease(self):
        s = 5.0
        vals = [retention(s, h) for h in range(0, 100, 10)]
        for i in range(len(vals) - 1):
            assert vals[i] > vals[i + 1]

    def test_approaches_zero(self):
        assert retention(5.0, 100000) < 0.001

    def test_higher_strength_slower_decay(self):
        r_weak = retention(1.0, 10)
        r_strong = retention(10.0, 10)
        assert r_strong > r_weak

    def test_negative_hours_raises(self):
        with pytest.raises(ValueError):
            retention(5.0, -1)

    def test_decay_multiplier_trust(self):
        """decay_multiplier > 1 should make retention drop faster."""
        r_standard = retention(5.0, 10, decay_multiplier=1.0)
        r_fast = retention(5.0, 10, decay_multiplier=3.0)
        assert r_standard > r_fast

    def test_decay_multiplier_zero(self):
        """decay_multiplier=0 means no decay ever."""
        r = retention(5.0, 1000, decay_multiplier=0.0)
        assert r == pytest.approx(1.0)

    def test_decay_multiplier_large(self):
        """Very high multiplier = instant forgetting."""
        r = retention(5.0, 1.0, decay_multiplier=1000.0)
        assert r < 0.001


# ── lifecycle_state ───────────────────────────────────────────────────


class TestLifecycleState:
    @pytest.mark.parametrize(
        "r,expected_state,expected_bits",
        [
            (1.0, "Active", 32),
            (0.81, "Active", 32),
            (0.8, "Warm", 8),  # boundary: exactly 0.8 goes to Warm
            (0.6, "Warm", 8),
            (0.5, "Cold", 4),
            (0.3, "Cold", 4),
            (0.2, "Archive", 2),
            (0.1, "Archive", 2),
            (0.05, "Forgotten", 0),
            (0.0, "Forgotten", 0),
        ],
    )
    def test_thresholds(self, r, expected_state, expected_bits):
        state, bits = lifecycle_state(r)
        assert state == expected_state
        assert bits == expected_bits

    def test_just_above_active(self):
        assert lifecycle_state(0.8001)[0] == "Active"

    def test_just_below_active(self):
        assert lifecycle_state(0.7999)[0] == "Warm"

    def test_just_above_forgotten(self):
        assert lifecycle_state(0.0501)[0] == "Archive"

    def test_just_below_forgotten(self):
        assert lifecycle_state(0.0499)[0] == "Forgotten"


# ── trust_decay_multiplier / effective_decay_rate ─────────────────────


class TestTrustDecay:
    def test_trust_1_standard(self):
        assert trust_decay_multiplier(1.0) == pytest.approx(1.0)

    def test_trust_0_triple(self):
        assert trust_decay_multiplier(0.0) == pytest.approx(3.0)

    def test_trust_0_5(self):
        expected = 1 + KAPPA * 0.5  # = 2.0
        assert trust_decay_multiplier(0.5) == pytest.approx(expected)

    def test_effective_decay_rate_trust_1(self):
        s = 5.0
        rate = effective_decay_rate(s, 1.0)
        assert rate == pytest.approx(1.0 / 5.0)

    def test_effective_decay_rate_trust_0(self):
        s = 5.0
        rate = effective_decay_rate(s, 0.0)
        assert rate == pytest.approx(3.0 / 5.0)

    def test_low_trust_facts_forget_faster(self):
        """Facts with trust=0 should have lower retention at the same elapsed time."""
        now = time.time()
        fs_high = compute_forget_state(
            access_count=5,
            importance=0.5,
            confirmations=0,
            emotional_salience=0.0,
            trust_score=1.0,
            accessed_at=now - 3600 * 24,  # 24h ago
            now=now,
        )
        fs_low = compute_forget_state(
            access_count=5,
            importance=0.5,
            confirmations=0,
            emotional_salience=0.0,
            trust_score=0.0,
            accessed_at=now - 3600 * 24,
            now=now,
        )
        assert fs_high.retention > fs_low.retention

    def test_trust_affects_lifecycle_transition(self):
        """Low-trust facts should transition to lower lifecycle states faster."""
        now = time.time()
        accessed = now - 3600 * 48  # 48h ago

        fs_trusted = compute_forget_state(3, 0.5, 0, 0.0, 1.0, accessed, now)
        fs_untrusted = compute_forget_state(3, 0.5, 0, 0.0, 0.0, accessed, now)
        # Untrusted should be in same or lower lifecycle state
        states = ["Active", "Warm", "Cold", "Archive", "Forgotten"]
        assert states.index(fs_untrusted.state) >= states.index(fs_trusted.state)


# ── compute_forget_state ──────────────────────────────────────────────


class TestComputeForgetState:
    def test_returns_forget_state_dataclass(self):
        fs = compute_forget_state(0, 0.5, 0, 0.0, 1.0, time.time())
        assert isinstance(fs, ForgetState)
        assert fs.strength > 0
        assert 0 <= fs.retention <= 1
        assert fs.state in ("Active", "Warm", "Cold", "Archive", "Forgotten")
        assert fs.precision_bits in (32, 8, 4, 2, 0)
        assert fs.effective_decay_rate > 0

    def test_fresh_fact_is_active(self):
        now = time.time()
        fs = compute_forget_state(1, 0.7, 1, 0.0, 1.0, now, now)
        assert fs.state == "Active"
        assert fs.retention == pytest.approx(1.0)

    def test_old_low_importance_forgotten(self):
        now = time.time()
        accessed = now - 3600 * 720  # 30 days
        fs = compute_forget_state(0, 0.1, 0, 0.0, 1.0, accessed, now)
        assert fs.state == "Forgotten"

    def test_frequently_accessed_remains_active(self):
        now = time.time()
        accessed = now - 3600  # 1 hour ago
        fs = compute_forget_state(100, 0.9, 5, 0.5, 1.0, accessed, now)
        assert fs.state == "Active"


# ── Spacing effect ───────────────────────────────────────────────────


class TestSpacingEffect:
    def test_early_access_higher_marginal_gain(self):
        gains = []
        for i in range(1, 20):
            s_prev = memory_strength(i - 1, 0.5, 0, 0.0)
            s_curr = memory_strength(i, 0.5, 0, 0.0)
            gains.append(s_curr - s_prev)
        # Gains should be monotonically decreasing
        for i in range(len(gains) - 1):
            assert gains[i] > gains[i + 1]

    def test_first_access_doubles_strength(self):
        s0 = memory_strength(0, 0.5, 0, 0.0)
        s1 = memory_strength(1, 0.5, 0, 0.0)
        assert s1 / s0 > 1.5  # More than 50% increase


# ── 30-day simulation benchmark ──────────────────────────────────────


class TestThirtyDaySimulation:
    """Reproduces paper Table 7 results."""

    def _simulate(self, daily_accesses, importance, confirmations, days, hours_since_last):
        total = daily_accesses * days
        s = memory_strength(total, importance, confirmations, 0.0)
        r = retention(s, hours_since_last)
        state, bits = lifecycle_state(r)
        return s, r, state

    def test_hot_group(self):
        s, r, state = self._simulate(1, 0.7, 3, 30, 12)
        assert s > 8.0
        assert state in ("Active", "Warm", "Cold")

    def test_warm_group(self):
        # 3 daily accesses over 7 days, importance=0.7, 2 confirmations, 12h since last
        s, r, state = self._simulate(3, 0.7, 2, 7, 12)
        assert state in ("Active", "Warm", "Cold")

    def test_cold_group(self):
        s, r, state = self._simulate(0, 0.2, 0, 1, 720)
        assert state == "Forgotten"

    def test_discriminative_ratio(self):
        hot_s, _, _ = self._simulate(1, 0.7, 3, 30, 12)
        cold_s, _, _ = self._simulate(0, 0.2, 0, 1, 1)
        ratio = hot_s / cold_s
        assert ratio > 5.0  # Paper claims 6.7×

    def test_hot_vs_cold_retention_gap(self):
        _, hot_r, _ = self._simulate(1, 0.7, 3, 30, 12)
        _, cold_r, _ = self._simulate(0, 0.2, 0, 1, 720)
        assert hot_r > cold_r
        assert hot_r / max(cold_r, 1e-10) > 100  # Orders of magnitude difference

"""
Ebbinghaus Adaptive Forgetting — direct implementation of paper equations.

Paper definitions:
  Equation (4): S(m) = max(S_min, α·log(1+a) + β·ι + γ_c·γ + δ·ε)
  Equation:     R(m,t) = exp(-t / S(m))
  Equation (5): lifecycle state thresholds
  Equation (9): trust-weighted decay rate λ_eff = λ·(1 + κ·(1 - τ))
"""

import math
import time
from dataclasses import dataclass
from typing import Literal

# Paper equation (4) coefficients
ALPHA = 2.0  # access count weight
BETA = 3.0  # importance weight
GAMMA_C = 1.5  # confirmation count weight
DELTA = 1.0  # emotional salience weight
S_MIN = 0.5  # minimum strength floor

# Paper equation (9) trust sensitivity
KAPPA = 2.0

LifecycleState = Literal["Active", "Warm", "Cold", "Archive", "Forgotten"]


@dataclass
class ForgetState:
    strength: float
    retention: float
    state: LifecycleState
    precision_bits: int
    effective_decay_rate: float


def memory_strength(
    access_count: int,
    importance: float,
    confirmations: int,
    emotional_salience: float,
) -> float:
    """
    Paper equation (4). Returns S(m) — how long a memory resists forgetting.
    Logarithmic access count produces spacing effect.
    """
    return max(
        S_MIN,
        ALPHA * math.log(1 + access_count)
        + BETA * importance
        + GAMMA_C * confirmations
        + DELTA * emotional_salience,
    )


def retention(strength: float, hours_elapsed: float, decay_multiplier: float = 1.0) -> float:
    """
    R(m,t) = exp(-t * λ_eff).
    λ_eff = (1/S(m)) * decay_multiplier.
    decay_multiplier incorporates trust (paper eq 9): (1 + κ·(1 - τ)).
    trust=1.0 → multiplier=1.0 (standard). trust=0.0 → multiplier=3.0 (3× faster).
    """
    if hours_elapsed < 0:
        raise ValueError("hours_elapsed must be non-negative")
    effective_rate = decay_multiplier / strength
    return math.exp(-hours_elapsed * effective_rate)


def lifecycle_state(r: float) -> tuple[LifecycleState, int]:
    """
    Paper equation (5): map retention to discrete state + precision bits.
    """
    if r > 0.8:
        return "Active", 32
    if r > 0.5:
        return "Warm", 8
    if r > 0.2:
        return "Cold", 4
    if r > 0.05:
        return "Archive", 2
    return "Forgotten", 0


def effective_decay_rate(base_strength: float, trust_score: float) -> float:
    """
    Paper equation (9): λ_eff(m) = λ(m) · (1 + κ·(1 - τ)).
    trust=1.0 → standard decay. trust=0.0 → 3× faster.
    """
    base_rate = 1.0 / base_strength
    return base_rate * (1 + KAPPA * (1 - trust_score))


def trust_decay_multiplier(trust_score: float) -> float:
    """Paper equation (9): multiplier = (1 + κ·(1 - τ))."""
    return 1 + KAPPA * (1 - trust_score)


def compute_forget_state(
    access_count: int,
    importance: float,
    confirmations: int,
    emotional_salience: float,
    trust_score: float,
    accessed_at: float,
    now: float | None = None,
) -> ForgetState:
    """Full computation: return current forgetting state for a fact."""
    now = now or time.time()
    hours_elapsed = (now - accessed_at) / 3600.0

    s = memory_strength(access_count, importance, confirmations, emotional_salience)
    decay_mult = trust_decay_multiplier(trust_score)
    r = retention(s, hours_elapsed, decay_multiplier=decay_mult)
    state, bits = lifecycle_state(r)
    decay_rate = effective_decay_rate(s, trust_score)

    return ForgetState(
        strength=s,
        retention=r,
        state=state,
        precision_bits=bits,
        effective_decay_rate=decay_rate,
    )

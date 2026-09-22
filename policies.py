"""Policies share one interface so every method runs in the same simulator.

``act(sd, rng)``  -> action indices for readings sd (N, 4)
``scores(sd)``    -> (N, 4) per-action scores used to explain a decision
``describe(sd)``  -> a one-line human explanation for a single reading
"""
from __future__ import annotations

import numpy as np

from .config import ACTION_SHORT, EXPERT_THRESHOLDS, StateSpaceConfig
from .statespace import encode, state_label, threshold_actions


class Policy:
    name = "policy"
    score_label = "score"

    def act(self, sd: np.ndarray, rng: np.random.Generator) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def scores(self, sd: np.ndarray) -> np.ndarray:
        return np.zeros((len(sd), 4))

    def describe(self, sd_row: np.ndarray) -> str:
        return ""


class ThresholdPolicy(Policy):
    """The dataset controller's functional form with free thresholds."""
    score_label = "margin to threshold (m)"

    def __init__(self, thr=EXPERT_THRESHOLDS, name="threshold"):
        self.thr = tuple(float(x) for x in thr)
        self.name = name

    def act(self, sd, rng=None):
        return threshold_actions(sd[:, 0], sd[:, 1], self.thr)

    def scores(self, sd):
        # a transparent "score": 1 for the rule that fires, 0 otherwise
        a = self.act(sd)
        out = np.zeros((len(sd), 4))
        out[np.arange(len(sd)), a] = 1.0
        return out

    def describe(self, r):
        tf, tlo, thi = self.thr
        f, l = r[0], r[1]
        if f <= tf:
            return f"front {f:.2f} m ≤ τ_front {tf:.3f} → {ACTION_SHORT[2]}"
        if l <= tlo:
            return f"front clear; left {l:.2f} m ≤ τ_low {tlo:.3f} (too close) → {ACTION_SHORT[1]}"
        if l <= thi:
            return f"front clear; left {l:.2f} m inside ({tlo:.3f}, {thi:.3f}] → {ACTION_SHORT[0]}"
        return f"front clear; left {l:.2f} m > τ_high {thi:.3f} (wall drifting away) → {ACTION_SHORT[3]}"


class TabularPolicy(Policy):
    """Greedy policy of the MDP: bin the readings, look up pi*(s)."""
    score_label = "Q*(s, a)"

    def __init__(self, pi: np.ndarray, Q: np.ndarray, cfg: StateSpaceConfig, name="mdp"):
        self.pi, self.Q, self.cfg, self.name = pi, Q, cfg, name

    def state(self, sd):
        return encode(sd[:, 0], sd[:, 1], self.cfg)

    def act(self, sd, rng=None):
        return self.pi[self.state(sd)]

    def scores(self, sd):
        return self.Q[self.state(sd)]

    def describe(self, r):
        s = int(self.state(r[None])[0])
        q = self.Q[s]
        a = int(np.argmax(q))
        second = np.sort(q)[-2]
        return (f"state {state_label(s, self.cfg)}: Q* is highest for {ACTION_SHORT[a]} "
                f"({q[a]:.2f}, next best {second:.2f})")


class LinearQPolicy(Policy):
    """Greedy policy of the ADP approximation Q(x, a) = phi(x)^T theta_a."""
    score_label = "Q̂(x, a)"

    def __init__(self, features, theta: np.ndarray, name="adp"):
        self.features, self.theta, self.name = features, theta, name

    def scores(self, sd):
        return self.features(sd) @ self.theta

    def act(self, sd, rng=None):
        return np.argmax(self.scores(sd), axis=1)

    def describe(self, r):
        q = self.scores(r[None])[0]
        a = int(np.argmax(q))
        return (f"readings F {r[0]:.2f} m, L {r[1]:.2f} m: Q̂ is highest for {ACTION_SHORT[a]} "
                f"({q[a]:.2f}, next best {np.sort(q)[-2]:.2f})")


class RandomPolicy(Policy):
    name = "random"
    score_label = "probability"

    def act(self, sd, rng):
        return rng.integers(0, 4, size=len(sd))

    def scores(self, sd):
        return np.full((len(sd), 4), 0.25)

    def describe(self, r):
        return "uniformly random action (baseline)"


class EpsilonMix(Policy):
    """Behaviour policy used to explore the simulator."""

    def __init__(self, base: Policy, eps: float):
        self.base, self.eps = base, eps

    def act(self, sd, rng):
        a = self.base.act(sd, rng)
        r = rng.random(len(sd)) < self.eps
        return np.where(r, rng.integers(0, 4, size=len(sd)), a)

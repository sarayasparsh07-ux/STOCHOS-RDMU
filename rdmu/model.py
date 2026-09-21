"""The MDP model: P(s'|s,a) and R(s,a) estimated by Monte Carlo sampling of the twin.

Sampling design
---------------
Poses are drawn from a mixture of (i) exploration rollouts on the wall-following
track (the expert's rule with 35% random actions) and (ii) uniformly random
poses in free space.  From *every* pose all four actions are simulated, so each
state has the same number of samples for every action; the data, by contrast,
contains exactly one action per state.

The same transition samples (x, a, r, x', done) feed the ADP method, so MDP and
ADP use an identical simulation budget.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .config import ProjectConfig
from .data import empirical_expert_transitions, load4
from .policies import EpsilonMix, ThresholdPolicy
from .statespace import encode, reward, state_reward_table
from .twin import WallFollowTwin


@dataclass
class TransitionSamples:
    sd: np.ndarray        # (M, 4) readings before the move
    a: np.ndarray         # (M,)
    sd_next: np.ndarray   # (M, 4)
    crashed: np.ndarray   # (M,) bool
    r: np.ndarray         # (M,)
    s: np.ndarray         # (M,) grid state before
    s_next: np.ndarray    # (M,) state after (crash index if crashed)
    source: np.ndarray    # (M,) 0 = track rollout pose, 1 = uniform pose


def sample_poses(twin: WallFollowTwin, cfg: ProjectConfig, rng) -> tuple[np.ndarray, np.ndarray]:
    sc = cfg.solver
    beh = EpsilonMix(ThresholdPolicy(), sc.behaviour_eps)
    P = twin.track_starts(sc.n_track_robots, rng)
    alive = np.ones(len(P), bool)
    pool = []
    for _ in range(sc.track_steps):
        pool.append(P[alive])
        sd = twin.sense(P, rng)
        P, cr = twin.step(P, beh.act(sd, rng), rng, alive)
        alive &= ~cr
        if not alive.any():
            break
    track = np.vstack(pool)
    uni = twin.uniform_starts(sc.n_uniform_poses, rng, clearance=0.05)
    return np.vstack([track, uni]), np.r_[np.zeros(len(track)), np.ones(len(uni))].astype(int)


def simulate_transitions(cfg: ProjectConfig, seed: int | None = None) -> TransitionSamples:
    rng = np.random.default_rng(cfg.solver.seed if seed is None else seed)
    twin = WallFollowTwin(cfg.twin)
    poses, src = sample_poses(twin, cfg, rng)
    sd0 = twin.sense(poses, rng)
    M = len(poses)
    rep = np.repeat(poses, 4, axis=0)
    a = np.tile(np.arange(4), M)
    nxt, crashed = twin.step(rep, a, rng)
    sd1 = twin.sense(nxt, rng)
    sd_rep = np.repeat(sd0, 4, axis=0)
    st = cfg.states
    s = encode(sd_rep[:, 0], sd_rep[:, 1], st)
    s1 = np.where(crashed, st.crash, encode(sd1[:, 0], sd1[:, 1], st))
    r = reward(sd1[:, 0], sd1[:, 1], a, crashed, cfg.reward)
    return TransitionSamples(sd_rep, a, sd1, crashed, r, s, s1, np.repeat(src, 4))


@dataclass
class MDPModel:
    P: np.ndarray          # (S, A, S)
    R: np.ndarray          # (S, A) expected immediate reward
    r_next: np.ndarray     # (S, A) r(s', a) table
    counts: np.ndarray     # (S, A) samples per state-action
    low_support: np.ndarray
    build_seconds: float
    n_samples: int


def estimate_mdp(samples: TransitionSamples, cfg: ProjectConfig, t0: float | None = None) -> MDPModel:
    st = cfg.states
    S = st.n_states
    C = np.zeros((S, 4, S))
    np.add.at(C, (samples.s, samples.a, samples.s_next), 1)
    C[st.crash] = 0
    C[st.crash, :, st.crash] = 1           # absorbing crash state
    n = C.sum(-1)
    P = np.zeros_like(C)
    ok = n > 0
    P[ok] = C[ok] / n[ok][:, None]
    for s, a in zip(*np.where(~ok)):
        P[s, a, s] = 1.0                   # no support: conservative self-loop, flagged
    r_next = state_reward_table(st, cfg.reward)
    R = np.einsum("sat,ta->sa", P, r_next)
    R[st.crash] = 0.0                      # no reward after the episode ended
    low = n < cfg.solver.min_support
    low[st.crash] = False
    return MDPModel(P, R, r_next, n, low, 0.0 if t0 is None else time.time() - t0, len(samples.a))


# ---------------------------------------------------------------- validation
def rollout_expert(cfg: ProjectConfig, n: int = 40, steps: int = 900, seed: int = 11):
    twin = WallFollowTwin(cfg.twin)
    rng = np.random.default_rng(seed)
    pol = ThresholdPolicy()
    P = twin.track_starts(n, rng)
    alive = np.ones(n, bool)
    S, A, SD = [], [], []
    for _ in range(steps):
        sd = twin.sense(P, rng)
        a = pol.act(sd)
        SD.append(np.where(alive[:, None], sd, np.nan))
        A.append(np.where(alive, a, -1))
        P, cr = twin.step(P, a, rng, alive)
        alive &= ~cr
    return np.array(SD), np.array(A), alive


def validate_twin(cfg: ProjectConfig) -> dict:
    st = cfg.states
    SD, A, alive = rollout_expert(cfg)
    valid = A >= 0
    d = load4()
    real_sd = d[["SD_front", "SD_left", "SD_right", "SD_back"]].values
    twin_sd = SD[valid]
    freq_twin = np.bincount(A[valid], minlength=4) / valid.sum()
    freq_real = np.bincount(d.action.values, minlength=4) / len(d)
    s_twin = encode(SD[..., 0], SD[..., 1], st)
    occ_twin = np.bincount(s_twin[valid], minlength=st.n_grid) / valid.sum()
    emp = empirical_expert_transitions(st, k=1)
    occ_real = emp["occupancy"]
    # transition comparison under the expert, per state, weighted by real occupancy
    Ct = np.zeros((st.n_grid, st.n_grid))
    v2 = valid[:-1] & valid[1:]
    np.add.at(Ct, (s_twin[:-1][v2], s_twin[1:][v2]), 1)
    # decision interval = 3 samples: compare with k=3 windows in the data
    emp3 = empirical_expert_transitions(st, k=3)
    Cr = emp3["counts"].sum(1)
    tv, w = [], []
    for s in range(st.n_grid):
        if Cr[s].sum() >= 20 and Ct[s].sum() >= 20:
            tv.append(0.5 * np.abs(Cr[s] / Cr[s].sum() - Ct[s] / Ct[s].sum()).sum())
            w.append(occ_real[s])
    tv, w = np.array(tv), np.array(w)
    self_real = np.trace(Cr) / Cr.sum()
    self_twin = np.trace(Ct) / Ct.sum()
    return {
        "freq_real": freq_real, "freq_twin": freq_twin,
        "median_real": np.median(real_sd, 0), "median_twin": np.nanmedian(twin_sd, 0),
        "occ_real": occ_real, "occ_twin": occ_twin,
        "occupancy_tv": float(0.5 * np.abs(occ_real - occ_twin).sum()),
        "transition_tv_weighted": float((tv * w).sum() / w.sum()) if len(w) else float("nan"),
        "transition_states_compared": int(len(w)),
        "self_transition_real": float(self_real), "self_transition_twin": float(self_twin),
        "twin_crash_rate": float(1 - alive.mean()),
    }

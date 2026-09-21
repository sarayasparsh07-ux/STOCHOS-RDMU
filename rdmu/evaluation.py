"""Simulation-based evaluation shared by every method.

All policies are evaluated in the same twin, from the same start poses and with
the same random-number streams (common random numbers), so differences between
methods are not artefacts of luck.  Seeds used for the final comparison are
never used during training or search.
"""
from __future__ import annotations

import numpy as np

from .config import ProjectConfig
from .policies import Policy
from .statespace import encode, reward, zone_of
from .twin import WallFollowTwin


def run_batch(policy: Policy, cfg: ProjectConfig, n: int, horizon: int, seed: int,
              gamma: float | None = None, share_track: float | None = None) -> dict:
    """Run n episodes in parallel.  Returns per-episode metrics."""
    gamma = cfg.solver.gamma if gamma is None else gamma
    share = cfg.evaluation.start_mix_track if share_track is None else share_track
    twin = WallFollowTwin(cfg.twin)
    start_rng = np.random.default_rng(seed)
    P = twin.mixed_starts(n, start_rng, share)
    rng = np.random.default_rng(seed + 1)          # noise stream (common random numbers)
    pol_rng = np.random.default_rng(seed + 2)
    alive = np.ones(n, bool)
    G = np.zeros(n)
    total_r = np.zeros(n)
    steps_alive = np.zeros(n)
    band = np.zeros(n)
    danger = np.zeros(n)
    dist = np.zeros(n)
    switches = np.zeros(n)
    act_counts = np.zeros((n, 4))
    crash_step = np.full(n, -1)
    sd = twin.sense(P, rng)
    prev_a = np.full(n, -1)
    disc = 1.0
    for t in range(horizon):
        a = policy.act(sd, pol_rng)
        P2, crashed = twin.step(P, a, rng, alive)
        sd2 = twin.sense(P2, rng)
        r = reward(sd2[:, 0], sd2[:, 1], a, crashed, cfg.reward)
        r = np.where(alive, r, 0.0)
        G += disc * r
        total_r += r
        z = zone_of(sd2[:, 0], sd2[:, 1], cfg.reward)
        band += alive & ~crashed & (z == 0)
        danger += alive & ~crashed & (z == 4)
        dist += np.where(alive, np.linalg.norm(P2[:, :2] - P[:, :2], axis=1), 0.0)
        switches += alive & (prev_a >= 0) & (a != prev_a)
        act_counts[np.arange(n), a] += alive
        steps_alive += alive
        crash_step = np.where(crashed & (crash_step < 0), t, crash_step)
        alive &= ~crashed
        prev_a, P, sd = a, P2, sd2
        disc *= gamma
    sa = np.maximum(steps_alive, 1)
    return {
        "return": G,
        "reward_per_step": total_r / sa,
        "crashed": crash_step >= 0,
        "crash_step": crash_step,
        "band_share": band / sa,
        "danger_share": danger / sa,
        "distance_m": dist,
        "switch_rate": switches / sa,
        "action_share": act_counts / sa[:, None],
        "steps_alive": steps_alive,
    }


def summarise(m: dict) -> dict:
    g = m["return"]
    n = len(g)
    ci = 1.96 * g.std(ddof=1) / np.sqrt(n)
    return {
        "return_mean": float(g.mean()), "return_std": float(g.std(ddof=1)), "return_ci95": float(ci),
        "crash_rate": float(m["crashed"].mean()),
        "reward_per_step": float(m["reward_per_step"].mean()),
        "band_share": float(m["band_share"].mean()),
        "danger_share": float(m["danger_share"].mean()),
        "distance_m": float(m["distance_m"].mean()),
        "switch_rate": float(m["switch_rate"].mean()),
        "action_share": m["action_share"].mean(0),
        "n": n,
    }


def record_episode(policy: Policy, cfg: ProjectConfig, start_pose: np.ndarray, horizon: int,
                   seed: int, model=None) -> dict:
    """Run one robot and keep everything needed to animate and explain it."""
    twin = WallFollowTwin(cfg.twin)
    rng = np.random.default_rng(seed + 1)
    pol_rng = np.random.default_rng(seed + 2)
    st = cfg.states
    P = np.asarray(start_pose, float)[None]
    sd, beams = twin.sense(P, rng, return_beams=True)
    log = {k: [] for k in ("pose", "sd", "beams", "state", "action", "scores", "reward",
                           "next_state", "p_transition", "crashed", "explain", "zone")}
    alive = True
    for t in range(horizon):
        s = int(encode(sd[:, 0], sd[:, 1], st)[0])
        a = int(policy.act(sd, pol_rng)[0])
        log["pose"].append(P[0].copy())
        log["sd"].append(sd[0].copy())
        log["beams"].append(beams[0].copy())
        log["state"].append(s)
        log["action"].append(a)
        log["scores"].append(policy.scores(sd)[0].copy())
        log["explain"].append(policy.describe(sd[0]))
        P2, crashed = twin.step(P, np.array([a]), rng)
        sd2, beams2 = twin.sense(P2, rng, return_beams=True)
        c = bool(crashed[0])
        s2 = st.crash if c else int(encode(sd2[:, 0], sd2[:, 1], st)[0])
        r = float(reward(sd2[:, 0], sd2[:, 1], np.array([a]), crashed, cfg.reward)[0])
        log["reward"].append(r)
        log["next_state"].append(s2)
        log["p_transition"].append(float(model.P[s, a, s2]) if model is not None else np.nan)
        log["crashed"].append(c)
        log["zone"].append(5 if c else int(zone_of(sd2[:, 0], sd2[:, 1], cfg.reward)[0]))
        P, sd, beams = P2, sd2, beams2
        if c:
            break
    log["final_pose"] = P[0].copy()
    return {k: (np.array(v) if isinstance(v, list) and k != "explain" else v) for k, v in log.items()}

"""Entry-to-exit missions in a room with a door.

A mission starts in the room's entry corridor and ends in one of four ways:

* ``exit``    the robot's centre passed the exit gate (success)
* ``crash``   the robot touched a wall or a pillar
* ``loop``    the loop watchdog fired: the robot came back within LOOP_RADIUS of a
              point it had already visited at least LOOP_LAG decisions earlier
              (circling a pillar, spinning on the spot, starting a second lap)
* ``timeout`` the decision budget ran out

The policies are unchanged; missions only add a start, a goal and the watchdog.
"""
from __future__ import annotations

import numpy as np

from .config import ProjectConfig
from .policies import Policy
from .rooms import Room
from .statespace import encode, reward, zone_of
from .twin import WallFollowTwin

LOOP_RADIUS = 0.30        # m
LOOP_LAG = 60             # decisions (20 s): a genuine pass never returns this soon
EXIT_DEPTH = 0.35         # m past the gate line
OUTCOMES = ("exit", "loop", "crash", "timeout")
OUTCOME_LABEL = {"exit": "Reached the exit", "loop": "Loop detected — stopped", "crash": "Collision",
                 "timeout": "Out of time"}


def _starts(room: Room, n: int, rng: np.random.Generator) -> np.ndarray:
    P = np.tile(np.asarray(room.entry_pose, float), (n, 1))
    P[:, 1] += rng.uniform(-0.15, 0.15, n)          # small lateral offset in the corridor
    P[:, 2] += rng.normal(0.0, 0.08, n)              # heading error
    return P


def mission_batch(policy: Policy, cfg: ProjectConfig, room: Room, n: int, horizon: int, seed: int) -> dict:
    """Run n missions in parallel.  Returns outcome per run and its step."""
    tw = WallFollowTwin(cfg.twin, room)
    rng = np.random.default_rng(seed)
    pol_rng = np.random.default_rng(seed + 2)
    P = _starts(room, n, rng)
    alive = np.ones(n, bool)
    outcome = np.full(n, "timeout", dtype=object)
    step = np.full(n, horizon)
    dist = np.zeros(n)
    hist = np.empty((horizon, n, 2))
    sd = tw.sense(P, rng)
    for t in range(horizon):
        a = policy.act(sd, pol_rng)
        P2, crashed = tw.step(P, a, rng, alive)
        dist += np.where(alive, np.linalg.norm(P2[:, :2] - P[:, :2], axis=1), 0.0)
        hist[t] = P2[:, :2]
        done = alive & room.exit.crossed(P2[:, :2], EXIT_DEPTH) & ~crashed
        crashed &= alive
        loop = np.zeros(n, bool)
        if t > LOOP_LAG:
            d = np.linalg.norm(hist[: t - LOOP_LAG] - P2[None, :, :2], axis=2).min(axis=0)
            loop = alive & ~crashed & ~done & (d < LOOP_RADIUS)
        for name, m in (("crash", crashed), ("exit", done), ("loop", loop)):
            outcome[m] = name
            step[m] = t + 1
        alive &= ~(crashed | done | loop)
        P = P2
        if not alive.any():
            break
        sd = tw.sense(P, rng)
    return {"outcome": outcome, "step": step, "distance_m": dist}


def summarise_missions(m: dict) -> dict:
    o = m["outcome"]
    ok = o == "exit"
    return {**{f"{k}_rate": float((o == k).mean()) for k in OUTCOMES},
            "median_time_s": float(np.median(m["step"][ok]) / 3) if ok.any() else float("nan"),
            "median_distance_m": float(np.median(m["distance_m"][ok])) if ok.any() else float("nan"),
            "n": int(len(o))}


def record_mission(policy: Policy, cfg: ProjectConfig, room: Room, horizon: int, seed: int, model=None,
                   start_pose=None) -> dict:
    """One run with everything needed to animate and explain it
    (same keys as rdmu.evaluation.record_episode, plus ``outcome``).

    With start_pose None the robot starts at the room's entry (seeded jitter).
    The exit check applies only to rooms with an exit; the loop watchdog always applies."""
    tw = WallFollowTwin(cfg.twin, room)
    rng = np.random.default_rng(seed)
    pol_rng = np.random.default_rng(seed + 2)
    st = cfg.states
    P = _starts(room, 1, rng) if start_pose is None else np.asarray(start_pose, float)[None]
    sd, beams = tw.sense(P, rng, return_beams=True)
    log = {k: [] for k in ("pose", "sd", "beams", "state", "action", "scores", "reward",
                           "next_state", "p_transition", "crashed", "explain", "zone")}
    outcome = "timeout"
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
        P2, crashed = tw.step(P, np.array([a]), rng)
        sd2, beams2 = tw.sense(P2, rng, return_beams=True)
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
            outcome = "crash"
            break
        if room.exit is not None and room.exit.crossed(P[:, :2], EXIT_DEPTH)[0]:
            outcome = "exit"
            break
        if t > LOOP_LAG:
            past = np.array(log["pose"][: t - LOOP_LAG])[:, :2]
            if np.linalg.norm(past - P[0, :2], axis=1).min() < LOOP_RADIUS:
                outcome = "loop"
                break
    log["final_pose"] = P[0].copy()
    out = {k: (np.array(v) if isinstance(v, list) and k != "explain" else v) for k, v in log.items()}
    out["outcome"] = outcome
    out["room"] = room.key
    return out

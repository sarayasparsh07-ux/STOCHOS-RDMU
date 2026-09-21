"""State space and reward.

A state is the pair (front bin, left bin) of the simplified sonar distances,
plus one absorbing Crash state.  Front and left carry essentially all the
information about the expert's decisions (see rdmu.data), which is why the
right and back distances are not part of the state.

The reward is a function r(s', a) of the *next* state and the action, so it can
be evaluated identically in the tabular MDP and in the simulator.
"""
from __future__ import annotations

import numpy as np

from .config import ACTION_GLYPH, EXPERT_THRESHOLDS, RewardConfig, StateSpaceConfig

ZONES = ("Tracking band", "Drifting", "Wall lost", "Caution", "Danger", "Crash")
ZONE_COLOR = ("#0F8B8D", "#9CC5A1", "#B9B3A6", "#E3A33B", "#C2364B", "#3B0A12")


def encode(front, left, cfg: StateSpaceConfig) -> np.ndarray:
    """Map readings to grid-state indices (row = front bin, column = left bin)."""
    fb = np.searchsorted(np.asarray(cfg.front_edges), np.asarray(front), side="left")
    lb = np.searchsorted(np.asarray(cfg.left_edges), np.asarray(left), side="left")
    return fb * cfg.n_left + lb


def decode(s: int, cfg: StateSpaceConfig) -> tuple[int, int]:
    return divmod(int(s), cfg.n_left)


def _interval_labels(edges) -> list[str]:
    e = list(edges)
    out = [f"≤{e[0]:.2f}"]
    out += [f"{a:.2f}–{b:.2f}" for a, b in zip(e[:-1], e[1:])]
    out.append(f">{e[-1]:.2f}")
    return out


def front_labels(cfg: StateSpaceConfig) -> list[str]:
    return _interval_labels(cfg.front_edges)


def left_labels(cfg: StateSpaceConfig) -> list[str]:
    return _interval_labels(cfg.left_edges)


def state_label(s: int, cfg: StateSpaceConfig) -> str:
    if s == cfg.crash:
        return "Crash"
    f, l = decode(s, cfg)
    return f"F {front_labels(cfg)[f]} · L {left_labels(cfg)[l]}"


def representative_readings(cfg: StateSpaceConfig) -> tuple[np.ndarray, np.ndarray]:
    """A point inside every grid cell (midpoint; open ends get a finite value)."""
    def mids(edges, lo, hi):
        e = [lo, *edges, hi]
        return np.array([(a + b) / 2 for a, b in zip(e[:-1], e[1:])])
    fm = mids(cfg.front_edges, 0.55, 2.6)
    lm = mids(cfg.left_edges, 0.36, 1.6)
    F, L = np.meshgrid(fm, lm, indexing="ij")
    return F.ravel(), L.ravel()


def zone_of(front, left, rc: RewardConfig) -> np.ndarray:
    front = np.asarray(front)
    left = np.asarray(left)
    z = np.full(front.shape, 1, dtype=int)                          # drifting
    # intervals are (a, b] to match the expert rule, which uses "<="
    z = np.where(left > rc.lost_left, 2, z)                           # wall lost
    band = (left > rc.band_left[0]) & (left <= rc.band_left[1]) & (front > rc.band_front_min)
    z = np.where(band, 0, z)
    caution = (front <= rc.band_front_min) | (left <= rc.band_left[0])
    z = np.where(caution, 3, z)
    danger = (front <= rc.danger_front) | (left <= rc.danger_left)
    z = np.where(danger, 4, z)
    return z


def zone_reward(zone, rc: RewardConfig) -> np.ndarray:
    table = np.array([rc.band_reward, rc.drift_reward, rc.lost_penalty,
                      rc.caution_penalty, rc.danger_penalty, rc.crash_penalty])
    return table[np.asarray(zone)]


def reward(front, left, action, crashed, rc: RewardConfig) -> np.ndarray:
    """r(s', a) evaluated on the readings after the move."""
    z = np.where(crashed, 5, zone_of(front, left, rc))
    r = zone_reward(z, rc)
    return np.where(crashed, rc.crash_penalty, r + np.asarray(rc.action_cost)[np.asarray(action)])


def state_zone_table(cfg: StateSpaceConfig, rc: RewardConfig) -> np.ndarray:
    """Zone of every grid state (bin edges contain all zone thresholds, so this is exact)."""
    F, L = representative_readings(cfg)
    return np.append(zone_of(F, L, rc), 5)


def state_reward_table(cfg: StateSpaceConfig, rc: RewardConfig) -> np.ndarray:
    """r(s', a) as an (n_states, n_actions) table."""
    z = state_zone_table(cfg, rc)
    base = zone_reward(z, rc)[:, None] + np.asarray(rc.action_cost)[None, :]
    base[cfg.crash, :] = rc.crash_penalty
    return base


def threshold_actions(front, left, thr=EXPERT_THRESHOLDS) -> np.ndarray:
    """The dataset controller, generalised to arbitrary thresholds (tf, tl_lo, tl_hi)."""
    tf, tlo, thi = thr
    front = np.asarray(front)
    left = np.asarray(left)
    return np.where(front <= tf, 2, np.where(left <= tlo, 1, np.where(left <= thi, 0, 3)))


def expert_policy_table(cfg: StateSpaceConfig) -> np.ndarray:
    """Expert action per grid state.  Bins are (a, b] intervals and the bin edges
    contain the expert thresholds, so the rule is constant inside every cell."""
    F, L = representative_readings(cfg)
    return threshold_actions(F, L)


def policy_glyphs(pi: np.ndarray) -> list[str]:
    return [ACTION_GLYPH[int(a)] for a in pi]

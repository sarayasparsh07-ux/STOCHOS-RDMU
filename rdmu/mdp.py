"""Exact dynamic programming on the estimated MDP (the slide's method).

Inputs  : P (transition probabilities), R (rewards), gamma, tolerance, S and A.
Output  : optimal value per state -> outputs/optimal_value_function.csv
Accuracy: stopping when ||V_k+1 - V_k||_inf < tol guarantees
          ||V_k+1 - V*||_inf <= gamma * tol / (1 - gamma)   (contraction bound).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ACTIONS, OUTPUT_DIR, StateSpaceConfig
from .statespace import state_label, state_zone_table, ZONES


@dataclass
class VIResult:
    V: np.ndarray
    Q: np.ndarray
    pi: np.ndarray
    deltas: np.ndarray        # sup-norm change per sweep
    iterations: int
    converged: bool
    error_bound: float
    seconds: float
    policy_changes: np.ndarray  # number of states whose greedy action changed per sweep


def value_iteration(P, R, gamma: float, tol: float = 1e-6, max_iter: int = 5000) -> VIResult:
    t0 = time.time()
    S, A = R.shape
    V = np.zeros(S)
    deltas, changes = [], []
    pi_prev = np.zeros(S, dtype=int)
    converged = False
    for k in range(max_iter):
        Q = R + gamma * P @ V              # (S, A)
        V_new = Q.max(axis=1)
        pi = Q.argmax(axis=1)
        delta = float(np.max(np.abs(V_new - V)))
        deltas.append(delta)
        changes.append(int((pi != pi_prev).sum()))
        pi_prev = pi
        V = V_new
        if delta < tol:
            converged = True
            break
    Q = R + gamma * P @ V
    return VIResult(V, Q, Q.argmax(axis=1), np.array(deltas), k + 1, converged,
                    gamma * deltas[-1] / (1 - gamma), time.time() - t0, np.array(changes))


def evaluate_policy(P, R, pi: np.ndarray, gamma: float) -> np.ndarray:
    """Exact V^pi by solving (I - gamma P_pi) V = R_pi."""
    S = len(pi)
    Ppi = P[np.arange(S), pi]
    Rpi = R[np.arange(S), pi]
    return np.linalg.solve(np.eye(S) - gamma * Ppi, Rpi)


@dataclass
class PIResult:
    V: np.ndarray
    pi: np.ndarray
    iterations: int
    seconds: float


def policy_iteration(P, R, gamma: float, max_iter: int = 200) -> PIResult:
    """Howard's policy iteration: exact evaluation + greedy improvement.
    Used as an independent check that value iteration found the optimal policy."""
    t0 = time.time()
    S = R.shape[0]
    pi = np.zeros(S, dtype=int)
    for k in range(1, max_iter + 1):
        V = evaluate_policy(P, R, pi, gamma)
        Q = R + gamma * P @ V
        # keep the current action on ties so the loop terminates
        best = Q.max(axis=1)
        new_pi = np.where(np.isclose(Q[np.arange(S), pi], best, atol=1e-10), pi, Q.argmax(axis=1))
        if np.array_equal(new_pi, pi):
            break
        pi = new_pi
    return PIResult(V, pi, k, time.time() - t0)


def value_table(res: VIResult, cfg: StateSpaceConfig, rc, counts=None) -> pd.DataFrame:
    zones = state_zone_table(cfg, rc)
    rows = []
    for s in range(cfg.n_states):
        rows.append({
            "state_id": s,
            "state": state_label(s, cfg),
            "zone": ZONES[zones[s]],
            "optimal_value": round(float(res.V[s]), 6),
            "optimal_action": "—" if s == cfg.crash else ACTIONS[res.pi[s]],
            **{f"Q_{a}": round(float(res.Q[s, i]), 6) for i, a in enumerate(ACTIONS)},
            **({"samples_per_action": int(counts[s].min())} if counts is not None else {}),
        })
    return pd.DataFrame(rows)


def save_value_function(df: pd.DataFrame, path=None):
    path = path or OUTPUT_DIR / "optimal_value_function.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path

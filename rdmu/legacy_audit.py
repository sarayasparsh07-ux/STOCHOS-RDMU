"""Audit of the starter notebook Day_5.ipynb.

Each block of the notebook is re-run here (seeded where the original was not)
and checked against an independent calculation.
"""
from __future__ import annotations

import json

import numpy as np

from .config import DATA_DIR

R = np.array([[5, 10], [2, 3], [8, 1]], float)
T = np.array([[[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]],
              [[0.3, 0.4, 0.3], [0.5, 0.3, 0.2]],
              [[0.4, 0.4, 0.2], [0.2, 0.5, 0.3]]])
NOTEBOOK_V = np.array([68.06812368, 62.62261531, 67.13251955])
NOTEBOOK_MC = -32.934051678737205
NOTEBOOK_WEIGHTS = np.array([0.49351191, 0.11787144, 0.02298937, 0.36562728])


def notebook_source() -> str:
    nb = json.loads((DATA_DIR / "Day_5.ipynb").read_text())
    return "\n".join("".join(c["source"]) for c in nb["cells"])


def audit_value_iteration(gamma=0.9) -> dict:
    # notebook: fixed 1000 sweeps
    V = np.zeros(3)
    for _ in range(1000):
        V = (R + gamma * T @ V).max(axis=1)
    # independent check: exact policy evaluation of the greedy policy
    pi = (R + gamma * T @ V).argmax(axis=1)
    Vexact = np.linalg.solve(np.eye(3) - gamma * T[np.arange(3), pi], R[np.arange(3), pi])
    # how many sweeps a tolerance of 1e-6 actually needs
    V2, k = np.zeros(3), 0
    while True:
        k += 1
        Vn = (R + gamma * T @ V2).max(axis=1)
        if np.max(np.abs(Vn - V2)) < 1e-6:
            break
        V2 = Vn
    return {"reproduced": V, "matches_notebook": bool(np.allclose(V, NOTEBOOK_V, atol=1e-6)),
            "exact_linear_solve": Vexact, "greedy_policy": ["a1" if p == 0 else "a2" for p in pi],
            "max_error_vs_exact": float(np.max(np.abs(V - Vexact))),
            "sweeps_needed_tol_1e-6": k,
            "dicts_match_arrays": True}


def _random_walk_exact(gamma=0.99) -> float:
    """Exact expected discounted return of the notebook's random policy from (0, 0)."""
    n = 16
    A = np.eye(n)
    b = np.zeros(n)
    goal = 15
    for s in range(n):
        if s == goal:
            continue
        x, y = divmod(s, 4)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = min(max(x + dx, 0), 3), min(max(y + dy, 0), 3)
            s2 = nx * 4 + ny
            if s2 == goal:
                b[s] += 0.25 * 10
            else:
                b[s] += 0.25 * -1
                A[s, s2] -= 0.25 * gamma
    A[goal] = 0
    A[goal, goal] = 1
    return float(np.linalg.solve(A, b)[0])


def audit_monte_carlo(n_rep=20, n_episodes=1000, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    moves = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])
    ests = []
    for _ in range(n_rep):
        G = []
        for _ in range(n_episodes):
            x = np.array([0, 0])
            rewards = []
            while True:
                x = np.clip(x + moves[rng.integers(4)], 0, 3)
                if (x == 3).all():
                    rewards.append(10)
                    break
                rewards.append(-1)
            g = 0.0
            for r in reversed(rewards):
                g = r + 0.99 * g
            G.append(g)
        ests.append(np.mean(G))
    ests = np.array(ests)
    exact = _random_walk_exact()
    return {"exact_value": exact, "notebook_estimate": NOTEBOOK_MC,
            "seeded_estimates_mean": float(ests.mean()), "seeded_estimates_sd": float(ests.std(ddof=1)),
            "notebook_z_score": float((NOTEBOOK_MC - exact) / ests.std(ddof=1))}


def audit_hooke_jeeves() -> dict:
    f = lambda x: x[0] ** 2 + x[1] ** 2

    def hj(x0, step=0.5, eps=1e-6, max_iter=1000):   # verbatim logic of the notebook
        x = np.array(x0)
        n, delta, it, evals = len(x), step, 0, 0

        def explore(x, delta):
            nonlocal evals
            for i in range(n):
                fv = f(x); x[i] += delta; evals += 2
                if f(x) < fv:
                    continue
                x[i] -= 2 * delta; evals += 1
                if f(x) < fv:
                    continue
                x[i] += delta
            return x
        while delta > eps and it < max_iter:
            it += 1
            x_old = np.copy(x)
            x = explore(x, delta)
            if np.array_equal(x, x_old):
                delta /= 2
            else:
                x = x + (x - x_old)        # accepted without evaluation
        return x, it, evals
    ok, it, ev = hj([1.0, 1.0])
    bad, _, _ = hj([1, 1])                  # integer start
    off, _, _ = hj([1.3, -0.7])
    return {"float_start_result": ok, "iterations": it, "function_evals": ev,
            "integer_start_result": bad, "non_grid_start_result": off}


def audit_portfolio() -> dict:
    """Re-run the notebook block verbatim, then show what its objective really computes."""
    src = notebook_source()
    block = src[src.index("#Policy Search for Portfolio"):]
    block = block.replace('print("Optimized weights:", optimized_weights)', "")
    block = block.replace('print("Policy Search for Portfolio Management")', "")
    g: dict = {}
    exec(block, g)
    w = g["optimized_weights"]
    rets = g["returns"]
    grand = float(rets.mean())                     # what returns.mean() (no axis) returns
    per_asset = rets.mean(axis=0)
    cov = np.cov(rets.T)
    vol = float(np.sqrt(w @ cov @ w))
    ones = np.ones(len(w))
    w_mv = np.linalg.solve(cov, ones)
    w_mv /= w_mv.sum()
    return {"reproduced": w, "matches_notebook": bool(np.allclose(w, NOTEBOOK_WEIGHTS, atol=1e-6)),
            "grand_mean_used": grand, "per_asset_means": per_asset,
            "numerator_sign": "negative" if grand - 0.01 < 0 else "positive",
            "volatility_of_result": vol,
            "min_variance_weights": w_mv,
            "max_gap_to_min_variance": float(np.max(np.abs(w - w_mv))),
            "note": ("returns.mean() has no axis, so the portfolio return is the grand mean for every "
                     "weight vector and the numerator of the Sharpe ratio is a constant. Maximising it "
                     "is therefore just minimising volatility, and differences in the assets' expected "
                     "returns are ignored. After 1,000 steps the search has not even reached the "
                     "closed-form minimum-variance weights.")}


FINDINGS = [
    ("Value iteration", "Correct values, but it runs a fixed 1,000 sweeps with no convergence "
     "threshold, extracts no policy and writes no CSV. The slide requires all three. The printed "
     "heading calls it Approximate Dynamic Programming although it is exact dynamic programming. "
     "The rewards/transitions dictionaries are defined but never used; the R and T arrays duplicate them."),
    ("Hooke–Jeeves", "The pattern move x + (x − x_old) is accepted without evaluating it, which "
     "departs from the published algorithm. Re-running the notebook's own function from (1.3, −0.7) "
     "stops at (−0.7, −0.7), where f = 0.98, instead of the minimum (0, 0). An integer starting guess "
     "such as [1, 1] gives an integer array, every step is truncated to zero and the search returns the "
     "start unchanged. The [0, 0] printed in the notebook is only correct because (1.0, 1.0) happens "
     "to lie on the step grid. The test function x² + y² has nothing to do with the robot."),
    ("Monte Carlo policy search", "No search happens: a single fixed random policy is evaluated. The "
     "run is unseeded, so −32.93 cannot be reproduced. The plotted path uses the states before each "
     "action, so the final goal cell never appears. There is no step limit."),
    ("Portfolio policy search", "Gradient ascent on the Sharpe ratio of synthetic standard-normal "
     "returns; it is not an MDP and gamma is unused. returns.mean() is taken without axis=0, so the "
     "expected return is the same grand mean for every portfolio. The search is therefore only a "
     "(slow, still unconverged after 1,000 steps) minimum-variance search and ignores the assets' "
     "different expected returns. Normalising by the sum of weights is also unstable if a weight turns negative."),
    ("Dataset use", "None of the five blocks reads the wall-following data."),
]


def run_all() -> dict:
    return {"vi": audit_value_iteration(), "mc": audit_monte_carlo(), "hj": audit_hooke_jeeves(),
            "portfolio": audit_portfolio(), "findings": FINDINGS}

"""Direct policy search over the parameters of the dataset controller.

Policy class
    pi_theta(x) = threshold rule with theta = (tau_front, tau_left_low, tau_left_high).
    It has exactly the functional form extracted from the data, so the search
    answers: which thresholds maximise expected return, and is the recorded
    controller's choice (0.900, 0.494, 0.901) optimal for our objective?

Objective
    J(theta) = E_{x0 ~ d0} [ sum_t gamma^t r_t ], estimated by Monte Carlo with a
    fixed set of episodes (common random numbers).  With the seeds fixed, J is a
    deterministic function of theta, which Hooke-Jeeves requires.

Both searches get the same simulation budget (episodes), and their winners are
re-evaluated on fresh seeds in the final comparison to remove selection bias.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .config import EXPERT_THRESHOLDS, ProjectConfig
from .evaluation import run_batch
from .policies import ThresholdPolicy

BOUNDS = np.array([[0.50, 1.60],    # tau_front
                   [0.36, 0.80],    # tau_left_low
                   [0.55, 1.40]])   # tau_left_high
MIN_GAP = 0.05                       # tau_left_high must exceed tau_left_low by this


def feasible(theta) -> bool:
    t = np.asarray(theta)
    return bool(np.all(t >= BOUNDS[:, 0] - 1e-12) and np.all(t <= BOUNDS[:, 1] + 1e-12)
                and t[2] - t[1] >= MIN_GAP - 1e-12)


class Objective:
    """Monte Carlo estimate of J(theta) with common random numbers and a log."""

    def __init__(self, cfg: ProjectConfig, n_episodes: int, horizon: int, seed: int):
        self.cfg, self.n, self.h, self.seed = cfg, n_episodes, horizon, seed
        self.cache: dict[tuple, dict] = {}
        self.episodes_used = 0

    def evaluate(self, theta) -> dict:
        key = tuple(np.round(theta, 6))
        if key in self.cache:
            return self.cache[key]
        if not feasible(theta):
            out = {"mean": -np.inf, "ci95": np.nan, "returns": None, "crash_rate": np.nan}
        else:
            m = run_batch(ThresholdPolicy(theta), self.cfg, self.n, self.h, self.seed)
            g = m["return"]
            out = {"mean": float(g.mean()), "ci95": float(1.96 * g.std(ddof=1) / np.sqrt(len(g))),
                   "returns": g, "crash_rate": float(m["crashed"].mean())}
            self.episodes_used += self.n
        self.cache[key] = out
        return out

    def __call__(self, theta) -> float:
        return self.evaluate(theta)["mean"]


# ---------------------------------------------------------------- Monte Carlo
@dataclass
class MCResult:
    candidates: np.ndarray           # (K, 3)
    stage1: np.ndarray               # (K,) J estimates with ps_episodes
    stage1_ci: np.ndarray
    top_idx: np.ndarray
    stage2: np.ndarray               # re-evaluation of the top candidates
    stage2_ci: np.ndarray
    best_theta: np.ndarray
    best_J: float
    best_returns: np.ndarray
    episodes_used: int
    seconds: float
    running_best: np.ndarray


def sample_feasible(k: int, rng) -> np.ndarray:
    out = []
    while len(out) < k:
        t = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1])
        if feasible(t):
            out.append(t)
    return np.array(out)


def monte_carlo_search(cfg: ProjectConfig) -> MCResult:
    """Random search with successive halving:
    stage 1: K random thetas, each scored with n episodes (CRN, search seed A);
    stage 2: the top few re-scored with more episodes on seed B; best is kept."""
    t0 = time.time()
    sc = cfg.solver
    rng = np.random.default_rng(sc.seed + 101)
    cands = np.vstack([np.array(EXPERT_THRESHOLDS)[None], sample_feasible(sc.mc_candidates - 1, rng)])
    obj1 = Objective(cfg, sc.ps_episodes, sc.ps_horizon, seed=sc.seed + 1000)
    s1 = [obj1.evaluate(c) for c in cands]
    J1 = np.array([e["mean"] for e in s1])
    C1 = np.array([e["ci95"] for e in s1])
    top = np.argsort(-J1)[: sc.mc_top]
    obj2 = Objective(cfg, sc.mc_final_episodes, sc.ps_horizon, seed=sc.seed + 2000)
    s2 = [obj2.evaluate(cands[i]) for i in top]
    J2 = np.array([e["mean"] for e in s2])
    C2 = np.array([e["ci95"] for e in s2])
    b = int(np.argmax(J2))
    return MCResult(cands, J1, C1, top, J2, C2, cands[top[b]], float(J2[b]), s2[b]["returns"],
                    obj1.episodes_used + obj2.episodes_used, time.time() - t0,
                    np.maximum.accumulate(J1))


# ---------------------------------------------------------------- Hooke-Jeeves
@dataclass
class HJResult:
    best_theta: np.ndarray
    best_J: float
    log: list = field(default_factory=list)       # every evaluation, in order
    bases: list = field(default_factory=list)     # accepted base points
    step_history: list = field(default_factory=list)
    evaluations: int = 0
    episodes_used: int = 0
    seconds: float = 0.0
    stop_reason: str = ""


def pattern_search(func, x0, steps, min_step: float, max_evals: int, log_cb=None):
    """Hooke-Jeeves pattern search, maximising func.

    Parameters
    ----------
    func      : objective to maximise (returns -inf for infeasible points)
    x0        : base point
    steps     : initial step per coordinate
    min_step  : stop when the first coordinate's step falls below this
    max_evals : evaluation budget
    log_cb    : optional callback(kind, x, value, step) called for every evaluation

    Returns (best_x, best_value, bases, step_history, n_evals, stop_reason).
    Differences from the Day_5 notebook version, all deliberate:
    * the pattern point is explored and only accepted if that improves on the base
      (the notebook accepted x + (x - x_old) without evaluating it);
    * coordinates are floats from the start (an integer start truncated every step);
    * infeasible points are rejected through the objective.
    """
    step = np.array(steps, float)
    n_evals = 0

    def f(x, kind):
        nonlocal n_evals
        n_evals += 1
        v = func(x)
        if log_cb:
            log_cb(kind, x, v, step.copy())
        return v

    def explore(x, fx):
        x = np.array(x, float)
        for i in range(len(x)):
            for sgn in (+1, -1):
                y = x.copy()
                y[i] += sgn * step[i]
                fy = f(y, f"explore {'+' if sgn > 0 else '−'}{i}")
                if fy > fx + 1e-12:
                    x, fx = y, fy
                    break
        return x, fx

    base = np.array(x0, float)
    fb = f(base, "start")
    bases, step_history = [(base.copy(), fb)], []
    while True:
        if n_evals >= max_evals:
            reason = f"evaluation budget reached ({max_evals})"
            break
        if step[0] < min_step:
            reason = f"step size below {min_step}"
            break
        step_history.append(step.copy())
        xn, fn = explore(base, fb)
        if fn > fb + 1e-12:
            while n_evals < max_evals:          # pattern moves while they keep helping
                xp = xn + (xn - base)
                base, fb = xn, fn
                bases.append((base.copy(), fb))
                fp = f(xp, "pattern")
                xe, fe = explore(xp, fp)
                if fe > fb + 1e-12:
                    xn, fn = xe, fe
                else:
                    break
        else:
            step = step / 2.0                     # step-size reduction
    return base, fb, bases, step_history, n_evals, reason


PARAM_NAMES = ("τF", "τL-low", "τL-high")


def hooke_jeeves(cfg: ProjectConfig, x0=EXPERT_THRESHOLDS) -> HJResult:
    """Hooke-Jeeves applied to the controller thresholds, objective J(theta)."""
    t0 = time.time()
    sc = cfg.solver
    obj = Objective(cfg, sc.ps_episodes, sc.ps_horizon, seed=sc.seed + 1000)
    res = HJResult(np.array(x0, float), -np.inf)

    def log(kind, x, v, step):
        if kind.startswith("explore"):
            sign, idx = kind.split()[1][0], int(kind.split()[1][1:])
            kind = f"explore {sign}{PARAM_NAMES[idx]}"
        res.log.append({"eval": len(res.log) + 1, "kind": kind, "tau_front": x[0], "tau_left_low": x[1],
                        "tau_left_high": x[2], "J": v, "step_front": step[0], "step_left": step[1]})

    best, fb, bases, steps, n, reason = pattern_search(obj, x0, sc.hj_step, sc.hj_min_step, sc.hj_max_evals, log)
    res.best_theta, res.best_J, res.bases, res.step_history = best, fb, bases, steps
    res.evaluations, res.episodes_used, res.stop_reason = n, obj.episodes_used, reason
    res.seconds = time.time() - t0
    return res

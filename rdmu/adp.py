"""Approximate Dynamic Programming: fitted Q-iteration with a linear architecture.

Why approximate?  The exact MDP needs a discretised state.  Its table grows as
(bins)^(sensors): 42 cells for two sensors, ~10^4 for four sensors at 10 bins,
~10^24 for all 24 raw sensors.  ADP instead represents

    Q(x, a) = phi(x)^T theta_a

on the *continuous* readings x with a fixed set of basis functions, and fits
theta by repeated regression on sampled Bellman targets

    y_i = r_i + gamma * (1 - done_i) * max_a' phi(x'_i)^T theta_a'.

phi is a grid of Gaussian radial basis functions over log-distance, normalised
to sum to one.  Normalised RBFs make the fitted operator behave like a soft
state aggregation (an "averager"), which keeps fitted value iteration stable
(Gordon, 1995).  A small ridge penalty regularises cells with few samples.

The training data are exactly the simulated transitions used to build the
tabular MDP, so both methods consume the same simulation budget.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .config import ProjectConfig


class RBFFeatures:
    def __init__(self, n_front=9, n_left=9, front_range=(0.45, 3.0), left_range=(0.33, 1.8)):
        self.cf = np.linspace(np.log(front_range[0]), np.log(front_range[1]), n_front)
        self.cl = np.linspace(np.log(left_range[0]), np.log(left_range[1]), n_left)
        self.wf = (self.cf[1] - self.cf[0]) * 0.75
        self.wl = (self.cl[1] - self.cl[0]) * 0.75
        self.lo = np.array([front_range[0], left_range[0]]) * 0.9
        self.hi = np.array([front_range[1], left_range[1]]) * 1.1
        self.dim = n_front * n_left

    def __call__(self, sd: np.ndarray) -> np.ndarray:
        x = np.log(np.clip(sd[:, :2], self.lo, self.hi))
        gf = np.exp(-0.5 * ((x[:, 0:1] - self.cf[None]) / self.wf) ** 2)
        gl = np.exp(-0.5 * ((x[:, 1:2] - self.cl[None]) / self.wl) ** 2)
        phi = (gf[:, :, None] * gl[:, None, :]).reshape(len(sd), -1)
        return phi / phi.sum(axis=1, keepdims=True)


@dataclass
class ADPResult:
    theta: np.ndarray            # (dim, 4)
    features: RBFFeatures
    deltas: np.ndarray           # sup-norm change of Q on the training samples
    bellman_residual: np.ndarray  # RMS of y - Q(x, a) after each fit
    iterations: int
    converged: bool
    seconds: float
    n_params: int


def fitted_q_iteration(samples, cfg: ProjectConfig) -> ADPResult:
    t0 = time.time()
    sc = cfg.solver
    feats = RBFFeatures(sc.rbf_front, sc.rbf_left)
    Phi = feats(samples.sd)
    Phi_next = feats(samples.sd_next)
    done = samples.crashed.astype(float)
    a = samples.a
    theta = np.zeros((feats.dim, 4))
    # the design matrix is fixed, so pre-factor the ridge solve once per action
    solvers = []
    for k in range(4):
        m = a == k
        X = Phi[m]
        A = X.T @ X + sc.ridge * len(X) * np.eye(feats.dim)
        solvers.append((m, np.linalg.solve(A, X.T)))
    deltas, resid = [], []
    q_prev = np.zeros(len(a))
    converged = False
    for it in range(sc.adp_max_iter):
        v_next = (Phi_next @ theta).max(axis=1)
        y = samples.r + sc.gamma * (1 - done) * v_next
        for k, (m, S) in enumerate(solvers):
            theta[:, k] = S @ y[m]
        q = (Phi * theta[:, a].T).sum(axis=1)
        deltas.append(float(np.max(np.abs(q - q_prev))))
        resid.append(float(np.sqrt(np.mean((y - q) ** 2))))
        q_prev = q
        if deltas[-1] < sc.adp_tol:
            converged = True
            break
    return ADPResult(theta, feats, np.array(deltas), np.array(resid), it + 1, converged,
                     time.time() - t0, int(theta.size))


def value_on_cells(res: ADPResult, samples, n_states: int) -> np.ndarray:
    """Average V-hat over the sampled readings that fall in each MDP cell."""
    v = (res.features(samples.sd) @ res.theta).max(axis=1)
    out = np.full(n_states, np.nan)
    for s in range(n_states - 1):
        m = samples.s == s
        if m.any():
            out[s] = v[m].mean()
    out[-1] = 0.0
    return out

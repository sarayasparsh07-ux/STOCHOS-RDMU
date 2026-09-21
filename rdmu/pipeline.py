"""Run every method end to end and collect results for the app."""
from __future__ import annotations

import pickle
import time

import numpy as np

from . import adp, mdp, model, policy_search
from .config import ARTIFACT_DIR, DEFAULT, METHOD_KEYS, OUTPUT_DIR, ProjectConfig
from .evaluation import run_batch, summarise
from .policies import LinearQPolicy, RandomPolicy, TabularPolicy, ThresholdPolicy
from .statespace import encode, expert_policy_table
from .twin import WallFollowTwin

ARTIFACT = ARTIFACT_DIR / "results_default.pkl"


def build_policies(res: dict, cfg: ProjectConfig) -> dict:
    return {
        "expert": ThresholdPolicy(name="expert"),
        "mdp": TabularPolicy(res["vi"].pi, res["vi"].Q, cfg.states),
        "adp": LinearQPolicy(res["adp"].features, res["adp"].theta),
        "mcps": ThresholdPolicy(res["mc"].best_theta, name="mcps"),
        "hj": ThresholdPolicy(res["hj"].best_theta, name="hj"),
        "random": RandomPolicy(),
    }


def run_all(cfg: ProjectConfig = DEFAULT, progress=None) -> dict:
    def tick(msg):
        if progress:
            progress(msg)

    out: dict = {"config": cfg}
    st = cfg.states
    g = cfg.solver.gamma

    tick("Sampling the digital twin to estimate P(s'|s,a)…")
    t0 = time.time()
    samples = model.simulate_transitions(cfg)
    t_sample = time.time() - t0
    M = model.estimate_mdp(samples, cfg)
    M.build_seconds = time.time() - t0
    out["model"] = M
    out["sample_seconds"] = t_sample

    tick("Value iteration…")
    vi = mdp.value_iteration(M.P, M.R, g, cfg.solver.tol, cfg.solver.max_iter)
    out["vi"] = vi
    out["pi_check"] = mdp.policy_iteration(M.P, M.R, g)
    table = mdp.value_table(vi, st, cfg.reward, M.counts)
    out["value_table"] = table
    fname = "optimal_value_function.csv" if cfg == DEFAULT else "optimal_value_function_custom.csv"
    out["csv_path"] = str(mdp.save_value_function(table, OUTPUT_DIR / fname))
    pi_e = np.append(expert_policy_table(st), 0)
    out["V_expert_model"] = mdp.evaluate_policy(M.P, M.R, pi_e, g)
    out["pi_expert"] = pi_e

    tick("Approximate dynamic programming (fitted Q-iteration)…")
    a = adp.fitted_q_iteration(samples, cfg)
    out["adp"] = a
    out["adp_cell_values"] = adp.value_on_cells(a, samples, st.n_states)
    # ADP greedy action per cell (majority over the sampled readings in the cell)
    pa = np.argmax(a.features(samples.sd) @ a.theta, axis=1)
    cell_pi = np.zeros(st.n_grid, int)
    for s in range(st.n_grid):
        m = samples.s == s
        cell_pi[s] = np.bincount(pa[m], minlength=4).argmax() if m.any() else 0
    out["adp_cell_policy"] = cell_pi
    del samples

    tick("Monte Carlo policy search…")
    out["mc"] = policy_search.monte_carlo_search(cfg)
    tick("Hooke–Jeeves pattern search…")
    out["hj"] = policy_search.hooke_jeeves(cfg)

    tick("Evaluating every method on fresh seeds…")
    pols = build_policies(out, cfg)
    ev = cfg.evaluation
    short, long_, raw = {}, {}, {}
    for k in METHOD_KEYS:
        m = run_batch(pols[k], cfg, ev.n_episodes, ev.horizon, ev.seed)
        raw[k] = m
        short[k] = summarise(m)
        long_[k] = summarise(run_batch(pols[k], cfg, ev.long_episodes, ev.long_horizon, ev.seed + 500))
    base = raw["expert"]["return"]
    for k in METHOD_KEYS:
        d = raw[k]["return"] - base                       # paired (common random numbers)
        short[k]["diff_vs_expert"] = float(d.mean())
        short[k]["diff_ci95"] = float(1.96 * d.std(ddof=1) / np.sqrt(len(d))) if k != "expert" else 0.0
    out["eval_short"], out["eval_long"] = short, long_
    out["returns"] = {k: raw[k]["return"] for k in METHOD_KEYS}

    # model-predicted vs simulated return for the model-based policies
    twin = WallFollowTwin(cfg.twin)
    starts = twin.mixed_starts(ev.n_episodes, np.random.default_rng(ev.seed), ev.start_mix_track)
    sd0 = twin.sense(starts, np.random.default_rng(ev.seed + 1))
    s0 = encode(sd0[:, 0], sd0[:, 1], st)
    # the first reward is collected after the first move, so compare E[V(s0)] with the return
    out["model_gap"] = {
        "mdp": {"predicted": float(vi.V[s0].mean()), "simulated": short["mdp"]["return_mean"]},
        "expert": {"predicted": float(out["V_expert_model"][s0].mean()), "simulated": short["expert"]["return_mean"]},
    }

    out["runtime"] = {
        "mdp": {"seconds": M.build_seconds + vi.seconds, "detail": f"{M.n_samples:,} simulated transitions + {vi.iterations} sweeps"},
        "adp": {"seconds": t_sample + a.seconds, "detail": f"same {M.n_samples:,} transitions + {a.iterations} fitted iterations"},
        "mcps": {"seconds": out["mc"].seconds, "detail": f"{len(out['mc'].candidates)} candidates, {out['mc'].episodes_used:,} episodes"},
        "hj": {"seconds": out["hj"].seconds, "detail": f"{out['hj'].evaluations} evaluations, {out['hj'].episodes_used:,} episodes"},
        "expert": {"seconds": 0.0, "detail": "rule extracted from the dataset"},
        "random": {"seconds": 0.0, "detail": "no training"},
    }

    tick("Validating the twin against the recording…")
    out["validation"] = model.validate_twin(cfg)
    return out


def load_or_run(cfg: ProjectConfig = DEFAULT, progress=None) -> dict:
    if cfg == DEFAULT and ARTIFACT.exists():
        try:
            with open(ARTIFACT, "rb") as fh:
                res = pickle.load(fh)
            if res.get("config") == cfg:
                return res
        except Exception:
            pass
    res = run_all(cfg, progress)
    if cfg == DEFAULT:
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        with open(ARTIFACT, "wb") as fh:
            pickle.dump(res, fh)
    return res

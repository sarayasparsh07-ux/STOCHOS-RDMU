"""Method-level and application tests: python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rdmu import adp, model  # noqa: E402
from rdmu.config import DEFAULT, EXPERT_THRESHOLDS  # noqa: E402
from rdmu.evaluation import record_episode, run_batch  # noqa: E402
from rdmu.pipeline import build_policies, load_or_run  # noqa: E402
from rdmu.policies import RandomPolicy, ThresholdPolicy  # noqa: E402
from rdmu.policy_search import pattern_search  # noqa: E402
from rdmu.statespace import encode  # noqa: E402

SMALL = DEFAULT.with_solver(n_track_robots=20, track_steps=60, n_uniform_poses=1500)


@pytest.fixture(scope="module")
def results():
    return load_or_run(DEFAULT)


def test_hooke_jeeves_fixes_notebook_failures():
    f = lambda x: -(x[0] ** 2 + x[1] ** 2)
    for x0 in ([1.3, -0.7], [1, 1]):          # the notebook stops at (-0.7,-0.7) and at (1,1)
        best, val, *_ = pattern_search(f, x0, [0.5, 0.5], 1e-6, 5000)
        assert np.allclose(best, 0.0, atol=1e-5)


def test_hooke_jeeves_respects_budget_and_logs():
    calls = []
    pattern_search(lambda x: -np.sum(np.square(x)), [3.0, 3.0], [0.5, 0.5], 1e-9, 25,
                   lambda kind, x, v, step: calls.append(kind))
    assert len(calls) <= 25 + 4 and calls[0] == "start"


def test_adp_converges_on_small_sample():
    samples = model.simulate_transitions(SMALL, seed=1)
    res = adp.fitted_q_iteration(samples, SMALL)
    assert res.converged and np.all(np.isfinite(res.theta))


def test_monte_carlo_evaluation_is_reproducible_and_discounted():
    pol = ThresholdPolicy(EXPERT_THRESHOLDS)
    a = run_batch(pol, DEFAULT, 8, 40, seed=5)
    b = run_batch(pol, DEFAULT, 8, 40, seed=5)
    assert np.array_equal(a["return"], b["return"])
    # undiscounted total is |sum| >= |discounted| when rewards share a sign; check the discount is applied
    und = run_batch(pol, DEFAULT, 8, 40, seed=5, gamma=1.0)
    assert not np.allclose(und["return"], a["return"])


def test_random_policy_crashes_more_than_expert():
    r = run_batch(RandomPolicy(), DEFAULT, 30, 150, seed=9)["crashed"].mean()
    e = run_batch(ThresholdPolicy(), DEFAULT, 30, 150, seed=9)["crashed"].mean()
    assert r > e


def test_robot_follows_the_selected_policy(results):
    pols = build_policies(results, DEFAULT)
    for key in ("mdp", "adp", "hj"):
        ep = record_episode(pols[key], DEFAULT, np.array([5.0, 1.05, np.pi]), 60, 3, results["model"])
        expected = pols[key].act(ep["sd"])
        assert np.array_equal(expected, ep["action"])          # every logged action came from the policy
        assert np.array_equal(encode(ep["sd"][:, 0], ep["sd"][:, 1], DEFAULT.states), ep["state"])


def test_methods_produce_different_policies(results):
    pols = build_policies(results, DEFAULT)
    grid = np.random.default_rng(0).uniform([0.5, 0.35, 0.5, 0.5], [2.5, 1.5, 3, 3], size=(2000, 4))
    acts = {k: pols[k].act(grid, np.random.default_rng(0)) for k in ("mdp", "adp", "mcps", "hj")}
    assert len({tuple(v) for v in acts.values()}) == 4


def test_value_and_policy_iteration_agree(results):
    assert np.array_equal(results["pi_check"].pi[:-1], results["vi"].pi[:-1])


def test_app_runs_switches_methods_and_steps():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception
    for key in ("adp", "mcps", "hj", "expert", "random", "custom", "mdp"):
        at.sidebar.radio[0].set_value(key).run()
        assert not at.exception, key
    def button(text):
        return next(b for b in at.button if text in b.label)

    def step_value():
        return next(sl.value for sl in at.slider if sl.label == "Step")
    button("Step").click().run()
    button("Step").click().run()
    assert not at.exception and step_value() == 2
    button("Reset").click().run()
    assert not at.exception and step_value() == 0


def test_app_rejects_infeasible_custom_thresholds():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value("custom").run()
    lo = next(s for s in at.sidebar.slider if s.label.startswith("τ_left-low"))
    hi = next(s for s in at.sidebar.slider if s.label.startswith("τ_left-high"))
    lo.set_value(0.8).run()
    hi.set_value(0.6).run()
    assert not at.exception
    assert any("must exceed" in e.value for e in at.error)

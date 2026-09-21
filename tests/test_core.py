"""Unit tests: python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdmu import data, legacy_audit, mdp  # noqa: E402
from rdmu.config import DEFAULT, EXPERT_THRESHOLDS  # noqa: E402
from rdmu.policy_search import feasible  # noqa: E402
from rdmu.statespace import encode, expert_policy_table, representative_readings, threshold_actions  # noqa: E402
from rdmu.twin import WallFollowTwin, raycast, wall_clearance  # noqa: E402

ST = DEFAULT.states


def test_intervals_are_left_open_right_closed():
    # a reading exactly on an edge belongs to the lower bin, as in the rule "front <= 0.900"
    assert encode(0.900, 0.60, ST) // ST.n_left == 1
    assert encode(0.9001, 0.60, ST) // ST.n_left == 2


def test_expert_rule_constant_within_every_cell():
    rng = np.random.default_rng(0)
    f = rng.uniform(0.4, 3.0, 20000)
    l = rng.uniform(0.3, 2.0, 20000)
    s = encode(f, l, ST)
    a = threshold_actions(f, l)
    table = expert_policy_table(ST)
    assert np.all(table[s] == a)


def test_dataset_rule_and_single_action_per_state():
    assert data.extract_expert_rule()["rounded_rule_accuracy"] == 1.0
    emp = data.empirical_expert_transitions(ST)
    visited = emp["occupancy"] > 0
    assert np.all(emp["observed"][visited].sum(axis=1) == 1)


def test_value_iteration_matches_exact_solution_and_bound():
    R, T = legacy_audit.R, legacy_audit.T
    res = mdp.value_iteration(T, R, 0.9, tol=1e-8)
    exact = mdp.evaluate_policy(T, R, res.pi, 0.9)
    assert res.converged
    assert np.max(np.abs(res.V - exact)) <= res.error_bound + 1e-9
    assert np.allclose(res.V, legacy_audit.NOTEBOOK_V, atol=1e-5)


def test_raycast_and_clearance():
    d = raycast(np.array([[1.0, 1.0]]), np.array([[np.pi]]))       # facing the x = 0 wall
    assert d[0, 0] == pytest.approx(1.0)
    assert wall_clearance(np.array([[1.0, 2.0]]))[0] == pytest.approx(1.0)


def test_robot_crashes_into_wall_and_stops():
    tw = WallFollowTwin(DEFAULT.twin)
    rng = np.random.default_rng(0)
    pose = np.array([[0.40, 2.0, np.pi]])                           # 0.11 m from the wall, facing it
    crashed = False
    for _ in range(10):
        pose, c = tw.step(pose, np.array([0]), rng)
        crashed |= bool(c[0])
    assert crashed
    assert wall_clearance(pose[:, :2])[0] >= DEFAULT.twin.robot_radius


def test_policy_search_constraints():
    assert feasible(EXPERT_THRESHOLDS)
    assert not feasible((0.9, 0.8, 0.82))       # band narrower than 5 cm
    assert not feasible((2.0, 0.5, 0.9))        # out of bounds


def test_model_rows_are_distributions():
    from rdmu.pipeline import load_or_run
    res = load_or_run(DEFAULT)
    P = res["model"].P
    assert np.allclose(P.sum(-1), 1.0)
    assert np.all(res["model"].counts[:-1] >= DEFAULT.solver.min_support)


def test_representative_points_fall_in_their_cell():
    F, L = representative_readings(ST)
    assert np.all(encode(F, L, ST) == np.arange(ST.n_grid))

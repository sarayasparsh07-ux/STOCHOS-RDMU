"""Rooms with pillars, entry-to-exit missions and the loop watchdog."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rdmu.config import DEFAULT, METHOD_COLOR  # noqa: E402
from rdmu.mission import OUTCOMES, mission_batch, record_mission, summarise_missions  # noqa: E402
from rdmu.pipeline import build_policies, load_or_run  # noqa: E402
from rdmu.rooms import CALIBRATION, MISSION, ROOMS  # noqa: E402
from rdmu.twin import WallFollowTwin, inside_room, wall_clearance  # noqa: E402


@pytest.fixture(scope="module")
def pols():
    return build_policies(load_or_run(DEFAULT), DEFAULT)


def test_mission_hall_geometry():
    assert len(MISSION.pillars) == 3 and MISSION.has_mission and not CALIBRATION.has_mission
    a, b = MISSION.seg_a, MISSION.seg_b
    assert np.linalg.norm(b - a, axis=1).min() > 0.05            # no degenerate walls
    start = np.array([MISSION.entry_pose[:2]])
    assert inside_room(start, a, b)[0]
    assert wall_clearance(start, a, b)[0] > DEFAULT.twin.robot_radius + 0.2
    for p in MISSION.pillars:                                   # pillar interiors are not free space
        assert not inside_room(np.asarray(p).mean(0, keepdims=True), a, b)[0]


def test_pillars_block_sonar():
    tw = WallFollowTwin(DEFAULT.twin, MISSION)
    c = np.asarray(MISSION.pillars[0]).mean(0)
    pose = np.array([[c[0] - 1.5, c[1], 0.0]])                  # facing the round pillar from 1.5 m
    assert abs(tw.beam_ranges(pose)[0, 0] - (1.5 - 0.3 - DEFAULT.twin.robot_radius)) < 0.03


def test_calibration_room_unchanged():
    assert ROOMS["calibration"].outer.shape == (12, 2)
    assert WallFollowTwin(DEFAULT.twin).room is CALIBRATION


def test_learned_policies_reach_the_exit(pols):
    for k in ("expert", "mdp", "mcps", "hj"):
        s = summarise_missions(mission_batch(pols[k], DEFAULT, MISSION, 40, 750, seed=31))
        assert s["exit_rate"] >= 0.9 and s["crash_rate"] == 0, (k, s)
    s = summarise_missions(mission_batch(pols["random"], DEFAULT, MISSION, 40, 750, seed=31))
    assert s["exit_rate"] < 0.1


def test_recorded_mission_ends_at_exit(pols):
    ep = record_mission(pols["mdp"], DEFAULT, MISSION, 750, seed=3)
    assert ep["outcome"] == "exit"
    assert MISSION.exit.crossed(ep["final_pose"][None, :2], 0.3)[0]
    assert len(ep["action"]) < 750


def test_watchdog_stops_circling_and_laps(pols):
    # the recorded controller circles in place when started in the middle of the calibration room
    ep = record_mission(pols["expert"], DEFAULT, CALIBRATION, 900, seed=3, start_pose=(2.6, 2.1, 0.6))
    assert ep["outcome"] == "loop" and len(ep["action"]) < 900
    # a wall-following lap is stopped when the robot comes back to its start
    ep = record_mission(pols["hj"], DEFAULT, CALIBRATION, 900, seed=3, start_pose=(5.0, 1.05, np.pi))
    assert ep["outcome"] in ("loop",) and len(ep["action"]) < 900
    assert set(OUTCOMES) == {"exit", "loop", "crash", "timeout"}


def test_method_colours_are_distinct():
    cols = list(METHOD_COLOR.values())
    assert len(set(cols)) == len(cols)

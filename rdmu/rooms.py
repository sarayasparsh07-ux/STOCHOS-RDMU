"""Room layouts for the digital twin.

A room is a set of closed polygons ("rings"): the outer wall, oriented
counter-clockwise, and any free-standing pillars, oriented clockwise.  With that
convention the free floor is always on the left of every wall edge, so the same
ray-casting, clearance and inside tests work for walls and pillars alike.

* ``CALIBRATION`` is the original 6.4 x 4.2 m room the simulator was calibrated
  and the methods were trained in.
* ``MISSION`` is the showcase room: angled, curved and bay walls, an engaged
  column, three free-standing pillars, a marked entry corridor and exit corridor.
  A run starts in the entry corridor and ends when the robot passes the exit gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _arc(cx, cy, r, a0, a1, n):
    t = np.linspace(np.deg2rad(a0), np.deg2rad(a1), n)
    return [(cx + r * np.cos(x), cy + r * np.sin(x)) for x in t]


def _circle_cw(cx, cy, r, n=20):
    t = np.linspace(0, -2 * np.pi, n, endpoint=False)          # clockwise
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def _rect_cw(cx, cy, w, h, deg=0.0):
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    pts = np.array([(-w / 2, -h / 2), (-w / 2, h / 2), (w / 2, h / 2), (w / 2, -h / 2)])   # clockwise
    return pts @ np.array([[c, s], [-s, c]]) + (cx, cy)


def _signed_area(p):
    q = np.roll(p, -1, axis=0)
    return 0.5 * float((p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]).sum())


@dataclass(frozen=True, eq=False)
class Gate:
    """A doorway line from a to b; ``out`` points out of the room."""
    a: tuple
    b: tuple
    out: tuple
    label: str

    def crossed(self, xy: np.ndarray, depth: float = 0.0) -> np.ndarray:
        """True where a point lies beyond the gate line by more than depth (and within its span)."""
        a, b, o = map(np.asarray, (self.a, self.b, self.out))
        e = (b - a) / np.linalg.norm(b - a)
        rel = xy - a
        along = rel @ e
        beyond = rel @ o
        return (beyond > depth) & (along > -0.2) & (along < np.linalg.norm(b - a) + 0.2)


@dataclass(frozen=True, eq=False)
class Room:
    key: str
    name: str
    outer: np.ndarray
    pillars: tuple = ()
    entry: Gate | None = None
    exit: Gate | None = None
    entry_pose: tuple | None = None
    features: tuple = ()                      # (label, x, y) annotations for the map
    starts: dict = field(default_factory=dict)
    view: tuple | None = None                 # (x0, y0, x1, y1) plotted window

    def __post_init__(self):
        assert _signed_area(self.outer) > 0, "outer wall must be counter-clockwise"
        for p in self.pillars:
            assert _signed_area(p) < 0, "pillars must be clockwise"

    @property
    def rings(self):
        return (self.outer, *self.pillars)

    @property
    def seg_a(self):
        return np.vstack(self.rings)

    @property
    def seg_b(self):
        return np.vstack([np.roll(r, -1, axis=0) for r in self.rings])

    @property
    def bounds(self):
        lo, hi = self.outer.min(0), self.outer.max(0)
        return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])

    @property
    def has_mission(self):
        return self.exit is not None and self.entry_pose is not None


# ---------------------------------------------------------------- calibration room
CALIBRATION = Room(
    key="calibration",
    name="Calibration room (training)",
    outer=np.array([
        (0.0, 0.0), (2.6, 0.0), (2.6, 0.7), (3.8, 0.7), (3.8, 0.0), (6.4, 0.0),
        (6.4, 4.2), (4.6, 4.2), (4.6, 3.6), (3.4, 3.6), (3.4, 4.2), (0.0, 4.2),
    ]),
    starts={
        "Bottom wall, beside the pillar": (5.0, 1.05, np.pi),
        "Left wall, heading north": (0.85, 1.4, np.pi / 2),
        "Top wall, before the recess": (1.6, 3.55, 0.0),
        "Right wall, heading south": (5.6, 3.0, -np.pi / 2),
        "Room centre, no wall in range": (2.6, 2.1, 0.6),
    },
)

# -------------------------------------------------------------------- mission room
# 8.4 x 5.6 m hall.  Traversal is clockwise with the wall on the robot's left, as in
# the recording: in through the bottom corridor, west along the bottom, round the
# chamfer, north past the engaged column, round the curved corner, east through the
# bay, south down the right wall and out through the exit corridor.
_W, _H = 8.4, 5.6
_D = 2.5                                           # corridor width
_L = 6.0                                           # corridors run on beyond the view, so they read as open
_outer = [
    (1.2, 0.0),                                    # end of the 45° chamfer
    (_W + 1.75, 0.0), (_W + 1.75, _D), (_W, _D),   # entry corridor (door closes behind the robot), in line with the bottom wall
    (_W, _H - _D), (_W + _L, _H - _D), (_W + _L, _H),   # exit corridor, in line with the top wall
    (6.8, _H), (4.4, _H + 0.55),                                  # pitched (angled) top wall
    (2.0, _H), *_arc(1.2, _H - 1.2, 1.2, 90, 180, 9)[1:-1],   # curved corner, r = 1.2 m
    (0.0, _H - 1.2),
    (0.0, 3.3), (0.45, 3.3), (0.45, 2.5), (0.0, 2.5),      # engaged column on the left wall
    (0.0, 1.2),
]
MISSION = Room(
    key="mission",
    name="Mission hall (entry → exit)",
    outer=np.array(_outer, dtype=float),
    pillars=(                                     # kept >= 0.8 m sideways from the wall-following lane
        _circle_cw(3.6, 2.85, 0.3),
        _rect_cw(5.2, 2.95, 0.5, 0.5, 45.0),
        _rect_cw(6.7, 2.8, 0.8, 0.35, 0.0),
    ),
    entry=Gate((_W + 1.6, 0.0), (_W + 1.6, _D), (1.0, 0.0), "ENTRY"),
    exit=Gate((_W, _H - _D), (_W, _H), (1.0, 0.0), "EXIT"),
    entry_pose=(_W + 1.2, 0.95, np.pi),
    features=(("Chamfered wall", 1.75, 0.35), ("Curved corner", 1.05, 4.75), ("Pitched wall", 4.4, 5.85),
              ("Column", 0.95, 2.9)),
    starts={"Entry door (mission)": (_W + 1.2, 0.95, np.pi)},
    view=(-0.35, -0.35, _W + 1.75, _H + 1.05),
)

ROOMS = {r.key: r for r in (MISSION, CALIBRATION)}

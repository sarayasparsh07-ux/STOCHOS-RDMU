"""Digital twin of the SCITOS-G5 wall-following experiment.

Why a simulator is needed
-------------------------
In the dataset the action is (almost) a fixed function of the state, so for
every state only one action was ever taken.  The consequences of the other
actions, P(s' | s, a) for a != expert(s), are never observed.  A decision
method must compare actions, so we need a model that can answer "what if".

The twin is a planar unicycle robot in a polygonal room with 24 simulated
sonar beams (15 deg apart, as on the real robot).  The four simplified
distances are the minimum over 60 deg arcs, exactly as the dataset defines
them.  Data-derived parameters (speed, sensor cap, spike rate) are in
rdmu.config; the twin is validated against the real recording in
rdmu.model.validate_twin.

Everything is vectorised over N robots so that thousands of episodes can be
simulated in seconds.
"""
from __future__ import annotations

import numpy as np

from .config import TwinConfig

# Room outline, counter-clockwise, metres.  [assumption] The real room's plan is
# not published.  We use a 6.4 x 4.2 m room with one wall-mounted pillar (bottom)
# and one recess (top).  Features sit >= 1.8 m from the corners so the robot can
# complete each manoeuvre; the pillar and recess create the convex corners that
# require left turns, the four room corners require sharp right turns.
SCITOS_ROOM = np.array([
    (0.0, 0.0), (2.6, 0.0), (2.6, 0.7), (3.8, 0.7), (3.8, 0.0), (6.4, 0.0),
    (6.4, 4.2), (4.6, 4.2), (4.6, 3.6), (3.4, 3.6), (3.4, 4.2), (0.0, 4.2),
])
ROOM_BOUNDS = (6.4, 4.2)

SEG_A = SCITOS_ROOM
SEG_B = np.roll(SCITOS_ROOM, -1, axis=0)

N_BEAMS = 24
BEAM_ANGLES = np.deg2rad(np.arange(N_BEAMS) * 15.0)   # 0 = front, +90 = left (CCW)
ARC_CENTRES = {"front": 0.0, "left": 90.0, "right": -90.0, "back": 180.0}
SENSOR_NAMES = ("SD_front", "SD_left", "SD_right", "SD_back")


def arc_beams(arc_deg: float = 60.0) -> list[np.ndarray]:
    """Indices of the beams inside each 60-degree arc (front, left, right, back)."""
    ang = np.rad2deg(BEAM_ANGLES)
    out = []
    for c in ARC_CENTRES.values():
        diff = (ang - c + 180.0) % 360.0 - 180.0
        out.append(np.where(np.abs(diff) <= arc_deg / 2 + 1e-9)[0])
    return out


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def raycast(origins: np.ndarray, angles: np.ndarray, max_range: float = 5.0) -> np.ndarray:
    """Distance from each origin along each angle to the first wall.

    origins: (N, 2); angles: (N, B) absolute angles.  Returns (N, B)."""
    d = np.stack([np.cos(angles), np.sin(angles)], axis=-1)[:, :, None, :]   # N,B,1,2
    o = origins[:, None, None, :]                                               # N,1,1,2
    a = SEG_A[None, None, :, :]
    s = (SEG_B - SEG_A)[None, None, :, :]
    denom = _cross(d, s)                                                        # N,B,W
    ao = a - o
    with np.errstate(divide="ignore", invalid="ignore"):
        t = _cross(ao, s) / denom
        u = _cross(ao, d) / denom
    hit = (np.abs(denom) > 1e-12) & (t > 1e-9) & (u >= -1e-9) & (u <= 1 + 1e-9)
    t = np.where(hit, t, np.inf)
    return np.minimum(t.min(axis=-1), max_range + 10.0)


def wall_clearance(points: np.ndarray) -> np.ndarray:
    """Euclidean distance from each point (N, 2) to the nearest wall."""
    p = points[:, None, :]
    a = SEG_A[None]
    ab = (SEG_B - SEG_A)[None]
    t = np.clip(((p - a) * ab).sum(-1) / (ab * ab).sum(-1), 0.0, 1.0)
    proj = a + t[..., None] * ab
    return np.linalg.norm(p - proj, axis=-1).min(axis=1)


def inside_room(points: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon test."""
    x, y = points[:, 0:1], points[:, 1:2]
    x1, y1 = SEG_A[:, 0][None], SEG_A[:, 1][None]
    x2, y2 = SEG_B[:, 0][None], SEG_B[:, 1][None]
    cond = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = (x2 - x1) * (y - y1) / (y2 - y1) + x1
    return (cond & (x < xint)).sum(axis=1) % 2 == 1


class WallFollowTwin:
    """Vectorised simulator.  A pose is (x, y, heading)."""

    def __init__(self, cfg: TwinConfig = TwinConfig()):
        self.cfg = cfg
        self.arcs = arc_beams(cfg.arc_deg)

    # ------------------------------------------------------------------ sensing
    def beam_ranges(self, poses: np.ndarray) -> np.ndarray:
        """Noise-free range from the robot's surface along all 24 beams (N, 24)."""
        ang = poses[:, 2:3] + BEAM_ANGLES[None, :]
        out = np.empty((len(poses), N_BEAMS))
        for i in range(0, len(poses), 4000):
            out[i:i + 4000] = raycast(poses[i:i + 4000, :2], ang[i:i + 4000], self.cfg.max_range)
        return np.clip(out - self.cfg.robot_radius, 0.0, self.cfg.max_range)

    def sense(self, poses: np.ndarray, rng: np.random.Generator, return_beams: bool = False):
        """Noisy simplified distances (N, 4): front, left, right, back."""
        c = self.cfg
        beams = self.beam_ranges(poses)
        noisy = beams + rng.normal(0.0, 1.0, beams.shape) * (c.sonar_noise_abs + c.sonar_noise_rel * beams)
        sd = np.stack([noisy[:, idx].min(axis=1) for idx in self.arcs], axis=1)
        spikes = rng.random(sd.shape) < c.spike_prob
        sd = np.where(spikes, c.max_range, sd)
        sd = np.clip(sd, 0.0, c.max_range)
        return (sd, beams) if return_beams else sd

    # ----------------------------------------------------------------- dynamics
    def step(self, poses: np.ndarray, actions: np.ndarray, rng: np.random.Generator,
             alive: np.ndarray | None = None):
        """Advance every robot by one decision interval.  Returns (poses, crashed)."""
        c = self.cfg
        n = len(poses)
        actions = np.asarray(actions)
        v = c.forward_speed * np.asarray(c.speed_factor)[actions] * (1 + c.speed_noise * rng.normal(size=n))
        w = np.asarray(c.turn_rate)[actions] + c.turn_noise * rng.normal(size=n)
        v = np.maximum(v, 0.0)
        new = poses.copy()
        crashed = np.zeros(n, dtype=bool)
        h = c.dt / c.substeps
        for _ in range(c.substeps):
            mid = new[:, 2] + 0.5 * w * h
            cand = new.copy()
            cand[:, 0] += v * h * np.cos(mid)
            cand[:, 1] += v * h * np.sin(mid)
            cand[:, 2] += w * h
            hit = (wall_clearance(cand[:, :2]) < c.robot_radius + c.collision_margin) | ~inside_room(cand[:, :2])
            hit &= ~crashed
            crashed |= hit
            new = np.where(crashed[:, None], new, cand)
        if alive is not None:
            new = np.where(alive[:, None], new, poses)
            crashed &= alive
        new[:, 2] = (new[:, 2] + np.pi) % (2 * np.pi) - np.pi
        return new, crashed

    # ----------------------------------------------------------- start states
    def track_starts(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Poses on the wall-following track: 0.5-0.8 m from a wall, wall on the
        left (clockwise traversal, as in the recording), heading +-20 deg."""
        lengths = np.linalg.norm(SEG_B - SEG_A, axis=1)
        out = []
        while len(out) < n:
            k = rng.choice(len(SEG_A), size=4 * n, p=lengths / lengths.sum())
            u = rng.uniform(0.2, 0.8, size=4 * n)
            e = SEG_B[k] - SEG_A[k]
            e = e / np.linalg.norm(e, axis=1, keepdims=True)
            inward = np.stack([-e[:, 1], e[:, 0]], axis=1)       # CCW polygon: interior on the left
            off = self.cfg.robot_radius + rng.uniform(0.5, 0.8, size=4 * n)
            p = SEG_A[k] + u[:, None] * (SEG_B[k] - SEG_A[k]) + inward * off[:, None]
            head = np.arctan2(-e[:, 1], -e[:, 0]) + np.deg2rad(rng.uniform(-20, 20, size=4 * n))
            ok = inside_room(p) & (wall_clearance(p) > self.cfg.robot_radius + 0.25)
            out.extend(np.column_stack([p, head])[ok].tolist())
        return np.array(out[:n])

    def uniform_starts(self, n: int, rng: np.random.Generator, clearance: float = 0.12) -> np.ndarray:
        """Poses uniformly in free space with a uniform heading."""
        out = []
        while len(out) < n:
            p = rng.uniform([0, 0], list(ROOM_BOUNDS), size=(4 * n, 2))
            ok = inside_room(p) & (wall_clearance(p) > self.cfg.robot_radius + clearance)
            h = rng.uniform(-np.pi, np.pi, size=4 * n)
            out.extend(np.column_stack([p, h])[ok].tolist())
        return np.array(out[:n])

    def mixed_starts(self, n: int, rng: np.random.Generator, share_track: float = 0.7) -> np.ndarray:
        k = int(round(n * share_track))
        parts = []
        if k > 0:
            parts.append(self.track_starts(k, rng))
        if n - k > 0:
            parts.append(self.uniform_starts(n - k, rng, clearance=0.3))
        return np.vstack(parts)

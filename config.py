"""Central configuration for STOCHOS.

Every tunable number lives here with its provenance:

* ``[data]``        measured from the Kaggle / UCI SCITOS-G5 dataset
* ``[slide]``       required by the class exercise slide
* ``[assumption]``  a modelling choice, documented and adjustable
* ``[physical]``    a published property of the SCITOS-G5 robot

All config objects are frozen dataclasses, so they are hashable and can be
used directly as Streamlit cache keys.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
ARTIFACT_DIR = ROOT / "artifacts"

# [data] the four class labels of the dataset, in a fixed order used everywhere
ACTIONS = ("Move-Forward", "Slight-Right-Turn", "Sharp-Right-Turn", "Slight-Left-Turn")
ACTION_SHORT = ("Forward", "Slight right", "Sharp right", "Slight left")
ACTION_GLYPH = ("↑", "↗", "↻", "↖")
N_ACTIONS = len(ACTIONS)

# [data] the controller that produced the labels (extracted with a depth-3
# decision tree, see rdmu.data.extract_expert_rule)
EXPERT_THRESHOLDS = (0.900, 0.494, 0.901)  # (front, left-low, left-high) in metres, tree split points


@dataclass(frozen=True)
class StateSpaceConfig:
    # Bin edges in metres.  The expert thresholds 0.900 / 0.494 / 0.901 are
    # deliberately bin edges so the expert policy is exactly representable.
    front_edges: tuple = (0.70, 0.90, 1.10, 1.50, 2.00)        # [data] quantiles + rule
    left_edges: tuple = (0.42, 0.494, 0.60, 0.75, 0.901, 1.20)   # [data] quantiles + rule

    @property
    def n_front(self) -> int:
        return len(self.front_edges) + 1

    @property
    def n_left(self) -> int:
        return len(self.left_edges) + 1

    @property
    def n_grid(self) -> int:
        return self.n_front * self.n_left

    @property
    def n_states(self) -> int:  # grid cells + absorbing Crash state
        return self.n_grid + 1

    @property
    def crash(self) -> int:
        return self.n_grid


@dataclass(frozen=True)
class RewardConfig:
    # zone thresholds [data]: the band is exactly where the expert drives forward
    band_left: tuple = (0.494, 0.901)
    band_front_min: float = 0.90
    danger_front: float = 0.70      # [data] ~3% of samples, expert never goes lower for long
    danger_left: float = 0.42       # [data] ~4% of samples
    lost_left: float = 1.20         # [data] above the 95th pct of left while tracking
    # reward values [assumption]
    band_reward: float = 1.0
    caution_penalty: float = -0.5
    danger_penalty: float = -2.0
    lost_penalty: float = -0.5
    drift_reward: float = 0.0       # front clear, left in [0.90, 1.20)
    action_cost: tuple = (0.0, -0.05, -0.20, -0.05)  # control effort per action
    crash_penalty: float = -25.0


@dataclass(frozen=True)
class TwinConfig:
    dt: float = 1.0 / 3.0            # [assumption] one decision every 3 samples at 9 Hz [data]
    substeps: int = 4
    robot_radius: float = 0.29       # [physical] SCITOS-G5 base is ~0.58 m wide
    forward_speed: float = 0.155     # [data] median front closing speed while Move-Forward
    speed_factor: tuple = (1.0, 1.0, 0.25, 1.0)       # [calibrated] scripts/calibrate_twin.py
    turn_rate: tuple = (0.0, -0.35, -0.50, 0.35)       # [calibrated] rad/s, + = left
    speed_noise: float = 0.05        # [assumption] relative 1 s.d.
    turn_noise: float = 0.06         # [assumption] rad/s 1 s.d.
    sonar_noise_abs: float = 0.005   # [assumption] m
    sonar_noise_rel: float = 0.01    # [assumption] fraction of range
    spike_prob: float = 0.004        # [data] isolated max-range spikes, 0.2-0.8% per sensor
    max_range: float = 5.0           # [data] readings are capped at 5.0 m
    beam_step_deg: float = 15.0      # [data] 24 sensors around the waist
    arc_deg: float = 60.0            # [slide] each simplified distance is a 60 deg arc
    collision_margin: float = 0.02


@dataclass(frozen=True)
class SolverConfig:
    gamma: float = 0.95              # [slide] discount factor (input)
    tol: float = 1e-6                # [slide] convergence threshold (input)
    max_iter: int = 5000
    # transition-model estimation (Monte Carlo sampling of the twin)
    n_track_robots: int = 150
    track_steps: int = 200
    behaviour_eps: float = 0.35      # prob. of a random action in the exploration rollouts
    n_uniform_poses: int = 12000
    min_support: int = 15            # (s,a) pairs with fewer samples are flagged
    # ADP (fitted Q-iteration)
    rbf_front: int = 9
    rbf_left: int = 9
    ridge: float = 1e-3
    adp_max_iter: int = 400
    adp_tol: float = 1e-4
    # policy search
    ps_episodes: int = 24            # episodes per objective evaluation (common random numbers)
    ps_horizon: int = 150
    mc_candidates: int = 60
    mc_top: int = 5
    mc_final_episodes: int = 96
    hj_step: tuple = (0.10, 0.05, 0.10)
    hj_min_step: float = 0.0125
    hj_max_evals: int = 80        # 80 x 24 episodes = MC budget (60 x 24 + 5 x 96)
    seed: int = 7


@dataclass(frozen=True)
class EvalConfig:
    n_episodes: int = 200            # fresh seeds, never used during training/search
    horizon: int = 150               # long enough that gamma^H is negligible
    long_episodes: int = 40
    long_horizon: int = 900          # 300 s, roughly two laps
    seed: int = 2026
    start_mix_track: float = 0.7     # share of starts placed on the wall-following track


@dataclass(frozen=True)
class ProjectConfig:
    states: StateSpaceConfig = field(default_factory=StateSpaceConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    twin: TwinConfig = field(default_factory=TwinConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    def with_solver(self, **kw) -> "ProjectConfig":
        return replace(self, solver=replace(self.solver, **kw))

    def with_eval(self, **kw) -> "ProjectConfig":
        return replace(self, evaluation=replace(self.evaluation, **kw))


DEFAULT = ProjectConfig()

METHOD_KEYS = ("expert", "mdp", "adp", "mcps", "hj", "random")
METHOD_LABEL = {
    "expert": "Dataset expert rule",
    "mdp": "MDP · Value Iteration",
    "adp": "ADP · Fitted Q-Iteration",
    "mcps": "Monte Carlo Policy Search",
    "hj": "Hooke–Jeeves Policy Search",
    "random": "Random baseline",
}
METHOD_COLOR = {                 # one clearly different colour per method (robot body, trail, charts)
    "expert": "#7C3AED",         # violet
    "mdp": "#2563EB",            # blue
    "adp": "#059669",            # emerald
    "mcps": "#D97706",           # amber
    "hj": "#DB2777",             # magenta
    "random": "#64748B",         # slate
}

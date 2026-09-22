"""STOCHOS — decisions under uncertainty for a wall-following robot.

    streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from rdmu import arena, data, legacy_audit, plots
from rdmu.config import (ACTION_GLYPH, ACTION_SHORT, ACTIONS, DATA_DIR, DEFAULT, EXPERT_THRESHOLDS, METHOD_COLOR,
                         METHOD_KEYS, METHOD_LABEL)
from rdmu.evaluation import run_batch, summarise
from rdmu.mission import OUTCOME_LABEL, mission_batch, record_mission, summarise_missions
from rdmu.pipeline import build_policies, load_or_run
from rdmu.policies import ThresholdPolicy
from rdmu.policy_search import BOUNDS, feasible
from rdmu.statespace import (ZONES, ZONE_COLOR, representative_readings, state_label, state_zone_table,
                             threshold_actions)
from rdmu.rooms import MISSION, ROOMS
from rdmu.twin import WallFollowTwin

st.set_page_config(page_title="STOCHOS · RDMU robot", page_icon="🛰️", layout="wide",
                   initial_sidebar_state="expanded")

# =========================================================================== style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root { --ink:#1B2433; --muted:#5B6476; --line:#E3E7EE; --soft:#F5F7FA; --teal:#0F8B8D; --indigo:#3552C7; }
html, body, [class*="css"], .stMarkdown, button, input, select, textarea {
  font-family: 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif; }
code, pre, .mono { font-family: 'IBM Plex Mono', ui-monospace, monospace; }
.block-container { padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1400px; }
#MainMenu, footer { visibility: hidden; }
h1, h2, h3, h4 { color: var(--ink); letter-spacing: -0.01em; }

/* hero */
.hero { background: linear-gradient(120deg, #13233F 0%, #1B3A5C 55%, #0F6E73 100%); border-radius: 16px;
        padding: 1.3rem 1.6rem 1.2rem; color: #F2F6FA; margin-bottom: 0.9rem; }
.hero .eyebrow { font-size: 0.74rem; letter-spacing: 0.16em; text-transform: uppercase; color: #9FD6D2; font-weight: 600; }
.hero h1 { color: #FFFFFF; font-size: 2.1rem; margin: 0.15rem 0 0.25rem; font-weight: 700; letter-spacing: 0.04em; }
.hero p { color: #D5DFEA; margin: 0; max-width: 78ch; font-size: 0.98rem; line-height: 1.5; }
.chips { margin-top: 0.75rem; display: flex; flex-wrap: wrap; gap: 0.4rem; }
.chip { display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.18rem 0.65rem; border-radius: 999px;
        background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.18); font-size: 0.8rem; color: #EEF3F8; }
.dot { width: 0.55rem; height: 0.55rem; border-radius: 50%; display: inline-block; }

/* section headers */
.sec { margin: 0.4rem 0 0.6rem; }
.sec h2 { font-size: 1.35rem; margin: 0; font-weight: 600; }
.sec p { color: var(--muted); margin: 0.2rem 0 0; font-size: 0.95rem; max-width: 95ch; }

/* KPI grid */
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr)); gap: 0.65rem; margin: 0.2rem 0 0.9rem; }
.kpi { background: #FFFFFF; border: 1px solid var(--line); border-radius: 12px; padding: 0.7rem 0.9rem;
       border-top: 3px solid var(--accent, var(--teal)); box-shadow: 0 1px 2px rgba(27,36,51,0.04); }
.kpi .l { font-size: 0.74rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }
.kpi .v { font-size: 1.45rem; font-weight: 600; color: var(--ink); margin-top: 0.1rem; line-height: 1.2; }
.kpi .s { font-size: 0.8rem; color: var(--muted); margin-top: 0.1rem; }

/* cards & decision panel */
.card { border: 1px solid var(--line); border-radius: 12px; padding: 0.85rem 1rem; background: #FFFFFF; }
.method { border-left: 4px solid var(--accent); background: var(--soft); border-radius: 10px; padding: 0.7rem 1rem;
          margin-bottom: 0.8rem; color: var(--ink); font-size: 0.95rem; line-height: 1.5; }
.readings, .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.35rem; margin: 0.2rem 0 0.5rem; }
.stats { grid-template-columns: repeat(3, 1fr); margin: 0.5rem 0 0.3rem; }
.readings div, .stats div { background: var(--soft); border-radius: 8px; padding: 0.3rem 0.45rem; }
.readings span, .stats span, .decision .row span { display: block; font-size: 0.7rem; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }
.readings b, .stats b { font-family: 'IBM Plex Mono', monospace; font-size: 0.9rem; color: var(--ink); font-weight: 500; }
.decision .row { margin-top: 0.35rem; }
.stats span, .decision .row span { text-transform: none; letter-spacing: 0; font-size: 0.76rem; }
.decision .row b { font-family: 'IBM Plex Mono', monospace; font-size: 0.86rem; font-weight: 500; color: var(--ink); }
.decision .act { font-size: 1.3rem; font-weight: 600; margin: 0.25rem 0 0.15rem; }
.kv { font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; color: #394256; line-height: 1.55; }
.why { color: var(--muted); font-size: 0.9rem; }
.note { color: var(--muted); font-size: 0.9rem; line-height: 1.5; }
.mcard { height: 100%; }
.mcard .t { font-weight: 600; font-size: 0.95rem; }
.trait { margin-top: 0.45rem; font-size: 0.87rem; color: var(--ink); line-height: 1.4; }
.trait .l { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }
.pill { display: inline-block; padding: 0.05rem 0.55rem; border-radius: 999px; font-size: 0.76rem; color: #FFFFFF; font-weight: 500; }

/* widgets */
div[data-testid="stSegmentedControl"] button { font-weight: 500; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px; }
section[data-testid="stSidebar"] { background: #F4F6F9; }
section[data-testid="stSidebar"] .brand { font-weight: 700; letter-spacing: 0.08em; font-size: 1.15rem; color: var(--ink); }
section[data-testid="stSidebar"] .brand span { color: var(--teal); }
section[data-testid="stSidebar"] h3 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em;
                                      color: var(--muted); margin: 1rem 0 0.3rem; }
.stTabs [data-baseweb="tab-list"] { gap: 0.3rem; }
.stTabs [data-baseweb="tab"] { padding: 0.4rem 0.9rem; border-radius: 8px 8px 0 0; }
</style>
""", unsafe_allow_html=True)

LABELS = {**METHOD_LABEL, "custom": "Custom thresholds (manual)"}
COLORS = {**METHOD_COLOR, "custom": "#0891B2"}
REQUIRED_FILES = ("sensor_readings_2.csv", "sensor_readings_4.csv", "sensor_readings_24.csv", "Day_5.ipynb")
PAGES = ["Live robot", "Compare methods", "Data", "MDP", "ADP", "Policy search", "Notebook audit", "About"]

METHOD_EXPLAIN = {
    "mdp": ("Finds the robot's (front, left) cell and takes the action with the highest Q*(s,a) from value iteration. "
            "Q* already includes the expected discounted future reward under the estimated transition probabilities."),
    "adp": ("Evaluates Q̂(x,a) = φ(x)ᵀθ_a on the raw, undiscretised readings and takes the largest. θ was fitted by "
            "repeated regression on sampled Bellman targets, so its decision boundaries are smooth, not tied to bins."),
    "mcps": ("Runs the recorded controller's rule with thresholds found by Monte Carlo random search: 60 candidate "
             "threshold sets were simulated on the same episodes and the best, re-checked on new seeds, was kept."),
    "hj": ("Runs the recorded controller's rule with thresholds tuned by Hooke–Jeeves pattern search, which started at "
           "the recorded values and moved one threshold at a time while the simulated return improved."),
    "expert": ("The controller that produced the dataset, recovered exactly from its labels: sharp right if front ≤ 0.900 m; "
               "otherwise slight right if left ≤ 0.494 m, forward if left ≤ 0.901 m, else slight left."),
    "random": "Chooses one of the four actions uniformly at random. It is the lower bound for every other method.",
    "custom": "The recorded controller's rule with thresholds you set in the sidebar, for exploring the objective by hand.",
}

RANDOM_START = "Random start (uses the seed)"
SPEED_MS = {"0.5×": 660, "1×": 330, "2×": 165, "4×": 80, "8×": 40}   # per decision
MISSION_HORIZON = 750


# =========================================================================== UI helpers
def chart(fig, key=None):
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False}, key=key)


def section(title, lede=None):
    st.markdown(f"<div class='sec'><h2>{title}</h2>{f'<p>{lede}</p>' if lede else ''}</div>", unsafe_allow_html=True)


def kpis(items):
    """items: (label, value, sub, colour) tuples, rendered as a responsive card grid."""
    cells = "".join(
        f"<div class='kpi' style='--accent:{c or '#0F8B8D'}'><div class='l'>{l}</div><div class='v'>{v}</div>"
        f"{f'<div class=s>{s}</div>' if s else ''}</div>" for l, v, s, c in items)
    st.markdown(f"<div class='kpis'>{cells}</div>", unsafe_allow_html=True)


def note(text):
    st.markdown(f"<div class='note'>{text}</div>", unsafe_allow_html=True)


def table(df, height=None):
    kw = {"height": height} if height else {}
    st.dataframe(df, width="stretch", hide_index=True, **kw)


# =========================================================================== cached computations
@st.cache_resource(show_spinner=False)
def get_results(cfg):
    return load_or_run(cfg)


@st.cache_resource(show_spinner=False)
def get_data_analysis():
    return {
        "df": data.load4(), "quality": data.quality_report(), "describe": data.describe(),
        "classes": data.class_distribution(), "ts": data.time_series_evidence(),
        "relevance": data.feature_relevance(), "rule": data.extract_expert_rule(),
        "arcs": data.arc_membership(), "emp": data.empirical_expert_transitions(DEFAULT.states),
    }


@st.cache_resource(show_spinner=False)
def get_audit():
    return legacy_audit.run_all()


def make_policy(res, cfg, method, custom_thr=None):
    if method == "custom":
        return ThresholdPolicy(custom_thr, name="custom")
    return build_policies(res, cfg)[method]


@st.cache_data(show_spinner=False)
def get_episode(_res, cfg, method, room_key, start, seed, horizon, custom_thr=None):
    """start None = the room's entry door (mission).  Every run has the loop watchdog."""
    return record_mission(make_policy(_res, cfg, method, custom_thr), cfg, ROOMS[room_key], horizon, seed,
                          _res["model"], None if start is None else np.array(start))


@st.cache_data(show_spinner=False)
def mission_stats(_res, cfg, method, n, seed, custom_thr=None):
    m = mission_batch(make_policy(_res, cfg, method, custom_thr), cfg, MISSION, n, MISSION_HORIZON, seed)
    return summarise_missions(m), m["step"][m["outcome"] == "exit"] / 3


@st.cache_data(show_spinner=False)
def batch_run(_res, cfg, method, n, horizon, seed, custom_thr=None):
    m = run_batch(make_policy(_res, cfg, method, custom_thr), cfg, n, horizon, seed)
    return summarise(m), m["return"], m["crashed"]


def start_pose(name, seed, cfg, room):
    """None means: start at the entry door (with seeded jitter)."""
    if room.has_mission:
        return None
    if name not in room.starts:
        return tuple(WallFollowTwin(cfg.twin, room).mixed_starts(1, np.random.default_rng(seed), 0.6)[0])
    return tuple(room.starts[name])


# =========================================================================== sidebar
with st.sidebar:
    st.markdown("<div class='brand'>STOCH<span>OS</span></div>", unsafe_allow_html=True)
    st.caption("RDMU · wall-following robot")
    st.markdown("### Decision method")
    method = st.radio("Controller driving the robot", list(METHOD_KEYS) + ["custom"], format_func=lambda k: LABELS[k],
                      index=1, label_visibility="collapsed")
    custom_thr = None
    if method == "custom":
        st.markdown("**Policy parameters** θ = (τ_front, τ_low, τ_high)")
        c_tf = st.slider("τ_front (m)", float(BOUNDS[0, 0]), float(BOUNDS[0, 1]), EXPERT_THRESHOLDS[0], 0.005)
        c_lo = st.slider("τ_left-low (m)", float(BOUNDS[1, 0]), float(BOUNDS[1, 1]), EXPERT_THRESHOLDS[1], 0.005)
        c_hi = st.slider("τ_left-high (m)", float(BOUNDS[2, 0]), float(BOUNDS[2, 1]), EXPERT_THRESHOLDS[2], 0.005)
        custom_thr = (round(c_tf, 3), round(c_lo, 3), round(c_hi, 3))
        if not feasible(custom_thr):
            st.error("τ_left-high must exceed τ_left-low by at least 0.05 m. Using the recorded thresholds instead.")
            custom_thr = EXPERT_THRESHOLDS

    st.markdown("### Robot run")
    room_key = st.selectbox("Room", list(ROOMS), format_func=lambda k: ROOMS[k].name)
    room = ROOMS[room_key]
    view = st.segmented_control("View", ["Single robot", "Race: all methods"], default="Single robot",
                                key="view") or "Single robot"
    start_name = st.selectbox("Start position", list(room.starts) + ([] if room.has_mission else [RANDOM_START]))
    c1, c2 = st.columns(2)
    seed = int(c1.number_input("Noise seed", 0, 10_000, 3, step=1))
    speed = c2.selectbox("Playback", ["0.5×", "1×", "2×", "4×", "8×"], index=2)
    horizon = st.slider("Time limit (decisions, 3 per second)", 60, 900, MISSION_HORIZON, step=30)
    n_batch = st.slider("Number of episodes (batch run)", 10, 200, 50, 10)

    with st.expander("Model settings"):
        with st.form("model_form", border=False):
            gamma = st.slider("Discount factor γ", 0.80, 0.99, DEFAULT.solver.gamma, 0.01)
            tol = st.select_slider("Convergence threshold", [1e-3, 1e-4, 1e-5, 1e-6, 1e-8], value=DEFAULT.solver.tol,
                                   format_func=lambda x: f"{x:g}")
            ps_eps = st.slider("Episodes per policy-search evaluation", 8, 64, DEFAULT.solver.ps_episodes, 8)
            ev_eps = st.slider("Evaluation episodes per method", 50, 400, DEFAULT.evaluation.n_episodes, 50)
            msd = int(st.number_input("Training seed", 0, 10_000, DEFAULT.solver.seed))
            applied = st.form_submit_button("Apply and recompute (≈40 s)", width="stretch")
        if applied:
            st.session_state.cfg = DEFAULT.with_solver(gamma=float(gamma), tol=float(tol), ps_episodes=int(ps_eps),
                                                       seed=msd).with_eval(n_episodes=int(ev_eps))
        if st.session_state.get("cfg", DEFAULT) != DEFAULT:
            st.caption("Custom settings active.")
            if st.button("Restore defaults", width="stretch"):
                st.session_state.cfg = DEFAULT
                st.rerun()

    with st.expander("Goal and environment"):
        st.caption("**Goal:** a region, not a single cell. Keep the wall on the left at 0.494–0.901 m with more than "
                   "0.900 m clear ahead (the tracking band), never collide, keep moving.")
        st.caption("**Rooms:** the mission hall (8.4 × 5.6 m: chamfered, curved and pitched walls, a wall column, "
                   "three pillars, entry and exit corridors) and the 6.4 × 4.2 m calibration room the methods were "
                   "trained in. 24 sonar beams in 60° arcs; one decision every 1/3 s; sensor and motion noise.")
        st.caption("**Runs end** at the exit, on a collision, at the time limit, or when the loop watchdog sees the "
                   "robot return to a spot it passed 20 s or more earlier.")

cfg = st.session_state.get("cfg", DEFAULT)

# =========================================================================== load
missing = [f for f in REQUIRED_FILES if not (DATA_DIR / f).exists()]
if missing:
    st.error(f"Missing input file(s) in `data/`: {', '.join(missing)}. Restore them from the Kaggle archive and reload.")
    st.stop()
try:
    with st.spinner("Loading data and results…"):
        D = get_data_analysis()
except (ValueError, KeyError) as exc:
    st.error(f"The dataset could not be read as expected ({exc}). Check that the CSV files are unmodified.")
    st.stop()
try:
    with st.spinner("Solving the MDP, fitting ADP and running both policy searches (≈40 s the first time)…"):
        res = get_results(cfg)
except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
    st.error(f"The computation failed for these settings ({exc}). Try a smaller discount factor or restore defaults.")
    st.stop()

stc = cfg.states
color = COLORS[method]
S_all, L_all = res["eval_short"], res["eval_long"]

# =========================================================================== hero + nav
chips = "".join(f"<span class='chip'><span class='dot' style='background:{METHOD_COLOR[k]}'></span>{METHOD_LABEL[k]}</span>"
                for k in ("mdp", "adp", "mcps", "hj"))
st.markdown(f"""
<div class="hero">
  <div class="eyebrow">Reasoning &amp; decision making under uncertainty</div>
  <h1>STOCHOS</h1>
  <p>A sonar robot learns to follow walls. Four decision methods are trained on a simulator calibrated to the
  SCITOS-G5 recording and compared with the robot's original controller under identical conditions.</p>
  <div class="chips">{chips}</div>
</div>""", unsafe_allow_html=True)

page = st.segmented_control("Section", PAGES, default="Live robot", key="page", label_visibility="collapsed")
page = page or "Live robot"


def method_kpis():
    if method == "custom":
        ev = cfg.evaluation
        es = batch_run(res, cfg, "custom", ev.n_episodes, ev.horizon, ev.seed, custom_thr)[0]
        el = batch_run(res, cfg, "custom", ev.long_episodes, ev.long_horizon, ev.seed + 500, custom_thr)[0]
        diff = es["return_mean"] - S_all["expert"]["return_mean"]
        train_s, iters = "—", "—"
    else:
        es, el = S_all[method], L_all[method]
        diff = es["diff_vs_expert"]
        train_s = f"{res['runtime'][method]['seconds']:.1f} s"
        iters = {"mdp": f"{res['vi'].iterations} sweeps", "adp": f"{res['adp'].iterations} fits",
                 "mcps": f"{len(res['mc'].candidates)} candidates", "hj": f"{res['hj'].evaluations} evaluations"}.get(method, "—")
    kpis([
        ("Discounted return", f"{es['return_mean']:.2f}", "—" if method == "expert" else f"{diff:+.2f} vs recorded controller", color),
        ("Success rate", f"{1 - el['crash_rate']:.0%}", "300 s runs without a collision", color),
        ("Time in tracking band", f"{el['band_share']:.0%}", "share of decisions in the goal region", color),
        ("Reward per step", f"{el['reward_per_step']:.3f}", "long-run average", color),
        ("Training", train_s, iters, color),
    ])


# =========================================================================== pages
def race_view():
    section("Race: every method in the same room",
            ("All six controllers enter through the same door one after another, each in its own colour. They are "
             "simulated independently (they do not see each other) with the same seed, so any difference comes "
             "from the decisions.") if room.has_mission else
            "All six controllers start from the same pose, one after another, each in its own colour.")
    start = start_pose(start_name, seed, cfg, room)
    eps = {k: get_episode(res, cfg, k, room_key, start, seed, horizon) for k in METHOD_KEYS}
    with st.container(border=True):
        chart(arena.race_animation(eps, room, cfg.twin.robot_radius, COLORS, stagger=15, stride=3,
                                   frame_ms=SPEED_MS[speed] * 3, height=640), key="race")
        note("▶ Play / ⏸ Pause animate the race; drag the slider to scrub. Badges: ✓ reached the exit · "
             "⟲ loop detected and stopped · ✕ collision · ⏱ out of time.")
    kpis([(LABELS[k], OUTCOME_LABEL[eps[k]["outcome"]],
           f"{len(eps[k]['action']) / 3:.0f} s · Σ reward {eps[k]['reward'].sum():+.1f}", COLORS[k]) for k in METHOD_KEYS])
    chart(arena.robot_closeup({k: COLORS[k] for k in METHOD_KEYS}, {k: arena.SHORT[k] for k in METHOD_KEYS}, height=170))
    note("Each controller drives an autonomous robot car of the same design — chassis, canopy, sensor pod, "
         "headlights and four wheels — in its own colour, so the fleet reads as one system. The <b>Random</b> car is "
         "drawn with a dashed canopy and a cloud of sample points to mark it as the probabilistic agent that picks its "
         "action at random. The drawings are top-down views of the twin's robot; they do not change the model.")


def page_robot():
    if view.startswith("Race"):
        return race_view()
    section(f"<span class='pill' style='background:{color}'>{LABELS[method]}</span> driving the robot",
            "Play the run, then step through it: every move below comes from the selected method's policy.")
    method_kpis()
    st.markdown(f"<div class='method' style='--accent:{color}'>{METHOD_EXPLAIN[method]}</div>", unsafe_allow_html=True)

    ep = get_episode(res, cfg, method, room_key, start_pose(start_name, seed, cfg, room), seed, horizon, custom_thr)
    n = len(ep["action"])
    stride = 1 if n <= 300 else 2 if n <= 600 else 3
    left, right = st.columns([1.7, 1], gap="medium")
    with left:
        with st.container(border=True):
            chart(arena.mission_animation(ep, room, cfg.twin.robot_radius, color, SPEED_MS[speed] * stride, stride, 600, kind="random" if method == "random" else "car"))
            legend = " ".join(f"<span class='pill' style='background:{ZONE_COLOR[i]}'>{ZONES[i]}</span>" for i in (0, 3, 4))
            note(f"▶ Play / ⏸ Pause animate the run; drag the slider to scrub. Red rays: front arc · teal rays: left arc. "
                 f"Path dots by zone: {legend}")
    with right:
        with st.container(border=True):
            st.markdown("**Decision inspector**")
            key = f"step_{method}_{room_key}_{start_name}_{seed}_{horizon}"
            if key not in st.session_state:
                st.session_state[key] = 0

            def _move(delta):
                st.session_state[key] = int(np.clip(st.session_state[key] + delta, 0, n - 1))

            def _reset():
                st.session_state[key] = 0
            b1, b2, b3 = st.columns(3)
            b1.button("↺ Reset", on_click=_reset, width="stretch")
            b2.button("◀ Back", on_click=_move, args=(-1,), width="stretch")
            b3.button("⏭ Step", on_click=_move, args=(1,), width="stretch")
            st.session_state[key] = min(st.session_state[key], n - 1)
            t = st.slider("Step", 0, n - 1, key=key)
            a, s, s2 = int(ep["action"][t]), int(ep["state"][t]), int(ep["next_state"][t])
            sd = ep["sd"][t]
            cells = "".join(f"<div><span>{lab}</span><b>{val:.2f} m</b></div>"
                            for lab, val in zip(("Front", "Left", "Right", "Back"), sd))
            st.markdown(f"""
<div class="decision">
  <div class="readings">{cells}</div>
  <div class="row"><span>Current state s · t = {t / 3:.1f} s</span><b>{state_label(s, stc)}</b></div>
  <div class="act" style="color:{plots.ACTION_COLOR[a]}">{ACTION_GLYPH[a]} {ACTIONS[a]}</div>
  <div class="why">{ep['explain'][t]}</div>
  <div class="row"><span>Next state s′</span><b>{state_label(s2, stc)}</b></div>
  <div class="stats"><div><span>P(s′ | s, a)</span><b>{ep['p_transition'][t]:.3f}</b></div>
  <div><span>Reward r</span><b>{ep['reward'][t]:+.2f}</b></div><div><span>Σ reward</span><b>{ep['reward'][:t + 1].sum():+.1f}</b></div></div>
</div>""", unsafe_allow_html=True)
            pol = make_policy(res, cfg, method, custom_thr)
            chart(plots.scores_bar(ep["scores"][t], a, pol.score_label, 200))
            P_row = res["model"].P[s]
            top = np.argsort(-P_row[a])[:3]
            note("<b>Uncertainty:</b> the same state and action can lead to different states. Most likely: " +
                 " · ".join(f"{state_label(j, stc)} ({P_row[a, j]:.2f})" for j in top))

    kpis([("Cumulative reward", f"{ep['reward'].sum():.1f}", "this run", color),
          ("Steps", f"{n}", f"{n / 3:.0f} s of driving", color),
          ("Final state", state_label(int(ep["next_state"][-1]), stc), None, color),
          ("Outcome", OUTCOME_LABEL[ep["outcome"]], None, color)])

    t1, t2, t3 = st.tabs(["Sensors & reward over time", "Decision log", f"Batch run ({n_batch} episodes)"])
    with t1:
        chart(plots.episode_timeline(ep, color))
    with t2:
        logdf = pd.DataFrame({
            "Step": np.arange(n), "Time (s)": np.round(np.arange(n) / 3, 2),
            "Current state": [state_label(x, stc) for x in ep["state"]],
            "Action": [ACTIONS[x] for x in ep["action"]],
            "Transition probability": ep["p_transition"].round(3),
            "Next state": [state_label(x, stc) for x in ep["next_state"]],
            "Reward": ep["reward"].round(2), "Cumulative reward": np.cumsum(ep["reward"]).round(2),
            "SD_front": ep["sd"][:, 0].round(3), "SD_left": ep["sd"][:, 1].round(3), "Why": ep["explain"],
        })
        table(logdf, 360)
        st.download_button("Download decision log (CSV)", logdf.to_csv(index=False), f"decision_log_{method}.csv", "text/csv")
    with t3:
        if room.has_mission:
            ms, times = mission_stats(res, cfg, method, n_batch, seed * 97 + 5, custom_thr)
            kpis([("Reached the exit", f"{ms['exit_rate']:.0%}", f"{n_batch} runs from the entry", color),
                  ("Loops stopped", f"{ms['loop_rate']:.0%}", "watchdog", color),
                  ("Collisions", f"{ms['crash_rate']:.0%}", None, color),
                  ("Median time to exit", "—" if np.isnan(ms['median_time_s']) else f"{ms['median_time_s']:.0f} s",
                   None, color)])
            if len(times):
                hf = go.Figure(go.Histogram(x=times, marker_color=color, nbinsx=25))
                plots._layout(hf, f"Time from entry to exit ({len(times)} successful runs)", 280)
                hf.update_xaxes(title="seconds")
                hf.update_yaxes(title="runs")
                chart(hf)
            return
        bs, br, _ = batch_run(res, cfg, method, n_batch, horizon, seed * 97 + 5, custom_thr)
        kpis([("Mean discounted return", f"{bs['return_mean']:.2f}", f"± {bs['return_ci95']:.2f} (95% CI)", color),
              ("Success rate", f"{1 - bs['crash_rate']:.0%}", None, color),
              ("Time in band", f"{bs['band_share']:.0%}", None, color),
              ("Reward per step", f"{bs['reward_per_step']:.3f}", None, color)])
        hf = go.Figure(go.Histogram(x=br, marker_color=color, nbinsx=25))
        plots._layout(hf, f"Discounted return per episode ({n_batch} episodes × {horizon} steps, seeded)", 280)
        hf.update_xaxes(title="discounted return")
        hf.update_yaxes(title="episodes")
        chart(hf)


def page_compare():
    ev = cfg.evaluation
    section("Compare methods",
            f"All controllers run in the same simulator from the same {ev.n_episodes} start poses with the same noise and "
            f"reward, on seeds never used for training. Long-run metrics come from {ev.long_episodes} runs of "
            f"{ev.long_horizon / 3:.0f} s. Values are simulation estimates with 95% intervals.")
    learned = ("mdp", "adp", "mcps", "hj")
    kb = max(learned, key=lambda k: S_all[k]["return_mean"])
    kpis([("Highest mean return", f"{S_all[kb]['return_mean']:.2f}", LABELS[kb], METHOD_COLOR[kb]),
          ("Recorded controller", f"{S_all['expert']['return_mean']:.2f}", "baseline", METHOD_COLOR["expert"]),
          ("Success, learned methods", f"{min(1 - L_all[k]['crash_rate'] for k in learned):.0%}", "300 s runs", "#0F8B8D"),
          ("Fastest training", f"{min(res['runtime'][k]['seconds'] for k in learned):.1f} s", None, "#0F8B8D"),
          ("VI sweeps / ADP fits", f"{res['vi'].iterations} / {res['adp'].iterations}", None, "#0F8B8D")])

    t1, t2, t3, t5, t4 = st.tabs(["Scoreboard", "Performance charts", "Behaviour & policies", "Mission hall",
                                  "Model check"])
    with t1:
        conv = {"mdp": ("Yes" if res["vi"].converged else "No") + f" (‖ΔV‖∞ < {cfg.solver.tol:g})",
                "adp": ("Yes" if res["adp"].converged else "No") + f" (max|ΔQ̂| < {cfg.solver.adp_tol:g})",
                "mcps": "Fixed budget completed", "hj": f"Yes ({res['hj'].stop_reason})", "expert": "—", "random": "—"}
        iters = {"mdp": f"{res['vi'].iterations} sweeps", "adp": f"{res['adp'].iterations} fits",
                 "mcps": f"{len(res['mc'].candidates)} + {cfg.solver.mc_top}", "hj": f"{res['hj'].evaluations}",
                 "expert": "—", "random": "—"}
        objective = {"mdp": "max V(s) on the estimated model", "adp": "max Q̂(x,a) on sampled targets",
                     "mcps": "max Ĵ(θ), random candidates", "hj": "max Ĵ(θ), pattern search",
                     "expert": "none (recorded rule)", "random": "none"}
        approx = {"mdp": "Exact on a discretised model", "adp": "324-parameter RBF value function",
                  "mcps": "Sampled objective, 3 parameters", "hj": "Sampled objective, 3 parameters", "expert": "—", "random": "—"}
        st.markdown("##### Performance (fresh seeds)")
        table(pd.DataFrame([{
            "Algorithm": LABELS[k],
            "Return (±95% CI)": f"{S_all[k]['return_mean']:.2f} ± {S_all[k]['return_ci95']:.2f}",
            "Δ vs recorded (paired)": "—" if k == "expert" else f"{S_all[k]['diff_vs_expert']:+.2f} ± {S_all[k]['diff_ci95']:.2f}",
            "Success (300 s)": f"{1 - L_all[k]['crash_rate']:.0%}", "In band": f"{L_all[k]['band_share']:.0%}",
            "Reward / step": f"{L_all[k]['reward_per_step']:.3f}", "Distance (m)": f"{L_all[k]['distance_m']:.1f}"}
            for k in METHOD_KEYS]))
        st.markdown("##### Computation")
        table(pd.DataFrame([{
            "Algorithm": LABELS[k], "Objective": objective[k], "Runtime": f"{res['runtime'][k]['seconds']:.1f} s",
            "Iterations": iters[k], "Converged?": conv[k], "Approximation?": approx[k]} for k in METHOD_KEYS]))
        st.markdown("##### Method characteristics")
        traits = {
            "mdp": [("Core idea", "Bellman optimality on estimated P and R (43 states)"),
                    ("Exact?", "Exact for the model; the model aggregates states"),
                    ("Convergence", f"Guaranteed · bound {res['vi'].error_bound:.1e}"),
                    ("Scalability", "Poor: grows as bins^sensors"), ("Interpretability", "High: one action per cell")],
            "adp": [("Core idea", "Fit Q̂(x,a) = φ(x)ᵀθ to sampled Bellman targets"),
                    ("Exact?", "Approximate, continuous input"),
                    ("Convergence", "Observed here; not guaranteed in general"),
                    ("Scalability", "Good: grows with features"), ("Interpretability", "Medium: smooth regions")],
            "mcps": [("Core idea", "Score random θ by simulated return; re-check the best"),
                     ("Exact?", "Sampled objective, 3 parameters"), ("Convergence", "Improves with more candidates"),
                     ("Scalability", "Parallel; weak in many dimensions"), ("Interpretability", "High: three thresholds")],
            "hj": [("Core idea", "Coordinate probes + pattern moves, step halving"),
                   ("Exact?", "Sampled objective (fixed seeds)"), ("Convergence", "To a local optimum"),
                   ("Scalability", "Evaluations grow linearly with parameters"), ("Interpretability", "High: three thresholds")],
        }
        cols = st.columns(4, gap="small")
        for col, k in zip(cols, learned):
            rows = "".join(f"<div class='trait'><div class='l'>{name}</div><div>{text}</div></div>" for name, text in traits[k])
            col.markdown(f"<div class='kpi mcard' style='--accent:{METHOD_COLOR[k]}'>"
                         f"<div class='t' style='color:{METHOD_COLOR[k]}'>{LABELS[k]}</div>{rows}"
                         f"<div class='trait'><b>Mean return {S_all[k]['return_mean']:.2f}</b></div></div>",
                         unsafe_allow_html=True)
        sig = [LABELS[k] for k in learned if S_all[k]["diff_vs_expert"] - S_all[k]["diff_ci95"] > 0]
        note(f"The paired difference against the recorded controller is above zero with 95% confidence for "
             f"{len(sig)} of 4 learned methods. Where two methods' intervals overlap, the data do not separate them.")
    with t2:
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            chart(plots.paired_diff(S_all))
        with c2:
            chart(plots.metric_bars(L_all, "band_share", "Time in the tracking band (300 s runs)", pct=True))
        c3, c4 = st.columns(2, gap="medium")
        with c3:
            chart(plots.return_distribution(res["returns"]))
        with c4:
            chart(plots.metric_bars(L_all, "reward_per_step", "Average reward per step (300 s runs)"))
        c5, c6 = st.columns(2, gap="medium")
        with c5:
            chart(plots.metric_bars(L_all, "crash_rate", "Crash rate (300 s runs)", pct=True))
        with c6:
            chart(plots.metric_bars({k: {"s": res["runtime"][k]["seconds"]} for k in METHOD_KEYS}, "s", "Computation time (s)"))
        chart(plots.convergence_overview(res))
    with t3:
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            chart(plots.action_mix(L_all))
        with c2:
            chart(plots.metric_bars(L_all, "distance_m", "Distance travelled in 300 s (m)"))
        st.markdown("##### Policy maps on the MDP grid")
        F_, L_ = representative_readings(stc)
        maps = {"expert": res["pi_expert"][:-1], "mdp": res["vi"].pi[:-1], "adp": res["adp_cell_policy"],
                "mcps": threshold_actions(F_, L_, res["mc"].best_theta), "hj": threshold_actions(F_, L_, res["hj"].best_theta)}
        chart(plots.policy_maps(maps, stc))
        note("Threshold policies are shown at each cell's midpoint; ADP shows the majority action per cell.")
        st.markdown(f"##### Same start, six controllers · {ROOMS[room_key].name} · {start_name}")
        start = start_pose(start_name, seed, cfg, room)
        eps = {k: get_episode(res, cfg, k, room_key, start, seed, horizon) for k in METHOD_KEYS}
        chart(arena.paths_grid(eps, room, COLORS, LABELS))
    with t5:
        n_m = 100
        with st.spinner("Running the mission-hall test (≈15 s the first time)…"):
            stats = {k: mission_stats(res, cfg, k, n_m, ev.seed + 900)[0] for k in METHOD_KEYS}
        note(f"{n_m} runs per method from the entry door of the mission hall ({MISSION_HORIZON / 3:.0f} s limit), on "
             "fresh seeds. The policies were trained in the calibration room, so this is a test of how well each one "
             "carries over to a room it has never seen.")
        chart(arena.outcome_bars(stats, LABELS))
        table(pd.DataFrame([{
            "Algorithm": LABELS[k], "Reached exit": f"{stats[k]['exit_rate']:.0%}",
            "Loops stopped": f"{stats[k]['loop_rate']:.0%}", "Collisions": f"{stats[k]['crash_rate']:.0%}",
            "Median time to exit": "—" if np.isnan(stats[k]["median_time_s"]) else f"{stats[k]['median_time_s']:.0f} s",
            "Median path length": "—" if np.isnan(stats[k]["median_distance_m"]) else f"{stats[k]['median_distance_m']:.1f} m"}
            for k in METHOD_KEYS]))
    with t4:
        mg = res["model_gap"]
        kpis([("MDP: model prediction", f"{mg['mdp']['predicted']:.2f}", "E[V*(s₀)]", METHOD_COLOR["mdp"]),
              ("MDP: simulated", f"{mg['mdp']['simulated']:.2f}", "mean return", METHOD_COLOR["mdp"]),
              ("Recorded: model prediction", f"{mg['expert']['predicted']:.2f}", "E[V^expert(s₀)]", METHOD_COLOR["expert"]),
              ("Recorded: simulated", f"{mg['expert']['simulated']:.2f}", "mean return", METHOD_COLOR["expert"])])
        note("The gap is the price of state aggregation: two readings in the same cell can hide different headings, so "
             "the Markov assumption holds only approximately. Methods acting on continuous readings avoid this, which "
             "is consistent with their higher measured return.")
        with st.expander("Observed trade-offs"):
            st.markdown("""
* **Hooke–Jeeves** changes only three numbers in the original rule, so the result stays interpretable; it needs a
  noise-free objective (fixed seeds) and finds a local optimum.
* **Monte Carlo search** needs no smoothness and parallelises trivially, but spends most of its budget on poor candidates.
* **ADP** handles continuous inputs and scales to more sensors; fitted iteration is not guaranteed to converge for
  every feature set, and its regions have no compact rule.
* **Value iteration** is exact for the model, converges in milliseconds with an error bound, and exposes the problem's
  structure, but inherits the discretisation's limits.""")


def page_data():
    q, df = D["quality"], D["df"]
    section("Data", "Kaggle / UCI SCITOS-G5 wall-following recording: the robot circles a room clockwise four times with "
                    "the wall on its left.")
    kpis([("Time steps", f"{q['rows']['4']:,}", "identical across the 3 files", None),
          ("Duration", f"{D['ts']['duration_s']:.0f} s", "9 Hz, four laps", None),
          ("Missing values", f"{q['missing']['4']}", None, None),
          ("Duplicate rows", f"{q['duplicate_rows_4']}", f"{q['consecutive_duplicates_4']} back-to-back, kept", None),
          ("Labels explained", f"{D['rule']['tree_train_accuracy']:.0%}", "by three thresholds", None)])
    t1, t2, t3, t4 = st.tabs(["Overview", "Distributions", "Quality & conflicts", "Simulator check"])
    with t1:
        c1, c2 = st.columns([1, 1.3], gap="medium")
        with c1:
            chart(plots.class_bars(D["classes"]))
        with c2:
            chart(plots.rule_scatter(df))
        st.markdown("##### What the recording implies")
        st.markdown(f"""
* **The label is a threshold rule.** front ≤ 0.900 m → sharp right; else left ≤ 0.494 m → slight right;
  left ≤ 0.901 m → forward; else slight left.
* **The rows are a trajectory.** Lag-1 autocorrelation {min(D['ts']['lag1_autocorr'].values()):.2f}–{max(D['ts']['lag1_autocorr'].values()):.2f};
  the label repeats {D['ts']['label_persistence']:.1%} of the time, so transitions s → s′ can be measured.
* **One action per state.** All {int((D['emp']['occupancy'] > 0).sum())} visited states show exactly one action, so
  P(s′|s,a) for other actions is never observed. A simulator calibrated on the data supplies it.
* **Front and left carry the information.** Random-forest importance: right {D['relevance']['rf_importance']['SD_right']:.2f},
  back {D['relevance']['rf_importance']['SD_back']:.2f}.""")
        chart(plots.time_series(df))
    with t2:
        table(D["describe"].reset_index().rename(columns={"index": "Sensor"}))
        note("Medians set the state bins and reward zones; skew and outlier counts show that long readings (open space, "
             "sonar spikes) are common, hence the uneven, open-ended bins; 5 m is the sensor's maximum range.")
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            chart(plots.sensor_hist(df, "SD_front"))
        with c2:
            chart(plots.sensor_hist(df, "SD_left"))
        c3, c4 = st.columns([1, 1.2], gap="medium")
        with c3:
            chart(plots.correlation_heatmap(df))
        with c4:
            rel = D["relevance"]
            table(pd.DataFrame({"Features": list(rel["blocked_cv_accuracy"]),
                                "Tree accuracy (blocked 5-fold CV)": [f"{v:.1%}" for v in rel["blocked_cv_accuracy"].values()]}))
            table(pd.DataFrame({"Sensor": list(rel["rf_importance"]),
                                "RF importance": list(rel["rf_importance"].values()),
                                "Mutual information": list(rel["mutual_information"].values())}).round(3))
    with t3:
        arcs = D["arcs"]
        st.markdown(f"""
* **Arc widths.** The README and slide say 60° arcs. The numbers show `SD_front` = min({', '.join(arcs['SD_front']['sensors'])}) and
  `SD_right` = min({', '.join(arcs['SD_right']['sensors'])}) (60°), but `SD_left` = min({', '.join(arcs['SD_left']['sensors'])})
  ({arcs['SD_left']['arc_deg']}°) and `SD_back` = min({', '.join(arcs['SD_back']['sensors'])}) ({arcs['SD_back']['arc_deg']}°).
* **Angle labels.** The README puts the front sensor at 180° and the back at 0°, yet `SD_front` is centred on US13.
* **Sensor cap.** {q['values_above_5_in_24']} raw readings exceed 5 m (max {q['max_24']:.3f} m); simplified distances cap at 5.0 m.
* **Duplicates.** {q['duplicate_rows_4']} repeated rows ({q['consecutive_duplicates_4']} consecutive) come from sub-millimetre motion at 9 Hz and are kept.
* **File overlap.** The 2-sensor file equals the first two columns of the 4-sensor file: {q['file2_subset_of_file4']}.

The simulator uses the documented 60° arcs; the arc width is a parameter in `rdmu/config.py`.""")
    with t4:
        v = res["validation"]
        chart(plots.validation_bars(v))
        kpis([("Median front", f"{v['median_twin'][0]:.2f} m", f"recording {v['median_real'][0]:.2f} m", None),
              ("Median left", f"{v['median_twin'][1]:.2f} m", f"recording {v['median_real'][1]:.2f} m", None),
              ("Sharp-right share", f"{v['freq_twin'][2]:.0%}", f"recording {v['freq_real'][2]:.0%}", "#C2364B"),
              ("Transition TV distance", f"{v['transition_tv_weighted']:.2f}", f"{v['transition_states_compared']} states", None)])
        note("The twin reproduces the recorded front and left distances and never collides, but uses fewer sharp right "
             "turns: the real room had more corners than the assumed floor plan. Treat absolute returns as "
             "simulator-specific; the comparison under identical conditions is the transferable result.")


def page_mdp():
    vi, M = res["vi"], res["model"]
    pe = res["pi_expert"]
    section("MDP · value iteration", "𝓜 = (S, A, P, R, γ) with 43 states, 4 actions, transitions estimated by simulation "
                                     "and value iteration stopping on the slide's convergence threshold.")
    pic = res["pi_check"]
    kpis([("Sweeps to converge", f"{vi.iterations}", f"tolerance {cfg.solver.tol:g}", METHOD_COLOR["mdp"]),
          ("Final ‖ΔV‖∞", f"{vi.deltas[-1]:.1e}", None, METHOD_COLOR["mdp"]),
          ("Error bound", f"{vi.error_bound:.1e}", "γε / (1 − γ)", METHOD_COLOR["mdp"]),
          ("Policy iteration agrees", "Yes" if np.array_equal(pic.pi[:-1], vi.pi[:-1]) else "No",
           f"{pic.iterations} improvement steps", METHOD_COLOR["mdp"]),
          ("Agreement with recorded rule", f"{(vi.pi[:-1] == pe[:-1]).mean():.0%}", "of states", METHOD_COLOR["mdp"])])
    t1, t2, t3, t4 = st.tabs(["Formulation", "Solution", "Transitions", "Output CSV"])
    with t1:
        c1, c2 = st.columns([1.1, 1], gap="medium")
        with c1:
            st.markdown(f"""
* **S** ({stc.n_states}): {stc.n_front} front bins × {stc.n_left} left bins + absorbing *Crash*. Bin edges contain the
  recorded thresholds, so that controller is exactly a policy in this MDP.
* **A** (4): {', '.join(ACTIONS)}.
* **P(s′|s,a)**: Monte Carlo estimate from {M.n_samples:,} simulated moves; every sampled pose tries all four actions
  (min {int(M.counts[:-1].min())} samples per state-action).
* **r(s′,a)**: +1 tracking band · −0.5 caution · −2 danger · −0.5 wall lost · −25 crash · turning cost 0.05 / 0.20.
  Zone thresholds from data; values are assumptions.
* **γ = {cfg.solver.gamma}**, horizon ≈ {1 / (1 - cfg.solver.gamma):.0f} decisions ≈ {1 / (1 - cfg.solver.gamma) / 3:.0f} s.""")
            st.latex(r"V_{k+1}(s)=\max_{a}\Big[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\,V_k(s')\Big]")
            st.latex(r"\lVert V_{k+1}-V_k\rVert_\infty<\varepsilon\;\Rightarrow\;\lVert V_{k+1}-V^*\rVert_\infty\le\tfrac{\gamma\varepsilon}{1-\gamma}")
            st.latex(r"\pi^*(s)=\arg\max_a Q^*(s,a)")
        with c2:
            chart(plots.zone_grid(state_zone_table(stc, cfg.reward), stc))
        chart(plots.reward_heatmaps(M.R, stc))
    with t2:
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            chart(plots.convergence({"‖V_k+1 − V_k‖∞": (vi.deltas, METHOD_COLOR["mdp"])}, "Value-iteration convergence",
                                    "sup-norm change", hline=cfg.solver.tol))
        with c2:
            pc = vi.policy_changes
            last = int(np.max(np.nonzero(pc)[0]) + 1) if pc.any() else 0
            chart(plots.convergence({"states whose greedy action changed": (np.maximum(pc, 0.5), "#6B7280")},
                                    f"Greedy policy stable after sweep {last}", "states (log)"))
        c3, c4 = st.columns(2, gap="medium")
        with c3:
            chart(plots.state_grid(vi.V, vi.pi, stc, "Optimal value V*(s) and policy π*(s)"))
        with c4:
            chart(plots.state_grid(res["V_expert_model"], pe, stc, "Recorded controller: V^expert(s)"))
        note(f"Glyphs: {' · '.join(f'{g} {s}' for g, s in zip(ACTION_GLYPH, ACTION_SHORT))}. With the default settings π* "
             "turns away from a wall ahead earlier (front 0.90–1.10 m) and steers back to the left wall earlier "
             "(left 0.60–0.90 m), keeping the robot in the band around convex corners.")
    with t3:
        chart(plots.transition_diagram(M.P, vi.pi, vi.V, stc))
        c1, c2 = st.columns([1, 2], gap="medium")
        with c1:
            s_sel = st.selectbox("State s", list(range(stc.n_grid)), index=3 * stc.n_left + 3,
                                 format_func=lambda s: state_label(s, stc))
            a_sel = st.radio("Action a", range(4), format_func=lambda a: f"{ACTION_GLYPH[a]} {ACTIONS[a]}")
            observed = ", ".join(ACTIONS[i] for i in np.flatnonzero(D["emp"]["observed"][s_sel])) or "state never visited"
            st.markdown(f"""<div class='card kv'>samples: {int(M.counts[s_sel, a_sel])}<br>R(s,a) = {M.R[s_sel, a_sel]:+.3f}<br>
Q*(s,a) = {vi.Q[s_sel, a_sel]:+.3f}<br>π*(s) = {ACTIONS[vi.pi[s_sel]]}<br>recorded(s) = {ACTIONS[pe[s_sel]]}<br>
in the data: {observed}</div>""", unsafe_allow_html=True)
        with c2:
            chart(plots.transition_heatmap(M.P[s_sel, a_sel], stc, f"P(s′ | s, {ACTION_SHORT[a_sel]})"))
    with t4:
        table(res["value_table"].round(3), 420)
        st.download_button("Download optimal_value_function.csv", res["value_table"].to_csv(index=False),
                           "optimal_value_function.csv", "text/csv")


def page_adp():
    a = res["adp"]
    section("ADP · fitted Q-iteration",
            f"The tabular MDP grows as bins^sensors ({stc.n_grid} cells for 2 sensors, ~10⁴ for 4, ~10²⁴ for all 24). ADP "
            f"replaces the table with a {a.n_params}-parameter function of the continuous readings.")
    kpis([("Fitted iterations", f"{a.iterations}", None, METHOD_COLOR["adp"]),
          ("Converged", "Yes" if a.converged else "No", f"max|ΔQ̂| < {cfg.solver.adp_tol:g}", METHOD_COLOR["adp"]),
          ("Parameters", f"{a.n_params}", f"{cfg.solver.rbf_front}×{cfg.solver.rbf_left} RBF × 4 actions", METHOD_COLOR["adp"]),
          ("Fit time", f"{a.seconds:.1f} s", f"{res['model'].n_samples:,} transitions", METHOD_COLOR["adp"])])
    t1, t2, t3 = st.tabs(["Method & convergence", "Accuracy vs exact MDP", "Decision regions"])
    with t1:
        st.latex(r"\hat Q(x,a)=\phi(x)^{\top}\theta_a,\quad y_i=r_i+\gamma(1-d_i)\max_{a'}\phi(x_i')^{\top}\theta_{a'},\quad "
                 r"\theta_a\leftarrow\arg\min_\theta\sum_{i:a_i=a}(y_i-\phi(x_i)^\top\theta)^2+\lambda\lVert\theta\rVert^2")
        note("φ(x) are normalised Gaussian radial basis functions over log-distance; normalisation makes the fitted "
             "operator behave like soft state aggregation, which keeps the iteration stable. Training data are the same "
             "simulated transitions used for the MDP.")
        chart(plots.convergence({"max |ΔQ̂| on samples": (a.deltas, METHOD_COLOR["adp"]),
                                 "RMS Bellman residual": (a.bellman_residual, "#6B7280")},
                                "Fitted Q-iteration convergence", "value (log)", hline=cfg.solver.adp_tol))
    with t2:
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            chart(plots.actual_vs_approx(res["vi"].V, res["adp_cell_values"]))
        with c2:
            chart(plots.state_grid(res["adp_cell_values"] - res["vi"].V, None, stc, "V̂ − V* per cell", colorscale="RdBu",
                                   zmid=0, fmt="{:+.1f}", colorbar_title="Δ"))
        c3, c4 = st.columns(2, gap="medium")
        with c3:
            chart(plots.state_grid(res["adp_cell_values"], np.append(res["adp_cell_policy"], 0), stc,
                                   "ADP: V̂ per cell, majority action"))
        with c4:
            chart(plots.state_grid(res["model"].counts.sum(axis=1), None, stc, "State-space coverage (samples per cell)",
                                   colorscale="Blues", fmt="{:.0f}", colorbar_title="samples"))
    with t3:
        ff, ll = np.meshgrid(np.linspace(0.45, 2.6, 160), np.linspace(0.33, 1.6, 160), indexing="ij")
        grid = np.column_stack([ff.ravel(), ll.ravel(), ff.ravel(), ff.ravel()])
        acts = build_policies(res, cfg)["adp"].act(grid).reshape(ff.shape)
        fig = go.Figure()
        for i in range(4):
            fig.add_trace(go.Heatmap(z=np.where(acts == i, 1, np.nan), x=ll[0], y=ff[:, 0], showscale=False,
                                     colorscale=[[0, plots.ACTION_COLOR[i]], [1, plots.ACTION_COLOR[i]]], opacity=0.85,
                                     name=ACTION_SHORT[i], hovertemplate=f"{ACTION_SHORT[i]}<extra></extra>"))
        tf, tlo, thi = EXPERT_THRESHOLDS
        for x0, x1, y0, y1 in ((0.33, 1.6, tf, tf), (tlo, tlo, tf, 2.6), (thi, thi, tf, 2.6)):
            fig.add_shape(type="line", x0=x0, x1=x1, y0=y0, y1=y1, line=dict(color="black", dash="dash", width=1.5))
        plots._layout(fig, "ADP decision regions on continuous readings (dashed: recorded thresholds)", 460)
        fig.update_xaxes(title="SD_left (m)")
        fig.update_yaxes(title="SD_front (m)")
        chart(fig)
        legend = " ".join(f"<span class='pill' style='background:{plots.ACTION_COLOR[i]}'>{ACTION_SHORT[i]}</span>" for i in range(4))
        note(f"{legend} &nbsp; ADP is not tied to bin edges, so its boundaries come out smooth rather than rectangular.")


def page_search():
    mc, hj = res["mc"], res["hj"]
    section("Policy search",
            f"Both searches tune the recorded controller's rule, θ = (τ_front, τ_left-low, τ_left-high), starting from "
            f"({', '.join(f'{x:.3f}' for x in EXPERT_THRESHOLDS)}). J(θ) is the mean discounted return of "
            f"{cfg.solver.ps_episodes} simulated episodes × {cfg.solver.ps_horizon} decisions with fixed seeds.")
    st.latex(r"G^{(i)}=\sum_{t=0}^{H-1}\gamma^{t}r^{(i)}_t,\qquad \hat J(\theta)=\frac1N\sum_{i=1}^{N}G^{(i)},\qquad "
             r"\text{CI}_{95\%}=\hat J\pm1.96\,\frac{s_G}{\sqrt N}")
    t1, t2, t3 = st.tabs(["Monte Carlo search", "Hooke–Jeeves", "Selection-bias check"])
    with t1:
        kpis([("Candidates", f"{len(mc.candidates)}", f"top {cfg.solver.mc_top} re-scored", METHOD_COLOR["mcps"]),
              ("Selected θ", ", ".join(f"{x:.3f}" for x in mc.best_theta), None, METHOD_COLOR["mcps"]),
              ("Episodes used", f"{mc.episodes_used:,}", None, METHOD_COLOR["mcps"]),
              ("Time", f"{mc.seconds:.1f} s", None, METHOD_COLOR["mcps"])])
        c1, c2 = st.columns([1.3, 1], gap="medium")
        with c1:
            chart(plots.mc_scatter(mc))
        with c2:
            table(pd.DataFrame({"τ_front": mc.candidates[mc.top_idx, 0], "τ_low": mc.candidates[mc.top_idx, 1],
                                "τ_high": mc.candidates[mc.top_idx, 2], "J stage 1": mc.stage1[mc.top_idx],
                                "J stage 2": mc.stage2, "± 95% CI": mc.stage2_ci}).round(3))
            note("Stage 1 scores random feasible θ (the recorded θ is candidate 0). Stage 2 re-scores the top candidates "
                 "on new seeds with more episodes, because the stage-1 winner is biased upwards by selection.")
        c3, c4 = st.columns(2, gap="medium")
        with c3:
            chart(plots.mc_progress(mc))
        with c4:
            chart(plots.return_histograms(res["returns"]))
    with t2:
        kpis([("Start θ", ", ".join(f"{x:.3f}" for x in EXPERT_THRESHOLDS), "recorded controller", METHOD_COLOR["hj"]),
              ("Final θ", ", ".join(f"{x:.3f}" for x in hj.best_theta), None, METHOD_COLOR["hj"]),
              ("Evaluations", f"{hj.evaluations}", f"{hj.episodes_used:,} episodes", METHOD_COLOR["hj"]),
              ("Stop", hj.stop_reason.split(" (")[0].capitalize(), None, METHOD_COLOR["hj"])])
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            chart(plots.hj_trace(hj.log))
        with c2:
            chart(plots.hj_params(hj.log))
        chart(plots.hj_steps(hj))
        note(f"Exploratory moves try ±Δ on each threshold (Δ = {cfg.solver.hj_step} m); a pattern move jumps to "
             "x + (x − x_base) and is kept only if exploring around it improves J; otherwise Δ halves.")
        with st.expander("Full evaluation log"):
            table(pd.DataFrame(hj.log).round(4), 320)
    with t3:
        kpis([("Hooke–Jeeves on search seeds", f"{hj.best_J:.2f}", None, METHOD_COLOR["hj"]),
              ("Hooke–Jeeves on fresh seeds", f"{S_all['hj']['return_mean']:.2f}", None, METHOD_COLOR["hj"]),
              ("Monte Carlo on search seeds", f"{mc.best_J:.2f}", None, METHOD_COLOR["mcps"]),
              ("Monte Carlo on fresh seeds", f"{S_all['mcps']['return_mean']:.2f}", None, METHOD_COLOR["mcps"])])
        note("Scores on the seeds used for searching are optimistic because the winner was selected on them. The "
             "comparison page uses only the fresh-seed numbers.")


def page_audit():
    with st.spinner("Re-running the notebook blocks (about 10 s the first time)…"):
        au = get_audit()
    vi_a, mc_a, hj_a, pf = au["vi"], au["mc"], au["hj"], au["portfolio"]
    section("Notebook audit", "The starter notebook `Day_5.ipynb` was re-executed block by block and each result was "
                              "checked against an independent calculation.")
    kpis([("Value iteration", "Verified", f"max error {vi_a['max_error_vs_exact']:.0e}", "#0F8B8D"),
          ("Random-walk value", f"{mc_a['exact_value']:.2f}", f"notebook {mc_a['notebook_estimate']:.2f}", "#0F8B8D"),
          ("Hooke–Jeeves from (1.3, −0.7)", str(np.round(hj_a["non_grid_start_result"], 2).tolist()), "true minimum (0, 0)", "#C2364B"),
          ("Portfolio objective", "Flawed", "ignores asset means", "#C2364B")])
    for title, text in au["findings"]:
        with st.container(border=True):
            st.markdown(f"**{title}** · {text}")
    with st.expander("Checks in numbers"):
        st.markdown(f"""
* **Value iteration:** [68.068, 62.623, 67.133] equals the exact linear solve for policy ({', '.join(vi_a['greedy_policy'])});
  a 10⁻⁶ tolerance would stop after {vi_a['sweeps_needed_tol_1e-6']} sweeps instead of 1,000.
* **Monte Carlo:** exact expected return {mc_a['exact_value']:.3f}; 20 seeded repeats average {mc_a['seeded_estimates_mean']:.2f}
  (s.d. {mc_a['seeded_estimates_sd']:.2f}), so −32.93 is sampling noise (z = {mc_a['notebook_z_score']:.2f}).
* **Hooke–Jeeves:** integer start [1, 1] returns {hj_a['integer_start_result'].tolist()}.
* **Portfolio:** the "return" is the grand mean {pf['grand_mean_used']:.4f} for every portfolio; per-asset means range
  {pf['per_asset_means'].min():.4f} to {pf['per_asset_means'].max():.4f}.""")
    with st.expander("What STOCHOS changes"):
        st.markdown("Value iteration stops on the slide's convergence threshold, reports the error bound, extracts the "
                    "policy and writes `optimal_value_function.csv`. The ADP label is used only for a genuinely "
                    "approximate method. Monte Carlo policy search actually searches. Hooke–Jeeves evaluates the pattern "
                    "point before accepting it, works in floats, respects bounds and logs every evaluation. All "
                    "randomness is seeded.")
    with st.expander("Original notebook source"):
        st.code(legacy_audit.notebook_source(), language="python")


def page_about():
    section("About the method", "How the pieces fit together, what was assumed, and where the limits are.")
    t1, t2, t3, t4 = st.tabs(["Pipeline", "Assumptions", "Interpretation", "Complexity & reproducibility"])
    with t1:
        st.markdown(f"""
1. **Data** → controller rule, state bins, reward zones, speed, sensor cap and spike rate (`rdmu/data.py`).
2. **Digital twin** → unicycle robot, 24 simulated sonar beams, polygonal rooms with pillars (`rdmu/twin.py`, `rdmu/rooms.py`), checked against the recording.
3. **Model** → P(s′|s,a) and R(s,a) from all four actions simulated at {res['model'].n_samples // 4:,} poses (`rdmu/model.py`).
4. **Methods** → value iteration (`mdp.py`), fitted Q-iteration (`adp.py`), Monte Carlo and Hooke–Jeeves (`policy_search.py`).
5. **Evaluation** → same starts and noise for every controller, on seeds never used in training (`evaluation.py`).
6. **Missions** → entry-to-exit runs in the mission hall with a loop watchdog (`mission.py`), drawn by `arena.py`.""")
    with t2:
        table(pd.DataFrame([
            ("Decision interval", "1/3 s (3 samples)", "Assumption; 9 Hz sampling from data"),
            ("Forward speed", f"{cfg.twin.forward_speed} m/s", "Data: front closing speed while moving forward"),
            ("Turn rates (slight / sharp)", f"{abs(cfg.twin.turn_rate[1])} / {abs(cfg.twin.turn_rate[2])} rad/s", "Calibrated to recorded medians"),
            ("Robot radius", f"{cfg.twin.robot_radius} m", "SCITOS-G5 footprint"),
            ("Sensor noise / spike rate", f"0.5 cm + 1% / {cfg.twin.spike_prob:.1%}", "Assumption / data"),
            ("Training room", "6.4 × 4.2 m, pillar and recess", "Assumption; real plan not published"),
            ("Mission hall", "8.4 × 5.6 m, 3 pillars, entry and exit", "Showcase and transfer test"),
            ("Loop watchdog", "back within 0.30 m of a spot passed ≥ 20 s earlier", "Assumption; stops laps and orbits"),
            ("Reward values", "+1 / −0.5 / −2 / −25", "Assumption; zone thresholds from data"),
            ("γ, tolerance", f"{cfg.solver.gamma}, {cfg.solver.tol:g}", "Slide inputs, adjustable"),
        ], columns=["Item", "Value", "Basis"]))
        note("Limitations: the simulator is calibrated, not identified (fewer sharp turns than the recording); the "
             "(front, left) state is not fully Markov because heading is hidden; policy search covers one controller family.")
    with t3:
        table(pd.DataFrame({
            "": ["Problem solved", "Use of uncertainty", "Future consequences", "Key assumption", "Strength", "Limitation", "Read the result as"],
            "MDP · value iteration": ["Optimal action per sensor cell", "Expectation over estimated P", "γ-discounted V(s′)",
                                      "(front, left) cell is Markov", "Exact, fast, error bound", "Discretisation error", "Best policy for the model"],
            "ADP · fitted Q-iteration": ["Optimal action for continuous readings", "Expectation through sampled targets",
                                         "γ max Q̂(x′,·)", "Features can represent Q", "Continuous, scalable",
                                         "No general convergence guarantee", "Smooth approximation of the optimum"],
            "Monte Carlo search": ["Best thresholds for the rule", "Average over episodes", "Discounted return",
                                   "Controller family is adequate", "Global, parallel", "Wasteful sampling", "Best of the sampled θ"],
            "Hooke–Jeeves": ["Best thresholds for the rule", "Average over fixed-seed episodes", "Discounted return",
                             "Objective locally well-behaved", "Few evaluations, interpretable", "Local optimum only",
                             "Local optimum near the recorded θ"],
        }))
    with t4:
        table(pd.DataFrame([
            ("Value iteration", "O(k · S² · A)", f"k = {res['vi'].iterations}, S = {stc.n_states}, A = 4 → milliseconds"),
            ("Model estimation", "O(poses · A · beams · walls)", f"{res['model'].n_samples:,} simulated moves, vectorised"),
            ("Fitted Q-iteration", "O(k · N · d) per action", f"k = {res['adp'].iterations}, N = {res['model'].n_samples:,}, d = {res['adp'].theta.shape[0]}"),
            ("Monte Carlo search", "O(K · N_ep · H)", f"{res['mc'].episodes_used:,} episodes × {cfg.solver.ps_horizon} steps"),
            ("Hooke–Jeeves", "O(evaluations · N_ep · H)", f"{res['hj'].episodes_used:,} episodes × {cfg.solver.ps_horizon} steps"),
        ], columns=["Method", "Cost", "Here"]))
        note(f"Every random draw uses a seeded NumPy generator: training seed {cfg.solver.seed}, evaluation seed "
             f"{cfg.evaluation.seed}, and the sidebar seed for the robot demo. Expensive results are cached and the "
             "default results ship precomputed.")


{"Live robot": page_robot, "Compare methods": page_compare, "Data": page_data, "MDP": page_mdp, "ADP": page_adp,
 "Policy search": page_search, "Notebook audit": page_audit, "About": page_about}[page]()

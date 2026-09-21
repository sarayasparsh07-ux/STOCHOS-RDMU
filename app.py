"""STOCHOS — decisions under uncertainty for a wall-following robot.

    streamlit run app.py
"""
from __future__ import annotations

import io
from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

from rdmu import data, legacy_audit, plots
from rdmu.config import (ACTION_GLYPH, ACTION_SHORT, ACTIONS, DEFAULT, EXPERT_THRESHOLDS, METHOD_COLOR,
                         METHOD_KEYS, METHOD_LABEL)
from rdmu.evaluation import record_episode, run_batch, summarise
from rdmu.pipeline import build_policies, load_or_run
from rdmu.policies import ThresholdPolicy
from rdmu.policy_search import BOUNDS, feasible
from rdmu.statespace import (ZONES, expert_policy_table, representative_readings, state_label,
                             state_zone_table, threshold_actions)
from rdmu.twin import WallFollowTwin

st.set_page_config(page_title="STOCHOS · wall-following robot", page_icon="🛰️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, .stDataFrame, button, input, select {
  font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif; }
code, pre, .mono { font-family: 'IBM Plex Mono', ui-monospace, monospace; }
.block-container { padding-top: 1.6rem; max-width: 1380px; }
.hero { border-left: 5px solid #0F8B8D; padding: 0.2rem 0 0.2rem 1rem; margin-bottom: 0.8rem; }
.hero h1 { font-size: 2.3rem; font-weight: 600; letter-spacing: 0.02em; margin: 0; color: #1B2433; }
.hero p { color: #4A5366; margin: 0.2rem 0 0 0; font-size: 1.02rem; max-width: 70ch; }
.card { border: 1px solid #E3E7EE; border-radius: 10px; padding: 0.8rem 1rem; background: #FFFFFF; }
.decision { border: 1px solid #E3E7EE; border-radius: 10px; padding: 0.7rem 0.9rem; background: #FBFCFD; }
.decision .act { font-size: 1.35rem; font-weight: 600; }
.kv { font-family: 'IBM Plex Mono', monospace; font-size: 0.86rem; color: #394256; }
.note { color: #4A5366; font-size: 0.92rem; }
.pill { display:inline-block; padding: 0.05rem 0.55rem; border-radius: 999px; font-size: 0.8rem;
        color: white; margin-right: 0.3rem; }
div[data-testid="stMetricValue"] { font-size: 1.55rem; }
</style>
""", unsafe_allow_html=True)

FULL = "stretch"
LABELS = {**METHOD_LABEL, "custom": "Custom thresholds (manual)"}
COLORS = {**METHOD_COLOR, "custom": "#7A4FB5"}
REQUIRED_FILES = ("sensor_readings_2.csv", "sensor_readings_4.csv", "sensor_readings_24.csv", "Day_5.ipynb")

METHOD_EXPLAIN = {
    "mdp": ("Looks up the current (front, left) cell and takes the action with the highest Q*(s,a) from value "
            "iteration. Q* already contains the expected discounted future reward, computed from the estimated "
            "transition probabilities, so the robot trades today's reward against where each action is likely to lead."),
    "adp": ("Evaluates Q̂(x,a) = φ(x)ᵀθ_a on the raw, undiscretised readings and takes the largest. θ was fitted "
            "by repeated regression on sampled Bellman targets, so the decision boundaries are smooth rather than "
            "tied to bin edges."),
    "mcps": ("Uses the recorded controller's rule with thresholds chosen by Monte Carlo random search: 60 candidate "
             "threshold sets were each simulated over the same episodes and the best, re-checked on new seeds, was kept."),
    "hj": ("Uses the recorded controller's rule with thresholds tuned by Hooke–Jeeves pattern search, starting "
           "from the recorded values and moving one threshold at a time while simulated return improved."),
    "expert": ("The controller that produced the dataset, recovered exactly from the labels: sharp right if front ≤ 0.900 m; "
               "else slight right if left ≤ 0.494 m; forward if left ≤ 0.901 m; otherwise slight left."),
    "random": "Picks one of the four actions uniformly at random. It is a lower bound for every other method.",
    "custom": "The recorded controller's rule with thresholds you set in the sidebar, for exploring the objective by hand.",
}


# ------------------------------------------------------------------ caching
@st.cache_resource(show_spinner=False)
def get_results(cfg):
    return load_or_run(cfg)


@st.cache_resource(show_spinner=False)
def get_data_analysis():
    return {
        "df": data.load4(), "quality": data.quality_report(), "describe": data.describe(),
        "classes": data.class_distribution(), "ts": data.time_series_evidence(),
        "relevance": data.feature_relevance(), "rule": data.extract_expert_rule(),
        "arcs": data.arc_membership(),
        "emp": data.empirical_expert_transitions(DEFAULT.states),
    }


@st.cache_resource(show_spinner=False)
def get_audit():
    return legacy_audit.run_all()


def make_policy(res, cfg, method, custom_thr=None):
    if method == "custom":
        return ThresholdPolicy(custom_thr, name="custom")
    return build_policies(res, cfg)[method]


@st.cache_data(show_spinner=False)
def get_episode(_res, cfg, method, start, seed, horizon, custom_thr=None):
    return record_episode(make_policy(_res, cfg, method, custom_thr), cfg, np.array(start), horizon, seed, _res["model"])


@st.cache_data(show_spinner=False)
def batch_run(_res, cfg, method, n, horizon, seed, custom_thr=None):
    m = run_batch(make_policy(_res, cfg, method, custom_thr), cfg, n, horizon, seed)
    return summarise(m), m["return"], m["crashed"]


START_POSES = {
    "Bottom wall, beside the pillar": (5.0, 1.05, np.pi),
    "Left wall, heading north": (0.85, 1.4, np.pi / 2),
    "Top wall, before the recess": (1.6, 3.55, 0.0),
    "Right wall, heading south": (5.6, 3.0, -np.pi / 2),
    "Room centre, no wall in range": (2.6, 2.1, 0.6),   # rule-like policies may orbit here
    "Random start (uses the seed)": None,
}


def start_pose(name, seed, cfg):
    p = START_POSES[name]
    if p is None:
        return tuple(WallFollowTwin(cfg.twin).mixed_starts(1, np.random.default_rng(seed), 0.6)[0])
    return p


# ------------------------------------------------------------------ sidebar
with st.sidebar:
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
    st.markdown("### Simulation")
    start_name = st.selectbox("Start position", list(START_POSES))
    seed = int(st.number_input("Random seed (sensor and motion noise)", 0, 10_000, 3, step=1))
    horizon = st.slider("Episode length (decisions, 3 per second)", 60, 900, 450, step=30)
    speed = st.select_slider("Playback speed", ["0.5×", "1×", "2×", "4×", "8×"], value="2×")
    n_batch = st.slider("Number of episodes (batch run)", 10, 200, 50, 10)
    st.markdown("### Goal")
    st.caption("Wall following has a goal *region*, not a goal cell: keep the wall on the left at "
               "0.494–0.901 m with more than 0.900 m clear ahead (the tracking band), never collide, keep moving.")
    st.markdown("### Environment")
    st.caption("6.4 × 4.2 m room with a pillar and a recess · 24 sonar beams, 60° arcs · decisions every 1/3 s · "
               "sensor noise and motion noise on.")
    with st.expander("Model settings (recomputes everything)"):
        with st.form("model_form"):
            gamma = st.slider("Discount factor γ", 0.80, 0.99, DEFAULT.solver.gamma, 0.01)
            tol = st.select_slider("Convergence threshold", [1e-3, 1e-4, 1e-5, 1e-6, 1e-8], value=DEFAULT.solver.tol,
                                   format_func=lambda x: f"{x:g}")
            ps_eps = st.slider("Episodes per policy-search evaluation", 8, 64, DEFAULT.solver.ps_episodes, 8)
            ev_eps = st.slider("Evaluation episodes per method", 50, 400, DEFAULT.evaluation.n_episodes, 50)
            msd = int(st.number_input("Training seed", 0, 10_000, DEFAULT.solver.seed))
            applied = st.form_submit_button("Apply and recompute")
    if applied:
        st.session_state.cfg = DEFAULT.with_solver(gamma=float(gamma), tol=float(tol), ps_episodes=int(ps_eps),
                                                   seed=msd).with_eval(n_episodes=int(ev_eps))
    cfg = st.session_state.get("cfg", DEFAULT)
    if cfg != DEFAULT:
        st.caption("Custom settings active.")
        if st.button("Restore defaults"):
            st.session_state.cfg = DEFAULT
            st.rerun()

from rdmu.config import DATA_DIR  # noqa: E402

missing = [f for f in REQUIRED_FILES if not (DATA_DIR / f).exists()]
if missing:
    st.error(f"Missing input file(s) in `data/`: {', '.join(missing)}. Restore them from the Kaggle archive and reload.")
    st.stop()
try:
    D = get_data_analysis()
except (ValueError, KeyError) as exc:
    st.error(f"The dataset could not be read as expected ({exc}). Check that the CSV files are unmodified.")
    st.stop()
try:
    with st.spinner("Solving the MDP, running ADP and both policy searches (about 40 s on first run)…"):
        res = get_results(cfg)
except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
    st.error(f"The computation failed for these settings ({exc}). Try a smaller discount factor or restore defaults.")
    st.stop()
stc = cfg.states
color = COLORS[method]

# ------------------------------------------------------------------ header
st.markdown("""
<div class="hero"><h1>STOCHOS</h1>
<p>A sonar robot learns to follow walls. Four decision methods (value iteration, approximate dynamic
programming, Monte Carlo policy search and Hooke–Jeeves) are trained on a data-calibrated simulator of the
SCITOS-G5 recording, then raced against the robot's original controller.</p></div>
""", unsafe_allow_html=True)

if method == "custom":
    ev_ = cfg.evaluation
    es = batch_run(res, cfg, "custom", ev_.n_episodes, ev_.horizon, ev_.seed, custom_thr)[0]
    el = batch_run(res, cfg, "custom", ev_.long_episodes, ev_.long_horizon, ev_.seed + 500, custom_thr)[0]
    es["diff_vs_expert"] = es["return_mean"] - res["eval_short"]["expert"]["return_mean"]
    train_s, iters = 0.0, "—"
else:
    es, el = res["eval_short"][method], res["eval_long"][method]
    train_s = res["runtime"][method]["seconds"]
    iters = {"mdp": res["vi"].iterations, "adp": res["adp"].iterations, "mcps": len(res["mc"].candidates),
             "hj": res["hj"].evaluations}.get(method, "—")
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Discounted return", f"{es['return_mean']:.2f}", f"{es['diff_vs_expert']:+.2f} vs expert" if method != "expert" else None)
k2.metric("Success rate (300 s, no crash)", f"{1 - el['crash_rate']:.0%}")
k3.metric("Time in tracking band", f"{el['band_share']:.0%}")
k4.metric("Reward per step", f"{el['reward_per_step']:.3f}")
k5.metric("Training time", f"{train_s:.1f} s")
k6.metric("Iterations / evaluations", iters)

tabs = st.tabs(["Robot", "Dataset", "MDP", "ADP", "Policy search", "Comparison", "Notebook audit", "Method notes"])

# ================================================================== ROBOT
with tabs[0]:
    ep = get_episode(res, cfg, method, start_pose(start_name, seed, cfg), seed, horizon, custom_thr)
    n = len(ep["action"])
    st.markdown(f"<div class='card'><b style='color:{color}'>{LABELS[method]}</b> — {METHOD_EXPLAIN[method]}</div>",
                unsafe_allow_html=True)
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Cumulative reward (this run)", f"{ep['reward'].sum():.1f}")
    e2.metric("Steps", f"{n}  ({n / 3:.0f} s)")
    e3.metric("Final state", state_label(int(ep["next_state"][-1]), stc))
    e4.metric("Outcome", "collision" if ep["crashed"][-1] else "no collision")
    frame_ms = {"0.5×": 660, "1×": 330, "2×": 165, "4×": 80, "8×": 40}[speed]
    stride = 1 if n <= 450 else 2
    left, right = st.columns([1.65, 1])
    with left:
        st.plotly_chart(plots.arena_animation(ep, cfg.twin.robot_radius, color, LABELS[method], frame_ms, stride),
                        width=FULL, config={"displayModeBar": False})
        st.markdown(f"<span class='note'>▶ Play and ⏸ Pause run the animation in your browser; drag the slider to scrub. "
                    f"Red rays are the 60° front arc, teal rays the 60° left arc. Dots mark every visited position, coloured by "
                    f"reward zone: teal = tracking band (the goal), amber = caution, red = danger. "
                    f"{'The run ended in a collision at step ' + str(n - 1) + '.' if ep['crashed'][-1] else f'{n} decisions, no collision.'}</span>",
                    unsafe_allow_html=True)
    with right:
        st.markdown("#### Decision inspector")
        key = f"step_{method}_{start_name}_{seed}_{horizon}"
        if key not in st.session_state:
            st.session_state[key] = 0

        def _mv(d):
            st.session_state[key] = int(np.clip(st.session_state[key] + d, 0, n - 1))

        def _reset():
            st.session_state[key] = 0
        b1, b2, b3 = st.columns(3)
        b1.button("↺ Reset", on_click=_reset, width=FULL)
        b2.button("◀ Back", on_click=_mv, args=(-1,), width=FULL)
        b3.button("⏭ Step", on_click=_mv, args=(1,), width=FULL)
        st.session_state[key] = min(st.session_state[key], n - 1)
        t = st.slider("Step", 0, n - 1, key=key)
        a = int(ep["action"][t])
        s, s2 = int(ep["state"][t]), int(ep["next_state"][t])
        st.markdown(f"""
<div class="decision">
<div class="kv">t = {t / 3:.1f} s · readings F {ep['sd'][t][0]:.3f} m · L {ep['sd'][t][1]:.3f} m · R {ep['sd'][t][2]:.2f} · B {ep['sd'][t][3]:.2f}</div>
<div class="kv">state s = {state_label(s, stc)}</div>
<div class="act" style="color:{plots.ACTION_COLOR[a]}">{ACTION_GLYPH[a]} {ACTIONS[a]}</div>
<div class="note">{ep['explain'][t]}</div>
<div class="kv" style="margin-top:0.35rem">next state s′ = {state_label(s2, stc)}<br>
model P(s′ | s, a) = {ep['p_transition'][t]:.3f} · reward r = {ep['reward'][t]:+.2f} · zone {ZONES[int(ep['zone'][t])]}</div>
</div>""", unsafe_allow_html=True)
        pol = make_policy(res, cfg, method, custom_thr)
        st.plotly_chart(plots.scores_bar(ep["scores"][t], a, pol.score_label), width=FULL,
                        config={"displayModeBar": False})
        P_row = res["model"].P[s]
        top = np.argsort(-P_row[a])[:4]
        st.caption("Uncertainty: the same state and action can lead to different next states. Most likely outcomes "
                   "under the estimated model: " + " · ".join(f"{state_label(j, stc)} ({P_row[a, j]:.2f})" for j in top))
    st.plotly_chart(plots.episode_timeline(ep, color), width=FULL, config={"displayModeBar": False})
    logdf = pd.DataFrame({
        "step": np.arange(n), "time_s": np.round(np.arange(n) / 3, 2),
        "SD_front": ep["sd"][:, 0].round(3), "SD_left": ep["sd"][:, 1].round(3),
        "state": [state_label(x, stc) for x in ep["state"]],
        "action": [ACTIONS[x] for x in ep["action"]],
        **{f"score_{ACTION_SHORT[i]}": ep["scores"][:, i].round(3) for i in range(4)},
        "next_state": [state_label(x, stc) for x in ep["next_state"]],
        "P(s'|s,a)": ep["p_transition"].round(4), "reward": ep["reward"].round(3),
        "cumulative_reward": np.cumsum(ep["reward"]).round(3), "why": ep["explain"],
    })
    with st.expander(f"Batch run: {n_batch} episodes with this method"):
        bs, br, bc = batch_run(res, cfg, method, n_batch, horizon, seed * 97 + 5, custom_thr)
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Mean discounted return", f"{bs['return_mean']:.2f} ± {bs['return_ci95']:.2f}")
        b2.metric("Success rate", f"{1 - bs['crash_rate']:.0%}")
        b3.metric("Time in band", f"{bs['band_share']:.0%}")
        b4.metric("Mean reward per step", f"{bs['reward_per_step']:.3f}")
        import plotly.graph_objects as go
        hf = go.Figure(go.Histogram(x=br, marker_color=color, nbinsx=25))
        plots._layout(hf, f"Discounted return per episode ({n_batch} episodes, {horizon} steps, seeded)", 260)
        hf.update_xaxes(title="discounted return")
        st.plotly_chart(hf, width=FULL)
    with st.expander("Decision log (every step)"):
        st.dataframe(logdf, width=FULL, height=320, hide_index=True)
        st.download_button("Download decision log (CSV)", logdf.to_csv(index=False), f"decision_log_{method}.csv", "text/csv")

# ================================================================== DATASET
with tabs[1]:
    q, df = D["quality"], D["df"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Time steps", f"{q['rows']['4']:,}")
    c2.metric("Recording length", f"{D['ts']['duration_s']:.0f} s (9 Hz)")
    c3.metric("Missing values", q["missing"]["4"])
    c4.metric("Labels explained by 3 thresholds", f"{D['rule']['tree_train_accuracy']:.0%}")
    st.markdown("""
The Kaggle archive holds the UCI *Wall-Following navigation task with mobile robot SCITOS-G5* recording: the robot
circles a room clockwise four times with the wall on its left. Three files contain the same 5,456 time steps,
with 24 raw sonar readings, 4 simplified distances (`SD_front`, `SD_left`, `SD_right`, `SD_back`: the minimum
reading within a 60° arc) and 2 simplified distances respectively. Each row carries one of four movement labels.
""")
    a1, a2 = st.columns([1, 1.25])
    with a1:
        st.plotly_chart(plots.class_bars(D["classes"]), width=FULL)
        st.dataframe(D["describe"], width=FULL)
        st.caption("Why these statistics matter: medians define the state bins and reward zones; skew and the IQR "
                   "outlier counts show that long readings (open space, sonar spikes) are common, which is why the bins "
                   "are uneven and the top bins open-ended; the 5 m cap is the sensor's maximum range.")
    with a2:
        st.plotly_chart(plots.rule_scatter(df), width=FULL)
    st.markdown("#### What the recording implies for decision-making")
    st.markdown(f"""
* **The label is a threshold rule.** A depth-3 tree on `SD_front` and `SD_left` reproduces every label
  ({D['rule']['tree_train_accuracy']:.2%}): front ≤ 0.900 m → sharp right; otherwise left ≤ 0.494 m → slight right;
  left ≤ 0.901 m → forward; else slight left. Right and back add nothing (random-forest importance
  {D['relevance']['rf_importance']['SD_right']:.2f} and {D['relevance']['rf_importance']['SD_back']:.2f}).
* **The rows are a trajectory, not independent samples.** Lag-1 autocorrelation is
  {min(D['ts']['lag1_autocorr'].values()):.2f}–{max(D['ts']['lag1_autocorr'].values()):.2f} and the label repeats
  {D['ts']['label_persistence']:.1%} of the time, so transitions s → s′ can be measured.
* **Only one action is ever observed per state.** Of the {stc.n_grid} states, {int((D['emp']['occupancy'] > 0).sum())}
  are visited and every one of them shows exactly one action (the rule's). P(s′|s,a) for the other three actions
  is never observed, so an MDP cannot be estimated from the data alone. STOCHOS therefore uses a simulator whose
  speed ({D['ts']['forward_speed_mps']} m/s), sensor cap (5 m) and spike rate come from the data, and validates it
  against the recording (below).
* **Duplicates are not errors.** {q['duplicate_rows_4']} rows repeat an earlier row, {q['consecutive_duplicates_4']}
  of them back-to-back: at 9 Hz the robot sometimes moves less than the sensors' 1 mm resolution. They are kept.
""")
    r1, r2 = st.columns([1, 1.2])
    r1.plotly_chart(plots.correlation_heatmap(df), width=FULL)
    rel = D["relevance"]
    r2.dataframe(pd.DataFrame({"Features": list(rel["blocked_cv_accuracy"]),
                               "Decision-tree accuracy (blocked 5-fold CV)": [f"{v:.1%}" for v in rel["blocked_cv_accuracy"].values()]}),
                 width=FULL, hide_index=True)
    r2.dataframe(pd.DataFrame({"Sensor": list(rel["rf_importance"]),
                               "Random-forest importance": list(rel["rf_importance"].values()),
                               "Mutual information with label": list(rel["mutual_information"].values())}).round(3),
                 width=FULL, hide_index=True)
    r2.caption("Blocked folds keep time order, so neighbouring samples cannot leak between training and test. "
               "Front and left carry the information; this justifies a two-variable state.")
    h1, h2 = st.columns(2)
    h1.plotly_chart(plots.sensor_hist(df, "SD_front"), width=FULL)
    h2.plotly_chart(plots.sensor_hist(df, "SD_left"), width=FULL)
    st.plotly_chart(plots.time_series(df), width=FULL)
    with st.expander("Inconsistencies between the documentation and the numbers"):
        arcs = D["arcs"]
        st.markdown(f"""
* The README and the class slide say every simplified distance covers a 60° arc. Matching against the raw sensors
  shows `SD_front` = min({', '.join(arcs['SD_front']['sensors'])}) (60°), `SD_right` = min({', '.join(arcs['SD_right']['sensors'])})
  (60°), but `SD_left` = min({', '.join(arcs['SD_left']['sensors'])}) spans only {arcs['SD_left']['arc_deg']}° and
  `SD_back` = min({', '.join(arcs['SD_back']['sensors'])}) only {arcs['SD_back']['arc_deg']}°.
* The README calls US1 the front sensor (reference angle 180°) and US13 the back sensor (0°), yet `SD_front` is built
  from the arc centred on US13. The angle labels are internally inconsistent.
* {q['values_above_5_in_24']} raw readings exceed the stated 5 m cap (maximum {q['max_24']:.3f} m); the simplified
  distances are capped at exactly 5.0 m.
* The 2-sensor file is an exact copy of the first two columns of the 4-sensor file: {q['file2_subset_of_file4']}.

The simulator follows the documented 60° arcs; the arc width is a parameter in `rdmu/config.py`.
""")
    st.markdown("#### Does the simulator behave like the real robot?")
    v = res["validation"]
    st.plotly_chart(plots.validation_bars(v), width=FULL)
    st.markdown(f"""
<span class='note'>Driven by the recorded controller, the twin reproduces the median front and left distances
({v['median_twin'][0]:.2f} vs {v['median_real'][0]:.2f} m, {v['median_twin'][1]:.2f} vs {v['median_real'][1]:.2f} m)
and never collides. It under-uses sharp right turns ({v['freq_twin'][2]:.0%} vs {v['freq_real'][2]:.0%}) and reads
larger right distances: the real room had more inside corners and clutter than the assumed floor plan, whose exact
geometry is not published. State-level transitions under the expert differ by a total-variation distance of
{v['transition_tv_weighted']:.2f} (weighted over {v['transition_states_compared']} states); occupancy differs by
{v['occupancy_tv']:.2f}. Treat absolute returns as simulator-specific; the ranking of methods is the robust result.</span>
""", unsafe_allow_html=True)

# ================================================================== MDP
with tabs[2]:
    vi, M = res["vi"], res["model"]
    st.markdown("#### MDP formulation")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown(f"""
* **States S** ({stc.n_states}): {stc.n_front} front bins × {stc.n_left} left bins, plus an absorbing *Crash* state.
  The bin edges contain the recorded controller's thresholds, so that controller is exactly a policy in this MDP.
* **Actions A** (4): the dataset's classes {', '.join(ACTIONS)}.
* **Transitions P(s′|s,a)**: Monte Carlo estimate from {M.n_samples:,} simulated moves. Every sampled pose tries all
  four actions; the smallest cell has {int(M.counts[:-1].min())} samples per action.
* **Reward r(s′,a)**: +1 in the tracking band (front > 0.90 m, 0.494 < left ≤ 0.901 m), −0.5 in caution, −2 in
  danger, −0.5 when the wall is lost, −25 for a crash, minus a small turning cost (0.05 slight, 0.20 sharp).
  Zone thresholds come from the data; the values are an assumption.
* **Discount γ = {cfg.solver.gamma}**, **convergence threshold = {cfg.solver.tol:g}** (inputs on the slide).
""")
        st.latex(r"V_{k+1}(s)=\max_{a}\Big[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\,V_k(s')\Big]")
        st.latex(r"\text{stop when } \lVert V_{k+1}-V_k\rVert_\infty<\varepsilon \;\Rightarrow\; \lVert V_{k+1}-V^*\rVert_\infty\le \tfrac{\gamma\varepsilon}{1-\gamma}")
    with c2:
        st.plotly_chart(plots.zone_grid(state_zone_table(stc, cfg.reward), stc), width=FULL)
    st.plotly_chart(plots.reward_heatmaps(M.R, stc), width=FULL)
    m0, m1, m2, m3, m4 = st.columns(5)
    pic = res["pi_check"]
    m0.metric("Policy iteration agrees", "yes" if np.array_equal(pic.pi[:-1], vi.pi[:-1]) else "no",
              f"{pic.iterations} improvement steps", delta_color="off")
    m1.metric("Sweeps to converge", vi.iterations)
    m2.metric("Final ‖ΔV‖∞", f"{vi.deltas[-1]:.1e}")
    m3.metric("Guaranteed error bound", f"{vi.error_bound:.1e}")
    m4.metric("Solve time", f"{vi.seconds * 1000:.0f} ms")
    g1, g2 = st.columns(2)
    g1.plotly_chart(plots.convergence({"‖V_k+1 − V_k‖∞": (vi.deltas, METHOD_COLOR["mdp"])}, "Value-iteration convergence",
                                      "sup-norm change", hline=cfg.solver.tol), width=FULL)
    pc = vi.policy_changes.copy()
    g2.plotly_chart(plots.convergence({"states whose greedy action changed": (np.maximum(pc, 0.5), "#6B7280")},
                                      f"Greedy policy stops changing after sweep {int(np.max(np.nonzero(pc)[0]) + 1) if pc.any() else 0}",
                                      "states (log)", log=True), width=FULL)
    p1, p2 = st.columns(2)
    p1.plotly_chart(plots.state_grid(vi.V, vi.pi, stc, "Optimal value V*(s) and policy π*(s)"), width=FULL)
    pe = res["pi_expert"]
    agree = float((vi.pi[:-1] == pe[:-1]).mean())
    p2.plotly_chart(plots.state_grid(res["V_expert_model"], pe, stc, f"Recorded controller: V^expert(s) · agrees with π* in {agree:.0%} of states"),
                    width=FULL)
    st.markdown(f"""
<span class='note'>Glyphs: {' · '.join(f'{g} {s}' for g, s in zip(ACTION_GLYPH, ACTION_SHORT))}.
With the default settings, π* departs from the recorded controller in two consistent ways: it starts turning away from a wall ahead earlier
(front 0.90–1.10 m) and it steers back towards the left wall earlier (left 0.60–0.90 m with open space ahead), which
keeps the robot inside the tracking band around convex corners.</span>""", unsafe_allow_html=True)
    st.plotly_chart(plots.transition_diagram(M.P, vi.pi, vi.V, stc), width=FULL)
    st.markdown("#### Transition explorer")
    e1, e2 = st.columns([1, 2])
    with e1:
        s_sel = st.selectbox("State s", list(range(stc.n_grid)), index=3 * stc.n_left + 3,
                             format_func=lambda s: state_label(s, stc))
        a_sel = st.radio("Action a", range(4), format_func=lambda a: f"{ACTION_GLYPH[a]} {ACTIONS[a]}")
        st.markdown(f"""<div class='kv'>samples for (s,a): {int(M.counts[s_sel, a_sel])}<br>
R(s,a) = {M.R[s_sel, a_sel]:+.3f}<br>Q*(s,a) = {vi.Q[s_sel, a_sel]:+.3f}<br>
π*(s) = {ACTIONS[vi.pi[s_sel]]}<br>expert(s) = {ACTIONS[pe[s_sel]]}<br>
data: action observed here = {', '.join(ACTIONS[i] for i in np.flatnonzero(D['emp']['observed'][s_sel])) or 'state never visited'}</div>""",
                    unsafe_allow_html=True)
    with e2:
        st.plotly_chart(plots.transition_heatmap(M.P[s_sel, a_sel], stc, f"P(s′ | s, {ACTION_SHORT[a_sel]})"), width=FULL)
    st.markdown("#### Output: optimal value function")
    st.dataframe(res["value_table"], width=FULL, height=300, hide_index=True)
    st.download_button("Download optimal_value_function.csv", res["value_table"].to_csv(index=False),
                       "optimal_value_function.csv", "text/csv")

# ================================================================== ADP
with tabs[3]:
    a = res["adp"]
    st.markdown("#### Why approximate?")
    st.markdown(f"""
The tabular MDP needs a discretised state. Its table grows as bins^sensors: {stc.n_grid} cells for two sensors,
about 10⁴ for four sensors at 10 bins, and about 10²⁴ for the 24 raw sensors. Approximate dynamic programming
replaces the table with a parametric function of the **continuous** readings and fits it to sampled Bellman targets
(fitted Q-iteration). Here φ(x) is a {cfg.solver.rbf_front} × {cfg.solver.rbf_left} grid of normalised Gaussian radial
basis functions over log-distance, giving {a.n_params} parameters in total. It trains on the same
{res['model'].n_samples:,} simulated transitions as the MDP.
""")
    st.latex(r"\hat Q(x,a)=\phi(x)^{\top}\theta_a,\qquad y_i=r_i+\gamma(1-d_i)\max_{a'}\phi(x_i')^{\top}\theta_{a'},\qquad \theta_a\leftarrow\arg\min_\theta\sum_{i:a_i=a}(y_i-\phi(x_i)^\top\theta)^2+\lambda\lVert\theta\rVert^2")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fitted iterations", a.iterations)
    m2.metric("Converged", "yes" if a.converged else "no")
    m3.metric("Parameters", a.n_params)
    m4.metric("Fit time", f"{a.seconds:.1f} s")
    st.plotly_chart(plots.convergence({"max |ΔQ̂| on samples": (a.deltas, METHOD_COLOR["adp"]),
                                       "RMS Bellman residual": (a.bellman_residual, "#6B7280")},
                                      "Fitted Q-iteration convergence", "value (log)", hline=cfg.solver.adp_tol), width=FULL)
    v1, v2 = st.columns(2)
    v1.plotly_chart(plots.state_grid(res["adp_cell_values"], np.append(res["adp_cell_policy"], 0), stc,
                                     "ADP: V̂ averaged per cell, majority action"), width=FULL)
    diff = res["adp_cell_values"] - res["vi"].V
    v2.plotly_chart(plots.state_grid(diff, None, stc, "V̂ − V* (approximation gap per cell)", colorscale="RdBu", zmid=0,
                                     fmt="{:+.1f}", colorbar_title="Δ"), width=FULL)
    w1, w2 = st.columns(2)
    w1.plotly_chart(plots.actual_vs_approx(res["vi"].V, res["adp_cell_values"]), width=FULL)
    w2.plotly_chart(plots.state_grid(res["model"].counts.sum(axis=1), None, stc,
                                     "State-space coverage: training samples per cell", colorscale="Blues", fmt="{:.0f}",
                                     colorbar_title="samples"), width=FULL)
    # continuous decision map
    ff, ll = np.meshgrid(np.linspace(0.45, 2.6, 160), np.linspace(0.33, 1.6, 160), indexing="ij")
    grid = np.column_stack([ff.ravel(), ll.ravel(), ff.ravel(), ff.ravel()])
    pols = build_policies(res, cfg)
    import plotly.graph_objects as go
    fig = go.Figure()
    for i in range(4):
        m = (pols["adp"].act(grid) == i).reshape(ff.shape)
        fig.add_trace(go.Heatmap(z=np.where(m, 1, np.nan), x=ll[0], y=ff[:, 0], showscale=False,
                                 colorscale=[[0, plots.ACTION_COLOR[i]], [1, plots.ACTION_COLOR[i]]], opacity=0.8,
                                 name=ACTION_SHORT[i], hovertemplate=f"{ACTION_SHORT[i]}<extra></extra>"))
    tf, tlo, thi = EXPERT_THRESHOLDS
    for x0, x1, y0, y1 in ((0.33, 1.6, tf, tf), (tlo, tlo, tf, 2.6), (thi, thi, tf, 2.6)):
        fig.add_shape(type="line", x0=x0, x1=x1, y0=y0, y1=y1, line=dict(color="black", dash="dash", width=1.5))
    plots._layout(fig, "ADP decision regions on continuous readings (dashed: recorded controller's thresholds)", 420)
    fig.update_xaxes(title="SD_left (m)")
    fig.update_yaxes(title="SD_front (m)")
    st.plotly_chart(fig, width=FULL)
    st.markdown("<span class='note'>Colours: teal forward, amber slight right, red sharp right, blue slight left. "
                "Because it is not tied to bin edges, ADP can place its boundaries anywhere; its boundaries come out "
                "smooth rather than rectangular.</span>", unsafe_allow_html=True)

# ================================================================== POLICY SEARCH
with tabs[4]:
    mc, hj = res["mc"], res["hj"]
    st.markdown(f"""
Both searches optimise the **same three-parameter controller** that produced the dataset,
θ = (τ_front, τ_left-low, τ_left-high), starting from the recorded values ({', '.join(f'{x:.3f}' for x in EXPERT_THRESHOLDS)}).
The objective is the expected discounted return J(θ) from the start distribution, estimated by simulating
{cfg.solver.ps_episodes} episodes of {cfg.solver.ps_horizon} decisions. The same seeds are used for every θ (common
random numbers), which makes J a deterministic function of θ and removes noise from the comparisons.
Bounds: τ_front ∈ [{BOUNDS[0,0]}, {BOUNDS[0,1]}], τ_low ∈ [{BOUNDS[1,0]}, {BOUNDS[1,1]}], τ_high ∈ [{BOUNDS[2,0]}, {BOUNDS[2,1]}],
with τ_high − τ_low ≥ 0.05 m.
""")
    st.markdown("#### Monte Carlo policy search")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.plotly_chart(plots.mc_scatter(mc), width=FULL)
    with c2:
        top = pd.DataFrame({"τ_front": mc.candidates[mc.top_idx, 0], "τ_low": mc.candidates[mc.top_idx, 1],
                            "τ_high": mc.candidates[mc.top_idx, 2],
                            f"J stage 1 ({cfg.solver.ps_episodes} ep.)": mc.stage1[mc.top_idx],
                            f"J stage 2 ({cfg.solver.mc_final_episodes} ep.)": mc.stage2,
                            "± 95% CI": mc.stage2_ci}).round(3)
        st.dataframe(top, width=FULL, hide_index=True)
        st.markdown(f"""<span class='note'>Stage 1 scores {len(mc.candidates)} random feasible θ (the recorded
controller included as candidate 0). Stage 2 re-scores the top {cfg.solver.mc_top} on new seeds with more episodes,
because the stage-1 winner is biased upwards by selection. Selected θ = ({', '.join(f'{x:.3f}' for x in mc.best_theta)}).
Budget: {mc.episodes_used:,} episodes, {mc.seconds:.1f} s.</span>""", unsafe_allow_html=True)
    q1, q2 = st.columns(2)
    q1.plotly_chart(plots.mc_progress(mc), width=FULL)
    q2.plotly_chart(plots.return_histograms(res["returns"]), width=FULL)
    st.latex(r"G^{(i)}=\sum_{t=0}^{H-1}\gamma^{t}r^{(i)}_t,\qquad \hat J(\theta)=\frac1N\sum_{i=1}^{N}G^{(i)},\qquad \text{CI}_{95\%}=\hat J\pm1.96\,\frac{s_G}{\sqrt N}")
    st.markdown("#### Hooke–Jeeves pattern search")
    h1, h2 = st.columns(2)
    h1.plotly_chart(plots.hj_trace(hj.log), width=FULL)
    h2.plotly_chart(plots.hj_params(hj.log), width=FULL)
    st.markdown(f"""<span class='note'>Exploratory moves try ±Δ on each threshold in turn (Δ = {cfg.solver.hj_step}
m); after a successful exploration a pattern move jumps to x + (x − x_base) and is kept only if exploring
around it improves J. When no move helps, Δ halves. Stop: {hj.stop_reason}. Result θ =
({', '.join(f'{x:.3f}' for x in hj.best_theta)}) after {hj.evaluations} evaluations ({hj.episodes_used:,} episodes,
{hj.seconds:.1f} s).</span>""", unsafe_allow_html=True)
    st.plotly_chart(plots.hj_steps(hj), width=FULL)
    with st.expander("Hooke–Jeeves evaluation log"):
        st.dataframe(pd.DataFrame(hj.log).round(4), width=FULL, height=300, hide_index=True)
    fresh = res["eval_short"]
    st.info(f"Selection bias check: on the search seeds Hooke–Jeeves reached J = {hj.best_J:.2f} and Monte Carlo "
            f"J = {mc.best_J:.2f}. On {cfg.evaluation.n_episodes} fresh episodes they score {fresh['hj']['return_mean']:.2f} "
            f"and {fresh['mcps']['return_mean']:.2f}. The Comparison tab only uses the fresh-seed numbers.")

# ================================================================== COMPARISON
with tabs[5]:
    S, L = res["eval_short"], res["eval_long"]
    ev = cfg.evaluation
    st.markdown(f"""<span class='note'>Evaluation protocol: every controller runs in the same simulator, from the same
{ev.n_episodes} start poses (70% on the wall-following track, 30% anywhere in the room), with the same sensor and motion
noise, for {ev.horizon} decisions ({ev.horizon / 3:.0f} s), under the same reward and γ = {cfg.solver.gamma}. A second set of
{ev.long_episodes} runs of {ev.long_horizon} decisions ({ev.long_horizon / 3:.0f} s) measures long-run behaviour. None of these
seeds were used for training or search. Numbers are simulation estimates, reported with 95% intervals.</span>""",
                unsafe_allow_html=True)
    learned = ("mdp", "adp", "mcps", "hj")
    kb = max(learned, key=lambda k: S[k]["return_mean"])
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Best observed mean return", f"{S[kb]['return_mean']:.2f}", LABELS[kb], delta_color="off")
    c2.metric("Recorded controller", f"{S['expert']['return_mean']:.2f}")
    c3.metric("Success rate, learned methods", f"{min(1 - L[k]['crash_rate'] for k in learned):.0%}–{max(1 - L[k]['crash_rate'] for k in learned):.0%}")
    c4.metric("Decisions per 300 s run", f"{ev.long_horizon}")
    c5.metric("Fastest training", f"{min(res['runtime'][k]['seconds'] for k in learned):.1f} s")
    c6.metric("VI sweeps / ADP iterations", f"{res['vi'].iterations} / {res['adp'].iterations}")

    st.markdown("#### Summary table")
    conv = {"mdp": ("yes" if res["vi"].converged else "no") + f" (‖ΔV‖∞ < {cfg.solver.tol:g})",
            "adp": ("yes" if res["adp"].converged else "no") + f" (max|ΔQ̂| < {cfg.solver.adp_tol:g})",
            "mcps": "search completes (fixed budget)", "hj": f"yes ({res['hj'].stop_reason})",
            "expert": "—", "random": "—"}
    iters = {"mdp": f"{res['vi'].iterations} sweeps", "adp": f"{res['adp'].iterations} fits",
             "mcps": f"{len(res['mc'].candidates)} + {cfg.solver.mc_top} evaluations", "hj": f"{res['hj'].evaluations} evaluations",
             "expert": "—", "random": "—"}
    objective = {"mdp": "max V(s) on estimated model", "adp": "max Q̂(x,a) on sampled Bellman targets",
                 "mcps": "max Ĵ(θ), random candidates", "hj": "max Ĵ(θ), pattern search", "expert": "none (recorded rule)",
                 "random": "none"}
    approx = {"mdp": "exact on a discretised model", "adp": "yes: 324-parameter RBF value function",
              "mcps": "sampled objective; 3 parameters", "hj": "sampled objective; 3 parameters", "expert": "—", "random": "—"}
    st.dataframe(pd.DataFrame([{
        "Algorithm": LABELS[k], "Objective": objective[k],
        "Discounted return": f"{S[k]['return_mean']:.2f} ± {S[k]['return_ci95']:.2f}",
        "Δ vs expert (paired)": "—" if k == "expert" else f"{S[k]['diff_vs_expert']:+.2f} ± {S[k]['diff_ci95']:.2f}",
        "Success (300 s)": f"{1 - L[k]['crash_rate']:.0%}", "Reward / step": f"{L[k]['reward_per_step']:.3f}",
        "In band": f"{L[k]['band_share']:.0%}", "In danger": f"{L[k]['danger_share']:.1%}",
        "Distance (m)": f"{L[k]['distance_m']:.1f}", "Switches / step": f"{L[k]['switch_rate']:.2f}",
        "Runtime": f"{res['runtime'][k]['seconds']:.1f} s", "Iterations": iters[k], "Converged?": conv[k],
        "Approximation?": approx[k]} for k in METHOD_KEYS]), width=FULL, hide_index=True)

    st.markdown("#### Method characteristics")
    st.dataframe(pd.DataFrame([
        {"Method": LABELS["mdp"], "Core idea": "Bellman optimality on an estimated P and R over 43 states",
         "Exact / approximate": "Exact for the model; the model aggregates states",
         "Computational cost": f"{res['runtime']['mdp']['seconds']:.1f} s (sampling dominates; VI {res['vi'].seconds * 1000:.0f} ms)",
         "Convergence": f"Guaranteed (contraction); bound {res['vi'].error_bound:.1e}",
         "Scalability": "Poor: table grows as bins^sensors", "Interpretability": "High: one action per cell, Q-values visible",
         "Final return": f"{S['mdp']['return_mean']:.2f}"},
        {"Method": LABELS["adp"], "Core idea": "Fit Q̂(x,a)=φ(x)ᵀθ to sampled Bellman targets",
         "Exact / approximate": "Approximate (324 parameters, continuous input)",
         "Computational cost": f"{res['runtime']['adp']['seconds']:.1f} s", "Convergence": "Observed here; not guaranteed in general",
         "Scalability": "Good: parameters grow with features, not states", "Interpretability": "Medium: smooth regions, no rule",
         "Final return": f"{S['adp']['return_mean']:.2f}"},
        {"Method": LABELS["mcps"], "Core idea": "Score random θ by simulated return; re-check the best",
         "Exact / approximate": "Sampled objective, 3 parameters",
         "Computational cost": f"{res['runtime']['mcps']['seconds']:.1f} s · {res['mc'].episodes_used:,} episodes",
         "Convergence": "Improves with more candidates; no step-wise guarantee", "Scalability": "Embarrassingly parallel; weak in many dimensions",
         "Interpretability": "High: three thresholds", "Final return": f"{S['mcps']['return_mean']:.2f}"},
        {"Method": LABELS["hj"], "Core idea": "Coordinate probes + pattern moves on θ, step halving",
         "Exact / approximate": "Sampled objective (fixed seeds), 3 parameters",
         "Computational cost": f"{res['runtime']['hj']['seconds']:.1f} s · {res['hj'].episodes_used:,} episodes",
         "Convergence": "To a local optimum of Ĵ", "Scalability": "Evaluations grow linearly with parameters",
         "Interpretability": "High: three thresholds", "Final return": f"{S['hj']['return_mean']:.2f}"},
    ]), width=FULL, hide_index=True)

    sig = [LABELS[k] for k in learned if S[k]["diff_vs_expert"] - S[k]["diff_ci95"] > 0]
    crashers = [LABELS[k] for k in METHOD_KEYS if k != "random" and L[k]["crash_rate"] > 0]
    st.markdown(f"""
**What the measurements show.** The paired difference against the recorded controller has a 95% interval above zero
for {len(sig)} of the 4 learned methods{(': ' + ', '.join(sig)) if sig else ''}. Where the intervals of two methods
overlap, the data do not separate them. The differences come from time spent in the tracking band (right-hand chart)
{"rather than from collisions: no method other than the random baseline collides." if not crashers else f"and from collisions, which occur for {', '.join(crashers)}."}
""")
    c1, c2 = st.columns(2)
    c1.plotly_chart(plots.paired_diff(S), width=FULL)
    c2.plotly_chart(plots.metric_bars(L, "band_share", "Share of time in the tracking band (300 s runs)", pct=True), width=FULL)
    c3, c4 = st.columns(2)
    c3.plotly_chart(plots.return_distribution(res["returns"]), width=FULL)
    c4.plotly_chart(plots.metric_bars(L, "reward_per_step", "Average reward per step (300 s runs)"), width=FULL)
    c5, c6 = st.columns(2)
    c5.plotly_chart(plots.metric_bars(L, "crash_rate", "Crash rate (300 s runs)", pct=True), width=FULL)
    rt = {k: {"seconds": res["runtime"][k]["seconds"]} for k in METHOD_KEYS}
    c6.plotly_chart(plots.metric_bars(rt, "seconds", "Computation time (s)"), width=FULL)
    c7, c8 = st.columns(2)
    c7.plotly_chart(plots.action_mix(L), width=FULL)
    c8.plotly_chart(plots.metric_bars(L, "distance_m", "Distance travelled in 300 s (m)"), width=FULL)
    st.markdown("#### Convergence of each method")
    st.plotly_chart(plots.convergence_overview(res), width=FULL)
    st.markdown("#### Policy maps on the MDP grid")
    F_, L_ = representative_readings(stc)
    maps = {"expert": res["pi_expert"][:-1], "mdp": res["vi"].pi[:-1], "adp": res["adp_cell_policy"],
            "mcps": threshold_actions(F_, L_, res["mc"].best_theta), "hj": threshold_actions(F_, L_, res["hj"].best_theta)}
    st.plotly_chart(plots.policy_maps(maps, stc), width=FULL, config={"displayModeBar": False})
    st.caption("Threshold policies are shown at each cell's midpoint; their real boundaries fall between bin edges. "
               "ADP shows the majority action over the sampled readings in each cell.")
    st.markdown("#### Same start, six controllers")
    eps = {k: get_episode(res, cfg, k, start_pose(start_name, seed, cfg), seed, horizon) for k in METHOD_KEYS}
    st.plotly_chart(plots.arena_paths(eps, 240), width=FULL, config={"displayModeBar": False})
    st.markdown("#### How good is the model itself?")
    mg = res["model_gap"]
    st.markdown(f"""
The MDP predicts E[V(s₀)] = {mg['mdp']['predicted']:.2f} for its own policy; simulated, it earns
{mg['mdp']['simulated']:.2f}. For the recorded controller the prediction is {mg['expert']['predicted']:.2f}, against
{mg['expert']['simulated']:.2f} simulated. The gap is the price of state aggregation: two readings in the same cell
can hide different headings, so the Markov assumption holds only approximately. Methods that act on continuous
readings (ADP and the threshold searches) do not pay this price, which is consistent with their higher measured return.
""")
    with st.expander("Observed trade-offs"):
        st.markdown("""
* **Hooke–Jeeves** changes only three numbers in the robot's original rule, so the result stays fully interpretable.
  It needs a noise-free objective (fixed seeds) and finds a local optimum.
* **Monte Carlo search** optimises the same three numbers without assuming smoothness and parallelises trivially,
  but most of its budget is spent on poor candidates.
* **ADP** handles continuous inputs and would scale to more sensors; fitted iteration is not guaranteed to converge
  for every feature set, and its decision regions have no compact rule.
* **Value iteration** is exact for the model, converges in milliseconds with an error bound and makes the problem's
  structure visible, but it inherits the discretisation's limits.
""")

# ================================================================== NOTEBOOK AUDIT
with tabs[6]:
    au = get_audit()
    st.markdown("The starter notebook `Day_5.ipynb` was re-executed block by block and each result was checked "
                "against an independent calculation.")
    vi_a, mc_a, hj_a, pf = au["vi"], au["mc"], au["hj"], au["portfolio"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Value iteration output", "verified", f"max error {vi_a['max_error_vs_exact']:.0e}")
    c2.metric("Random-walk value (exact)", f"{mc_a['exact_value']:.2f}", f"notebook {mc_a['notebook_estimate']:.2f}")
    c3.metric("Hooke–Jeeves from (1.3, −0.7)", str(np.round(hj_a['non_grid_start_result'], 2)), "true minimum (0, 0)",
              delta_color="off")
    c4.metric("Portfolio output", "reproduced" if pf["matches_notebook"] else "differs", "objective ignores asset means",
              delta_color="off")
    for title, text in au["findings"]:
        st.markdown(f"**{title}.** {text}")
    st.markdown(f"""
**Checks in numbers.** Value iteration: the notebook's [68.068, 62.623, 67.133] equals the exact linear solve of the
greedy policy ({', '.join(vi_a['greedy_policy'])}); a tolerance of 10⁻⁶ would have stopped after
{vi_a['sweeps_needed_tol_1e-6']} sweeps instead of 1,000. Monte Carlo: the exact expected return of the random walk is
{mc_a['exact_value']:.3f}; twenty seeded repeats of the notebook's 1,000-episode estimate average
{mc_a['seeded_estimates_mean']:.2f} with s.d. {mc_a['seeded_estimates_sd']:.2f}, so −32.93 is ordinary sampling noise
(z = {mc_a['notebook_z_score']:.2f}). Hooke–Jeeves: integer start [1, 1] returns {hj_a['integer_start_result'].tolist()}.
Portfolio: the grand mean used as "return" is {pf['grand_mean_used']:.4f} for every portfolio; per-asset means range
from {pf['per_asset_means'].min():.4f} to {pf['per_asset_means'].max():.4f}.

**What STOCHOS changes.** Value iteration stops on the slide's convergence threshold, reports the error bound,
extracts the policy and writes `optimal_value_function.csv`. The ADP label is used only for a genuinely approximate
method. Monte Carlo policy search actually searches. Hooke–Jeeves evaluates the pattern point before accepting it,
works in floats, respects bounds and logs every evaluation. All randomness is seeded.
""")
    with st.expander("Original notebook source"):
        st.code(legacy_audit.notebook_source(), language="python")

# ================================================================== NOTES
with tabs[7]:
    st.markdown(f"""
#### Pipeline
1. **Data** → controller rule, state bins, reward zones, speed, sensor cap and spike rate (`rdmu/data.py`).
2. **Digital twin** → a unicycle robot with 24 simulated sonar beams in a polygonal room (`rdmu/twin.py`),
   checked against the recording.
3. **Model** → P(s′|s,a) and R(s,a) estimated by simulating all four actions from {res['model'].n_samples // 4:,} poses (`rdmu/model.py`).
4. **Methods** → value iteration (`mdp.py`), fitted Q-iteration (`adp.py`), Monte Carlo search and Hooke–Jeeves (`policy_search.py`).
5. **Evaluation** → every controller runs from the same starts with the same noise on seeds never used in training (`evaluation.py`).

#### Assumptions (each is a parameter in `rdmu/config.py`)
| Item | Value | Basis |
|---|---|---|
| Decision interval | 1/3 s (3 samples) | assumption, 9 Hz sampling from data |
| Forward speed | {cfg.twin.forward_speed} m/s | data: front-distance closing speed while moving forward |
| Turn rates (slight / sharp) | {abs(cfg.twin.turn_rate[1])} / {abs(cfg.twin.turn_rate[2])} rad/s | calibrated to the recording's sensor medians |
| Robot radius | {cfg.twin.robot_radius} m | SCITOS-G5 footprint |
| Sensor noise / spike rate | 0.5 cm + 1% / {cfg.twin.spike_prob:.1%} | assumption / data |
| Room | 6.4 × 4.2 m with a pillar and a recess | assumption; the real plan is not published |
| Reward values | +1 / −0.5 / −2 / −25 | assumption; zone thresholds from data |
| γ, tolerance | {cfg.solver.gamma}, {cfg.solver.tol:g} | slide inputs, adjustable in the sidebar |

#### Academic interpretation of each method
| | MDP · value iteration | ADP · fitted Q-iteration | Monte Carlo policy search | Hooke–Jeeves |
|---|---|---|---|---|
| Problem solved | Optimal action per sensor cell | Optimal action for continuous readings | Best thresholds for the recorded rule | Best thresholds for the recorded rule |
| Use of uncertainty | Expectation over estimated P(s′ given s,a) | Expectation through sampled targets | Average over simulated episodes | Average over simulated episodes (fixed seeds) |
| Future consequences | γ-discounted V(s′) inside the Bellman update | γ max Q̂(x′,·) in every target | Discounted episode return | Discounted episode return |
| Key assumption | (front, left) cell is Markov | Features can represent Q | Controller family is adequate | Objective is locally well-behaved |
| Strength | Exact, fast, error bound | Continuous, scalable | Global, parallel, simple | Few evaluations, interpretable result |
| Limitation | Discretisation error | No general convergence guarantee | Wasteful sampling | Local optimum only |
| Read the result as | Best policy *for the model* | Smooth approximation of the optimum | Best of the sampled θ | Local optimum near the recorded θ |

#### Computational complexity
| Method | Cost | Here |
|---|---|---|
| Value iteration | O(k · S² · A) | k = {res['vi'].iterations}, S = {stc.n_states}, A = 4 → milliseconds |
| Model estimation | O(poses · A · beams · walls) | {res['model'].n_samples:,} simulated moves, vectorised → about 3 s |
| Fitted Q-iteration | O(k · N · d) per action after one ridge factorisation | k = {res['adp'].iterations}, N = {res['model'].n_samples:,}, d = {res['adp'].theta.shape[0]} |
| Monte Carlo search | O(K · N_ep · H) | {res['mc'].episodes_used:,} episodes × {cfg.solver.ps_horizon} steps |
| Hooke–Jeeves | O(evaluations · N_ep · H) | {res['hj'].episodes_used:,} episodes × {cfg.solver.ps_horizon} steps |

Expensive results are cached (`st.cache_resource` / `st.cache_data`), and default results ship precomputed.

#### Reproducibility
Every random draw comes from a seeded NumPy generator. Training seed: {cfg.solver.seed}; evaluation seed:
{cfg.evaluation.seed}; the robot demo uses the sidebar seed. The same settings always reproduce the same numbers.

#### Limitations
* The simulator is calibrated, not identified: it matches the recording's distances but under-uses sharp turns,
  so absolute numbers are simulator-specific. The ranking of methods is the transferable finding.
* The (front, left) state is not fully Markov because heading is hidden; the model-gap section quantifies this.
* Policy search optimises a fixed controller family; a richer family could do better but would be harder to explain.
""")

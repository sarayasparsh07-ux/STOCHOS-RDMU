"""Plotly figures used by the Streamlit app (one consistent visual language)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import ACTION_GLYPH, ACTION_SHORT, ACTIONS, METHOD_COLOR, METHOD_KEYS, METHOD_LABEL
from .statespace import ZONE_COLOR, ZONES, front_labels, left_labels
from .twin import BEAM_ANGLES, ROOM_BOUNDS, SCITOS_ROOM, arc_beams

INK = "#1B2433"
MUTED = "#6B7385"
GRID = "#E3E7EE"
PAPER = "#FBFCFD"
FONT = "IBM Plex Sans, Inter, Segoe UI, sans-serif"
ACTION_COLOR = ("#0F8B8D", "#D08C0B", "#C2364B", "#3552C7")
ARC_COLOR = {"front": "#C2364B", "left": "#0F8B8D", "right": "#9AA3B2", "back": "#9AA3B2"}


def _layout(fig, title=None, height=380, **kw):
    margin = kw.pop("margin", dict(l=10, r=10, t=44 if title else 12, b=10))
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=INK)) if title else None,
        height=height, margin=margin,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PAPER,
        font=dict(family=FONT, size=12, color=INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font_family=FONT), **kw)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


# ------------------------------------------------------------------ robot arena
def _room_trace():
    r = np.vstack([SCITOS_ROOM, SCITOS_ROOM[:1]])
    return go.Scatter(x=r[:, 0], y=r[:, 1], mode="lines", fill="toself", fillcolor="#F1F4F8",
                      line=dict(color=INK, width=3), hoverinfo="skip", showlegend=False)


def _robot_shapes(pose, beams, radius, arcs):
    x, y, h = pose
    t = np.linspace(0, 2 * np.pi, 28)
    body = (x + radius * np.cos(t), y + radius * np.sin(t))
    nose = ([x, x + radius * 1.25 * np.cos(h)], [y, y + radius * 1.25 * np.sin(h)])
    rays = {}
    for name, idx in zip(("front", "left", "right", "back"), arcs):
        xs, ys = [], []
        for b in idx:
            ang = h + BEAM_ANGLES[b]
            r0 = radius
            r1 = radius + min(beams[b], 2.5)
            xs += [x + r0 * np.cos(ang), x + r1 * np.cos(ang), None]
            ys += [y + r0 * np.sin(ang), y + r1 * np.sin(ang), None]
        rays[name] = (xs, ys)
    return body, nose, rays


def arena_animation(ep: dict, radius: float, color: str, title: str, frame_ms: int = 90,
                    stride: int = 1, height: int = 520):
    arcs = arc_beams(60.0)
    poses = ep["pose"]
    n = len(poses)
    idx = list(range(0, n, stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    path = np.vstack([poses[:, :2], ep["final_pose"][None, :2]])

    def frame_data(i):
        body, nose, rays = _robot_shapes(poses[i], ep["beams"][i], radius, arcs)
        lo = max(0, i - 45)
        z = int(ep["zone"][i])
        data = [
            go.Scatter(x=rays["right"][0] + rays["back"][0], y=rays["right"][1] + rays["back"][1], mode="lines",
                       line=dict(color="#B8C0CC", width=1)),
            go.Scatter(x=rays["left"][0], y=rays["left"][1], mode="lines", line=dict(color=ARC_COLOR["left"], width=2)),
            go.Scatter(x=rays["front"][0], y=rays["front"][1], mode="lines", line=dict(color=ARC_COLOR["front"], width=2)),
            go.Scatter(x=path[lo:i + 1, 0], y=path[lo:i + 1, 1], mode="lines", line=dict(color=color, width=4)),
            go.Scatter(x=body[0], y=body[1], mode="lines", fill="toself", fillcolor=color,
                       line=dict(color=INK, width=1.5), opacity=0.9),
            go.Scatter(x=nose[0], y=nose[1], mode="lines", line=dict(color="white", width=3)),
        ]
        sd = ep["sd"][i]
        txt = (f"t = {i / 3:5.1f} s   ·   step {i}<br>{ACTION_GLYPH[ep['action'][i]]} {ACTIONS[ep['action'][i]]}"
               f"   ·   front {sd[0]:.2f} m  left {sd[1]:.2f} m   ·   {ZONES[z]}"
               f"   ·   r = {ep['reward'][i]:+.2f}   Σ = {ep['reward'][: i + 1].sum():+.1f}")
        return data, txt

    d0, t0 = frame_data(0)
    zc = [ZONE_COLOR[int(z)] for z in ep["zone"]]
    base = [_room_trace(),
            go.Scatter(x=path[1:, 0], y=path[1:, 1], mode="markers", marker=dict(color=zc, size=4, opacity=0.55),
                       hoverinfo="skip")]
    frames = []
    for i in idx:
        d, t = frame_data(i)
        frames.append(go.Frame(data=d, traces=list(range(2, 2 + len(d))), name=str(i),
                               layout=dict(annotations=[dict(text=t, x=0, y=1.07, xref="paper", yref="paper",
                                                             showarrow=False, xanchor="left", align="left",
                                                             font=dict(size=12, color=INK))])))
    fig = go.Figure(data=base + d0, frames=frames)
    for tr in fig.data:
        tr.update(showlegend=False, hoverinfo="skip")
    fig.update_layout(
        annotations=[dict(text=t0, x=0, y=1.07, xref="paper", yref="paper", showarrow=False,
                          xanchor="left", align="left", font=dict(size=12, color=INK))],
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.02, xanchor="left", yanchor="top",
                          pad=dict(t=6, r=6), bgcolor="#FFFFFF", bordercolor=GRID,
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=dict(duration=frame_ms, redraw=False),
                                                         transition=dict(duration=0), fromcurrent=True, mode="immediate")]),
                                   dict(label="⏸ Pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(active=0, x=0.2, len=0.8, y=-0.02, yanchor="top", pad=dict(t=6),
                      currentvalue=dict(visible=False), tickcolor=GRID, font=dict(size=9),
                      steps=[dict(label=str(i), method="animate",
                                  args=[[str(i)], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                             for i in idx])],
    )
    _layout(fig, None, height, margin=dict(l=10, r=10, t=56, b=60))
    fig.update_xaxes(range=[-0.25, ROOM_BOUNDS[0] + 0.25], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[-0.25, ROOM_BOUNDS[1] + 0.25], showgrid=False, zeroline=False, visible=False,
                     scaleanchor="x", scaleratio=1)
    return fig


def arena_paths(episodes: dict, height=330):
    """Small multiples: one trajectory per method from the same start."""
    keys = list(episodes)
    fig = make_subplots(rows=2, cols=3, subplot_titles=[METHOD_LABEL[k] for k in keys],
                        horizontal_spacing=0.02, vertical_spacing=0.08)
    for j, k in enumerate(keys):
        r, c = divmod(j, 3)
        ep = episodes[k]
        rm = np.vstack([SCITOS_ROOM, SCITOS_ROOM[:1]])
        fig.add_trace(go.Scatter(x=rm[:, 0], y=rm[:, 1], mode="lines", line=dict(color=INK, width=2),
                                 fill="toself", fillcolor="#F1F4F8", hoverinfo="skip"), r + 1, c + 1)
        p = np.vstack([ep["pose"][:, :2], ep["final_pose"][None, :2]])
        fig.add_trace(go.Scatter(x=p[:, 0], y=p[:, 1], mode="lines", line=dict(color=METHOD_COLOR[k], width=2),
                                 hoverinfo="skip"), r + 1, c + 1)
        end = "✕" if ep["crashed"][-1] else "●"
        fig.add_trace(go.Scatter(x=[p[-1, 0]], y=[p[-1, 1]], mode="text", text=[end],
                                 textfont=dict(size=16, color=METHOD_COLOR[k])), r + 1, c + 1)
    fig.update_xaxes(visible=False, range=[-0.2, ROOM_BOUNDS[0] + 0.2])
    fig.update_yaxes(visible=False, range=[-0.2, ROOM_BOUNDS[1] + 0.2])
    for i in range(1, 7):
        fig.update_yaxes(scaleanchor=f"x{i if i > 1 else ''}", row=(i - 1) // 3 + 1, col=(i - 1) % 3 + 1)
    _layout(fig, None, height * 2)
    fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
    return fig


# --------------------------------------------------------------- state grids
def state_grid(values, pi, cfg, title, colorscale="Tealgrn", zmid=None, fmt="{:.1f}", height=360,
               colorbar_title="value"):
    z = np.asarray(values[: cfg.n_grid], float).reshape(cfg.n_front, cfg.n_left)
    text = []
    for f in range(cfg.n_front):
        row = []
        for l in range(cfg.n_left):
            v = z[f, l]
            g = ACTION_GLYPH[int(pi[f * cfg.n_left + l])] if pi is not None else ""
            row.append(f"<b>{g}</b><br>{'' if np.isnan(v) else fmt.format(v)}")
        text.append(row)
    fig = go.Figure(go.Heatmap(z=z, x=left_labels(cfg), y=front_labels(cfg), text=text, texttemplate="%{text}",
                               colorscale=colorscale, zmid=zmid, hovertemplate="front %{y}<br>left %{x}<br>%{z:.3f}<extra></extra>",
                               colorbar=dict(title=colorbar_title, thickness=10)))
    _layout(fig, title, height)
    fig.update_xaxes(title="SD_left bin (m)", side="bottom")
    fig.update_yaxes(title="SD_front bin (m)")
    return fig


def zone_grid(zones, cfg, title="Reward zones", height=320):
    z = np.asarray(zones[: cfg.n_grid]).reshape(cfg.n_front, cfg.n_left)
    cs = []
    n = len(ZONES)
    for i, c in enumerate(ZONE_COLOR):
        cs += [[i / n, c], [(i + 1) / n, c]]
    fig = go.Figure(go.Heatmap(z=z, x=left_labels(cfg), y=front_labels(cfg), zmin=-0.5 + 0.5, zmax=n,
                               colorscale=cs, showscale=False, text=[[ZONES[v] for v in r] for r in z],
                               texttemplate="%{text}", textfont=dict(size=10, color="white"),
                               hovertemplate="front %{y}<br>left %{x}<br>%{text}<extra></extra>"))
    fig.data[0].update(z=z + 0.5)
    _layout(fig, title, height)
    fig.update_xaxes(title="SD_left bin (m)")
    fig.update_yaxes(title="SD_front bin (m)")
    return fig


def transition_heatmap(P_row, cfg, title, height=340):
    z = P_row[: cfg.n_grid].reshape(cfg.n_front, cfg.n_left)
    fig = go.Figure(go.Heatmap(z=z, x=left_labels(cfg), y=front_labels(cfg), colorscale="Blues", zmin=0,
                               text=np.where(z > 0.005, np.round(z, 2).astype(str), ""), texttemplate="%{text}",
                               hovertemplate="→ front %{y}, left %{x}<br>P = %{z:.3f}<extra></extra>",
                               colorbar=dict(title="P", thickness=10)))
    _layout(fig, f"{title}   ·   P(crash) = {P_row[-1]:.3f}", height)
    fig.update_xaxes(title="next SD_left bin (m)")
    fig.update_yaxes(title="next SD_front bin (m)")
    return fig


# ------------------------------------------------------------------ curves
def convergence(series: dict, title, ytitle, log=True, height=320, hline=None):
    fig = go.Figure()
    for name, (y, color) in series.items():
        fig.add_trace(go.Scatter(x=np.arange(1, len(y) + 1), y=y, mode="lines", name=name, line=dict(color=color, width=2)))
    if hline is not None:
        fig.add_hline(y=hline, line=dict(color=MUTED, dash="dash"), annotation_text=f"tolerance {hline:g}",
                      annotation_position="top right")
    _layout(fig, title, height)
    fig.update_xaxes(title="iteration")
    fig.update_yaxes(title=ytitle, type="log" if log else "linear")
    return fig


def scores_bar(scores, chosen, label, height=230):
    colors = [ACTION_COLOR[i] if i == chosen else "#CBD2DC" for i in range(4)]
    fig = go.Figure(go.Bar(x=list(scores), y=[f"{ACTION_GLYPH[i]} {ACTION_SHORT[i]}" for i in range(4)],
                           orientation="h", marker_color=colors, text=[f"{v:.2f}" for v in scores],
                           textposition="auto", hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    _layout(fig, None, height)
    fig.update_xaxes(title=label)
    fig.update_yaxes(autorange="reversed")
    return fig


def episode_timeline(ep, color, height=300):
    t = np.arange(len(ep["reward"])) / 3
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.55, 0.45])
    fig.add_trace(go.Scatter(x=t, y=ep["sd"][:, 0], name="front", line=dict(color=ARC_COLOR["front"], width=1.6)), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=ep["sd"][:, 1], name="left", line=dict(color=ARC_COLOR["left"], width=1.6)), 1, 1)
    for y, c in ((0.9, ARC_COLOR["front"]), (0.494, ARC_COLOR["left"]), (0.901, ARC_COLOR["left"])):
        fig.add_hline(y=y, line=dict(color=c, width=0.8, dash="dot"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=np.cumsum(ep["reward"]), name="cumulative reward", line=dict(color=color, width=2)), 2, 1)
    _layout(fig, None, height)
    fig.update_yaxes(title="distance (m)", range=[0, 2.6], row=1, col=1)
    fig.update_yaxes(title="Σ reward", row=2, col=1)
    fig.update_xaxes(title="time (s)", row=2, col=1)
    return fig


# ---------------------------------------------------------------- comparison
def metric_bars(summary: dict, metric: str, title, err: str | None = None, pct=False, height=320, keys=METHOD_KEYS):
    y = [summary[k][metric] * (100 if pct else 1) for k in keys]
    e = [summary[k][err] * (100 if pct else 1) for k in keys] if err else None
    fig = go.Figure(go.Bar(x=[METHOD_LABEL[k] for k in keys], y=y, marker_color=[METHOD_COLOR[k] for k in keys],
                           error_y=dict(type="data", array=e, color=INK, thickness=1.2) if e else None,
                           text=[f"{v:.1f}{'%' if pct else ''}" if abs(v) >= 1 or pct else f"{v:.2f}" for v in y],
                           textposition="outside", cliponaxis=False))
    _layout(fig, title, height)
    fig.update_xaxes(tickangle=0, tickfont=dict(size=10))
    return fig


def return_distribution(returns: dict, height=360):
    fig = go.Figure()
    for k in METHOD_KEYS:
        fig.add_trace(go.Box(y=returns[k], name=METHOD_LABEL[k], marker_color=METHOD_COLOR[k], boxmean=True,
                             boxpoints="outliers", line=dict(width=1.5)))
    _layout(fig, "Distribution of discounted return (200 fresh episodes per method)", height, showlegend=False)
    fig.update_yaxes(title="discounted return")
    fig.update_xaxes(tickfont=dict(size=10))
    return fig


def paired_diff(summary: dict, height=300):
    keys = [k for k in METHOD_KEYS if k != "expert"]
    fig = go.Figure(go.Bar(y=[METHOD_LABEL[k] for k in keys], x=[summary[k]["diff_vs_expert"] for k in keys],
                           error_x=dict(type="data", array=[summary[k]["diff_ci95"] for k in keys], color=INK),
                           orientation="h", marker_color=[METHOD_COLOR[k] for k in keys]))
    fig.add_vline(x=0, line=dict(color=INK, width=1))
    _layout(fig, "Improvement over the dataset expert (paired, same episodes, 95% CI)", height)
    fig.update_xaxes(title="Δ discounted return vs expert")
    fig.update_yaxes(autorange="reversed")
    return fig


def action_mix(summary: dict, height=300):
    fig = go.Figure()
    for i in range(4):
        fig.add_trace(go.Bar(y=[METHOD_LABEL[k] for k in METHOD_KEYS], x=[summary[k]["action_share"][i] * 100 for k in METHOD_KEYS],
                             name=ACTION_SHORT[i], orientation="h", marker_color=ACTION_COLOR[i]))
    _layout(fig, "Action mix (% of decisions, 300 s runs)", height, barmode="stack")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="%")
    return fig


# ---------------------------------------------------------------- policy search
def hj_trace(log, height=340):
    ev = [e["eval"] for e in log]
    J = np.array([e["J"] for e in log])
    kinds = [e["kind"].split()[0] for e in log]
    best = np.maximum.accumulate(np.where(np.isfinite(J), J, -1e9))
    cmap = {"start": INK, "explore": "#9AA3B2", "pattern": METHOD_COLOR["hj"]}
    fig = go.Figure()
    for kind, c in cmap.items():
        m = [i for i, k in enumerate(kinds) if k == kind]
        fig.add_trace(go.Scatter(x=[ev[i] for i in m], y=[J[i] for i in m], mode="markers", name=kind,
                                 marker=dict(color=c, size=7, line=dict(color="white", width=0.5)),
                                 customdata=[[log[i]["kind"], log[i]["tau_front"], log[i]["tau_left_low"], log[i]["tau_left_high"]] for i in m],
                                 hovertemplate="eval %{x}: %{customdata[0]}<br>τ = (%{customdata[1]:.3f}, %{customdata[2]:.3f}, %{customdata[3]:.3f})<br>J = %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=ev, y=best, mode="lines", name="best so far", line=dict(color=METHOD_COLOR["hj"], width=2, shape="hv")))
    _layout(fig, "Hooke–Jeeves: every objective evaluation", height)
    fig.update_xaxes(title="evaluation #")
    fig.update_yaxes(title="J(θ)  (search seeds)")
    return fig


def hj_params(log, height=300):
    fig = go.Figure()
    base = []
    best = -np.inf
    for e in log:
        if e["J"] > best:
            best = e["J"]
            base.append(e)
    names = [("tau_front", "τ front", ARC_COLOR["front"]), ("tau_left_low", "τ left low", "#0F8B8D"),
             ("tau_left_high", "τ left high", "#3552C7")]
    for key, label, c in names:
        fig.add_trace(go.Scatter(x=[b["eval"] for b in base], y=[b[key] for b in base], mode="lines+markers",
                                 name=label, line=dict(color=c, width=2, shape="hv")))
    _layout(fig, "Accepted thresholds (improvements only)", height)
    fig.update_xaxes(title="evaluation #")
    fig.update_yaxes(title="metres")
    return fig


def mc_scatter(mc, height=360):
    c = mc.candidates
    fig = go.Figure(go.Scatter(x=c[:, 0], y=c[:, 2] - c[:, 1], mode="markers",
                               marker=dict(color=mc.stage1, colorscale="YlOrBr", size=10, showscale=True,
                                           colorbar=dict(title="J", thickness=10), line=dict(color=INK, width=0.5)),
                               customdata=np.column_stack([c, mc.stage1]),
                               hovertemplate="τ = (%{customdata[0]:.3f}, %{customdata[1]:.3f}, %{customdata[2]:.3f})<br>J = %{customdata[3]:.2f}<extra></extra>",
                               name="candidates"))
    t = mc.candidates[mc.top_idx]
    fig.add_trace(go.Scatter(x=t[:, 0], y=t[:, 2] - t[:, 1], mode="markers", name="top 5 (re-scored)",
                             marker=dict(symbol="circle-open", size=18, color=METHOD_COLOR["mcps"], line=dict(width=2))))
    fig.add_trace(go.Scatter(x=[mc.best_theta[0]], y=[mc.best_theta[2] - mc.best_theta[1]], mode="markers", name="selected",
                             marker=dict(symbol="star", size=18, color=METHOD_COLOR["mcps"], line=dict(color=INK, width=1))))
    fig.add_trace(go.Scatter(x=[c[0, 0]], y=[c[0, 2] - c[0, 1]], mode="markers", name="expert thresholds",
                             marker=dict(symbol="x", size=14, color=INK)))
    _layout(fig, "Monte Carlo search: sampled thresholds and their estimated return", height)
    fig.update_xaxes(title="τ front (m)")
    fig.update_yaxes(title="width of the forward band  τ high − τ low (m)")
    return fig


# ------------------------------------------------------------------ dataset
def class_bars(dist, height=280):
    fig = go.Figure(go.Bar(x=dist.index, y=dist["count"], marker_color=list(ACTION_COLOR),
                           text=[f"{s:.1%}" for s in dist["share"]], textposition="outside", cliponaxis=False))
    _layout(fig, "Class distribution", height)
    return fig


def rule_scatter(df, height=420):
    fig = go.Figure()
    for i, a in enumerate(ACTIONS):
        d = df[df.Class == a]
        fig.add_trace(go.Scattergl(x=d.SD_left, y=d.SD_front, mode="markers", name=a,
                                   marker=dict(size=4, color=ACTION_COLOR[i], opacity=0.55)))
    fig.add_hline(y=0.900, line=dict(color=INK, dash="dash", width=1), annotation_text="front 0.900")
    fig.add_vline(x=0.494, line=dict(color=INK, dash="dash", width=1), annotation_text="left 0.494")
    fig.add_vline(x=0.901, line=dict(color=INK, dash="dash", width=1), annotation_text="left 0.901")
    _layout(fig, "The recorded controller is a threshold rule on SD_front and SD_left", height)
    fig.update_xaxes(title="SD_left (m)", range=[0.3, 2.0])
    fig.update_yaxes(title="SD_front (m)", range=[0.4, 3.0])
    return fig


def sensor_hist(df, col, height=260):
    fig = go.Figure()
    for i, a in enumerate(ACTIONS):
        fig.add_trace(go.Histogram(x=df[df.Class == a][col].clip(upper=3), name=ACTION_SHORT[i], marker_color=ACTION_COLOR[i],
                                   opacity=0.7, nbinsx=60))
    _layout(fig, f"{col} by class (clipped at 3 m)", height, barmode="overlay")
    fig.update_xaxes(title="metres")
    return fig


def time_series(df, start=0, n=900, height=300):
    d = df.iloc[start:start + n]
    fig = go.Figure()
    for c, col in (("SD_front", ARC_COLOR["front"]), ("SD_left", ARC_COLOR["left"])):
        fig.add_trace(go.Scatter(x=d.t_s, y=d[c].clip(upper=3), name=c, line=dict(color=col, width=1.4)))
    fig.add_trace(go.Scatter(x=d.t_s, y=np.full(len(d), 2.85), mode="markers", name="label",
                             marker=dict(symbol="square", size=5, color=[ACTION_COLOR[a] for a in d.action]),
                             text=d.Class, hovertemplate="%{text}<extra></extra>"))
    _layout(fig, "The rows are a 9 Hz trajectory (first 100 s shown; label strip on top)", height)
    fig.update_xaxes(title="time (s)")
    fig.update_yaxes(title="metres", range=[0, 3])
    return fig


def validation_bars(v, height=300):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Action frequency under the expert rule",
                                                        "Median simplified distance (m)"))
    for name, key, c in (("recording", "real", INK), ("twin", "twin", METHOD_COLOR["adp"])):
        fig.add_trace(go.Bar(x=list(ACTION_SHORT), y=v[f"freq_{key}"], name=name, marker_color=c), 1, 1)
        fig.add_trace(go.Bar(x=["front", "left", "right", "back"], y=v[f"median_{key}"], name=name,
                             marker_color=c, showlegend=False), 1, 2)
    _layout(fig, None, height, barmode="group")
    return fig


# ------------------------------------------------------------ added visuals
def reward_heatmaps(R, cfg, height=300):
    """Expected immediate reward R(s, a) for each action."""
    fig = make_subplots(rows=1, cols=4, subplot_titles=[f"{ACTION_GLYPH[i]} {ACTION_SHORT[i]}" for i in range(4)],
                        horizontal_spacing=0.03, shared_yaxes=True)
    zmin, zmax = float(R[:-1].min()), float(R[:-1].max())
    for a in range(4):
        z = R[: cfg.n_grid, a].reshape(cfg.n_front, cfg.n_left)
        fig.add_trace(go.Heatmap(z=z, x=left_labels(cfg), y=front_labels(cfg), zmin=zmin, zmax=zmax, zmid=0,
                                 colorscale="RdBu", showscale=a == 3, text=np.round(z, 1), texttemplate="%{text}",
                                 textfont=dict(size=9), colorbar=dict(title="R", thickness=10),
                                 hovertemplate="front %{y}<br>left %{x}<br>R = %{z:.3f}<extra></extra>"), 1, a + 1)
        fig.update_xaxes(tickfont=dict(size=8), tickangle=45, row=1, col=a + 1)
    _layout(fig, "Expected immediate reward R(s, a) = Σ P(s′|s,a) r(s′,a)", height)
    fig.update_yaxes(title="SD_front bin", col=1)
    return fig


def transition_diagram(P, pi, V, cfg, threshold=0.12, height=460):
    """State-transition graph under the optimal policy: nodes are grid states,
    arrows show P(s'|s, pi(s)) >= threshold (self-loops shown as ring size)."""
    nl = cfg.n_left
    xs = np.array([s % nl for s in range(cfg.n_grid)] + [nl + 0.8])
    ys = np.array([s // nl for s in range(cfg.n_grid)] + [-0.8])
    fig = go.Figure()
    for s in range(cfg.n_grid):
        row = P[s, pi[s]]
        for t in np.flatnonzero(row >= threshold):
            if t == s:
                continue
            fig.add_annotation(x=xs[t], y=ys[t], ax=xs[s], ay=ys[s], xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=0.6 + 4 * row[t],
                               arrowcolor="rgba(53,82,199,%.2f)" % min(0.95, 0.25 + row[t]), standoff=11, startstandoff=11)
    selfp = np.array([P[s, pi[s], s] for s in range(cfg.n_grid)] + [1.0])
    txt = [f"{ACTION_GLYPH[int(pi[s])]}" for s in range(cfg.n_grid)] + ["✕"]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=txt, textfont=dict(color="white", size=13),
                             marker=dict(size=14 + 22 * selfp, color=np.append(V[: cfg.n_grid], V[-1]),
                                         colorscale="Tealgrn", line=dict(color=INK, width=1),
                                         colorbar=dict(title="V*", thickness=10)),
                             customdata=np.column_stack([selfp, np.append(V[: cfg.n_grid], V[-1])]),
                             hovertemplate="stay probability %{customdata[0]:.2f}<br>V* = %{customdata[1]:.2f}<extra></extra>"))
    _layout(fig, f"State-transition graph under π* (arrows: P ≥ {threshold}; larger node = more likely to stay)", height)
    fig.update_xaxes(tickvals=list(range(nl)) + [nl + 0.8], ticktext=left_labels(cfg) + ["Crash"], title="SD_left bin (m)",
                     showgrid=False)
    fig.update_yaxes(tickvals=list(range(cfg.n_front)), ticktext=front_labels(cfg), title="SD_front bin (m)", showgrid=False)
    return fig


def actual_vs_approx(v_exact, v_approx, height=340):
    m = np.isfinite(v_approx[:-1])
    x, y = v_exact[:-1][m], v_approx[:-1][m]
    lo, hi = min(x.min(), y.min()) - 1, max(x.max(), y.max()) + 1
    rmse = float(np.sqrt(np.mean((x - y) ** 2)))
    corr = float(np.corrcoef(x, y)[0, 1])
    fig = go.Figure([go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=MUTED, dash="dash"), name="V̂ = V*"),
                     go.Scatter(x=x, y=y, mode="markers", name="state", marker=dict(size=9, color=METHOD_COLOR["adp"],
                                                                                     line=dict(color="white", width=1)))])
    _layout(fig, f"Exact V* vs ADP V̂ per state · RMSE {rmse:.2f} · correlation {corr:.3f}", height)
    fig.update_xaxes(title="V*(s) from value iteration")
    fig.update_yaxes(title="V̂(s) from fitted Q-iteration")
    return fig


def mc_progress(mc, height=320):
    k = np.arange(1, len(mc.stage1) + 1)
    fig = go.Figure([go.Scatter(x=k, y=mc.stage1, mode="markers", name="candidate J",
                                marker=dict(color=METHOD_COLOR["mcps"], size=6, opacity=0.6),
                                error_y=dict(type="data", array=mc.stage1_ci, color="rgba(208,140,11,0.35)", thickness=1)),
                     go.Scatter(x=k, y=mc.running_best, mode="lines", name="best so far",
                                line=dict(color=INK, width=2, shape="hv"))])
    _layout(fig, "Monte Carlo search: performance across candidates (±95% CI)", height)
    fig.update_xaxes(title="candidate #")
    fig.update_yaxes(title="J(θ)  (search seeds)")
    return fig


def return_histograms(returns: dict, keys=("expert", "mcps", "hj"), height=320):
    fig = go.Figure()
    for k in keys:
        fig.add_trace(go.Histogram(x=returns[k], name=METHOD_LABEL[k], marker_color=METHOD_COLOR[k], opacity=0.6, nbinsx=30))
    _layout(fig, "Monte Carlo return distributions on 200 fresh episodes", height, barmode="overlay")
    fig.update_xaxes(title="discounted return of one episode")
    fig.update_yaxes(title="episodes")
    return fig


def hj_steps(hj, height=300):
    st_ = np.array(hj.step_history)
    fig = go.Figure()
    for i, (lab, c) in enumerate((("Δ τ front", ARC_COLOR["front"]), ("Δ τ left-low", "#0F8B8D"), ("Δ τ left-high", "#3552C7"))):
        fig.add_trace(go.Scatter(x=np.arange(1, len(st_) + 1), y=st_[:, i], mode="lines+markers", name=lab,
                                 line=dict(color=c, width=2, shape="hv")))
    _layout(fig, "Hooke–Jeeves step-size progression (halves when no move improves J)", height)
    fig.update_xaxes(title="exploration round")
    fig.update_yaxes(title="step (m)", type="log")
    return fig


def convergence_overview(res, height=560):
    vi, a, mc, hj = res["vi"], res["adp"], res["mc"], res["hj"]
    J = np.array([e["J"] for e in hj.log])
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.16, horizontal_spacing=0.1,
                        subplot_titles=(f"MDP · value iteration ({vi.iterations} sweeps)",
                                        f"ADP · fitted Q-iteration ({a.iterations} iterations)",
                                        f"Monte Carlo search ({len(mc.stage1)} candidates)",
                                        f"Hooke–Jeeves ({hj.evaluations} evaluations)"))
    fig.add_trace(go.Scatter(y=vi.deltas, line=dict(color=METHOD_COLOR["mdp"], width=2)), 1, 1)
    fig.add_trace(go.Scatter(y=a.deltas, line=dict(color=METHOD_COLOR["adp"], width=2)), 1, 2)
    fig.add_trace(go.Scatter(y=mc.running_best, line=dict(color=METHOD_COLOR["mcps"], width=2, shape="hv")), 2, 1)
    fig.add_trace(go.Scatter(y=np.maximum.accumulate(J), line=dict(color=METHOD_COLOR["hj"], width=2, shape="hv")), 2, 2)
    fig.update_yaxes(type="log", title="‖ΔV‖∞", row=1, col=1)
    fig.update_yaxes(type="log", title="max |ΔQ̂|", row=1, col=2)
    fig.update_yaxes(title="best J", row=2, col=1)
    fig.update_yaxes(title="best J", row=2, col=2)
    _layout(fig, None, height)
    fig.update_layout(showlegend=False)
    return fig


def policy_maps(maps: dict, cfg, height=330):
    """Side-by-side action maps on the MDP grid (threshold policies evaluated at cell midpoints)."""
    keys = list(maps)
    fig = make_subplots(rows=1, cols=len(keys), subplot_titles=[METHOD_LABEL[k] for k in keys],
                        horizontal_spacing=0.02, shared_yaxes=True)
    cs = []
    for i, c in enumerate(ACTION_COLOR):
        cs += [[i / 4, c], [(i + 1) / 4, c]]
    for j, k in enumerate(keys):
        z = np.asarray(maps[k][: cfg.n_grid]).reshape(cfg.n_front, cfg.n_left)
        fig.add_trace(go.Heatmap(z=z + 0.5, zmin=0, zmax=4, x=left_labels(cfg), y=front_labels(cfg), colorscale=cs,
                                 showscale=False, text=[[ACTION_GLYPH[v] for v in r] for r in z], texttemplate="%{text}",
                                 textfont=dict(color="white", size=14),
                                 hovertemplate="front %{y}<br>left %{x}<extra></extra>"), 1, j + 1)
        fig.update_xaxes(tickfont=dict(size=8), tickangle=45, row=1, col=j + 1)
    _layout(fig, None, height)
    fig.update_yaxes(title="SD_front bin", col=1)
    return fig


def correlation_heatmap(df, height=320):
    c = df[["SD_front", "SD_left", "SD_right", "SD_back"]].corr()
    fig = go.Figure(go.Heatmap(z=c.values, x=c.columns, y=c.index, zmin=-1, zmax=1, colorscale="RdBu",
                               text=np.round(c.values, 2), texttemplate="%{text}", colorbar=dict(thickness=10)))
    _layout(fig, "Pearson correlation between simplified distances", height)
    fig.update_yaxes(autorange="reversed")
    return fig

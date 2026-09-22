"""Room and robot drawings for the Live robot view (Plotly).

The agents are drawn as realistic cars seen from a slightly raised
three-quarter angle: a rounded coloured body with two shaded side skirts for
depth, a hood, a cabin (windshield + roof), paired head- and tail-lights, four
rounded wheels and a soft ground shadow that plants each car on the floor.  All
cars share one design language and differ only in colour.  ``kind="random"``
swaps the solid roof for a translucent glass dome and a cloud of sample points,
marking the probabilistic agent.  These are purely visual representations of the
twin's robot and do not change the model.

* ``room_traces``      floor, walls, free-standing pillars, entry door, exit gate
* ``robot_traces``     one car (12 traces, fixed order) in the method's colour
* ``mission_animation`` one car with its sonar arcs, trail and reward zones
* ``race_animation``   every method's car in the same hall, each in its own colour
* ``room_map``         static, annotated floor plan
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import ACTION_GLYPH, ACTIONS
from .rooms import Room
from .statespace import ZONE_COLOR, ZONES
from .twin import BEAM_ANGLES, arc_beams

INK = "#1B2433"
MUTED = "#6B7385"
FONT = "IBM Plex Sans, Inter, Segoe UI, sans-serif"
OUTSIDE = "#D3DAE3"
FLOOR = "rgba(250,251,253,0.84)"
WALL = "#1F2937"
PILLAR = "#3A4556"
ENTRY_C = "#16A34A"
EXIT_C = "#E11D48"
RAY = {"front": "#EF4444", "left": "#0891B2"}
BADGE = {"exit": "✓", "loop": "⟲", "crash": "✕", "timeout": "⏱"}
N_SONAR = 24


# ---------------------------------------------------------------------- helpers
def _closed(p):
    p = np.asarray(p)
    return np.vstack([p, p[:1]])


def _join(polys):
    """Concatenate polygons into one trace with None separators."""
    xs, ys = [], []
    for p in polys:
        p = _closed(p)
        xs += p[:, 0].tolist() + [None]
        ys += p[:, 1].tolist() + [None]
    return xs, ys


def _view(room: Room):
    return room.view or (lambda b: (b[0] - 0.3, b[1] - 0.3, b[2] + 0.3, b[3] + 0.3))(room.bounds)


# ------------------------------------------------------------------------- room
def room_traces(room: Room, labels: bool = True):
    """Static traces and annotations for a room."""
    tr = []
    o = _closed(room.outer)
    tr.append(go.Scatter(x=o[:, 0], y=o[:, 1], mode="lines", fill="toself", fillcolor=FLOOR,
                         line=dict(color=WALL, width=6), hoverinfo="skip"))
    if room.pillars:
        xs, ys = _join(room.pillars)
        tr.append(go.Scatter(x=xs, y=ys, mode="lines", fill="toself", fillcolor=PILLAR,
                             line=dict(color=WALL, width=2), hoverinfo="skip"))
    ann = []
    if room.entry is not None:
        # the entry door: the corridor's closing wall, drawn as a green door
        x = max(p[0] for p in (room.entry.a, room.entry.b)) + 0.15
        y0, y1 = sorted((room.entry.a[1], room.entry.b[1]))
        tr.append(go.Scatter(x=[x, x], y=[y0 + 0.08, y1 - 0.08], mode="lines", line=dict(color=ENTRY_C, width=9),
                             hoverinfo="skip"))
        ann.append(dict(x=x - 0.12, y=y1 - 0.28, text="<b>ENTRY</b>", showarrow=False, xanchor="right",
                        font=dict(size=12, color=ENTRY_C)))
        ann.append(dict(x=x - 0.95, y=y1 - 0.62, ax=x - 0.2, ay=y1 - 0.62, xref="x", yref="y",
                        axref="x", ayref="y", text="", showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2.2,
                        arrowcolor=ENTRY_C, opacity=0.8))
    if room.exit is not None:
        gx = room.exit.a[0]
        y0, y1 = sorted((room.exit.a[1], room.exit.b[1]))
        tr.append(go.Scatter(x=[gx, gx], y=[y0 + 0.06, y1 - 0.06], mode="lines",
                             line=dict(color=EXIT_C, width=3, dash="dash"), hoverinfo="skip"))
        # checkered finish strip at the exit depth
        fx, sq = gx + 0.35, 0.1
        ny = int((y1 - y0 - 0.12) / sq)
        blk = []
        for j in range(ny):
            for i in range(2):
                if (i + j) % 2 == 0:
                    x0, yy = fx - sq + i * sq, y0 + 0.06 + j * sq
                    blk.append([(x0, yy), (x0 + sq, yy), (x0 + sq, yy + sq), (x0, yy + sq)])
        bx, by = _join(blk)
        tr.append(go.Scatter(x=bx, y=by, mode="lines", fill="toself", fillcolor=INK, line=dict(width=0),
                             hoverinfo="skip", opacity=0.75))
        ann.append(dict(x=fx + 0.2, y=y1 - 0.28, text="<b>EXIT</b>", showarrow=False, xanchor="left",
                        font=dict(size=12, color=EXIT_C)))
        ann.append(dict(x=fx + 1.05, y=(y0 + y1) / 2 - 0.55, ax=fx + 0.25, ay=(y0 + y1) / 2 - 0.55, xref="x",
                        yref="y", axref="x", ayref="y", text="", showarrow=True, arrowhead=2, arrowsize=1.2,
                        arrowwidth=2.2, arrowcolor=EXIT_C, opacity=0.8))
    if labels:
        for text, x, y in room.features:
            ann.append(dict(x=x, y=y, text=f"<i>{text}</i>", showarrow=False, font=dict(size=10, color=MUTED)))
        for k, p in enumerate(room.pillars):
            c = np.asarray(p).mean(0)
            ann.append(dict(x=c[0], y=c[1] - 0.5, text=f"<i>Pillar {k + 1}</i>", showarrow=False,
                            font=dict(size=10, color=MUTED)))
    x0, y0, _, _ = _view(room)
    tr.append(go.Scatter(x=[x0 + 0.15, x0 + 1.15], y=[y0 + 0.14] * 2, mode="lines+text",
                         line=dict(color=INK, width=3), text=["", "1 m"], textposition="middle right",
                         textfont=dict(size=10, color=INK), hoverinfo="skip"))
    return tr, ann


# ------------------------------------------------------------------------ robot
# The robots are drawn as futuristic autonomous robot cars, seen from a slightly
# raised three-quarter angle.  A pose is (x, y, heading) in world metres, heading
# in radians (0 = +x).  The car points along +heading (its "forward" axis u); the
# left axis v is +90 deg from it.  Depth is faked with a fixed screen-space "lift"
# so raised parts (body top, canopy) sit above their footprint and every car
# throws a ground shadow, which gives clear floor contact.  Numbers below are in
# units of the robot radius r, tuned once so a car fills the same footprint the
# old disc robot did (roughly 2r long, 1.5r wide).

LIFT = np.array([0.0, 0.34])          # screen-space rise (world y) per unit height
_CAR_L, _CAR_W = 1.9, 1.35            # body length / width in units of r


def _car_parts(pose, r):
    """Polygons of a realistic car at pose (x, y, heading), in world coordinates.

    Drawn from a slightly raised three-quarter view: a rounded body with a
    separate cabin (windshield + roof), a hood, paired head- and tail-lights,
    four rounded wheels and a soft ground shadow.  ``height`` (screen-space lift)
    lifts the roof and cabin above the body so the car reads three-dimensional.
    """
    x, y, h = pose
    u = np.array([np.cos(h), np.sin(h)])      # forward (car nose)
    v = np.array([-np.sin(h), np.cos(h)])     # left

    def w(fwd, lat, lift=0.0):
        p = np.asarray(fwd, float)[..., None] * u + np.asarray(lat, float)[..., None] * v + (x, y)
        return p + lift * LIFT * r

    def rounded(f0, f1, half_w, round_front, round_rear, n=7, lift=0.0):
        """A capsule-ish body outline from rear (f0) to front (f1)."""
        fwd, lat = [], []
        aa = np.linspace(-np.pi / 2, np.pi / 2, n)         # front cap
        for a in aa:
            fwd.append(f1 - round_front + round_front * np.cos(a)); lat.append(half_w * np.sin(a))
        ab = np.linspace(np.pi / 2, 3 * np.pi / 2, n)      # rear cap
        for a in ab:
            fwd.append(f0 + round_rear + round_rear * np.cos(a)); lat.append(half_w * np.sin(a))
        return w(fwd, lat, lift)

    L, W = _CAR_L * r, _CAR_W * r
    hf, hw = L / 2, W / 2

    # body: rounded rectangle, longer round at the nose than the tail
    footprint = rounded(-hf, hf, hw, 0.55 * hf, 0.32 * hf)
    body_top = rounded(-hf, hf, hw * 0.94, 0.55 * hf, 0.32 * hf, lift=0.55)

    # side skirt between footprint and body top on each side (depth)
    def skirt(sign):
        lo = footprint[footprint[:, 0] * 0 == 0]           # all points
        # take the outer edge on this side by lateral sign in body frame
        fwd = (footprint - (x, y)) @ u
        lat = (footprint - (x, y)) @ v
        m = np.where(lat * sign > 0.02 * r)[0]
        base = w(fwd[m], lat[m])
        top = w(fwd[m], lat[m], lift=0.55)
        return np.vstack([base, top[::-1]])
    left_side = skirt(+1)
    right_side = skirt(-1)

    # hood then cabin (windshield + roof) sitting on the body top
    hood = rounded(-hf * 0.05, hf * 0.62, hw * 0.7, 0.42 * hf, 0.05 * hf, n=5, lift=0.6)
    cabin = rounded(-hf * 0.62, hf * 0.12, hw * 0.66, 0.18 * hf, 0.30 * hf, n=6, lift=0.95)
    roof = rounded(-hf * 0.5, -hf * 0.02, hw * 0.5, 0.12 * hf, 0.22 * hf, n=6, lift=1.05)

    # paired lights: two headlamps at the nose, two tail-lamps at the rear
    def dot(cf, cl, rad, lift):
        t = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        return w(cf + rad * np.cos(t), cl + rad * np.sin(t), lift)
    head_l = dot(hf * 0.86, hw * 0.52, 0.12 * r, 0.58)
    head_r = dot(hf * 0.86, -hw * 0.52, 0.12 * r, 0.58)
    tail_l = dot(-hf * 0.9, hw * 0.55, 0.10 * r, 0.58)
    tail_r = dot(-hf * 0.9, -hw * 0.55, 0.10 * r, 0.58)

    # four rounded wheels straddling the sides
    def wheel(cf, cl):
        t = np.linspace(0, 2 * np.pi, 14, endpoint=False)
        return w(cf + 0.30 * r * np.cos(t), cl + 0.17 * r * np.sin(t), 0.12)
    wheels = [wheel(f, s * hw * 0.99) for f in (hf * 0.55, -hf * 0.58) for s in (1, -1)]

    return dict(footprint=footprint, wheels=wheels, left=left_side, right=right_side, body=body_top,
                hood=hood, cabin=cabin, roof=roof, head_l=head_l, head_r=head_r,
                tail_l=tail_l, tail_r=tail_r)


def _shade(hex_color, factor):
    """Lighten (factor>1) or darken (factor<1) a #rrggbb colour."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    f = lambda c: max(0, min(255, int(c * factor)))
    return f"#{f(r):02x}{f(g):02x}{f(b):02x}"


N_ROBOT_TRACES = 12   # fixed trace count per robot, so animation frames stay aligned


def robot_traces(pose, r, color, visible=True, label=None, badge=None, kind="car"):
    """A realistic car as a fixed list of 12 Plotly traces (fixed order, so
    animation frames can address them positionally; empty traces when hidden).

    Order: shadow, wheels, side_l, side_r, body, hood, cabin/glass, roof-or-cloud,
    headlights, taillights, footprint-outline, text label.  ``kind="random"``
    swaps the solid roof for a translucent glass dome and a cloud of sample
    points, marking the probabilistic agent.
    """
    if not visible or pose is None:
        return [go.Scatter(x=[], y=[], mode="lines") for _ in range(N_ROBOT_TRACES - 2)] + \
               [go.Scatter(x=[], y=[], mode="markers"), go.Scatter(x=[], y=[], mode="text")]

    p = _car_parts(pose, r)
    dark, light = _shade(color, 0.60), _shade(color, 1.30)
    glass = "rgba(191,219,254,0.92)"
    wx, wy = _join([_closed(w) for w in p["wheels"]])
    ft, bt = _closed(p["footprint"]), _closed(p["body"])
    ls, rs = _closed(p["left"]), _closed(p["right"])
    hood, cabin, roof = _closed(p["hood"]), _closed(p["cabin"]), _closed(p["roof"])
    hl, hr = _closed(p["head_l"]), _closed(p["head_r"])
    tl, tr = _closed(p["tail_l"]), _closed(p["tail_r"])
    txt = " ".join(x for x in (label, badge) if x)
    lx, ly = pose[0], pose[1] + _CAR_W * r * 0.62 + LIFT[1] * r * 1.9

    shadow = go.Scatter(x=ft[:, 0] + 0.10 * r, y=ft[:, 1] - 0.12 * r, mode="lines", fill="toself",
                        fillcolor="rgba(15,23,42,0.22)", line=dict(color="rgba(0,0,0,0)"))
    wheels = go.Scatter(x=wx, y=wy, mode="lines", fill="toself", fillcolor="#111827", line=dict(color="#0B0F19", width=1))
    side_l = go.Scatter(x=ls[:, 0], y=ls[:, 1], mode="lines", fill="toself", fillcolor=dark, line=dict(color=INK, width=1))
    side_r = go.Scatter(x=rs[:, 0], y=rs[:, 1], mode="lines", fill="toself", fillcolor=dark, line=dict(color=INK, width=1))
    body = go.Scatter(x=bt[:, 0], y=bt[:, 1], mode="lines", fill="toself", fillcolor=color, line=dict(color=INK, width=1.6))
    hood_t = go.Scatter(x=hood[:, 0], y=hood[:, 1], mode="lines", fill="toself", fillcolor=light, line=dict(color=color, width=1))
    heads = go.Scatter(x=hl[:, 0].tolist() + [None] + hr[:, 0].tolist(),
                       y=hl[:, 1].tolist() + [None] + hr[:, 1].tolist(), mode="lines", fill="toself",
                       fillcolor="#FEF3C7", line=dict(color="#F59E0B", width=1))
    tails = go.Scatter(x=tl[:, 0].tolist() + [None] + tr[:, 0].tolist(),
                       y=tl[:, 1].tolist() + [None] + tr[:, 1].tolist(), mode="lines", fill="toself",
                       fillcolor="#F87171", line=dict(color="#DC2626", width=1))
    outline = go.Scatter(x=ft[:, 0], y=ft[:, 1], mode="lines", line=dict(color=INK, width=1))
    label_t = go.Scatter(x=[lx], y=[ly], mode="text", text=[txt], textfont=dict(size=11, color=color, family=FONT))

    if kind == "random":
        rng = np.random.default_rng(int((pose[0] * 131 + pose[1] * 57) * 1000) % (2**32))
        centre = p["roof"].mean(0)
        cloud = centre + rng.normal(0, 0.15 * r, size=(24, 2))
        cabin_t = go.Scatter(x=cabin[:, 0], y=cabin[:, 1], mode="lines", fill="toself",
                             fillcolor="rgba(226,240,253,0.35)", line=dict(color="#FFFFFF", width=1.2, dash="dot"))
        roof_t = go.Scatter(x=cloud[:, 0], y=cloud[:, 1], mode="markers",
                           marker=dict(size=4, color=light, opacity=0.85, symbol="diamond", line=dict(width=0)))
    else:
        cabin_t = go.Scatter(x=cabin[:, 0], y=cabin[:, 1], mode="lines", fill="toself", fillcolor=glass,
                            line=dict(color="#1E3A5F", width=1))
        roof_t = go.Scatter(x=roof[:, 0], y=roof[:, 1], mode="lines", fill="toself", fillcolor=color,
                          line=dict(color=INK, width=1))
    return [shadow, wheels, side_l, side_r, body, hood_t, cabin_t, roof_t, heads, tails, outline, label_t]


# --------------------------------------------------------------------- layout
def _finish(fig, room, height, frame_ms, idx, top_text, tick_every):
    x0, y0, x1, y1 = _view(room)
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=58, b=64), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=OUTSIDE,
        font=dict(family=FONT, size=12, color=INK), showlegend=False,
        annotations=list(fig.layout.annotations or []) + [top_text],
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.02, xanchor="left", yanchor="top",
                          pad=dict(t=6, r=6), bgcolor="#FFFFFF", bordercolor="#E3E7EE",
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=dict(duration=frame_ms, redraw=False),
                                                         transition=dict(duration=0), fromcurrent=True,
                                                         mode="immediate")]),
                                   dict(label="⏸ Pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(active=0, x=0.2, len=0.8, y=-0.02, yanchor="top", pad=dict(t=6),
                      currentvalue=dict(visible=False), tickcolor="#E3E7EE", font=dict(size=9),
                      steps=[dict(label=f"{i / 3:.0f}s" if k % tick_every == 0 else "", method="animate",
                                  args=[[str(i)], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                             for k, i in enumerate(idx)])])
    fig.update_xaxes(range=[x0, x1], showgrid=True, dtick=1, gridcolor="#B9C3D0", zeroline=False,
                     showticklabels=False, fixedrange=True)
    fig.update_yaxes(range=[y0, y1], showgrid=True, dtick=1, gridcolor="#B9C3D0", zeroline=False,
                     showticklabels=False, scaleanchor="x", scaleratio=1, fixedrange=True)
    for tr in fig.data:
        tr.update(showlegend=False, hoverinfo="skip")
    return fig


def _hud(text):
    return dict(text=text, x=0, y=1.0, yshift=8, xref="paper", yref="paper", showarrow=False, xanchor="left",
                yanchor="bottom", align="left", font=dict(size=12, color=INK))


# ------------------------------------------------------------ single robot run
def mission_animation(ep: dict, room: Room, radius: float, color: str, frame_ms: int = 90,
                      stride: int = 1, height: int = 560, kind: str = "car"):
    arcs = arc_beams(60.0)
    poses = ep["pose"]
    n = len(poses)
    idx = list(range(0, n, stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    path = np.vstack([poses[:, :2], ep["final_pose"][None, :2]])
    outcome = ep.get("outcome")

    def rays(i):
        x, y, h = poses[i]
        out = {}
        for name, bi in zip(("front", "left"), (arcs[0], arcs[1])):
            xs, ys = [], []
            for b in bi:
                a = h + BEAM_ANGLES[b]
                r1 = radius + min(ep["beams"][i][b], 2.5)
                xs += [x + radius * np.cos(a), x + r1 * np.cos(a), None]
                ys += [y + radius * np.sin(a), y + r1 * np.sin(a), None]
            out[name] = (xs, ys)
        return out

    def frame(i):
        rr = rays(i)
        last = i == n - 1
        badge = BADGE.get(outcome) if last and outcome else None
        pose = ep["final_pose"] if last else poses[i]
        data = [go.Scatter(x=rr["left"][0], y=rr["left"][1], mode="lines", line=dict(color=RAY["left"], width=1.6)),
                go.Scatter(x=rr["front"][0], y=rr["front"][1], mode="lines", line=dict(color=RAY["front"], width=1.6)),
                go.Scatter(x=path[:i + 2, 0], y=path[:i + 2, 1], mode="lines",
                           line=dict(color=color, width=3.5), opacity=0.85)]
        data += robot_traces(pose, radius, color, badge=badge, kind=kind)
        sd = ep["sd"][i]
        z = int(ep["zone"][i])
        txt = (f"<b>t = {i / 3:5.1f} s</b> · step {i} · {ACTION_GLYPH[ep['action'][i]]} {ACTIONS[ep['action'][i]]}"
               f" · front {sd[0]:.2f} m · left {sd[1]:.2f} m · {ZONES[z]} · Σr = {ep['reward'][: i + 1].sum():+.1f}")
        if last and outcome:
            from .mission import OUTCOME_LABEL
            txt += f" · <b>{OUTCOME_LABEL[outcome]}</b>"
        return data, txt

    static, ann = room_traces(room)
    zc = [ZONE_COLOR[int(z)] for z in ep["zone"]]
    static.append(go.Scatter(x=path[1:, 0], y=path[1:, 1], mode="markers",
                             marker=dict(color=zc, size=4, opacity=0.5)))
    d0, t0 = frame(0)
    k0 = len(static)
    frames = [go.Frame(data=d, traces=list(range(k0, k0 + len(d))), name=str(i),
                       layout=dict(annotations=ann + [_hud(t)])) for i in idx for d, t in [frame(i)]]
    fig = go.Figure(data=static + d0, frames=frames, layout=dict(annotations=ann))
    return _finish(fig, room, height, frame_ms, idx, _hud(t0), max(1, len(idx) // 10))


# -------------------------------------------------------------------- race
SHORT = {"expert": "Recorded", "mdp": "MDP", "adp": "ADP", "mcps": "MC-PS", "hj": "HJ", "random": "Random",
         "custom": "Custom"}


def race_animation(eps: dict, room: Room, radius: float, colors: dict, stagger: int = 15, stride: int = 3,
                   frame_ms: int = 80, height: int = 600):
    """All methods in one hall.  Robot k is released stagger*k decisions after robot k-1.
    Each robot is simulated on its own (they do not see each other)."""
    keys = list(eps)
    release = {k: j * stagger for j, k in enumerate(keys)}
    T = max(release[k] + len(eps[k]["pose"]) for k in keys)
    idx = list(range(0, T, stride)) + ([T - 1] if (T - 1) % stride else [])

    def state(k, g):
        ep = eps[k]
        t = g - release[k]
        n = len(ep["pose"])
        if t < 0:
            return None, None, 0, False
        if t >= n - 1:
            return ep["final_pose"], ep["outcome"], n, True
        return ep["pose"][t], None, t + 1, False

    def frame(g):
        data, status = [], []
        for k in keys:
            ep = eps[k]
            pose, out, m, done = state(k, g)
            path = np.vstack([ep["pose"][:m, :2], ep["final_pose"][None, :2]]) if done else ep["pose"][:m, :2]
            data.append(go.Scatter(x=path[:, 0].tolist(), y=path[:, 1].tolist(), mode="lines",
                                   line=dict(color=colors[k], width=3), opacity=0.55))
        for k in keys:
            pose, out, m, done = state(k, g)
            data += robot_traces(pose, radius, colors[k], visible=pose is not None, label=SHORT[k],
                                 badge=BADGE.get(out) if done else None, kind="random" if k == "random" else "car")
            if pose is None:
                status.append(f"<span style='color:{colors[k]}'>●</span> {SHORT[k]} waiting")
            elif done:
                status.append(f"<span style='color:{colors[k]}'>●</span> {SHORT[k]} {BADGE[out]} "
                              f"{(m) / 3:.0f} s")
            else:
                status.append(f"<span style='color:{colors[k]}'>●</span> {SHORT[k]} {(m) / 3:.0f} s")
        return data, f"<b>t = {g / 3:5.1f} s</b>   " + "   ".join(status)

    static, ann = room_traces(room)
    d0, t0 = frame(0)
    k0 = len(static)
    frames = [go.Frame(data=d, traces=list(range(k0, k0 + len(d))), name=str(g),
                       layout=dict(annotations=ann + [_hud(t)])) for g in idx for d, t in [frame(g)]]
    fig = go.Figure(data=static + d0, frames=frames, layout=dict(annotations=ann))
    return _finish(fig, room, height, frame_ms, idx, _hud(t0), max(1, len(idx) // 10))


# ----------------------------------------------------------------- static map
def room_map(room: Room, height=460, show_start=True, radius=0.29, color="#2563EB"):
    tr, ann = room_traces(room)
    if show_start and room.entry_pose is not None:
        tr += robot_traces(room.entry_pose, radius, color)
    fig = go.Figure(data=tr, layout=dict(annotations=ann))
    x0, y0, x1, y1 = _view(room)
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor=OUTSIDE, showlegend=False, font=dict(family=FONT))
    fig.update_xaxes(range=[x0, x1], showgrid=True, dtick=1, gridcolor="#B9C3D0", zeroline=False,
                     showticklabels=False, fixedrange=True)
    fig.update_yaxes(range=[y0, y1], showgrid=True, dtick=1, gridcolor="#B9C3D0", zeroline=False,
                     showticklabels=False, scaleanchor="x", scaleratio=1, fixedrange=True)
    for t in fig.data:
        t.update(hoverinfo="skip")
    return fig


def robot_closeup(colors: dict, labels: dict, radius=0.29, height=220):
    """One robot per method, side by side, for the legend."""
    tr, ann = [], []
    for j, (k, c) in enumerate(colors.items()):
        tr += robot_traces((j * 1.6, 0.0, np.pi / 2), radius, c, kind="random" if k == "random" else "car")[:-1]
        ann.append(dict(x=j * 1.6, y=-0.95, text=labels[k], showarrow=False, font=dict(size=11, color=c)))
    fig = go.Figure(data=tr, layout=dict(annotations=ann))
    n = len(colors)
    fig.update_layout(height=height, margin=dict(l=4, r=4, t=4, b=4), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    fig.update_xaxes(range=[-1.1, (n - 1) * 1.6 + 1.1], visible=False, fixedrange=True)
    fig.update_yaxes(range=[-1.25, 0.95], visible=False, scaleanchor="x", fixedrange=True)
    for t in fig.data:
        t.update(hoverinfo="skip")
    return fig


def paths_grid(eps: dict, room: Room, colors: dict, labels: dict, height: int = 260):
    """One trajectory per method from the same start, with its outcome badge."""
    keys = list(eps)
    rows = (len(keys) + 2) // 3
    fig = make_subplots(rows=rows, cols=3, horizontal_spacing=0.015, vertical_spacing=0.07,
                        subplot_titles=[f"{labels[k]} · {BADGE[eps[k]['outcome']]} {len(eps[k]['action']) / 3:.0f} s"
                                        for k in keys])
    static, _ = room_traces(room, labels=False)
    for j, k in enumerate(keys):
        r, c = divmod(j, 3)
        for t in static[:-1]:
            fig.add_trace(go.Scatter(t), r + 1, c + 1)
        ep = eps[k]
        p = np.vstack([ep["pose"][:, :2], ep["final_pose"][None, :2]])
        fig.add_trace(go.Scatter(x=p[:, 0], y=p[:, 1], mode="lines", line=dict(color=colors[k], width=2.5)), r + 1, c + 1)
        fig.add_trace(go.Scatter(x=[p[-1, 0]], y=[p[-1, 1]], mode="markers",
                                 marker=dict(size=11, color=colors[k], line=dict(color=INK, width=1.5))), r + 1, c + 1)
    x0, y0, x1, y1 = _view(room)
    fig.update_xaxes(visible=False, range=[x0, x1], fixedrange=True)
    fig.update_yaxes(visible=False, range=[y0, y1], fixedrange=True)
    for i in range(1, rows * 3 + 1):
        fig.update_yaxes(scaleanchor=f"x{i if i > 1 else ''}", row=(i - 1) // 3 + 1, col=(i - 1) % 3 + 1)
    fig.update_layout(height=height * rows, margin=dict(l=4, r=4, t=30, b=4), showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=OUTSIDE, font=dict(family=FONT, size=12, color=INK))
    fig.update_annotations(font=dict(size=12, color=INK))
    for t in fig.data:
        t.update(hoverinfo="skip", showlegend=False)
    return fig


def outcome_bars(stats: dict, labels: dict, height: int = 320):
    """Stacked outcome shares per method (exit / loop / crash / timeout)."""
    cols = {"exit": "#16A34A", "loop": "#D97706", "crash": "#E11D48", "timeout": "#94A3B8"}
    names = {"exit": "Reached exit", "loop": "Loop detected", "crash": "Collision", "timeout": "Out of time"}
    keys = list(stats)
    fig = go.Figure([go.Bar(y=[labels[k] for k in keys], x=[stats[k][f"{o}_rate"] * 100 for k in keys],
                            name=names[o], orientation="h", marker_color=c,
                            text=[f"{stats[k][f'{o}_rate']:.0%}" if stats[k][f"{o}_rate"] >= 0.06 else "" for k in keys],
                            textposition="inside", insidetextanchor="middle")
                     for o, c in cols.items()])
    fig.update_layout(barmode="stack", height=height, margin=dict(l=8, r=8, t=36, b=8),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FBFCFD", font=dict(family=FONT, size=12, color=INK),
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(size=11)))
    fig.update_xaxes(range=[0, 100], ticksuffix="%", gridcolor="#E3E7EE")
    fig.update_yaxes(autorange="reversed", automargin=True)
    return fig

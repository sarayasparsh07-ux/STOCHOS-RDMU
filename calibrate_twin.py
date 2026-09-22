"""Grid search used to choose the twin's turn rates and turning speed.

The dataset pins down forward speed, sensor cap and spike rate directly.  Turn
rates are not observable, so they were chosen by driving the twin with the
recorded controller and minimising the distance between simulated and real
statistics: |action frequencies| (L1) + |median front and left distance|.

    python scripts/calibrate_twin.py
"""
import itertools
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdmu.config import TwinConfig  # noqa: E402
from rdmu.data import class_distribution, load4  # noqa: E402
from rdmu.statespace import threshold_actions  # noqa: E402
from rdmu.twin import WallFollowTwin  # noqa: E402

target_freq = class_distribution()["share"].values
target_med = load4()[["SD_front", "SD_left"]].median().values

rows = []
for sharp_speed, sharp_rate, slight_rate, slight_speed in itertools.product(
        [0.25, 0.5], [-0.5, -0.7, -0.95], [0.2, 0.35], [0.8, 1.0]):
    cfg = replace(TwinConfig(), speed_factor=(1.0, slight_speed, sharp_speed, slight_speed),
                  turn_rate=(0.0, -slight_rate, sharp_rate, slight_rate))
    tw = WallFollowTwin(cfg)
    rng = np.random.default_rng(1)
    P = tw.track_starts(24, rng)
    alive = np.ones(24, bool)
    A, S = [], []
    for _ in range(700):
        sd = tw.sense(P, rng)
        a = threshold_actions(sd[:, 0], sd[:, 1])
        A.append(a[alive]); S.append(sd[alive])
        P, cr = tw.step(P, a, rng, alive)
        alive &= ~cr
    A = np.concatenate(A); S = np.concatenate(S)
    f = np.bincount(A, minlength=4) / len(A)
    m = np.median(S[:, :2], axis=0)
    loss = np.abs(f - target_freq).sum() + np.abs(m - target_med).sum()
    rows.append((loss, sharp_speed, sharp_rate, slight_rate, slight_speed, f.round(3), m.round(3), 1 - alive.mean()))

for r in sorted(rows, key=lambda r: r[0])[:8]:
    print(f"loss={r[0]:.3f} sharp_speed={r[1]} sharp_rate={r[2]} slight_rate={r[3]} slight_speed={r[4]} "
          f"freq={r[5]} median_front_left={r[6]} crash={r[7]:.2f}")
print("Chosen (config.py): sharp_speed 0.25, sharp_rate -0.50, slight_rate 0.35, slight_speed 1.0 — "
      "within 0.06 of the best loss and with tighter, more wall-hugging corners than slight_rate 0.2.")

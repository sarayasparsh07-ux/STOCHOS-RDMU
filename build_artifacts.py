"""Precompute the default results so the Streamlit app opens instantly.

    python scripts/build_artifacts.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdmu.config import DEFAULT          # noqa: E402
from rdmu.pipeline import ARTIFACT, load_or_run  # noqa: E402

if __name__ == "__main__":
    if ARTIFACT.exists():
        ARTIFACT.unlink()
    t = time.time()
    res = load_or_run(DEFAULT, progress=lambda m: print("•", m, flush=True))
    print(f"done in {time.time() - t:.1f}s -> {ARTIFACT}")
    for k, v in res["eval_short"].items():
        print(f"{k:7s} J={v['return_mean']:6.2f}±{v['return_ci95']:.2f}  Δ vs expert={v['diff_vs_expert']:+.2f}±{v['diff_ci95']:.2f}"
              f"  crash={v['crash_rate']:.2f}  long crash={res['eval_long'][k]['crash_rate']:.2f}"
              f"  band={res['eval_long'][k]['band_share']:.2f}")
    print("model gap", res["model_gap"])
    print("HJ", res["hj"].best_theta.round(3), "MC", res["mc"].best_theta.round(3))

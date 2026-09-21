"""Kaggle / UCI SCITOS-G5 wall-following dataset: loading and analysis.

Key findings reproduced by this module
--------------------------------------
1. The three files share the same 5,456 time steps (9 Hz); the 2-sensor file is
   an exact subset of the 4-sensor file.
2. The rows are a genuine trajectory (high lag-1 autocorrelation, labels
   persist), so empirical transitions (s_t -> s_t+k) can be estimated.
3. The class label is (almost) a deterministic threshold rule of SD_front and
   SD_left.  Hence each state contains a single observed action and the data
   alone cannot identify P(s'|s,a) for the other actions.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text

from .config import ACTIONS, DATA_DIR, StateSpaceConfig
from .statespace import encode, threshold_actions

SD_COLS = ["SD_front", "SD_left", "SD_right", "SD_back"]


@lru_cache(maxsize=1)
def load_all() -> dict[str, pd.DataFrame]:
    d4 = pd.read_csv(DATA_DIR / "sensor_readings_4.csv", header=None, names=SD_COLS + ["Class"])
    d2 = pd.read_csv(DATA_DIR / "sensor_readings_2.csv", header=None, names=SD_COLS[:2] + ["Class"])
    d24 = pd.read_csv(DATA_DIR / "sensor_readings_24.csv", header=None,
                      names=[f"US{i}" for i in range(1, 25)] + ["Class"])
    for d in (d4, d2, d24):
        d["Class"] = d["Class"].str.strip()
    return {"4": d4, "2": d2, "24": d24}


def load4() -> pd.DataFrame:
    df = load_all()["4"].copy()
    df["t_s"] = np.arange(len(df)) / 9.0
    df["action"] = df["Class"].map({a: i for i, a in enumerate(ACTIONS)})
    return df


def quality_report() -> dict:
    d = load_all()
    d4, d2, d24 = d["4"], d["2"], d["24"]
    X = d4[SD_COLS]
    consecutive_dup = int((d4.iloc[1:].values == d4.iloc[:-1].values).all(axis=1).sum())
    return {
        "rows": {k: len(v) for k, v in d.items()},
        "columns": {k: v.shape[1] for k, v in d.items()},
        "missing": {k: int(v.isna().sum().sum()) for k, v in d.items()},
        "duplicate_rows_4": int(d4.duplicated().sum()),
        "consecutive_duplicates_4": consecutive_dup,
        "file2_subset_of_file4": bool(np.allclose(d2[SD_COLS[:2]].values, d4[SD_COLS[:2]].values)
                                      and (d2.Class.values == d4.Class.values).all()),
        "labels_identical_24_vs_4": bool((d24.Class.values == d4.Class.values).all()),
        "at_cap_5m": {c: int((X[c] >= 5.0).sum()) for c in SD_COLS},
        "values_above_5_in_24": int((d24.iloc[:, :24] > 5.0).sum().sum()),
        "max_24": float(d24.iloc[:, :24].values.max()),
    }


def describe() -> pd.DataFrame:
    """Location, spread, shape and outlier statistics of the simplified distances."""
    d = load4()[SD_COLS]
    q1, q3 = d.quantile(0.25), d.quantile(0.75)
    iqr = q3 - q1
    out = pd.DataFrame({
        "mean": d.mean(), "median": d.median(), "mode": d.mode().iloc[0], "std": d.std(), "variance": d.var(),
        "min": d.min(), "Q1": q1, "Q3": q3, "IQR": iqr, "max": d.max(), "skew": d.skew(),
        "outliers (1.5·IQR)": ((d < q1 - 1.5 * iqr) | (d > q3 + 1.5 * iqr)).sum(),
        "at 5 m cap": (d >= 5.0).sum(),
    })
    return out.round(3)


def class_distribution() -> pd.DataFrame:
    c = load4().Class.value_counts()
    return pd.DataFrame({"count": c, "share": (c / c.sum()).round(4)}).reindex(ACTIONS)


def time_series_evidence() -> dict:
    d = load4()
    X = d[SD_COLS].values
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(X), 5000), rng.integers(0, len(X), 5000)
    cls = d.Class.values
    runs = np.diff(np.flatnonzero(np.r_[True, cls[1:] != cls[:-1], True]))
    up = (X[1:-1] - X[:-2] > 0.8) & (X[1:-1] - X[2:] > 0.8)
    f = d.SD_front.values
    k = 3
    v = (f[k:] - f[:-k]) / (k / 9)
    m = (cls[:-k] == ACTIONS[0]) & (cls[k:] == ACTIONS[0]) & (f[:-k] < 1.3) & (np.abs(v) < 1)
    return {
        "lag1_autocorr": {c: round(float(d[c].autocorr(1)), 3) for c in SD_COLS},
        "median_abs_change_consecutive": dict(zip(SD_COLS, np.median(np.abs(np.diff(X, axis=0)), 0).round(3))),
        "median_abs_change_random_pairs": dict(zip(SD_COLS, np.median(np.abs(X[i] - X[j]), 0).round(3))),
        "label_persistence": round(float((cls[1:] == cls[:-1]).mean()), 4),
        "n_label_runs": int(len(runs)),
        "median_run_length": float(np.median(runs)),
        "spike_rate": dict(zip(SD_COLS, up.mean(0).round(4))),
        "forward_speed_mps": round(float(np.median(-v[m])), 3),
        "duration_s": round(len(d) / 9.0, 1),
        "lap_time_s": round(len(d) / 9.0 / 4, 1),
    }


def feature_relevance() -> dict:
    d = load4()
    y = d.Class.to_numpy(dtype=str)
    cv = KFold(5, shuffle=False)   # blocked folds: respects the time order
    subsets = {"front": ["SD_front"], "left": ["SD_left"], "front+left": SD_COLS[:2],
               "front+left+right": SD_COLS[:3], "all four": SD_COLS}
    acc = {k: round(float(cross_val_score(DecisionTreeClassifier(random_state=0),
                                          d[v].values, y, cv=cv).mean()), 4) for k, v in subsets.items()}
    mi = mutual_info_classif(d[SD_COLS].values, y, random_state=0)
    rf = RandomForestClassifier(200, random_state=0).fit(d[SD_COLS].values, y)
    return {"blocked_cv_accuracy": acc,
            "mutual_information": dict(zip(SD_COLS, mi.round(3))),
            "rf_importance": dict(zip(SD_COLS, rf.feature_importances_.round(3)))}


def extract_expert_rule() -> dict:
    d = load4()
    X = d[SD_COLS[:2]].values
    tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, d.Class.to_numpy(dtype=str))
    text = export_text(tree, feature_names=["SD_front", "SD_left"], decimals=3)
    pred = np.array(ACTIONS)[threshold_actions(X[:, 0], X[:, 1])]
    mism = np.flatnonzero(pred != d.Class.to_numpy(dtype=str))
    # do mismatches sit next to a label change? (sign of a one-sample lag)
    cls = d.Class.to_numpy(dtype=str)
    near_change = np.mean([(i > 0 and cls[i] != cls[i - 1]) or (i + 1 < len(cls) and cls[i] != cls[i + 1])
                           or (i > 1 and cls[i - 1] != cls[i - 2]) for i in mism]) if len(mism) else 0.0
    shifted = np.array(ACTIONS)[threshold_actions(X[1:, 0], X[1:, 1])]
    return {"tree_text": text,
            "tree_train_accuracy": round(float(tree.score(X, d.Class.to_numpy(dtype=str))), 4),
            "rounded_rule_accuracy": round(float((pred == cls).mean()), 4),
            "rule_vs_previous_label_accuracy": round(float((shifted == cls[:-1]).mean()), 4),
            "n_mismatches": int(len(mism)),
            "mismatch_share_near_label_change": round(float(near_change), 3)}


def arc_membership() -> dict:
    """Which of the 24 raw sensors reproduce each simplified distance (documents
    a README inconsistency)."""
    d = load_all()
    U = d["24"].iloc[:, :24].values
    X = d["4"][SD_COLS].values
    out = {}
    for j, c in enumerate(SD_COLS):
        best = (0, None)
        for w in range(1, 7):
            for s in range(24):
                idx = [(s + k) % 24 for k in range(w)]
                acc = np.isclose(U[:, idx].min(axis=1), X[:, j], atol=1e-3).mean()
                if acc > best[0] + 1e-9:
                    best = (acc, idx)
        out[c] = {"sensors": [f"US{i + 1}" for i in best[1]], "match": round(float(best[0]), 4),
                  "arc_deg": 15 * (len(best[1]) - 1)}
    return out


def empirical_expert_transitions(cfg: StateSpaceConfig, k: int = 3) -> dict:
    """Transitions observed in the recording at the decision interval (k samples)."""
    d = load4()
    s = encode(d.SD_front.values, d.SD_left.values, cfg)
    a = d.action.values
    n = cfg.n_grid
    C = np.zeros((n, 4, n))
    np.add.at(C, (s[:-k], a[:-k], s[k:]), 1)
    occupancy = np.bincount(s, minlength=n) / len(s)
    observed_actions = (C.sum(-1) > 0)
    return {"counts": C, "occupancy": occupancy, "observed": observed_actions,
            "states": s, "actions": a}

# STOCHOS: decisions under uncertainty for a wall-following robot

## Project overview
This is a course project for Reasoning and Decision Making Under Uncertainty (RDMU). A SCITOS-G5 robot follows the walls of a room using four sonar distances.

The recorded run is turned into one decision problem, which is solved with four methods:
- a Markov decision process solved by value iteration (checked with policy iteration)
- approximate dynamic programming (fitted Q-iteration)
- Monte Carlo policy search
- Hooke–Jeeves pattern search

All four drive the same simulated robot, together with the robot's original controller and a random baseline. They are compared under identical conditions. The Streamlit app shows the robot moving and explains each decision it makes.

## Dataset
**Source.** Kaggle / UCI, *Wall-Following navigation task with mobile robot SCITOS-G5* (Freire et al., 2009). Files are in `data/`.

**Structure.** Three files share the same 5,456 time steps, recorded at 9 Hz (606 s, four laps):
- 24 raw sonar readings
- 4 simplified distances
- 2 simplified distances

Each row carries one of four labels: Move-Forward 40.4%, Sharp-Right-Turn 38.4%, Slight-Right-Turn 15.1%, Slight-Left-Turn 6.0%.

**Quality.**
- No missing values.
- 50 repeated rows, 33 of them back-to-back. These come from sub-millimetre motion at 9 Hz and are kept.
- Readings are capped at 5 m (91 back readings sit at the cap).
- 48 raw readings exceed the cap slightly.
- Distributions are strongly right-skewed (skew of SD_left is 5.0).
- Nothing is removed or altered.

**Findings that shape the model.**
1. **The labels follow an exact three-threshold rule.** Sharp right if SD_front ≤ 0.900 m; otherwise slight right if SD_left ≤ 0.494 m; forward if SD_left ≤ 0.901 m; otherwise slight left. This reproduces 100% of labels.
2. **The rows are a trajectory.** Lag-1 autocorrelation is 0.78–0.91 and the label persists 92.8% of the time, so transitions between states can be measured.
3. **Each visited state shows exactly one action.** The data therefore cannot identify P(s′|s,a) for the other actions. A simulator calibrated on the data supplies these, and the recording is used to validate it.
4. **Front and left carry the information.** Blocked cross-validated accuracy is 99.96% with those two sensors. Random-forest importance is only 0.02 for right and 0.05 for back.

**Documented conflicts.**
- The README and the class slide say every distance is a 60° arc. The numbers show SD_left is a 30° arc (US18–US20) and SD_back a 15° arc (US23–US24).
- The README's angle labels put the front sensor at 180° and the back at 0°, yet SD_front is built around US13.
- The simulator follows the documented 60° arcs; the arc width is configurable.

## Methodology
**MDP 𝓜 = (S, A, P, R, γ).**
- **States S:** 6 front bins × 7 left bins plus an absorbing Crash state (43 in total). The bin edges contain the recorded thresholds, so the original controller is exactly a policy in this MDP.
- **Actions A:** the four dataset classes.
- **Transitions P:** a Monte Carlo estimate from about 167,000 simulated moves, trying all four actions from every sampled pose.
- **Reward R:**
  - +1 in the tracking band (front > 0.90 m, 0.494 < left ≤ 0.901 m)
  - −0.5 in the caution zone
  - −2 in the danger zone
  - −0.5 when the wall is lost
  - −25 for a crash
  - a small turning cost

  The zone thresholds come from the data; the values are assumptions.
- **Discount and tolerance:** γ = 0.95 and tolerance 10⁻⁶.

**Value iteration.** V_{k+1}(s) = max_a [R(s,a) + γ Σ P(s′|s,a) V_k(s′)], iterated until ‖ΔV‖∞ < tolerance. The error bound γ·tol/(1−γ) is reported. The policy is extracted as π(s) = argmax_a Q(s,a) and saved to `outputs/optimal_value_function.csv`. Policy iteration is run as an independent check and returns the same policy.

**ADP.** Q̂(x,a) = φ(x)ᵀθ_a on the continuous (front, left) readings.
- φ is an 81-function grid of normalised Gaussian radial basis functions over log-distance, giving 324 parameters.
- It is trained by fitted Q-iteration with ridge regression on the same simulated transitions as the MDP.

**Policy search.** The policy π_θ uses the recorded controller's rule with θ = (τ_front, τ_left-low, τ_left-high). The objective is J(θ) = E[Σ γᵗ r_t] over 150 decisions, estimated from 24 episodes. The seeds are fixed (common random numbers), so J is deterministic in θ.
- **Monte Carlo:** 60 random feasible θ, including the recorded one. The top 5 are re-scored on new seeds with 96 episodes.
- **Hooke–Jeeves:**
  - Starts from the recorded θ.
  - Exploratory moves try ±Δ on each threshold (Δ = 0.10 / 0.05 / 0.10 m).
  - Pattern moves are accepted only after they are evaluated.
  - The step halves when no move improves J.
  - It stops when Δ falls below 0.0125 m or after 80 evaluations.

**Evaluation.** All controllers use:
- the same 200 start poses, noise and reward, on seeds never used in training
- paired differences against the recorded controller, with 95% intervals
- 40 additional runs of 300 s for long-run metrics

## Results (default settings)
| Method | Discounted return (±95% CI) | Paired Δ vs recorded controller | Success (300 s, no crash) | Time in band | Runtime |
|---|---|---|---|---|---|
| Recorded controller | 5.34 ± 1.56 | — | 100% | 60% | 0.0 s |
| MDP · value iteration | 7.57 ± 1.61 | +2.24 ± 0.59 | 100% | 78% | 2.9 s |
| ADP · fitted Q-iteration | 9.32 ± 1.60 | +3.98 ± 0.52 | 100% | 82% | 15.0 s |
| Monte Carlo policy search | 9.08 ± 1.62 | +3.74 ± 0.40 | 100% | 86% | 7.3 s |
| Hooke–Jeeves policy search | 9.65 ± 1.62 | +4.31 ± 0.42 | 100% | 87% | 4.8 s |
| Random baseline | -10.46 ± 2.16 | -15.80 ± 1.47 | 0% | 13% | 0.0 s |

- All four learned methods improve on the recorded controller, and every paired interval excludes zero.
- The intervals of ADP, Monte Carlo search and Hooke–Jeeves overlap, so the data do not separate those three.
- The MDP's own prediction differs from its simulated return. This gap, caused by grouping readings into cells, is shown in the app.

The starter notebook was re-executed and checked:
- Value iteration is correct but has no tolerance, no policy output and no CSV.
- Its Hooke–Jeeves accepts pattern moves without evaluating them. It stops at (−0.7, −0.7) from (1.3, −0.7), and an integer start never moves.
- Its "Monte Carlo policy search" only evaluates one random policy; −32.93 agrees with the exact −32.68 within sampling noise.
- The portfolio block uses `returns.mean()` without an axis, so every portfolio has the same expected return.

## Installation
```bash
pip install -r requirements.txt
```

## Running the application
```bash
streamlit run app.py
```
Default results are precomputed in `artifacts/`, so the app opens in seconds. Changing the model settings in the sidebar recomputes everything, which takes about 35 s.

Rebuild the results, recalibrate the simulator, or run the tests:
```bash
python scripts/build_artifacts.py
python scripts/calibrate_twin.py
python -m pytest -q
```

## Project structure
| Path | Purpose |
|---|---|
| `app.py` | Streamlit app. Tabs: Robot, Dataset, MDP, ADP, Policy search, Comparison, Notebook audit, Method notes |
| `rdmu/config.py` | All parameters, each labelled with its provenance: data, slide, assumption, calibrated or physical |
| `rdmu/data.py` | Loading, quality audit, statistics, rule extraction, time-series evidence, empirical transitions |
| `rdmu/statespace.py` | State discretisation, reward zones, reward function |
| `rdmu/twin.py` | Vectorised robot simulator: kinematics, 24-beam sonar, collisions, start poses |
| `rdmu/model.py` | Estimation of P and R; validation of the simulator against the recording |
| `rdmu/mdp.py` | Value iteration, policy iteration, exact policy evaluation, CSV export |
| `rdmu/adp.py` | Radial basis features and fitted Q-iteration |
| `rdmu/policy_search.py` | Monte Carlo objective, random search, generic Hooke–Jeeves |
| `rdmu/policies.py` | Common policy interface with per-decision explanations |
| `rdmu/evaluation.py` | Batched evaluation, metrics, single-episode recorder |
| `rdmu/legacy_audit.py` | Verification of `Day_5.ipynb` |
| `rdmu/plots.py` | Every chart (Plotly, one colour system) |
| `rdmu/pipeline.py` | Runs all methods and caches the results |
| `scripts/` | Rebuilding the precomputed results and calibrating the simulator |
| `tests/` | 19 tests: data, states, rewards, simulator, all four methods, policy-driven robot, app switching, step/reset, invalid input |
| `outputs/optimal_value_function.csv` | Slide-required output |

## Methodological assumptions
- **Room:** 6.4 × 4.2 m with a pillar and a recess. The real floor plan is not published.
- **Decisions:** one every 1/3 s. The 9 Hz sampling rate comes from the data.
- **Motion:** forward speed of 0.155 m/s is measured from the data. Turn rates are calibrated to the recorded sensor medians (`scripts/calibrate_twin.py`). Motion and sensor noise are assumed.
- **Reward values:** assumed. The zone thresholds come from the data.
- **Discount:** γ = 0.95 gives an effective horizon of about 20 decisions (≈ 7 s), long enough to see a wall 1 m ahead.
- **Configuration:** every value above can be changed in `rdmu/config.py`, and γ, tolerance and budgets in the app sidebar.

## Limitations
- **The simulator is calibrated, not identified.** It matches the recorded front and left distances but uses sharp turns less often than the real robot (15% against 38%). Absolute returns are specific to the simulator; the comparison under identical conditions is the transferable result.
- **The state is not fully Markov.** The robot's heading is hidden within each cell.
- **Policy search covers one controller family.** A richer family could perform better but would be harder to interpret.

## Reproducibility
Every random draw uses a seeded NumPy generator:
- training seed 7
- evaluation seed 2026
- the demo seed set in the sidebar

The same settings always reproduce the same numbers. Tests check that repeated evaluations are identical.

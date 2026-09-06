# Third enhancement: overestimation comparison

From the repository root:

```powershell
python -m src.compare_overestimation
```

If Windows resolves `python` to a nonworking app alias, the verified installation on this computer is:

```powershell
& "$env:LOCALAPPDATA/Python/pythoncore-3.14-64/python.exe" -m src.compare_overestimation
```

Requires NumPy, PyTorch and Matplotlib. The default runs seeds 0–9, 5,000 diagnostic interactions and 81,000 grid interactions **per algorithm per seed**, with up to four CPU worker processes. Each run reports progress and writes its own JSON when complete. Results go into a new timestamped directory under `compare_figures/overestimation/`; existing figures are never overwritten. Figures open after training; use `--no-show` when running unattended.

Quick functionality check (not thesis evidence):

```powershell
python -m src.compare_overestimation --seeds 0 1 --diagnostic-steps 100 --grid-steps 200 --workers 1 --no-show
```

Options: `--seeds 0 1 ...`, `--diagnostic-steps N`, `--grid-steps N`, `--workers N`, `--output-dir PATH`, `--no-show`, `--reference-replay`.

The default uses a runner-local indexed replay adapter to avoid repeatedly scanning Python dictionaries and walking deque indices. It preserves the production buffer's chronological probabilities, random-number calls, transition order, importance weights and priority updates. Tests verify exact sample equality across buffer rollover and exact equality of training measurements. The production agent update methods remain unchanged. `--reference-replay` uses the original slower sampler for independent verification; this affects runtime, not the algorithm being compared.

## What the figures demonstrate

- `diagnostic_q_values.png`: a continuing one-state, four-action task with independent normal rewards of mean zero and standard deviation one. Every true action value is zero. Positive learned maximum Q-values expose overestimation. Ten-step blocks are reporting/schedule intervals, not terminal episodes. EQLBPW uses its production network with a constant 29-zero observation.
- `grid_overestimation.png`: the existing first 9×9 preset, static obstacles, fixed start and goal, four movement actions. Both algorithms receive expected rewards −1 for collision, +1 for goal, and zero otherwise; independent normal noise with standard deviation 0.5 is added to nonterminal rewards. The goal terminates, collisions leave the agent in place, and boundary moves clamp to the grid. Episodes truncate at 162 steps, while learning correctly continues to bootstrap across truncation. The budget counts actual interactions, not episode labels.
- `q_value_accuracy.png`: absolute error over all four actions and all reachable nonterminal grid states (one state for the diagnostic). This reveals underestimation or other inaccuracies that a positive-bias plot alone could hide.
- `final_comparison.png`: bias averaged over the final 20% of noninitial checkpoints, plus the fraction of seeds whose final greedy grid policy reaches the goal. Success is measured on this one fixed map/start, not on held-out maps.

At each evaluation, the chosen action is the greedy action of the learned agent. Signed error is `Q_estimated(s, a_greedy) - Q_reference(s, a_greedy)`. Positive overestimation averages `max(error, 0)`. Grid reference Q-values come from value iteration on the exact expected-reward model with discount 0.95 and Bellman residual below 1e-10. This is a comparison against **optimal action values**, not a Monte Carlo estimate of the learned policy's return. The diagnostic reference is exactly zero.

The shaded curves show pointwise 95% percentile bootstrap intervals across seeds. Summary intervals resample seed-level final-window averages, with paired QLBPW-minus-EQLBPW differences. A positive difference favors EQLBPW on positive bias or absolute error; signed bias must instead be interpreted relative to zero. A one-seed run has no uncertainty interval. All seeds, failures and unfavorable results are retained. These intervals are descriptive and not simultaneous confidence bands or a multiple-testing correction.

## How this relates to the enhancement

The runner calls the current agents' actual update methods. EQLBPW selects a next action with its main network and evaluates it with its target network. QLBPW bootstraps from the maximum of its own table. This comparison tests whether the complete EQLBPW implementation shows the lower overestimation intended by the third enhancement.

As requested, only these two algorithms are shown. They also differ in representation, replay, optimizer and exploration, so this experiment **does not isolate Double DQN's causal contribution**. Lower positive bias alone does not imply better accuracy or better paths: read it alongside absolute error and success. The stochastic grid reward is a diagnostic addition, not a claim that the original simulator already has noisy rewards. Expected rewards are shared in this runner, rather than comparing the production environments' incompatible scales.

Existing learning rates, network, replay methods/capacities and standalone exploration behavior are preserved. QLBPW uses fixed epsilon 0.9 and one ranked replay update per interaction (its configured batch size is not an update gate in the standalone simulator). EQLBPW decays epsilon after each episode/block, syncs every 20 episodes/blocks, and uses batches of 64 after warm-up. Its beta advances by interaction progress to reach 1. Both agents use gamma 0.95 here. Equal interactions do not mean equal updates or compute. Environment reward-noise draws use a dedicated generator paired by seed and interaction index, independent of agent/replay random draws.

The unchanged EQLBPW production replay sampler uses sampling without replacement with conventional PER importance weights. That known approximation is retained here to avoid changing other enhancements; it limits attributing outcomes solely to Double DQN.

## Reproducibility and verification

`config.json` records settings, versions, exact obstacles and hashes of the runner, agents and presets. `reference_q.json` contains grid reference values. Per-run JSON includes all measurements, update counts and final paths; `measurements.csv` combines the measurements; `summary.json` records uncertainty and the outcome without forcing a winner.

```powershell
python -m pytest tests/test_compare_overestimation.py -q
```

Tests exercise analytic grid values, zero-reference diagnostic values, production observation compatibility, actual production Double DQN targets and terminal masks, underestimation accounting, paired summaries, reproducibility, resets and exact interaction budgets. Existing trainers and the first/second-enhancement experiment files are not imported or changed.

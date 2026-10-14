# GitHub Copilot Instructions

## Project context

This repository implements and compares two reinforcement-learning approaches for grid-based path planning and obstacle avoidance:

- **QLBPW**: a tabular Q-learning baseline with rank-based prioritized experience selection and priority-adjusted learning rates.
- **EQLBPW**: an enhanced Deep Q-Learning approach using a neural Q-network, Double DQN targets, prioritized experience replay, importance-sampling weights, and path-aware priority weighting.

The broader research motivation is to study state-space growth, local-optimum / exploration difficulties, and Q-value overestimation in grid navigation. The project is an experimental research codebase, so preserving scientific validity, comparable evaluation conditions, reproducibility, and honest reporting of results is as important as code quality.

The README describes the work as an extension of Chunlei Wang, Xiao Yang, and He Li's _Improved Q-Learning Applied to Dynamic Obstacle Avoidance and Path Planning_ (2022). Treat claims about improvements as hypotheses to evaluate, not guaranteed outcomes.

## Main goals

When changing this repository, prioritize these goals in order:

1. Preserve the intended algorithm semantics and environment rules.
2. Keep QLBPW and EQLBPW comparisons fair and their metrics interpretable.
3. Make experiments reproducible from explicit seeds and recorded settings.
4. Keep training, evaluation, and visualization responsibilities understandable and separate.
5. Add focused tests for behavior changes and edge cases.
6. Avoid unrelated refactors, dependency churn, or changes to saved experiment artifacts.

## Repository map

- `README.md`: research motivation, objectives, and launch commands.
- `requirements.txt`: pinned dependencies currently listed by the project. Before adding or changing dependencies, check actual imports and the documented setup; do not assume this file already lists every runtime dependency.
- `docs/ORIGINAL ALGO.md`: notes describing the baseline algorithm.
- `docs/PROPOSED ALGO.md`: proposed enhancement design, including Double DQN, prioritized replay, importance sampling, reward shaping, and path-aware weights.
- `docs/overestimation-demo.md`, `docs/test_cases.md`, and `docs/commit-definitions.md`: experiment and development notes. Read the relevant document before changing the behavior it describes.
- `docs/QLBPW_Algorithm.jpg`, `docs/EQLBPW_Algorithm.jpg`: algorithm diagrams.
- `src/QLBPW/agent.py`: tabular QLBPW agent and its replay buffer.
- `src/QLBPW/environment.py`, `src/QLBPW/tracker.py`, `src/QLBPW/simulator.py`: QLBPW environment interaction, tracking, and simulation.
- `src/EQLBPW/agent.py`: neural-network agent, prioritized replay buffer, Double DQN update, and checkpointing.
- `src/EQLBPW/environment.py`, `src/EQLBPW/tracker.py`, `src/EQLBPW/simulator.py`: EQLBPW environment interaction, tracking, and simulation.
- `src/agent_config.py`, `src/env_config.py`, `src/env_settings.py`: agent/environment settings and preset environments.
- `src/app.py`: Streamlit dashboard that lets users switch between the algorithms.
- `src/Astar.py`: A\* path-planning implementation/reference.
- `src/compare.py`, `src/compare_overestimation.py`, `src/overestimation.py`, `src/overestimation_benchmark.py`: comparison and overestimation experiments. Inspect imports and each script's actual entry point before assuming they share identical evaluation logic.
- `src/visualizer.py`: plotting and visualization utilities.
- `tests/`: regression and experiment-correctness tests, including `tests/test_compare_overestimation.py`.
- `compare_figures/`: generated experiment outputs (plots, CSVs, JSON summaries, per-seed data). Treat these as research artifacts; do not overwrite, regenerate, or mass-edit them unless specifically requested.

The repository contains multiple scripts with similar names and multiple experiment pathways. Do not assume that a benchmark script, dashboard, simulator, and production agent all use exactly the same reward function or settings. Trace the relevant call path before making a change.

# Semantic Rules

### Environment semantics

- Actions are discrete grid moves; inspect the environment implementation for the exact action-to-direction mapping rather than assuming an ordering.
- Read `src/QLBPW/environment.py` and `src/EQLBPW/environment.py` before changing transition rules. The two environments may expose different state representations or return signatures.
- Goal arrival, obstacle collision, boundary handling, and maximum-step termination are distinct cases. Do not conflate clamped movement at a grid edge with an obstacle collision.
- Terminal handling must be consistent between the transition, reward, replay `done` flag, tracker, and evaluator.
- Dynamic-obstacle experiments can change the obstacle set during training or evaluation. Keep the timing and randomization explicit and reproducible.
- EQLBPW currently consumes a feature vector (the code and tests use a 29-element state representation in relevant experiment paths). If state encoding changes, update the network input dimension, every producer/consumer, fixtures, and tests together. Prefer a single source of truth over scattered hard-coded dimensions.

### Reward and metric semantics

- Reward functions are not necessarily identical across every script. Some comparison paths deliberately normalize rewards (for example, goal = +1, collision = -1, ordinary movement = 0) to make an experiment comparable. Confirm the exact experiment contract before changing rewards.
- Do not silently change reward shaping, terminal rewards, collision penalties, distance-progress calculations, discounting, or reset behavior.
- Keep estimated value, realized discounted return, signed bias, positive overestimation, absolute Q error, and goal success as separate concepts. Underestimation is not the same as accurate estimation.
- Clearly distinguish training metrics from held-out/evaluation metrics. Avoid evaluating only with exploratory actions when the intent is to measure the learned greedy policy.
- When comparing algorithms, align environment/preset, reward semantics, discount factor, episode or interaction budget, seed policy, and evaluation trials wherever the experiment requires a fair comparison. If alignment is impossible, document the difference.
- Never claim that EQLBPW outperforms QLBPW without evidence from the relevant experiment and its uncertainty/variation across seeds.

## Coding conventions

- Use Python and follow the style of the surrounding module. Prefer clear names, small functions, and type hints when they improve understanding without forcing a broad rewrite.
- Keep algorithm logic in the agent, transition logic in the environment, shared parameters in configuration modules, and presentation in the dashboard/visualizer.
- Avoid duplicating reward functions, action mappings, state encodings, and metric formulas. If centralizing shared logic, do so carefully so existing experiment semantics remain explicit.
- Avoid hidden global state and import-time training or expensive work. Keep scripts safe to import where practical.
- Use explicit random seeds for Python's `random`, NumPy, and PyTorch where relevant. If a code path uses CUDA or additional random generators, handle those as needed and document reproducibility limits.
- Preserve deterministic behavior in tests. Avoid relying on wall-clock time, network access, GUI interaction, or large training runs in unit tests.
- Handle empty buffers, insufficient samples, terminal transitions, zero/near-zero priorities, invalid probabilities, and non-finite values defensively.
- For prioritized sampling, probabilities must be finite, nonnegative, and normalized before passing them to NumPy. Avoid division by zero if priorities sum to zero; retain a positive priority epsilon.
- When adding tensors, check batch dimensions and broadcasting carefully. Importance weights should align with per-transition losses and should not accidentally broadcast across the wrong axis.
- Checkpoint formats are part of practical compatibility. If adding fields, load old checkpoints with sensible defaults where feasible. Do not silently discard optimizer state, replay contents, epsilon, or random states when resume-training behavior depends on them.
- Keep file paths portable. Avoid assuming the current working directory unless the existing command explicitly requires it; prefer paths derived from a known project/module root when touching path handling.
- Do not introduce a new library if the standard library or an existing dependency is sufficient. Update dependency documentation when a new runtime dependency is genuinely required.

## Experiment artifacts and reporting

- Treat `compare_figures/` as generated output, not source code. Do not edit result JSON/CSV files by hand to make outcomes look better.
- Preserve per-seed results and unfavorable results. Never drop failed runs or negative results solely because they weaken a conclusion.
- Record configuration, algorithm, task/preset, seed, interaction/episode budget, and relevant summary statistics for new experiments.
- If a change alters experiment methodology, update the relevant documentation and make the change visible in result metadata when appropriate.
- For statistical summaries, report sample counts and uncertainty when supported. Do not interpret a confidence interval as proof of general superiority if the experiment is underpowered or narrowly scoped.
- Prefer a small smoke test before a long benchmark. Do not launch a large experiment or regenerate many artifacts unless the user asks.

## Dependency and runtime guidance

- The README documents these launch commands from the `src` directory:
  - `streamlit run app.py`
  - `python -m EQLBPW.simulator`
  - `python -m QLBPW.simulator`
- Install dependencies from `requirements.txt`, but verify that the runtime dependencies needed by the specific command are installed. The dashboard and EQLBPW paths use Streamlit and PyTorch; comparison/visualization scripts may also import libraries not currently listed in the minimal requirements file.
- Avoid changing pinned versions as a side effect of unrelated work. If a dependency is missing from the manifest, propose or make a targeted manifest/documentation update rather than assuming the environment will provide it.

## How Copilot should approach a task

1. Restate the intended behavior and identify the affected algorithm path (QLBPW, EQLBPW, environment, comparison/benchmark, dashboard, or visualization).
2. Inspect the relevant implementation, configuration, documentation, and tests before proposing edits.
3. Trace data end-to-end: state representation -> action -> environment transition -> reward/done/info -> replay storage -> target/loss/priority -> tracker/metrics.
4. Make the smallest coherent change that solves the task. Preserve existing public interfaces unless the task requires a change.
5. Add or update tests for the behavior and edge cases; update docs if the algorithm or experiment methodology changes.
6. Run focused tests and then broader tests when practical.
7. Summarize files changed, algorithmic implications, tests actually run, and any remaining limitations.

## Do not do these things

- Do not silently replace Double DQN with standard DQN.
- Do not bootstrap from terminal states.
- Do not mix up QLBPW and EQLBPW semantics or rename them without a compatibility reason.
- Do not change reward functions or experiment budgets merely to improve reported metrics.
- Do not compare results from mismatched reward definitions as if they were directly comparable.
- Do not hard-code a new state dimension in only one place.
- Do not overwrite generated experiment outputs or trained checkpoints unless requested.
- Do not add large architectural abstractions or refactor unrelated modules for a narrow bug fix.
- Do not fabricate results, test outcomes, references, or claims of convergence.
- Do not assume that a design document's pseudocode perfectly matches production code; use the current implementation and tests as the source of truth, and update docs when a deliberate divergence is introduced.

## Source-of-truth order

When sources disagree, use this order and call out meaningful discrepancies:

1. The user's explicit task requirements.
2. Current production code and existing tests for actual runtime behavior.
3. Experiment-specific benchmark code for that benchmark's methodology.
4. Current configuration and README for intended usage.
5. Algorithm notes and diagrams as design intent.

If a discrepancy affects scientific validity or changes the reported results, do not guess. Explain the ambiguity and ask for clarification when needed.

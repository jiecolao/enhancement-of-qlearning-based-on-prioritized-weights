# Further Enhancement of Improved Q-Learning for Dynamic Obstacle Avoidance and Path Planning

An experimental reinforcement learning project exploring prioritized experience replay, Double Deep Q-Networks (Double DQN), reward shaping, and path-aware weighting for obstacle avoidance and path planning.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Reinforcement Learning](https://img.shields.io/badge/AI-Reinforcement%20Learning-purple)](https://en.wikipedia.org/wiki/Reinforcement_learning)
[![Status](https://img.shields.io/badge/Project-Research%20%26%20Development-orange)](https://github.com/jiecolao/enhancement-of-qlearning-based-on-prioritized-weights)

## Table of Contents

- [Overview](#overview)
- [Research Motivation](#research-motivation)
- [Project Objectives](#project-objectives)
- [Algorithms and Enhancements](#algorithms-and-enhancements)
- [How the Proposed Algorithm Works](#how-the-proposed-algorithm-works)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Using the Streamlit Dashboard](#using-the-streamlit-dashboard)
- [Configuration and Customization](#configuration-and-customization)
- [Experiments and Evaluation](#experiments-and-evaluation)
- [Running Tests](#running-tests)
- [Troubleshooting](#troubleshooting)
- [Guidelines for Users and Contributors](#guidelines-for-users-and-contributors)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [References](#references)
- [License](#license)

## Overview

This repository investigates a further enhancement of an improved Q-learning approach for dynamic obstacle avoidance and path planning.

The project builds upon the work of Chunlei Wang, Xiao Yang, and He Li, *Improved Q-Learning Applied to Dynamic Obstacle Avoidance and Path Planning* (2022). It explores how reinforcement learning agents can learn to navigate a grid-based environment, avoid obstacles, and reach a designated goal through repeated interaction with the environment.

The repository provides two implementations for experimentation:

| Implementation | Description |
|---|---|
| **QLBPW** | A tabular Q-learning implementation used for learning and evaluating path-planning behavior. |
| **EQLBPW** | An enhanced, neural-network-based implementation incorporating Double DQN, prioritized experience replay, importance-sampling weights, and path-aware priority weighting. |

The project also includes a Streamlit dashboard, simulation environments, visualization utilities, comparison scripts, configuration modules, algorithm documentation, and automated tests.

The primary purpose is to explore whether enhanced learning strategies can improve navigation behavior, training efficiency, and robustness in obstacle-filled environments.

> **Note:** This is a research and experimentation project. Performance improvements should be established through controlled experiments rather than assumed from the algorithm design alone.

## Research Motivation

Traditional tabular Q-learning is a model-free reinforcement learning algorithm that learns action values through interaction with an environment.

Although it is relatively simple to implement, several challenges can arise when applying it to path planning.

### 1. State-space growth

As an environment becomes more complex, the number of possible states and actions may increase substantially. Tabular methods can become expensive in terms of memory and learning efficiency.

### 2. Local optima and inefficient exploration

An agent may repeatedly select familiar actions or learn a suboptimal route. Insufficient exploration can prevent it from discovering better paths.

### 3. Q-value overestimation

Q-learning uses a maximum over estimated action values in its update target. Estimation errors can lead to overoptimistic values and potentially undesirable action selection.

### 4. Inefficient experience utilization

Not every interaction provides the same amount of learning value. Uniformly sampling past experiences may spend training updates on relatively uninformative transitions while important collisions or goal-reaching events receive insufficient attention.

### 5. Dynamic obstacles

An agent that performs well in a fixed environment may struggle when obstacle positions change. This motivates evaluating policies under both static and dynamic conditions.

This project explores techniques intended to address these challenges through enhanced value estimation, prioritized learning, reward design, and state representation.

## Project Objectives

The main objectives are to:

1. Implement and study Q-learning-based path planning in grid environments.
2. Compare a tabular baseline with a deep reinforcement learning approach.
3. Investigate Double DQN as a method for reducing overestimation bias.
4. Apply prioritized experience replay to emphasize informative transitions.
5. Incorporate collision, goal, and distance-progress information into experience prioritization.
6. Investigate reward shaping and Manhattan-distance-based path guidance.
7. Visualize agent training, navigation paths, and performance metrics.
8. Provide a reproducible foundation for algorithm comparison and further research.

## Algorithms and Enhancements

### QLBPW: Tabular Q-learning

The QLBPW implementation maintains a table of estimated action values and uses an exploration strategy to balance random actions with learned behavior.

Its action-value update follows the standard Q-learning formulation:

\[
Q(s,a)\leftarrow Q(s,a)+\alpha
\left[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\right]
\]

where:

- \(s\) is the current state.
- \(a\) is the selected action.
- \(r\) is the immediate reward.
- \(s'\) is the next state.
- \(\alpha\) is the learning rate.
- \(\gamma\) is the discount factor.

This implementation serves as a useful baseline for studying navigation behavior without a deep neural network.

### EQLBPW: Enhanced deep Q-learning

EQLBPW extends the learning process with neural-network-based value estimation and several additional learning mechanisms.

The proposed enhancements include:

| Technique | Purpose |
|---|---|
| Deep Q-Network (DQN) | Estimates action values using a neural network. |
| Double DQN | Separates next-action selection from next-action evaluation to help reduce overestimation bias. |
| Prioritized Experience Replay (PER) | Samples transitions according to their learning priority instead of sampling all experiences uniformly. |
| Importance-sampling weights | Compensate for the bias introduced by prioritized sampling. |
| Path-aware priority weighting | Incorporates collision, goal, and distance-progress signals into transition priority. |
| Decaying epsilon-greedy exploration | Gradually shifts behavior from exploration toward exploitation. |
| Reward shaping | Provides additional learning signals associated with navigation outcomes. |
| Manhattan distance | Provides a grid-based measure of distance between positions. |
| Target network | Stabilizes learning by periodically synchronizing a separate target network. |
| Weighted Huber loss | Combines a robust value-estimation loss with importance-sampling weights. |

These mechanisms work together to influence which experiences are learned from, how action values are estimated, and how the agent balances exploration with exploitation.

### Path-aware priority weighting

A central idea in EQLBPW is to incorporate information about a transition's navigation significance into its replay priority.

The documented proposed algorithm defines a weighting term of the form:

\[
W_i=1+w_cC_i+w_gG_i+w_d\max(D_i,0)
\]

where:

- \(W_i\) is the transition's path-aware weighting term.
- \(w_c\) is the collision weight.
- \(w_g\) is the goal weight.
- \(w_d\) is the distance-progress weight.
- \(C_i\) indicates whether a collision occurred.
- \(G_i\) indicates whether the goal was reached.
- \(D_i\) represents distance progress.

The corresponding proposed priority is:

\[
p_i=(|\delta_i|+\varepsilon_p)W_i
\]

where \(\delta_i\) is the temporal-difference (TD) error and \(\varepsilon_p\) is a small positive constant.

Under this formulation, experiences with large TD errors or significant navigation events can receive greater sampling priority.

The implementation's actual behavior depends on the code and its configured parameters. Refer to [`docs/PROPOSED ALGO.md`](docs/PROPOSED%20ALGO.md) for the full proposed algorithm.

## How the Proposed Algorithm Works

The enhanced learning process can be summarized as follows:

1. **Initialize the environment and agent.** Set the starting position, goal, obstacle configuration, learning parameters, replay memory, and neural networks.
2. **Observe the current state.** Obtain the state representation from the environment.
3. **Select an action.** Use epsilon-greedy exploration to choose between a random action and the action with the highest estimated value.
4. **Interact with the environment.** Execute the action and observe the reward, next state, terminal status, and relevant navigation events.
5. **Store the experience.** Save the transition in replay memory with an initial priority.
6. **Sample prioritized experiences.** Select a minibatch according to the experiences' priorities.
7. **Calculate the Double DQN target.** Use the main network to select the next action and the target network to evaluate it.
8. **Update priorities and weights.** Incorporate TD errors, path-aware signals, and importance-sampling weights.
9. **Optimize the main network.** Apply the weighted Huber loss and update network parameters.
10. **Update exploration and target networks.** Decay epsilon, adjust the importance-sampling exponent according to training progress, and periodically synchronize the target network.
11. **Evaluate the learned policy.** Examine the resulting path, reward, success status, and other relevant metrics.

The learning process repeats over multiple episodes until the configured training budget or termination conditions are reached.

## Repository Structure

The main repository files are organized as follows:

```text
enhancement-of-qlearning-based-on-prioritized-weights/
├── .github/
├── compare_figures/
│   └── overestimation/
├── docs/
│   ├── EQLBPW_Algorithm.jpg
│   ├── QLBPW_Algorithm.jpg
│   ├── ORIGINAL ALGO.md
│   ├── PROPOSED ALGO.md
│   ├── overestimation-demo.md
│   ├── test_cases.md
│   └── Improved_Q-Learning_Applied_to_Dynamic_Obstacle_Avoidance_and_Path_Planning.pdf
├── src/
│   ├── EQLBPW/
│   │   ├── agent.py
│   │   ├── environment.py
│   │   ├── simulator.py
│   │   └── tracker.py
│   ├── QLBPW/
│   │   ├── agent.py
│   │   ├── environment.py
│   │   ├── simulator.py
│   │   ├── tracker.py
│   │   └── SOPs/
│   ├── app.py
│   ├── agent_config.py
│   ├── env_config.py
│   ├── env_settings.py
│   ├── Astar.py
│   ├── compare.py
│   ├── compare_overestimation.py
│   ├── overestimation.py
│   ├── overestimation_benchmark.py
│   ├── visualizer.py
│   ├── EQLBPW_simulator.py
│   └── QLBPW_simulator.py
├── tests/
│   └── test_compare_overestimation.py
├── requirements.txt
├── .gitignore
└── README.md
```

### Important directories and files

- `src/QLBPW/`: Tabular agent, environment, simulation, and tracking components.
- `src/EQLBPW/`: Enhanced deep reinforcement learning agent, environment, simulation, and tracking components.
- `src/app.py`: Streamlit dashboard for selecting and running the available learning approaches.
- `src/agent_config.py`: Agent parameter defaults.
- `src/env_config.py` and `src/env_settings.py`: Environment configuration and preset environments.
- `src/compare.py`: Algorithm comparison utilities.
- `src/compare_overestimation.py`, `src/overestimation.py`, and `src/overestimation_benchmark.py`: Overestimation-related experiments and benchmarks.
- `src/Astar.py`: A* path-planning implementation.
- `src/visualizer.py`: Visualization utilities.
- `docs/`: Algorithm descriptions, diagrams, research material, and test documentation.
- `tests/`: Automated tests.

## Requirements

The project uses Python and scientific computing libraries. The enhanced implementation and interactive dashboard additionally require PyTorch and Streamlit.

The repository's `requirements.txt` currently specifies:

- `matplotlib`
- `mpmath`
- `numpy`
- `pytest`

The dashboard also imports `streamlit` and `torch`, which are not listed in the current requirements file.

### Recommended environment

- Python 3.10 or newer, subject to compatibility with the installed dependencies.
- `pip` for dependency installation.
- A terminal or command prompt.
- A web browser for the Streamlit dashboard.

PyTorch supports CPU-based execution, so a GPU is not required for basic experimentation. Training time will depend on the environment, parameters, and hardware.

## Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/jiecolao/enhancement-of-qlearning-based-on-prioritized-weights.git
cd enhancement-of-qlearning-based-on-prioritized-weights
```

### Step 2: Create a virtual environment

A virtual environment isolates the project's dependencies from other Python projects.

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

**Linux or macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 4: Install the repository dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 5: Install dashboard and deep-learning dependencies

```bash
python -m pip install streamlit torch
```

If you require a specific PyTorch build, follow the installation instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/).

### Step 6: Verify the installation

```bash
python --version
python -m pip show numpy matplotlib pytest streamlit torch
```

If the installed versions are incompatible with your Python version or operating system, consult the relevant package documentation and install a compatible set of versions.

## How to Run

Run the commands below from the repository root unless otherwise specified.

### Option 1: Launch the interactive dashboard

The dashboard is the recommended starting point for users who want to explore the project interactively.

```bash
cd src
python -m streamlit run app.py
```

After the server starts, open the local URL printed in the terminal. Streamlit commonly uses `http://localhost:8501`.

Use the dashboard's sidebar to switch between `EQLBPW` and `QLBPW`.

### Option 2: Run the enhanced EQLBPW simulator

From the repository root:

```bash
python -m EQLBPW.simulator
```

Run this command from inside `src`, where the package can be imported directly:

```bash
cd src
python -m EQLBPW.simulator
```

### Option 3: Run the tabular QLBPW simulator

From inside `src`:

```bash
python -m QLBPW.simulator
```

### Option 4: Run the standalone simulator scripts

The repository also includes top-level simulator scripts:

```bash
cd src
python EQLBPW_simulator.py
```

For the tabular implementation:

```bash
python QLBPW_simulator.py
```

These entry points may provide different execution or visualization behavior from the package-level simulators.

### Option 5: Run comparison experiments

The source directory contains scripts for comparing algorithms and investigating overestimation.

From inside `src`, you can start with:

```bash
python compare.py
```

For overestimation comparison experiments:

```bash
python compare_overestimation.py
```

Other available experiment scripts include:

```bash
python overestimation.py
python overestimation_benchmark.py
```

Review the relevant script before launching a long experiment to understand its configuration, runtime, and output behavior. The exact experiment and generated files depend on the script's current implementation.

## Using the Streamlit Dashboard

The dashboard provides an interactive way to run training and inspect the learning process.

### Recommended workflow

1. Launch the dashboard using the command above.
2. Select `EQLBPW` or `QLBPW` in the sidebar.
3. Review the available environment and training settings.
4. Configure the episode count, exploration settings, obstacle behavior, and other available parameters.
5. Start the training process.
6. Monitor training progress, episode rewards, exploration rate, and the training log.
7. Inspect the resulting navigation path and environment visualization.
8. Repeat the experiment with the other algorithm using comparable settings.

The dashboard includes functionality for displaying the environment, evaluating a learned path, and viewing training progress.

For a fair comparison, use equivalent start and goal positions, obstacle layouts, evaluation budgets, and success criteria wherever the implementations allow.

## Configuration and Customization

The project separates agent parameters from environment settings, making it easier to investigate different experimental conditions.

### Agent configuration

See [`src/agent_config.py`](src/agent_config.py).

The configuration classes include parameters such as:

| Parameter | Meaning |
|---|---|
| `learning_rate` / `alpha` | Controls the size of learning updates. |
| `gamma` | Discount factor for future rewards. |
| `priority_alpha` | Controls how strongly replay priorities influence sampling. |
| `beta_start` and `beta_end` | Define the importance-sampling exponent schedule. |
| `e` | Initial exploration rate. |
| `e_min` | Minimum exploration rate. |
| `e_decay` | Exploration decay factor. |
| `batch_size` | Number of transitions sampled for an update. |
| `max_buffer` | Replay memory capacity. |
| `target_sync_freq` | Target-network synchronization interval. |
| `collision_weight` | Weight associated with collision events in path-aware prioritization. |
| `goal_weight` | Weight associated with reaching the goal. |
| `distance_weight` | Weight associated with distance progress. |

The actual values used by an experiment may be defined in the configuration classes, the dashboard, or the relevant simulator. Check the active entry point before changing parameters.

### Environment configuration

See [`src/env_config.py`](src/env_config.py) and [`src/env_settings.py`](src/env_settings.py).

You can investigate:

- Grid size.
- Starting position.
- Goal position.
- Static obstacle locations.
- Dynamic obstacle behavior.
- Number of training episodes.
- Maximum steps per episode.
- Environment presets.

When changing the environment, verify that the starting and goal positions are valid and that obstacles do not occupy prohibited positions.

### Tips for tuning

- Start with a small number of episodes to verify that the environment and training loop work correctly.
- Change one or two parameters at a time.
- Use multiple random seeds when supported.
- Record the exact configuration used for each experiment.
- Avoid drawing conclusions from a single training run.
- Compare performance on consistent evaluation environments.

## Experiments and Evaluation

The repository includes comparison utilities and overestimation-related experiments. These can be used to investigate the strengths and weaknesses of different learning strategies.

### Recommended evaluation metrics

| Metric | What it measures |
|---|---|
| Goal-reaching success rate | The proportion of evaluation episodes that reach the goal. |
| Path length | The number of moves required to reach the goal. |
| Episode reward | The cumulative reward received during an episode. |
| Collision count or rate | The frequency of collisions with obstacles. |
| Steps to goal | The number of actions required for successful navigation. |
| Training duration | The time required to complete a training run. |
| Q-value overestimation | The difference between estimated values and an appropriate reference, when available. |
| Performance under dynamic obstacles | How reliably the learned policy navigates when obstacles move or change. |

### Suggested experiment procedure

1. Define the environment, start state, goal state, and obstacle configuration.
2. Select the algorithms to compare.
3. Keep the evaluation conditions consistent.
4. Train each algorithm using documented parameters.
5. Evaluate the learned policy separately from training where possible.
6. Repeat experiments across multiple seeds.
7. Calculate average performance and variability.
8. Save plots, logs, parameter configurations, and results.
9. Discuss limitations and possible sources of bias.

A lower path length alone does not necessarily indicate a better algorithm. Successful navigation, collision avoidance, consistency, and computational cost should also be considered.

### Visualizations and research documentation

The `docs/` directory contains algorithm diagrams and the reference research paper. The `compare_figures/` directory contains comparison-related visual assets.

Use these resources to understand the intended algorithms and organize your experiments. Verify that generated plots correspond to the correct experiment, configuration, and evaluation period before using them in reports or publications.

## Running Tests

The repository includes automated tests under `tests/`.

From the repository root, run:

```bash
python -m pytest
```

To run the existing overestimation comparison test specifically:

```bash
python -m pytest tests/test_compare_overestimation.py -v
```

To obtain more detailed test output:

```bash
python -m pytest -v
```

Tests help identify implementation errors and regressions. Passing the current test suite does not by itself establish that the algorithms converge, outperform baselines, or generalize to unseen environments.

## Troubleshooting

### `ModuleNotFoundError`

**Possible cause:** A dependency is missing, or Python is running from the wrong working directory.

**What to do:**

1. Activate the virtual environment.
2. Install the dependencies.
3. Change into `src` when running the package-level simulators.
4. Confirm which Python interpreter is active.

```bash
python -m pip install -r requirements.txt
python -m pip install streamlit torch
```

### `streamlit` command not found

Use Python's module invocation rather than relying on a separate executable:

```bash
cd src
python -m streamlit run app.py
```

### PyTorch installation problems

Check the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) for a compatible build. Start with CPU execution if GPU configuration is not necessary for your experiment.

### Dashboard fails to start

Confirm that all required dependencies are installed and that the command is being executed from `src`.

```bash
python -m pip show streamlit torch matplotlib numpy
```

If an exception occurs, inspect the full traceback in the terminal. The first relevant error often identifies the missing dependency or failing module.

### Training is slow

Deep reinforcement learning experiments can require many environment interactions and neural-network updates.

Try reducing the episode count or batch size for an initial smoke test. Use smaller environments while debugging, and increase the workload after confirming that the training loop behaves as expected.

### The agent fails to reach the goal

Check the start and goal positions, obstacle placement, reward configuration, exploration parameters, and maximum steps per episode. Also verify whether the environment is static or dynamic and whether the evaluation policy matches the intended training setup.

Failure to reach the goal in a small number of episodes does not necessarily indicate a software defect; it may also reflect the learning configuration or difficulty of the environment.

## Guidelines for Users and Contributors

### For users

- Read the algorithm documentation before modifying the learning process.
- Begin with the default configurations and verify the example workflows.
- Preserve experiment settings and logs so that results can be reproduced.
- Use controlled comparisons rather than relying on isolated runs.
- Report errors with the full traceback, Python version, operating system, and relevant configuration.
- Avoid treating simulation results as evidence of real-world safety without additional validation.

### For contributors

Contributions that improve correctness, reproducibility, usability, and experimental quality are welcome.

Potential areas for improvement include:

- Better experiment configuration and logging.
- Additional automated tests.
- Improved documentation and setup instructions.
- More rigorous evaluation metrics.
- Additional baseline algorithms.
- Reproducibility support through random seeds.
- Performance profiling and optimization.
- Better handling of dynamic obstacles and edge cases.

### Suggested contribution workflow

1. Fork the repository.
2. Create a feature branch.
3. Make a focused change.
4. Add or update tests when appropriate.
5. Run the available test suite.
6. Document any changes to configuration or behavior.
7. Submit a pull request describing the motivation, implementation, and validation results.

Use clear, descriptive commit messages. The repository's [`docs/commit-definitions.md`](docs/commit-definitions.md) documents a conventional format:

```text
<type>(<optional scope>): <summary>
```

Examples:

```text
feat: add additional path-planning evaluation metrics
fix: correct obstacle collision handling
docs: improve installation instructions
test: add environment edge-case coverage
refactor: simplify replay priority calculations
```

## Limitations and Future Improvements

This project is intended primarily for research, learning, and simulation.

Several limitations should be considered:

- **Grid-based environment:** Results from a discrete grid may not transfer directly to continuous navigation problems.
- **Training variability:** Reinforcement learning performance can vary across random seeds and configurations.
- **Hyperparameter sensitivity:** Learning rate, reward design, replay settings, and exploration schedules can materially affect results.
- **Dynamic environments:** Changes in obstacle positions can make previously learned paths less effective.
- **Computational requirements:** Neural-network training may require more computation than tabular learning.
- **Evaluation scope:** Performance claims require appropriate baselines, repeated experiments, and clearly defined metrics.
- **Real-world deployment:** Simulated collision avoidance does not guarantee safe navigation in physical environments.

Potential future work includes expanding benchmark environments, improving reproducibility, evaluating generalization, adding more baselines, and conducting statistically grounded comparisons.

## References

### Primary reference

Wang, C., Yang, X., & Li, H. (2022). *Improved Q-Learning Applied to Dynamic Obstacle Avoidance and Path Planning.*

[Read the paper on IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/9870811)

The original paper provides the research foundation for the path-planning approach explored in this repository.

### Project algorithm documentation

- [Original Algorithm](docs/ORIGINAL%20ALGO.md)
- [Proposed Algorithm](docs/PROPOSED%20ALGO.md)
- [EQLBPW Algorithm Diagram](docs/EQLBPW_Algorithm.jpg)
- [QLBPW Algorithm Diagram](docs/QLBPW_Algorithm.jpg)
- [Overestimation Demonstration](docs/overestimation-demo.md)
- [Test Cases](docs/test_cases.md)

### Additional learning resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/stable/)

## License

No license is currently specified in the repository metadata. Until a license is added, users should not assume that the project is available for unrestricted redistribution, modification, or commercial use.

If you intend to make the project open source, consider selecting an appropriate license and adding a `LICENSE` file to the repository.

---

**Repository:** [jiecolao/enhancement-of-qlearning-based-on-prioritized-weights](https://github.com/jiecolao/enhancement-of-qlearning-based-on-prioritized-weights)

If you find this project useful for research or learning, consider starring the repository, reporting issues, or contributing improvements that make its algorithms and experimental results easier to reproduce.
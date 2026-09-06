from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from src.compare_overestimation import (
    GAMMA, Task, bootstrap, evaluate, make_agent, summarize, train_run,
    IndexedReplayBuffer, EQLBPW_SETTINGS,
)
from src.EQLBPW.agent import ReplayBuffer
from src.EQLBPW.environment import Environment


def test_reference_values_and_boundary_collision_semantics():
    task = Task("grid", dict(grid_size=2, start_state=(0, 0), end_state=(1, 0), obstacles=set()))
    assert task.reference[task.indices[(0, 0)], 1] == pytest.approx(1.0)
    assert task.reference[task.indices[(0, 0)], 0] == pytest.approx(GAMMA)
    assert task.reference[task.indices[(0, 1)], 0] == pytest.approx(GAMMA)
    assert task.residual < 1e-10
    assert task.transition((1, 0), 0)[1:3] == (0.0, True)
    assert task.transition((0, 0), 0)[3] is False  # Clamped edge is not an obstacle collision.
    task = Task("grid", dict(grid_size=2, start_state=(0, 0), end_state=(1, 0), obstacles={(0, 1)}))
    assert task.transition((0, 0), 2) == ((0, 0), -1.0, False, True, 0.0)
    assert task.reference[task.indices[(0, 0)], 2] == pytest.approx(-1 + GAMMA)


def test_diagnostic_zero_reference_and_production_grid_encoding():
    diagnostic = Task("diagnostic")
    assert np.count_nonzero(diagnostic.reference) == 0
    assert diagnostic.transition((0, 0), 3) == ((0, 0), 0.0, False, False, 0.0)
    grid = Task("grid")
    for state in grid.states:
        env = SimpleNamespace(agent_pos=state, end_state=grid.goal, grid_rows=grid.grid,
                              grid_cols=grid.grid, obstacles=grid.obstacles)
        np.testing.assert_array_equal(grid.features[state], Environment.get_state(env))


def test_actual_production_double_dqn_target_and_terminal_mask():
    class ConstantQ(nn.Module):
        def __init__(self, values):
            super().__init__()
            self.values = nn.Parameter(torch.tensor(values, dtype=torch.float32))

        def forward(self, states):
            return self.values.expand(len(states), -1)

    class CaptureLoss(nn.Module):
        def forward(self, current, target):
            self.target = target.clone()
            return nn.functional.smooth_l1_loss(current, target, reduction="none")

    agent = make_agent("EQLBPW")
    agent.main_net = ConstantQ([1, 5, 2, 3])  # Main chooses action 1.
    agent.target_net = ConstantQ([10, 20, 30, 40])  # Target's own max is action 3.
    agent.optimizer = torch.optim.Adam(agent.main_net.parameters(), lr=.0005)
    agent.criterion = CaptureLoss()
    agent.batch_size = agent.memory.batch_size = 2
    for done in (False, True):
        agent.memory.push(np.zeros(29), 0, 2, np.zeros(29), done)
    agent.update()
    np.testing.assert_allclose(sorted(agent.criterion.target.flatten().tolist()), [2, 2 + GAMMA * 20])


def test_metrics_do_not_treat_underestimation_as_accuracy():
    task = Task("diagnostic")
    agent = make_agent("QLBPW")
    agent.Q[(0, 0)] = np.full(4, -5.0)
    row, _ = evaluate(agent, task, "QLBPW", 0, 10)
    assert row["positive_bias"] == 0
    assert row["signed_bias"] == -5
    assert row["mean_absolute_q_error"] == 5


def test_production_qlbpw_terminal_target_has_no_bootstrap():
    agent = make_agent("QLBPW")
    state, goal = (0, 0), (1, 0)
    agent.Q[goal] = np.full(4, 100.0)
    agent.memory.push(state, 1, 1.0, goal, 0.0)
    agent.update_Q(state, 1, 1.0, goal, 0.0, 0, 1.0, goal, set())
    assert agent.Q[state][1] == 1.0


@pytest.mark.parametrize("algorithm", ["QLBPW", "EQLBPW"])
@pytest.mark.parametrize("task", ["diagnostic", "grid"])
def test_repeatable_training_and_exact_budget(task, algorithm):
    first = train_run(task, algorithm, 12, 170)
    second = train_run(task, algorithm, 12, 170)
    assert first["rows"] == second["rows"]
    assert first["interactions"] == 170
    assert first["rows"][-1]["interactions"] == 170
    assert first["completed_episodes"] >= 1
    assert first["updates"] == (107 if algorithm == "EQLBPW" else 170)


def test_bootstrap_and_summary_preserve_unfavorable_results():
    assert bootstrap([1.0])["ci95"] is None
    runs = []
    for task in ("diagnostic", "grid"):
        for seed in (0, 1):
            for algorithm, error in (("QLBPW", 1), ("EQLBPW", 2)):
                runs.append(dict(task=task, seed=seed, algorithm=algorithm, rows=[
                    dict(interactions=10, positive_bias=error, signed_bias=error,
                         mean_absolute_q_error=error, goal_success=0)]))
    summary = summarize(runs)
    assert summary["grid"]["bias_conclusion"] == "QLBPW has lower positive overestimation"


def test_indexed_replay_exactly_matches_production_through_rollover():
    reference = ReplayBuffer(7, 3)
    indexed = IndexedReplayBuffer(7, 3)
    for step in range(40):
        for buffer in (reference, indexed):
            buffer.push([step], step % 4, float(step), [step + 1], False)
        if len(reference) < 3:
            continue
        np.random.seed(step)
        batch_a, indices_a, probabilities_a = reference.sample(.6)
        np.random.seed(step)
        batch_b, indices_b, probabilities_b = indexed.sample(.6)
        assert batch_a == batch_b
        np.testing.assert_array_equal(indices_a, indices_b)
        np.testing.assert_array_equal(probabilities_a, probabilities_b)
        torch.testing.assert_close(reference.importance_weights(probabilities_a, .7),
                                   indexed.importance_weights(probabilities_b, .7), rtol=0, atol=0)
        priorities = np.array([step + .001, .001, 5.7])
        reference.update_priorities(indices_a, priorities)
        indexed.update_priorities(indices_b, priorities)
        assert list(reference.buffer) == list(indexed.buffer)
        assert reference.max_priority == indexed.max_priority


def test_indexed_replay_preserves_entire_training_result(monkeypatch):
    monkeypatch.setitem(EQLBPW_SETTINGS, "max_buffer", 128)
    original = train_run("diagnostic", "EQLBPW", 3, 1000, reference_replay=True)
    optimized = train_run("diagnostic", "EQLBPW", 3, 1000)
    assert original["rows"] == optimized["rows"]
    assert original["updates"] == optimized["updates"]

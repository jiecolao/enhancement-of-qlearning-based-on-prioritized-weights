# Completed overestimation comparison

Ten predefined seeds (0-9), both production agents, 5,000 diagnostic and 81,000 grid interactions per agent per seed. All 40 runs completed: 1,720,000 training interactions in total.

## Results

Means below use the final 20% of noninitial evaluation checkpoints. Lower positive bias and absolute Q-error are better.

| Setting | Metric | QLBPW | EQLBPW | Reduction |
|---|---|---:|---:|---:|
| Diagnostic | Positive overestimation | 6.515221 | 0.223844 | 96.56% |
| Diagnostic | Absolute Q-error | 6.135587 | 0.187874 | 96.94% |
| Grid | Positive overestimation | 2.240927 | 0.022211 | 99.01% |
| Grid | Absolute Q-error | 2.058545 | 0.127458 | 93.81% |

The paired 95% bootstrap intervals for QLBPW minus EQLBPW positive bias are **[5.9106, 6.7119]** for the diagnostic and **[2.1725, 2.2653]** for the grid; both favor EQLBPW.

## Figures

- [Diagnostic Q-value estimates](diagnostic_q_values.png)
- [Grid overestimation](grid_overestimation.png)
- [Absolute Q-error](q_value_accuracy.png)
- [Final comparison and greedy success](final_comparison.png)

## Interpretation

Under these specified stochastic-reward conditions, EQLBPW has substantially lower overestimation and more accurate Q-values than the current QLBPW implementation. This supports the intended outcome of the third enhancement in the tested settings.

Neither algorithm reached the goal in any of the ten final greedy grid evaluations (0/10 each). This demo therefore does not show improved path success. The two algorithms also differ beyond Double DQN; without an ablation, the observed difference cannot be attributed exclusively to Double DQN.

The grid uses the existing static 9x9 map with shared expected rewards and explicitly added reward noise. This is a stochastic diagnostic, not a replication of every production simulation setting. See [method and commands](../../../docs/overestimation-demo.md) for the full protocol.

## Verification

- 12 automated tests passed, including actual production target calculations, reproducibility and replay equivalence.
- All 20 full diagnostic runs exactly matched the slower original sampler measurements.
- All 40 seed runs, CSV row counts, interaction budgets, source hashes and recomputed summary values verified.
- All four final figures visually inspected.

Raw results: [measurements.csv](measurements.csv), [summary.json](summary.json), [config.json](config.json), and per-seed JSON files.

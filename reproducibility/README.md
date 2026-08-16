# Reproducibility

> This file references `docs/PHASE_*_REPORT.md` and `docs/FINAL_RESEARCH_AUDIT.md` in several
> places below. Those narrative write-ups are kept in the project's private working repository and
> are not included here; every number and script path referenced here is still fully reproducible
> from this repository alone (`experiments/phase_*/run_*.py` and the archived `results/`/
> `summary*.json` files), the `docs/` references just point to additional prose discussion that
> did not ship with this export.

Tested on Python 3.13 and 3.14 (developed and verified on 3.14.5; a reviewer independently
verified successfully on 3.13.12). CI (`.github/workflows/tests.yml`) runs a matrix over both
3.13 and 3.14, the only two interpreters actually available and verified on the machine this
project was developed on. `pyproject.toml` declares `requires-python = ">=3.13,<3.15"`, matching
exactly what has been run and tested here -- earlier drafts of this file claimed `>=3.10` as an
unverified lower bound; that claim has been removed rather than left standing, since no 3.10-3.12
interpreter was ever available on this machine to check it. If you successfully run this project
on 3.10-3.12, that would be useful information, but it is not currently a supported/verified
configuration.

Requires numpy. Two install paths exist and they are **not equivalent**:

- `pip install -r requirements.txt` pins numpy to the exact version (`==2.5.1`) this repository's
  results were generated and verified against. This is the only path that is guaranteed to
  reproduce results bit-for-bit; use it for exact-match reproduction.
- `pip install -e .` (via `pyproject.toml`) uses a floor-and-ceiling range, `numpy>=2.5.1,<2.6`,
  not an exact pin. A plain `pip install -e .` today picks up whatever the latest 2.5.x release is
  (2.5.2 as of this writing) rather than exactly 2.5.1, so this path is **compatible but not
  guaranteed to be bit-for-bit identical** to the committed results -- a different patch-level
  numpy build could in principle shift floating-point results at the margins, even though the two
  install paths silently diverge on which exact numpy version gets installed. The `<2.6` ceiling
  was added specifically to reduce (not eliminate) this drift risk by ruling out a future numpy
  minor-version bump changing behavior under `pip install -e .`; it does not make the two paths
  equivalent. If exact reproduction matters, use `requirements.txt`.

No scipy/sklearn/pandas dependency anywhere in `src/`. `matplotlib` is optional, used only by
`analysis/build_summary_tables.py` for figure rendering, and the script falls back to a table if
it is not installed.

```
pip install -r requirements.txt          # exact-match numpy (2.5.1 exactly) -- use for bit-for-bit reproduction
# or
pip install -e .                          # pyproject.toml, numpy>=2.5.1,<2.6 -- compatible, not guaranteed identical
pip install -e ".[dev]"                   # adds pytest
pip install -e ".[plots]"                 # adds matplotlib, for analysis/ figures
```

## Tests

```
python3 -m pytest tests/ -q
```

## Analysis script and summary-JSON smoke tests (also run in CI)

```
python3 analysis/build_summary_tables.py     # builds consolidated tables/figures from committed summary JSON
python3 -m analysis.check_summary_json       # confirms every experiments/phase_*/*summary*.json parses and is non-empty
```

`build_summary_tables.py`'s PNG output is byte-deterministic: it passes a fixed `metadata` dict to
every `matplotlib.figure.Figure.savefig(...)` call (empty `Software`/`Creator`/`Author`/
`Description` strings) so two consecutive runs on identical input JSON produce byte-identical PNG
files -- verified directly by running the script twice and comparing
`hashlib.sha256(open(path, "rb").read()).hexdigest()` for both PNGs in `analysis/output/`; the
hashes matched. `check_summary_json.py` is a lightweight structural smoke test (valid JSON,
non-empty), not a full audit that every number in every phase report matches its source JSON --
that would be a larger undertaking and is explicitly out of scope here.

## Phase 1 -- baseline reproduction (B0/B1/B2)

```
python3 experiments/phase_a/run_baselines.py
```

Writes per-seed raw results to `experiments/phase_a/results/*.json` (gitignored) and a committed
aggregate to `experiments/phase_a/summary.json`. Also writes an explicit same-seed
reproducibility check to `experiments/phase_a/reproducibility_check.json`. Seeds: 1000-1019.
See `docs/PHASE_1_REPORT.md`.

## Phase 2 -- counterfactual dataset and offline oracle

```
python3 experiments/phase_b/run_oracle_comparison.py
```

Writes raw counterfactual samples to `experiments/phase_b/results/*.json` (gitignored) and a
committed aggregate to `experiments/phase_b/summary.json`. Seeds: 2000-2019 (disjoint from
Phase 1). See `docs/PHASE_2_REPORT.md`.

## Phase 3 -- learned router (B3) and distribution shift

```
python3 experiments/phase_c/run_learned_router.py
```

Writes a committed summary to `experiments/phase_c/summary.json`, including calibrated tau and
logistic regression weights for both configs, held-out test results, and the distribution-shift
cross-evaluation. Seed pools (disjoint from Phase 1/2 and from each other): development
3000-3009, calibration 3100-3109, validation 3200-3209, held-out test 3300-3319. See
`docs/PHASE_3_REPORT.md`.

## Phase 1/2/3 revised -- difficulty-axis cost regime

```
python3 experiments/phase_a/run_baselines_revised.py
python3 experiments/phase_b/run_oracle_comparison_revised.py
python3 experiments/phase_c/run_learned_router_revised.py
```

Same mechanics as the original Phase 1/2/3 scripts, but using the difficulty-axis fields on
`FleetConfig` (`difficulty_hard_fraction`, `hard_spread_multiplier`, `easy_spread_multiplier`,
see `docs/PHASE_2_REPORT.md` addendum) and fresh, disjoint seed pools: Phase 1 revised
11000-11019, Phase 2 revised 12000-12019, Phase 3 revised dev 13000-13009 / cal 13100-13109 /
val 13200-13209 / test 13300-13319. See the "## Revised" sections of
`docs/PHASE_1_REPORT.md`/`PHASE_2_REPORT.md`/`PHASE_3_REPORT.md`.

## Phase 4 -- distributed cognitive delegation

```
python3 experiments/phase_d/run_delegation.py
```

10 robots, 150 decisions/robot, delegation enabled vs disabled, seeds 14000-14014. Writes
`experiments/phase_d/summary.json` and `experiments/phase_d/results/raw_episodes.json`. See
`docs/PHASE_4_REPORT.md`.

## Phase 5 -- shared experience / cognitive memory

```
python3 experiments/phase_e/run_memory.py
```

C0-C3 memory conditions, seeds 15000-15014. Writes `experiments/phase_e/summary.json`. See
`docs/PHASE_5_REPORT.md`.

## Phase 6 -- specialization experiment

```
python3 experiments/phase_f/run_specialization.py
```

800 decisions/robot, 10 robots, seeds 16000-16005. Writes
`experiments/phase_f/results/specialization_summary.json` and `cross_seed_summary.json`. See
`docs/PHASE_6_REPORT.md`.

## Phase 7 -- emergence test and artifact controls

```
python3 experiments/phase_g/run_emergence_test.py
```

Reruns Phase 6's setup under randomized-ID and shuffled-order controls, same seeds 16000-16005.
Writes `experiments/phase_g/results/artifact_control_summary.json`. See `docs/PHASE_7_REPORT.md`.

## Delegation policy calibration (used by Phase 4/5/11 revised)

```
python3 -m experiments.phase_d.calibrate_delegation
```

Grid search over `self_load_penalty`/`lambda_msg`/`self_quality_factor` on calibration seeds
18000-18009. Writes `experiments/phase_d/results/calibration_grid.json`. See `docs/PHASE_4_REPORT.md`
"Revised (calibrated)".

## Phase 4/5 revised -- calibrated delegation policy

```
python3 -m experiments.phase_d.run_delegation_revised
python3 -m experiments.phase_e.run_memory_revised
```

Same mechanics as the original Phase 4/5 scripts but using the calibrated `DelegationPolicy`
parameters. Fresh held-out seeds: Phase 4 revised 19000-19014, Phase 5 revised 20000-20014 (both
disjoint from the original pools and from the calibration pool). Writes
`experiments/phase_d/summary_revised.json` and `experiments/phase_e/summary_revised.json`. See the
"## Revised (calibrated)" sections of `docs/PHASE_4_REPORT.md` / `docs/PHASE_5_REPORT.md`.

## Phase 6/7 revised -- RNG fix and tie-break artifact control

```
python3 -m experiments.phase_f.run_specialization_revised
python3 -m experiments.phase_g.run_emergence_test_revised
```

Reruns Phase 6/7 after fixing `FleetSimulator`'s per-robot RNG construction
(`src/environment/simulator.py`, now `SeedSequence.spawn()` instead of `base_draw + rid`), on a
fresh seed pool 21000-21005. Adds a third control not present in the original Phase 7 script:
randomized peer-evaluation order in `DelegationPolicy.decide()` (`tie_break_rng_seed`), which
directly tests the deterministic-tie-breaking artifact diagnosed during this rerun. Writes
`experiments/phase_f/results/specialization_summary_revised.json`,
`experiments/phase_g/results/artifact_control_summary_revised.json`. See the "## Revised (RNG fix)"
sections of `docs/PHASE_6_REPORT.md` / `docs/PHASE_7_REPORT.md`.

## Phase 8 -- perturbation / reorganization test

```
python3 -m experiments.phase_h.run_perturbation
```

Same seeds as Phase 6/7 revised (21000-21005). Removes each seed's delegate-in hub robot at the
episode midpoint and measures utility and delegate-in redistribution before/after. Writes
`experiments/phase_h/results/perturbation_summary.json`. See `docs/PHASE_8_REPORT.md`.

## Phase 8 revised -- larger seed pool (n=30) with bootstrap confidence intervals

```
python3 -m experiments.phase_h.run_perturbation_30seeds
```

Same mechanism as Phase 8 (identical `run_one_seed` logic, RNG-fixed simulator, uncalibrated
`DelegationPolicy` defaults, C3 local-memory condition, midpoint hub removal at round 400 of 800),
but on a much larger, fresh, disjoint seed pool (30 seeds, 50000-50029) with 95% bootstrap
confidence intervals (10000 resamples) added. Resolves the original 6-seed run's statistical-power
concern (std of the utility drop exceeded its mean) without changing the qualitative conclusion:
the utility drop is now confirmed real but small (95% CI [0.0165, 0.0288]), and the
redistribution-without-clean-successor pattern is confirmed with non-overlapping CIs (Gini 0.7744 ->
0.6975). Writes `experiments/phase_h/results/perturbation_summary_30seeds.json`. See the "##
Revised (larger seed pool, n=30, with bootstrap confidence intervals)" section of
`docs/PHASE_8_REPORT.md`.

## Phase 9 -- scaling

```
python3 -m experiments.phase_i.run_scaling
```

Fleet sizes 10/25/50, 4 seeds each (pool 22000s), 400 decisions/robot. Writes
`experiments/phase_i/results/scaling_summary.json`. See `docs/PHASE_9_REPORT.md`.

## Phase 9 revised -- N=100 added

```
python3 -m experiments.phase_i.run_scaling_n100
```

Adds a fourth fleet size, N=100, on a fresh seed pool (60000-60003, 4 seeds), reusing the identical
`run_episode` mechanism from `run_scaling.py` unchanged (same RNG-fixed simulator, uncalibrated
`DelegationPolicy` defaults, C3 local-memory condition, 400 decisions/robot -- kept identical to the
original N=10/25/50 run so the new point is directly comparable). Confirms both original Phase 9
trends (utility worsens with scale, delegate-in Gini weakens with scale) continue cleanly through
N=100 with no reversal or plateau. Writes `experiments/phase_i/results/scaling_n100_summary.json`
and `experiments/phase_i/results/scaling_n100_raw.json`. See the "## Revised (N=100 added)" section
of `docs/PHASE_9_REPORT.md`.

## Phase 10 -- heterogeneity (H0 vs H1)

```
python3 -m experiments.phase_j.run_heterogeneity
```

6 seeds (23000-23005). H1 gives 3/10 robots a faster, more reliable reasoning backend. Writes
`experiments/phase_j/results/heterogeneity_summary.json`. See `docs/PHASE_10_REPORT.md`.

## Phase 11 -- ablation matrix

```
python3 -m experiments.phase_k.run_ablation
```

A0/A1/A3/A4/A5/A6, 8 seeds (23100-23107), calibrated delegation parameters, freshly-trained B3 on
dev/cal seeds 23900-23904/23910-23914. Writes `experiments/phase_k/results/ablation_summary.json`.
See `docs/PHASE_11_REPORT.md`.

## Phase 12 -- centralized manager (A2) and full system (A7)

```
python3 -m experiments.phase_l.run_ablation_full
```

Runs the full A0-A7 matrix (adding the centralized manager and the composed full system to Phase
11's A0/A1/A3-A6) on 10 fleet-size-10 seeds (25100-25109), with B3 retrained on dev/cal seeds
25900-25904/25910-25914, then a separate centralized-vs-distributed scaling comparison (A2 vs A3)
at N=10 and N=50 on a fresh pool (base 26000). Writes
`experiments/phase_l/results/ablation_full_summary.json` and
`experiments/phase_l/results/scaling_centralized_vs_distributed.json`. See the "## A2 and A7
(completed)" section of `docs/PHASE_11_REPORT.md` and the "## Addendum (Phase 12)" section of
`docs/FINAL_RESEARCH_AUDIT.md`.

## Phase M -- large-N, null-model-controlled resolution

```
python3 -m experiments.phase_m.run_large_n_resolution
```

Reruns the fully-controlled (randomized ID, shuffled order, randomized tie-break) specialization
measurement from Phase 6/7 revised on 50 fresh seeds (30000-30049) and compares it against an
explicit null model (`src/delegation/null_policy.py`, uniform-random peer choice among willing
peers) on the same seeds, with 95% bootstrap confidence intervals (10000 resamples). Writes
`experiments/phase_m/results/large_n_resolution_summary.json`,
`experiments/phase_m/results/real_per_seed.json`, `experiments/phase_m/results/null_per_seed.json`.
See the "## Definitive resolution (large-N, null-model-controlled)" sections of
`docs/PHASE_6_REPORT.md` / `docs/PHASE_7_REPORT.md`.

## Phase N -- mechanism investigation

```
python3 -m experiments.phase_n.run_mechanism_investigation
```

Tests four candidate explanations for the confirmed residual (decision-opportunity-count
privilege, spatial/positional privilege, whole-episode success-rate reinforcement, unbounded vs
windowed `PeerMemory` via `src/delegation/windowed_peer_memory.py`) on a fresh 50-seed pool
(30100-30149). Writes `experiments/phase_n/results/mechanism_investigation_summary.json`. See the
"## Mechanism investigation" section of `docs/PHASE_7_REPORT.md`.

## Phase O -- path-dependence perturbation and mathematical analysis

```
python3 -m experiments.phase_o.run_path_dependence_test
python3 -m experiments.phase_o.run_injection_depth_test
python3 -m experiments.phase_o.run_peer_capacity_ablation
```

`run_path_dependence_test` reverses peer-evaluation order at the first reversible exact-utility
tie in an otherwise identical episode and measures the hub-identity flip rate, on 30 fresh seeds
(30200-30229). `run_injection_depth_test` repeats this at four injection depths (rounds 0/200/400/
600) on a separate 30-seed pool (40000-40029) to test the mathematical model's flip-rate-decay
prediction. `run_peer_capacity_ablation` sweeps `peer_capacity` in {1,2,3,5,10} on a 20-seed pool
(42000-42019) to test the model's Gini-saturation prediction. Writes
`experiments/phase_o/results/path_dependence_summary.json`,
`experiments/phase_o/results/injection_depth_summary.json`,
`experiments/phase_o/results/peer_capacity_ablation_summary.json` (each with a matching
`*_per_seed.json`). See the "## Mechanism investigation, part 2" and "## Mathematical analysis of
the self-reinforcing loop" sections of `docs/PHASE_7_REPORT.md`.

## Final audit

A written synthesis of every phase above (original and revised) into an RQ1-RQ7 structure exists
in the project's private working repository and is not included in this export. No script to run
here; it is a narrative synthesis of the JSON summaries already listed above, all of which are
included in this repository.

## Determinism

All simulator randomness derives from `np.random.default_rng(seed)` / `numpy.random.SeedSequence`
at the fleet level, with each robot given an independently-spawned child generator (see
`src/environment/simulator.py`; this was changed from a structured `(seed_draw + robot_id)`
construction to `SeedSequence(seed).spawn(num_robots)` during the RNG-artifact fix described above).
Re-running any script with the same seed reproduces bit-identical per-decision records; this is
checked explicitly by `experiments/phase_a/run_baselines.py` (`reproducibility_check.json`) and by
`tests/test_simulator.py::test_simulator_determinism_same_seed`.

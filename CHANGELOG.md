## [0.5.1] - Unreleased

### Changed

- Adopted `ConstantKernel * RationalQuadratic + WhiteKernel` as the v0.5.1
  Gaussian-process covariance kernel. The original RBF GP produced
  acquisition-relevant score ties in 163/400 controlled Uni-Mol rounds, with
  1,223/4,000 selected molecules belonging to tied groups. Under otherwise
  identical conditions, the Rational Quadratic diagnostic reduced this to
  1/400 rounds and 10/4,000 selections. This was an acquisition-stability
  decision; predictive and discovery performance were deliberately not used.
- Retained the existing GP preprocessing, optimizer, noise `alpha`, UCB,
  normalization, and deterministic tie policy. Rational Quadratic produced
  optimizer/bound warnings in 270/400 diagnostic fits, primarily white-noise
  lower-bound and kernel-alpha upper-bound behavior. These warnings were not
  equivalent to acquisition collapse, and no further GP tuning was performed.


## [0.5.0] - 2026-09-29

### Added

- Added structure-derived QED calculation and molecular-identity alignment to
  target preprocessing and regression data loading.
- Added `ucb_qed`, `ucb_diversity`, and `ucb_qed_diversity` acquisition
  strategies alongside the existing UCB strategy.
- Added diversity-aware batch construction within a high-priority acquisition
  shortlist using Morgan-fingerprint Tanimoto distance.
- Added affinity-only and joint affinity/QED discovery endpoints, discovery
  curve AUC, qualifying Bemis-Murcko scaffold counts, and acquired-molecule
  diversity metrics.
- Added a metadata-driven v0.5 benchmark analysis pipeline with endpoint
  validation and seed-matched strategy comparisons.
- Added focused tests for QED preprocessing, molecular alignment, v0.5
  acquisition, evaluation, history aggregation, and benchmark analysis.

### Changed

- Organized v0.5 benchmark and analysis paths under the established
  `results/regression/` hierarchy and publication figures under
  `results/figures/regression/`.
- Made repeated-history aggregation robust to DataFrame attributes containing
  array-valued campaign provenance.
- Updated regression plotting and publication scripts with v0.5 metrics,
  canonical output names, and explicit output-path overrides.


## [0.4.0] - 2026-07-27

### Added

- Added a regression active-learning workflow for continuous molecular
  affinity optimization.
- Added a common surrogate-model interface with support for:
  Bayesian Ridge, Gaussian Process, Random Forest, Extra Trees,
  Gradient Boosting, HistGradientBoosting, K-nearest neighbours,
  and Linear Regression.
- Added uncertainty-aware upper confidence bound (UCB) acquisition with
  configurable exploration weight `beta`.
- Added repeated regression benchmarking across models, seeds, and
  active-learning rounds.
- Added round-wise, final-round, and paired comparison outputs.
- Added publication-quality benchmark figures for predictive performance,
  molecular discovery, uncertainty, acquisition behaviour, and runtime.
- Added scripts to compare UCB exploration weights using absolute and
  relative final-round performance differences.
- Added regression workflow and benchmark notebooks.

### Changed

- Refactored regression models behind a unified fit, predict, and uncertainty
  interface.
- Standardized benchmark metrics and output tables across surrogate models.
- Improved regression plotting utilities with consistent model styling,
  multi-panel figures, external legends, and ranking plots.
- Expanded the documentation to cover both classification and regression
  active-learning workflows.

### Fixed

- Corrected beta comparison logic for aggregated final-round benchmark
  summaries.
- Added validation for missing metrics, duplicated models, and inconsistent
  model sets during benchmark comparison.

### Results

- Increasing UCB beta from 1 to 2 improved predictive metrics for several
  uncertainty-aware regressors, but did not consistently improve molecular
  discovery.
- Beta 1 generally favoured stronger exploitation, producing better best-
  discovered and mean-discovered affinity values.
- Several surrogate models were insensitive to the tested beta values.
- The results show a model-dependent trade-off between surrogate accuracy
  and discovery performance rather than a universally optimal beta.


## v0.3.2 — Campaign calibration diagnostics

### Classification calibration
- Added Brier score and log loss.
- Added Expected Calibration Error and Maximum Calibration Error.
- Added uniform and equal-frequency calibration bins.
- Added explicit handling of collapsed bins for repeated probabilities.

### Regression uncertainty diagnostics
- Added Pearson and Spearman uncertainty–error correlations.
- Added error summaries across uncertainty quantiles.
- Added empirical ensemble interval coverage.
- Added interval-width and coverage-gap diagnostics.

### Validation
- Added synthetic tests for calibrated, overconfident, random, informative, and degenerate uncertainty cases.
- Verified empirical interval coverage against nominal levels.

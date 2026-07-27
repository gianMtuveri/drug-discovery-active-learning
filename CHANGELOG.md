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

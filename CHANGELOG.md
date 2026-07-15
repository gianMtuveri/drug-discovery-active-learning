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

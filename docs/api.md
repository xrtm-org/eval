# xrtm-eval API Reference

## Metrics (xrtm.eval.kit.eval.metrics)

- **`BrierScoreEvaluator`**: Decomposed Brier scoring.
- **`ExpectedCalibrationErrorEvaluator`**: Calibration binning.

## Trust (xrtm.eval.core.epistemics)

- **`IntegrityGuardian`**: Source validation engine.
- **`SourceTrustRegistry`**: Whitelist/blacklist management.

## Contracts

- **`ForecastResolution`**: Canonical ground-truth resolution object keyed by `forecast_request_id` (`question_id` remains a compatibility alias).
- **`ExternalComparisonRecord`** / **`PublicScorecardRow`**: Benchmark scorecard contracts keyed by `evaluation_path` (`reporting_lane` remains a compatibility alias).

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.7] - 2026-05-11

### Added
- Add canonical binary log score and dashboard-ready binary forecast scoring aggregation helpers.

### Changed
- Standardize evaluation artifact terminology around evaluation paths and reasoning traces.
- Require the released `xrtm-data>=0.2.7` floor for the coordinated terminology train.

### Fixed
- Preserve compatibility with current data-main forecast question fields while accepting terminology-aligned aliases.

## [0.2.6] - 2026-05-10

### Added
- Export benchmark scorecard artifacts from the public package surface and document the new scorecard workflow.

### Changed
- Require the released `xrtm-data>=0.2.6` floor for the coordinated train.

## [0.2.5] - 2026-04-30

### Fixed
- Hardened metric input validation while preserving calibration-bin compatibility for finite out-of-bound predictions.

## [0.2.1] - 2026-02-17

### Fixed
- **CI/CD**: Fix `pythonpath` issue in CI workflow
- **CI/CD**: Pin `xrtm-data` dependency to `v0.2.1` in lockfile
- **Linting**: Formatting and docstring fixes for ruff compliance

## [0.2.0] - 2026-02-04

### Changed
- **Architecture**: Restructured to `core/kit/providers` hierarchy for consistency with xrtm-forecast
- **core/schemas/**: Moved `ForecastResolution` to new location
- **providers/**: Added empty directory for future evaluator backends
- **README**: Added Project Structure section

### Breaking Changes
- Import paths changed: `xrtm.eval.schemas` → `xrtm.eval.core.schemas`

## [0.1.2] - 2026-01-28

### Added
- Epistemic trust primitives (`IntegrityGuardian`, `SourceTrustRegistry`)

## [0.1.1] - 2026-01-27

### Added
- Expected Calibration Error (ECE) evaluator

## [0.1.0] - 2026-01-27

### Added
- Initial release
- `BrierScoreEvaluator` for probabilistic forecast accuracy
- `Evaluator` protocol and `EvaluationResult` schema
- `ForecastResolution` schema for ground truth

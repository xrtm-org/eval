# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

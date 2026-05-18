# xrtm-eval Documentation

**The Judge.**

## Quick Links
- **[API Reference](api.md)**
- **[Concepts](concepts/)**
    - [Calibration](concepts/calibration.md)
    - [Benchmark scorecards](concepts/benchmark_scorecards.md)
    - [Epistemic Security](concepts/epistemic_security.md)

## Overview
`xrtm-eval` provides scoring metrics, trust primitives, and analysis tools to grade forecasts objectively.

For benchmark work, this package is the scoring and scorecard layer: it judges
forecast results, calibration, and comparison outcomes, while leaving corpus
ownership to `xrtm-data` and orchestration workflows to `xrtm-train`.

# Benchmark scorecards

`xrtm-eval` owns the **benchmark scoring contract** for the XRTM stack.

That means benchmark logic in this package should stay focused on judging
forecast quality, not on importing corpora or running replay loops.

## What belongs here

- Brier and related probabilistic metrics
- calibration and reliability summaries
- cohort comparison and leaderboard math
- significance and confidence-interval helpers
- benchmark result schemas and scorecard generation primitives

Concrete score-layer artifacts include:

- `BenchmarkScoreSummary`
- `BenchmarkComparisonSnapshot`
- `ExternalComparisonRecord`
- `ExternalLeaderboardSnapshot`
- `InspectableOutputReference`
- `PublicScorecardSnapshot`

## Public external comparison lane

`xrtm-eval` also owns the **reporting contract** for public benchmark references
that should appear in scorecards without being mislabeled as locally reproducible
stress arms.

Use the external lane when a scorecard row comes from:

- a public human baseline
- a public leaderboard snapshot
- an inspectable third-party output artifact

Those rows travel through `PublicScorecardRow.reporting_lane` as one of:

- `public-human-baseline`
- `public-leaderboard`
- `public-inspectable-output`

The default internal lane remains `internal-stress-suite`, which is reserved for
reproducible XRTM benchmark artifacts such as repeated stress suites and frozen
baseline-vs-candidate compares.

`ExternalComparisonRecord` and `InspectableOutputReference` exist so public
references keep explicit provenance (`source_name`, URLs, capture time, output
artifact pointers) instead of pretending a third-party system was rerun locally.

## What does not belong here

- corpus registry and provenance rules
- live benchmark submission workflows
- backtesting or replay orchestration
- product-facing report shells

Those responsibilities belong in `xrtm-data`, `xrtm-train`, and `xrtm`.

## Design rule

If a benchmark feature answers **"how should this run be scored fairly?"**, it
probably belongs in `xrtm-eval`.

If it answers **"what data are we allowed to use?"**, it belongs in
`xrtm-data`.

If it answers **"how do we execute and compare many runs?"**, it belongs in
`xrtm-train`.

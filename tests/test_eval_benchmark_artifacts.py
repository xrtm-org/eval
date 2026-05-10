# coding=utf-8
# Copyright 2026 XRTM Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import datetime, timezone

from xrtm.eval.core.eval.benchmark_artifacts import (
    BenchmarkComparisonRow,
    BenchmarkComparisonSnapshot,
    BenchmarkScoreSummary,
    ExternalComparisonRecord,
    ExternalLeaderboardEntry,
    ExternalLeaderboardSnapshot,
    InspectableOutputReference,
    PublicScorecardRow,
    PublicScorecardSnapshot,
    ScoreInterval,
)
from xrtm.eval.core.eval.definitions import EvaluationReport, EvaluationResult


def test_benchmark_score_summary_from_evaluation_report() -> None:
    report = EvaluationReport(
        metric_name="Brier Score",
        mean_score=0.21,
        total_evaluations=12,
        results=[EvaluationResult(subject_id="q1", score=0.21, ground_truth=True, prediction=0.7)],
        summary_statistics={"ece": 0.03, "reliability": 0.01, "resolution": 0.07, "uncertainty": 0.25},
    )

    summary = BenchmarkScoreSummary.from_evaluation_report(
        report,
        confidence_interval=ScoreInterval(low=0.19, high=0.24),
        notes=["preview corpus"],
        metadata={"lane": "offline"},
    )

    assert summary.metric_name == "Brier Score"
    assert summary.primary_score == 0.21
    assert summary.sample_size == 12
    assert summary.calibration_error == 0.03
    assert summary.reliability == 0.01
    assert summary.resolution == 0.07
    assert summary.uncertainty == 0.25
    assert summary.confidence_interval is not None
    assert summary.confidence_interval.low == 0.19
    assert summary.notes == ["preview corpus"]
    assert summary.metadata["lane"] == "offline"


def test_external_leaderboard_snapshot_returns_top_ranked_entry() -> None:
    snapshot = ExternalLeaderboardSnapshot(
        benchmark_id="forecastbench",
        benchmark_name="ForecastBench",
        source_name="ForecastBench",
        captured_at=datetime(2026, 5, 7, tzinfo=timezone.utc),
        entries=[
            ExternalLeaderboardEntry(system_id="xrtm", display_name="XRTM", rank=2, score_name="brier", score=0.18),
            ExternalLeaderboardEntry(system_id="mantic", display_name="Mantic", rank=1, score_name="brier", score=0.17),
        ],
    )

    top = snapshot.top_entry()

    assert top is not None
    assert top.system_id == "mantic"


def test_external_leaderboard_snapshot_converts_to_external_records() -> None:
    snapshot = ExternalLeaderboardSnapshot(
        benchmark_id="forecastbench",
        benchmark_name="ForecastBench",
        source_name="ForecastBench",
        source_url="https://bench.example/leaderboard",
        source_version="2026-05-07",
        scoring_rule="lower-is-better",
        captured_at=datetime(2026, 5, 7, tzinfo=timezone.utc),
        entries=[
            ExternalLeaderboardEntry(
                system_id="xrtm",
                display_name="XRTM",
                rank=2,
                score_name="brier",
                score=0.18,
                metadata={"track": "open"},
            )
        ],
    )

    records = snapshot.to_external_records(metadata={"capture_id": "snap-001"}, notes=["captured from public leaderboard"])

    assert len(records) == 1
    assert records[0].reporting_lane == "public-leaderboard"
    assert records[0].source_name == "ForecastBench"
    assert records[0].source_url == "https://bench.example/leaderboard"
    assert records[0].metadata["track"] == "open"
    assert records[0].metadata["capture_id"] == "snap-001"
    assert records[0].metadata["scoring_rule"] == "lower-is-better"
    assert records[0].notes == ["captured from public leaderboard"]


def test_external_comparison_record_renders_public_scorecard_row() -> None:
    record = ExternalComparisonRecord(
        benchmark_id="forecastbench",
        benchmark_name="ForecastBench",
        system_id="public-human",
        display_name="Metaculus Community",
        reporting_lane="public-inspectable-output",
        primary_score_name="brier",
        primary_score=0.16,
        captured_at=datetime(2026, 5, 7, tzinfo=timezone.utc),
        source_name="Metaculus",
        source_id="metaculus-public-2026-05-07",
        source_url="https://www.metaculus.com/questions/",
        inspectable_output=InspectableOutputReference(
            artifact_uri="s3://benchmarks/public/metaculus-2026-05-07.jsonl",
            artifact_format="jsonl",
            viewer_url="https://bench.example/artifacts/metaculus-2026-05-07",
        ),
        notes=["external reference only"],
        metadata={"review_status": "triaged"},
    )

    row = record.to_scorecard_row(lane="public-review")

    assert row.lane == "public-review"
    assert row.reporting_lane == "public-inspectable-output"
    assert row.is_external_reference is True
    assert row.inspectable_output is not None
    assert row.inspectable_output.artifact_format == "jsonl"
    assert row.metadata["source_id"] == "metaculus-public-2026-05-07"
    assert row.notes == ["external reference only"]


def test_public_scorecard_snapshot_collects_benchmark_ids() -> None:
    snapshot = PublicScorecardSnapshot(
        rows=[
            PublicScorecardRow(
                benchmark_id="forecastbench",
                benchmark_name="ForecastBench",
                system_id="xrtm",
                display_name="XRTM",
                lane="offline",
                primary_score_name="brier",
                primary_score=0.18,
            ),
            PublicScorecardRow(
                benchmark_id="futureeval",
                benchmark_name="FutureEval",
                system_id="xrtm",
                display_name="XRTM",
                lane="live",
                primary_score_name="rank",
                primary_score=4.0,
            ),
            PublicScorecardRow(
                benchmark_id="forecastbench",
                benchmark_name="ForecastBench",
                system_id="metaculus-community",
                display_name="Metaculus Community",
                lane="public",
                reporting_lane="public-human-baseline",
                primary_score_name="brier",
                primary_score=0.16,
                source_name="Metaculus",
            ),
        ]
    )

    assert snapshot.benchmark_ids() == ["forecastbench", "futureeval"]
    assert snapshot.reporting_lanes() == ["internal-stress-suite", "public-human-baseline"]
    assert len(snapshot.internal_rows()) == 2
    assert len(snapshot.external_rows()) == 1


def test_benchmark_comparison_snapshot_tracks_rows() -> None:
    snapshot = BenchmarkComparisonSnapshot(
        benchmark_id="xrtm-real-binary-v1",
        benchmark_name="XRTM Real Binary",
        rows=[
            BenchmarkComparisonRow(
                metric_name="eval_brier",
                baseline_system_id="baseline",
                candidate_system_id="candidate",
                direction="lower-is-better",
                baseline_value=0.21,
                candidate_value=0.19,
                delta=-0.02,
                interpretation="lower is better; candidate improved",
            )
        ],
    )

    assert snapshot.rows[0].metric_name == "eval_brier"
    assert snapshot.rows[0].delta == -0.02

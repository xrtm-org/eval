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

r"""Typed benchmark scorecard and leaderboard artifacts for xrtm-eval."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from xrtm.eval.core.eval.definitions import EvaluationReport

BenchmarkReportingLane = Literal[
    "internal-stress-suite",
    "public-human-baseline",
    "public-leaderboard",
    "public-inspectable-output",
]
ExternalBenchmarkReportingLane = Literal[
    "public-human-baseline",
    "public-leaderboard",
    "public-inspectable-output",
]

INTERNAL_STRESS_REPORTING_LANE: BenchmarkReportingLane = "internal-stress-suite"
EXTERNAL_BENCHMARK_REPORTING_LANES = frozenset(
    {
        "public-human-baseline",
        "public-leaderboard",
        "public-inspectable-output",
    }
)


class ScoreInterval(BaseModel):
    """Confidence interval for an aggregate benchmark score."""

    low: float
    high: float
    level: float = Field(default=0.95, ge=0.0, le=1.0)


class BenchmarkScoreSummary(BaseModel):
    """Normalized summary of one benchmark run's scored outcome."""

    metric_name: str
    primary_score_name: str
    primary_score: float
    sample_size: int = Field(ge=0)
    calibration_error: Optional[float] = None
    reliability: Optional[float] = None
    resolution: Optional[float] = None
    uncertainty: Optional[float] = None
    confidence_interval: Optional[ScoreInterval] = None
    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_evaluation_report(
        cls,
        report: EvaluationReport,
        *,
        primary_score_name: Optional[str] = None,
        confidence_interval: Optional[ScoreInterval] = None,
        notes: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "BenchmarkScoreSummary":
        """Create a benchmark summary from the canonical evaluation report."""
        summary = report.summary_statistics or {}
        return cls(
            metric_name=report.metric_name,
            primary_score_name=primary_score_name or report.metric_name,
            primary_score=report.mean_score,
            sample_size=report.total_evaluations,
            calibration_error=summary.get("ece"),
            reliability=summary.get("reliability"),
            resolution=summary.get("resolution"),
            uncertainty=summary.get("uncertainty"),
            confidence_interval=confidence_interval,
            notes=list(notes or []),
            metadata=dict(metadata or {}),
        )


class ExternalLeaderboardEntry(BaseModel):
    """One system entry captured from an external leaderboard snapshot."""

    system_id: str
    display_name: str
    rank: Optional[int] = Field(default=None, ge=1)
    score_name: str
    score: float
    sample_size: Optional[int] = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExternalLeaderboardSnapshot(BaseModel):
    """Immutable snapshot of an external benchmark leaderboard."""

    benchmark_id: str
    benchmark_name: str
    source_name: str
    captured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_url: Optional[str] = None
    source_version: Optional[str] = None
    scoring_rule: Optional[str] = None
    entries: list[ExternalLeaderboardEntry] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def top_entry(self) -> Optional[ExternalLeaderboardEntry]:
        """Return the current first-ranked entry if the ranking is available."""
        ranked = [entry for entry in self.entries if entry.rank is not None]
        if not ranked:
            return None
        return min(ranked, key=lambda entry: entry.rank if entry.rank is not None else 0)

    def to_external_records(
        self,
        *,
        metadata: Optional[dict[str, Any]] = None,
        notes: Optional[list[str]] = None,
    ) -> list["ExternalComparisonRecord"]:
        """Convert one public leaderboard snapshot into external scorecard records."""
        record_notes = list(notes or [])
        return [
            ExternalComparisonRecord(
                benchmark_id=self.benchmark_id,
                benchmark_name=self.benchmark_name,
                system_id=entry.system_id,
                display_name=entry.display_name,
                evaluation_path="public-leaderboard",
                primary_score_name=entry.score_name,
                primary_score=entry.score,
                captured_at=self.captured_at,
                source_name=self.source_name,
                source_url=self.source_url,
                source_version=self.source_version,
                rank=entry.rank,
                sample_size=entry.sample_size,
                notes=record_notes,
                metadata={
                    **dict(metadata or {}),
                    **entry.metadata,
                    **({"scoring_rule": self.scoring_rule} if self.scoring_rule else {}),
                },
            )
            for entry in self.entries
        ]


class InspectableOutputReference(BaseModel):
    """Reference to an inspectable external output without implying local reruns."""

    artifact_uri: str
    artifact_format: Optional[str] = None
    viewer_url: Optional[str] = None
    checksum: Optional[str] = None
    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExternalComparisonRecord(BaseModel):
    """Typed public comparison record for human baselines and external systems."""

    model_config = ConfigDict(populate_by_name=True)

    benchmark_id: str
    benchmark_name: str
    system_id: str
    display_name: str
    evaluation_path: ExternalBenchmarkReportingLane = Field(
        ...,
        validation_alias=AliasChoices("evaluation_path", "reporting_lane"),
    )
    primary_score_name: str
    primary_score: float
    captured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_name: str
    source_id: Optional[str] = None
    source_url: Optional[str] = None
    source_version: Optional[str] = None
    rank: Optional[int] = Field(default=None, ge=1)
    sample_size: Optional[int] = Field(default=None, ge=0)
    baseline_name: Optional[str] = None
    delta_vs_baseline: Optional[float] = None
    score_summary: Optional[BenchmarkScoreSummary] = None
    inspectable_output: Optional[InspectableOutputReference] = None
    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def reporting_lane(self) -> ExternalBenchmarkReportingLane:
        """Backward compatibility alias for ``evaluation_path``."""
        return self.evaluation_path

    @reporting_lane.setter
    def reporting_lane(self, value: ExternalBenchmarkReportingLane) -> None:
        """Backward compatibility setter for ``evaluation_path``."""
        self.evaluation_path = value

    def to_scorecard_row(
        self,
        *,
        lane: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "PublicScorecardRow":
        """Render this external comparison into the public scorecard schema."""
        row_metadata = dict(self.metadata)
        if self.source_id is not None:
            row_metadata.setdefault("source_id", self.source_id)
        if metadata:
            row_metadata.update(metadata)
        return PublicScorecardRow(
            benchmark_id=self.benchmark_id,
            benchmark_name=self.benchmark_name,
            system_id=self.system_id,
            display_name=self.display_name,
            lane=lane or self.reporting_lane,
            evaluation_path=self.evaluation_path,
            primary_score_name=self.primary_score_name,
            primary_score=self.primary_score,
            captured_at=self.captured_at,
            rank=self.rank,
            sample_size=self.sample_size,
            baseline_name=self.baseline_name,
            delta_vs_baseline=self.delta_vs_baseline,
            score_summary=self.score_summary,
            source_name=self.source_name,
            source_url=self.source_url,
            source_version=self.source_version,
            inspectable_output=self.inspectable_output,
            notes=list(self.notes),
            metadata=row_metadata,
        )


class PublicScorecardRow(BaseModel):
    """One public-facing scorecard row for a system on a benchmark."""

    model_config = ConfigDict(populate_by_name=True)

    benchmark_id: str
    benchmark_name: str
    system_id: str
    display_name: str
    lane: str
    evaluation_path: BenchmarkReportingLane = Field(
        default=INTERNAL_STRESS_REPORTING_LANE,
        validation_alias=AliasChoices("evaluation_path", "reporting_lane"),
    )
    primary_score_name: str
    primary_score: float
    captured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    rank: Optional[int] = Field(default=None, ge=1)
    sample_size: Optional[int] = Field(default=None, ge=0)
    baseline_name: Optional[str] = None
    delta_vs_baseline: Optional[float] = None
    score_summary: Optional[BenchmarkScoreSummary] = None
    source_name: Optional[str] = None
    source_url: Optional[str] = None
    source_version: Optional[str] = None
    inspectable_output: Optional[InspectableOutputReference] = None
    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def reporting_lane(self) -> BenchmarkReportingLane:
        """Backward compatibility alias for ``evaluation_path``."""
        return self.evaluation_path

    @reporting_lane.setter
    def reporting_lane(self, value: BenchmarkReportingLane) -> None:
        """Backward compatibility setter for ``evaluation_path``."""
        self.evaluation_path = value

    @property
    def is_external_reference(self) -> bool:
        """Whether this row comes from the public external comparison lane."""
        return self.evaluation_path in EXTERNAL_BENCHMARK_REPORTING_LANES


class PublicScorecardSnapshot(BaseModel):
    """Versioned public scoreboard artifact."""

    schema_version: str = "xrtm.scorecard.v1"
    captured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    rows: list[PublicScorecardRow] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def benchmark_ids(self) -> list[str]:
        """Return the unique benchmark identifiers contained in this snapshot."""
        return sorted({row.benchmark_id for row in self.rows})

    def reporting_lanes(self) -> list[str]:
        """Return the unique reporting lanes represented in this snapshot."""
        return self.evaluation_paths()

    def evaluation_paths(self) -> list[str]:
        """Return the unique evaluation paths represented in this snapshot."""
        return sorted({row.evaluation_path for row in self.rows})

    def external_rows(self) -> list[PublicScorecardRow]:
        """Return rows sourced from public human or competitor references."""
        return [row for row in self.rows if row.is_external_reference]

    def internal_rows(self) -> list[PublicScorecardRow]:
        """Return rows sourced from reproducible internal stress artifacts."""
        return [row for row in self.rows if not row.is_external_reference]


class BenchmarkComparisonRow(BaseModel):
    """One comparative benchmark row between a baseline and another system."""

    metric_name: str
    baseline_system_id: str
    candidate_system_id: str
    direction: str
    baseline_value: Optional[float] = None
    candidate_value: Optional[float] = None
    delta: Optional[float] = None
    interpretation: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkComparisonSnapshot(BaseModel):
    """Versioned internal comparison artifact for repeated benchmark suites."""

    schema_version: str = "xrtm.benchmark-comparison.v1"
    captured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    benchmark_id: str
    benchmark_name: str
    rows: list[BenchmarkComparisonRow] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "BenchmarkReportingLane",
    "ScoreInterval",
    "BenchmarkScoreSummary",
    "ExternalBenchmarkReportingLane",
    "EXTERNAL_BENCHMARK_REPORTING_LANES",
    "INTERNAL_STRESS_REPORTING_LANE",
    "InspectableOutputReference",
    "ExternalComparisonRecord",
    "ExternalLeaderboardEntry",
    "ExternalLeaderboardSnapshot",
    "BenchmarkComparisonRow",
    "BenchmarkComparisonSnapshot",
    "PublicScorecardRow",
    "PublicScorecardSnapshot",
]

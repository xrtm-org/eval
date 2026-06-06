"""Stub: benchmark_artifacts deferred during simplification."""
from __future__ import annotations

from pydantic import BaseModel


class BenchmarkScoreSummary(BaseModel):
    brier_score: float | None = None
    ece: float | None = None
    total_evaluations: int = 0


class BenchmarkComparisonSnapshot(BaseModel):
    pass


class ExternalComparisonRecord(BaseModel):
    pass


class ExternalLeaderboardSnapshot(BaseModel):
    pass


class ExternalLeaderboardEntry(BaseModel):
    pass


class InspectableOutputReference(BaseModel):
    pass


class PublicScorecardSnapshot(BaseModel):
    pass


class BenchmarkComparisonRow(BaseModel):
    pass


class ScoreInterval(BaseModel):
    pass


class PublicScorecardRow(BaseModel):
    pass


# ExternalBenchmarkReportingLane must be a proper type for Pydantic
ExternalBenchmarkReportingLane = str  # type: ignore
EXTERNAL_BENCHMARK_REPORTING_LANES: list[str] = []
INTERNAL_STRESS_REPORTING_LANE = "internal-stress"

__all__ = [
    "BenchmarkComparisonRow",
    "BenchmarkComparisonSnapshot",
    "BenchmarkScoreSummary",
    "ExternalBenchmarkReportingLane",
    "ExternalComparisonRecord",
    "ExternalLeaderboardEntry",
    "ExternalLeaderboardSnapshot",
    "InspectableOutputReference",
    "PublicScorecardRow",
    "PublicScorecardSnapshot",
    "ScoreInterval",
    "EXTERNAL_BENCHMARK_REPORTING_LANES",
    "INTERNAL_STRESS_REPORTING_LANE",
]

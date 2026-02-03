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

r"""
Core interfaces and domain-agnostic logic for xrtm-eval.

This module exports evaluator protocols, epistemics utilities, and
core schemas. MUST NOT import from kit/ or providers/.
"""

from xrtm.eval.core.epistemics import (
    IntegrityGuardian,
    SourceTrustEntry,
    SourceTrustRegistry,
)
from xrtm.eval.core.eval import EvaluationReport, EvaluationResult, Evaluator
from xrtm.eval.core.schemas import ForecastResolution

__all__ = [
    # Evaluator protocol
    "Evaluator",
    "EvaluationResult",
    "EvaluationReport",
    # Epistemics
    "IntegrityGuardian",
    "SourceTrustRegistry",
    "SourceTrustEntry",
    # Schemas
    "ForecastResolution",
]

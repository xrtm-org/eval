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

r"""Kit-level re-exports of the aggregation utilities.

The implementations live in :mod:`xrtm.eval.core.eval.aggregation`; this module
keeps the historical ``xrtm.eval.kit.eval.aggregation`` import path working.
"""

from __future__ import annotations

from xrtm.eval.core.eval.aggregation import (
    inverse_variance_weighting,
    inverse_variance_weighting_with_variance,
    robustness_check_mad,
)

__all__ = [
    "inverse_variance_weighting",
    "inverse_variance_weighting_with_variance",
    "robustness_check_mad",
]

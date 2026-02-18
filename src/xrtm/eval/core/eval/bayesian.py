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
Bayesian probability helpers.

Converts between probability and odds representations and applies
Bayes factor updates.  Used by the epistemic security and calibration layers.
"""

import logging

logger = logging.getLogger(__name__)


def probability_to_odds(probability: float) -> float:
    r"""Convert a probability in [0, 1] to odds."""
    if probability >= 1.0:
        return float("inf")
    if probability <= 0.0:
        return 0.0
    return probability / (1.0 - probability)


def odds_to_probability(odds: float) -> float:
    r"""Convert odds back to a probability."""
    if odds == float("inf"):
        return 1.0
    return odds / (1.0 + odds)


def bayesian_update(prior_probability: float, bayes_factor: float) -> float:
    r"""Apply a Bayes factor update to a prior probability and return the posterior."""
    prior_odds = probability_to_odds(prior_probability)
    posterior_odds = prior_odds * bayes_factor
    return odds_to_probability(posterior_odds)


__all__ = ["probability_to_odds", "odds_to_probability", "bayesian_update"]

# coding=utf-8
# Copyright 2026 XRTM Team. All rights reserved.

import logging

logger = logging.getLogger(__name__)


def probability_to_odds(probability: float) -> float:
    if probability >= 1.0:
        return float("inf")
    if probability <= 0.0:
        return 0.0
    return probability / (1.0 - probability)


def odds_to_probability(odds: float) -> float:
    if odds == float("inf"):
        return 1.0
    return odds / (1.0 + odds)


def bayesian_update(prior_probability: float, bayes_factor: float) -> float:
    prior_odds = probability_to_odds(prior_probability)
    posterior_odds = prior_odds * bayes_factor
    return odds_to_probability(posterior_odds)


__all__ = ["probability_to_odds", "odds_to_probability", "bayesian_update"]

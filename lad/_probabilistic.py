"""Degree-bound computations from Gardy, Lardeux, and Saubion (2022)."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, log


@dataclass(frozen=True)
class DegreeProbability:
    degree: int
    log_probability: float


def _alpha(projection_values: int, observations: int, residual_values: int) -> int:
    """Equation (3): coefficient for a projection with ``projection_values``."""

    return sum(
        (-1) ** (projection_values - used)
        * comb(projection_values, used)
        * comb(used * residual_values, observations)
        for used in range(1, projection_values + 1)
        if used * residual_values >= observations
    )


def _reasonable_bound_counts(
    attribute_count: int,
    degree: int,
    positive_count: int,
    negative_count: int,
) -> tuple[int, int]:
    """Return the numerator/denominator of paper probability Pr(n1,n2/n)."""

    if not 1 <= degree <= attribute_count:
        raise ValueError("degree must be between one and attribute_count")
    if positive_count < 1 or negative_count < 1:
        raise ValueError("both groups must contain observations")
    total = positive_count + negative_count
    domain_y = 1 << degree
    domain_z = 1 << (attribute_count - degree)

    alpha_total = {
        values: _alpha(values, total, domain_z)
        for values in range(1, min(total, domain_y) + 1)
    }
    denominator = sum(
        (1 << values) * comb(domain_y, values) * coefficient
        for values, coefficient in alpha_total.items()
    )

    alpha_positive = {
        values: _alpha(values, positive_count, domain_z)
        for values in range(1, min(positive_count, domain_y - 1) + 1)
    }
    alpha_negative = {
        values: _alpha(values, negative_count, domain_z)
        for values in range(1, min(negative_count, domain_y - 1) + 1)
    }
    numerator = 0
    for positive_values, positive_coefficient in alpha_positive.items():
        maximum_negative = min(negative_count, domain_y - positive_values)
        for negative_values in range(1, maximum_negative + 1):
            negative_coefficient = alpha_negative.get(negative_values, 0)
            assignments = comb(domain_y, positive_values) * comb(
                domain_y - positive_values, negative_values
            )
            numerator += (
                assignments * positive_coefficient * negative_coefficient
            )
    return numerator, denominator


def reasonable_degree_probabilities(
    attribute_count: int,
    max_degree: int,
    positive_count: int,
    negative_count: int,
) -> tuple[DegreeProbability, ...]:
    """Evaluate the paper's Model-M1 degree-bound probability.

    Results are returned in log space because the exact rational probabilities
    can be far below floating-point range for realistic data sets.
    """

    if attribute_count < 1 or max_degree < 1:
        raise ValueError("attribute_count and max_degree must be positive")
    upper = min(attribute_count, max_degree)
    results = []
    for degree in range(1, upper + 1):
        numerator, denominator = _reasonable_bound_counts(
            attribute_count, degree, positive_count, negative_count
        )
        log_probability = (
            log(numerator) - log(denominator)
            if numerator and denominator
            else float("-inf")
        )
        results.append(DegreeProbability(degree, log_probability))
    return tuple(results)


def reasonable_degree_bound(
    attribute_count: int,
    max_degree: int,
    positive_count: int,
    negative_count: int,
) -> tuple[int, tuple[DegreeProbability, ...]]:
    """Select the degree maximizing the paper's reasonable-bound probability."""

    probabilities = reasonable_degree_probabilities(
        attribute_count, max_degree, positive_count, negative_count
    )
    selected = max(
        probabilities,
        key=lambda result: (result.log_probability, -result.degree),
    ).degree
    return selected, probabilities

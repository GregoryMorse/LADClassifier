from itertools import combinations, product

from lad._probabilistic import (
    _reasonable_bound_counts,
    reasonable_degree_bound,
)


def _direct_model_m1_counts(attribute_count, degree, positive_count, negative_count):
    domain = tuple(product((0, 1), repeat=attribute_count))
    total = positive_count + negative_count
    direct_denominator = 0
    direct_numerator = 0
    for observations in combinations(domain, total):
        projected_values = {observation[:degree] for observation in observations}
        for assignments in range(1 << len(projected_values)):
            direct_denominator += 1
            group_by_projection = {
                projected: (assignments >> index) & 1
                for index, projected in enumerate(sorted(projected_values))
            }
            groups = [
                group_by_projection[observation[:degree]]
                for observation in observations
            ]
            direct_numerator += int(
                groups.count(0) == positive_count
                and groups.count(1) == negative_count
            )
    return direct_numerator, direct_denominator


def test_model_m1_probability_matches_direct_tiny_domain_count():
    for attribute_count, degree, positive_count, negative_count in (
        (2, 1, 1, 1),
        (3, 1, 1, 2),
        (3, 2, 1, 2),
    ):
        assert _reasonable_bound_counts(
            attribute_count, degree, positive_count, negative_count
        ) == _direct_model_m1_counts(
            attribute_count, degree, positive_count, negative_count
        )


def test_reasonable_degree_bound_is_deterministic_and_bounded():
    selected, probabilities = reasonable_degree_bound(8, 4, 6, 10)
    assert 1 <= selected <= 4
    assert [result.degree for result in probabilities] == [1, 2, 3, 4]
    assert selected == max(
        probabilities,
        key=lambda result: (result.log_probability, -result.degree),
    ).degree

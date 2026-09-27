from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pulp

from lad import LADClassifier
from lad._ilp_patterns import (
    _solve,
    _solver_factory,
    hammer_maximum_patterns,
    maximum_pattern,
    minimum_pattern_cover,
)


def test_auto_uses_unlicensed_open_source_solver():
    chosen = _solver_factory(
        "auto", time_limit_seconds=None, relative_gap=0, threads=1
    )
    assert chosen.name in {"highs", "cbc"}


def test_time_limited_incumbent_is_not_claimed_as_optimal():
    class FakeProblem:
        sol_status = pulp.LpSolutionIntegerFeasible

        def solve(self, _solver):
            return pulp.LpStatusOptimal

    factory = SimpleNamespace(create=lambda: SimpleNamespace(close=lambda: None), pulp=pulp)
    assert _solve(FakeProblem(), factory) == "FeasibleNotProven"


def _cover(values, pattern):
    return np.logical_and.reduce(
        [values[:, feature] == value for feature, value in pattern]
    )


def _brute_maximum(positive, negative, observation, max_degree, robustness=1):
    feasible = []
    for degree in range(1, max_degree + 1):
        for attributes in combinations(range(positive.shape[1]), degree):
            if any(
                sum(observation[feature] != row[feature] for feature in attributes)
                < robustness
                for row in negative
            ):
                continue
            pattern = tuple(
                (feature, int(observation[feature])) for feature in attributes
            )
            feasible.append((int(np.sum(_cover(positive, pattern))), degree, pattern))
    if not feasible:
        return None
    maximum_coverage = max(value[0] for value in feasible)
    minimum_degree = min(
        value[1] for value in feasible if value[0] == maximum_coverage
    )
    return maximum_coverage, minimum_degree


def test_hammer_ilp_matches_independent_exhaustive_maximum_pattern_oracle():
    random = np.random.RandomState(23)
    for _ in range(6):
        matrix = random.randint(0, 2, size=(7, 5))
        positive = matrix[:3]
        negative = matrix[3:]
        for observation in positive:
            expected = _brute_maximum(positive, negative, observation, 3)
            actual = maximum_pattern(
                positive, negative, observation, 3, solver="cbc"
            )
            if expected is None:
                assert actual is None
                continue
            assert actual is not None
            assert (actual.positive_coverage, len(actual.pattern)) == expected
            assert not np.any(_cover(negative, actual.pattern))


def test_weighted_hammer_matches_brute_force_coverage_and_degree():
    positive = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1]])
    negative = np.array([[1, 0, 1], [1, 1, 0]])
    weights = np.array([1.0, 0.25, 8.0])
    observation = positive[0]
    feasible = []
    for degree in (1, 2):
        for attributes in combinations(range(positive.shape[1]), degree):
            pattern = tuple((feature, int(observation[feature])) for feature in attributes)
            if np.any(_cover(negative, pattern)):
                continue
            feasible.append((float(weights[_cover(positive, pattern)].sum()), degree))
    expected_weight = max(weight for weight, _ in feasible)
    expected_degree = min(degree for weight, degree in feasible
                          if weight == expected_weight)

    actual = maximum_pattern(
        positive, negative, observation, 2,
        solver="cbc", positive_weight=weights,
    )

    assert actual is not None
    assert actual.positive_coverage_weight == expected_weight
    assert len(actual.pattern) == expected_degree


def test_hammer_ilp_supports_finite_domains_and_robust_covering():
    positive = np.array([[0, 1, 2], [0, 2, 2], [1, 1, 2]])
    negative = np.array([[2, 0, 2], [2, 1, 0]])
    expected = _brute_maximum(
        positive, negative, positive[0], 3, robustness=2
    )
    actual = maximum_pattern(
        positive,
        negative,
        positive[0],
        3,
        solver="cbc",
        robustness=2,
    )
    assert actual is not None
    assert (actual.positive_coverage, len(actual.pattern)) == expected


def test_highs_and_cbc_prove_the_same_primary_and_secondary_optima():
    positive = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 1]])
    negative = np.array([[1, 1, 0], [1, 1, 1]])
    solutions = [
        maximum_pattern(positive, negative, positive[0], 2, solver=solver)
        for solver in ("highs", "cbc")
    ]
    assert all(solution is not None for solution in solutions)
    assert {
        (solution.positive_coverage, len(solution.pattern))
        for solution in solutions
    } == {(3, 1)}


def test_hammer_ilp_degree_bound_can_make_anchor_infeasible():
    positive = np.array([[0, 0, 0]])
    negative = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
    assert maximum_pattern(
        positive, negative, positive[0], 1, solver="cbc"
    ) is None
    assert maximum_pattern(
        positive, negative, positive[0], 2, solver="cbc"
    ) is not None
    with np.testing.assert_raises_regex(ValueError, "positive class"):
        maximum_pattern(positive, negative, negative[0], 2, solver="cbc")


def test_minimum_pattern_cover_matches_exhaustive_set_cover_cardinality():
    positive = np.array([[0, 0], [0, 1], [1, 0]])
    patterns = (
        ((0, 0),),
        ((1, 0),),
        ((0, 0), (1, 1)),
        ((0, 1), (1, 0)),
    )
    selected = minimum_pattern_cover(positive, patterns, solver="cbc")
    assert len(selected) == 2
    coverage = np.column_stack([_cover(positive, pattern) for pattern in selected])
    assert np.all(np.any(coverage, axis=1))


def test_hammer_candidate_generation_and_model_compression_preserve_coverage():
    positive = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 1]])
    negative = np.array([[1, 1, 0], [1, 1, 1]])
    result = hammer_maximum_patterns(
        positive,
        negative,
        2,
        solver="cbc",
        model_selection="minimum_cover",
    )
    assert result.solver == "cbc"
    assert len(result.patterns) <= len(result.candidate_patterns)
    assert result.covered_positive_count == len(positive)


def test_classifier_can_select_hammer_ilp_engine():
    features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    labels = np.array([0, 1, 1, 0])
    classifier = LADClassifier(
        degree=2,
        random=False,
        threshold_pct=1,
        pattern_method="hammer_ilp",
        ilp_solver="cbc",
        ilp_model_selection="minimum_cover",
        binarizer_params={"method": "equaldivisions", "divisions": 2},
    ).fit(features, labels)
    assert classifier.predict(features).tolist() == labels.tolist()
    assert all(entry["solver"] == "cbc" for entry in classifier.ilp_diagnostics_)

from itertools import combinations, product

import numpy as np

from lad import LADClassifier
from lad._prime_patterns import prime_patterns


def _brute_prime_patterns(positive, negative, max_degree):
    patterns = set()
    all_observations = np.vstack([positive, negative])
    for degree in range(1, max_degree + 1):
        for attributes in combinations(range(positive.shape[1]), degree):
            value_domains = [
                tuple(np.unique(all_observations[:, attribute]))
                for attribute in attributes
            ]
            for values in product(*value_domains):
                pattern = tuple(zip(attributes, values))
                positive_cover = np.logical_and.reduce(
                    [positive[:, feature] == value for feature, value in pattern]
                )
                negative_cover = np.logical_and.reduce(
                    [negative[:, feature] == value for feature, value in pattern]
                )
                if not positive_cover.any() or negative_cover.any():
                    continue
                prime = True
                for removed in range(degree):
                    reduced = pattern[:removed] + pattern[removed + 1 :]
                    if not reduced:
                        continue
                    reduced_positive = np.logical_and.reduce(
                        [positive[:, feature] == value for feature, value in reduced]
                    )
                    reduced_negative = np.logical_and.reduce(
                        [negative[:, feature] == value for feature, value in reduced]
                    )
                    if reduced_positive.any() and not reduced_negative.any():
                        prime = False
                if prime:
                    patterns.add(pattern)
    return patterns


def test_ppc2_matches_independent_exhaustive_prime_pattern_oracle():
    random = np.random.RandomState(7)
    for _ in range(20):
        matrix = random.randint(0, 2, size=(8, 5))
        positive = matrix[:3]
        negative = matrix[3:]
        assert set(prime_patterns(positive, negative, 3)) == _brute_prime_patterns(
            positive, negative, 3
        )


def test_ppc2_finite_domain_extension_matches_exhaustive_oracle():
    random = np.random.RandomState(11)
    for _ in range(10):
        matrix = random.randint(0, 3, size=(7, 4))
        positive = matrix[:3]
        negative = matrix[3:]
        assert set(prime_patterns(positive, negative, 3)) == _brute_prime_patterns(
            positive, negative, 3
        )


def test_strong_ppc2_patterns_have_undominated_positive_covers():
    positive = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 1]])
    negative = np.array([[1, 1, 0], [1, 1, 1]])
    strong = prime_patterns(positive, negative, 3, strong_only=True)
    covers = []
    for pattern in strong:
        covers.append(
            frozenset(
                np.flatnonzero(
                    np.logical_and.reduce(
                        [positive[:, feature] == value for feature, value in pattern]
                    )
                )
            )
        )
    assert all(not (left < right) for left in covers for right in covers)


def test_classifier_can_select_exact_ppc2_engine():
    features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    labels = np.array([0, 1, 1, 0])
    classifier = LADClassifier(
        degree=2,
        random=False,
        threshold_pct=1,
        pattern_method='chambon_ppc2_prime',
        binarizer_params={'method': 'equaldivisions', 'divisions': 2},
    ).fit(features, labels)
    assert classifier.predict(features).tolist() == labels.tolist()
    assert classifier.selected_degree_ == 2


def test_classifier_can_select_gardy_degree_strategy_without_test_data():
    features = np.array(
        [[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 0], [0, 1, 1]],
        dtype=float,
    )
    labels = np.array([0, 0, 1, 1, 1, 0])
    classifier = LADClassifier(
        degree=3,
        random=False,
        degree_strategy='gardy_2022',
        binarizer_params={'method': 'equaldivisions', 'divisions': 2},
    ).fit(features, labels)
    assert 1 <= classifier.selected_degree_ <= 3
    assert {entry['class'] for entry in classifier.degree_diagnostics_} == {0, 1}


def test_public_ppc2_rejects_undiscretized_values():
    with np.testing.assert_raises_regex(ValueError, "discretized integer"):
        prime_patterns(
            np.array([[0.1, 0.2]]),
            np.array([[0.8, 0.9]]),
            2,
        )

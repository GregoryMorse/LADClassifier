from itertools import product

import numpy as np

from lad import LADClassifier
from lad._intervals import (
    extended_gray_code,
    prevalence_matrices,
    upper_prevalence,
)


def _brute_prevalence(distribution, basis):
    expected = np.zeros_like(distribution)
    for corner in product(*(range(size) for size in distribution.shape)):
        lower = np.minimum(corner, basis)
        upper = np.maximum(corner, basis)
        region = tuple(
            slice(int(low), int(high) + 1)
            for low, high in zip(lower, upper)
        )
        expected[corner] = distribution[region].sum()
    return expected


def test_extended_gray_code_visits_every_basis_once():
    maxima = np.array([2, 3, 1])
    codes = extended_gray_code(maxima)
    bases = [tuple(code[0]) for code in codes]

    assert len(codes) == int(np.prod(maxima + 1))
    assert len(set(bases)) == len(codes)
    assert bases[0] == tuple(maxima)
    assert all(
        np.abs(current[0] - previous[0]).sum() == 1
        for previous, current in zip(codes, codes[1:])
    )


def test_incremental_prevalence_matches_brute_force_for_every_basis():
    distribution = np.array(
        [
            [[1, 0], [0, 2], [1, 0]],
            [[0, 1], [3, 0], [0, 1]],
        ],
        dtype=np.int64,
    )
    original = distribution.copy()

    for basis, prevalence in prevalence_matrices(distribution):
        assert np.array_equal(
            prevalence,
            _brute_prevalence(distribution, basis),
        )

    assert np.array_equal(distribution, original)
    assert np.array_equal(
        upper_prevalence(distribution),
        _brute_prevalence(distribution, np.subtract(distribution.shape, 1)),
    )


def test_paper_worked_example_exercises_canonical_implementation():
    LADClassifier._testpaper()


def test_classifier_does_not_skip_the_mixed_binary_quadrant():
    features = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    labels = np.array([0, 1, 0, 0])

    classifier = LADClassifier(
        degree=2,
        maxcombs=10,
        random=False,
        random_state=0,
    ).fit(features, labels)

    assert classifier.predict(features).tolist() == labels.tolist()
    assert len(classifier.booleqs_[1][2]) > 1

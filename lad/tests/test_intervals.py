from itertools import product

import numpy as np

from lad import LADClassifier
from lad._intervals import (
    extended_gray_code,
    prevalence_matrices,
    upper_prevalence,
)


def test_distinct_prefix_sharing_xor_patterns_survive_pruning():
    # All eight degree-three patterns are needed. The previous subset
    # comparator falsely merged equal-length patterns sharing a first literal.
    values=np.tile(np.array(list(product((0, 1), repeat=3))), (4, 1))
    target=values.sum(axis=1) % 2
    for fast in (False, True):
        model=LADClassifier(degree=3,random=False,threshold_pct=1,minmatch_pct=.01,
            max_patterns_per_class=32,max_projection_evaluations=1,
            binary_projection_fast_path=fast,
            binarizer_params=dict(method='equaldistribution',divisions=3,binarymode=True,interval=False))
        model.fit(values,target,sample_weight=np.ones(len(target)))
        assert sum(len(patterns) for _,_,patterns in model.booleqs_)==8
        assert np.array_equal(model.predict(values),target)
        assert not model.fit_timed_out_


def test_total_projection_budget_retains_valid_best_so_far_patterns():
    rng=np.random.default_rng(45)
    values=rng.integers(0,2,(1000,24));target=values[:,0]^values[:,1]
    model=LADClassifier(degree=4,random=True,threshold_pct=.55,maxcombs=80,
        minimum_precision_lift=.02,minimum_precision_floor=.52,
        max_projection_evaluations=10,binary_projection_fast_path=True,
        max_patterns_per_class=32,random_state=42,
        binarizer_params=dict(method='equaldistribution',divisions=3,binarymode=True,interval=False))
    model.fit(values,target)
    assert model.projection_evaluations_==10
    assert model.binary_fast_projection_evaluations_==10
    assert model.projection_budget_exhausted_ and not model.fit_timed_out_


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

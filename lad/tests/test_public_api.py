import importlib.util

import numpy as np

import lad
import lad.binarization as binarization


def test_complete_public_api_is_exported():
    expected = {
        'LADClassifier',
        'DiscretizingTransformer',
        'FeatureGroup',
        'BooleanEquationClassifier',
        'plot_confusion_matrix',
    }
    assert expected.issubset(set(lad.__all__))


def test_binarization_helpers_are_public_and_backward_compatible():
    expected = {
        'binarizer', 'binarize', 'binarizeall', 'postbinarize', 'binarizecompare'
    }
    assert expected.issubset(set(dir(lad.LADClassifier)))
    assert expected.issubset(set(lad.__all__))
    assert lad.binarizer is binarization.binarizer

    values = np.array([0.0, 0.2, 0.8, 1.0])
    labels = np.array([0, 0, 1, 1])
    parameters, transformed, names = lad.LADClassifier.binarize(
        values,
        'value',
        labels,
        method='equaldivisions',
        divisions=2,
    )
    assert parameters['divisions'] == 2
    # Exactly two divisions reduce to one sufficient Boolean feature.
    assert len(transformed) == 1
    assert len(names) == 2


def test_public_binarizer_round_trip_and_comparisons():
    features = np.array([
        [0.0, 1.0],
        [0.2, 0.8],
        [0.8, 0.2],
        [1.0, 0.0],
    ])
    labels = np.array([0, 0, 1, 1])
    params = {'method': 'equaldivisions', 'divisions': 2}

    transformed, _, values, bounds = lad.binarizeall(
        features,
        labels,
        feature_names=['left', 'right'],
        binarizer_params=params,
    )

    assert np.array_equal(transformed, lad.postbinarize(features, values))
    assert bounds == [2, 2]

    comparisons, names, mutex = lad.binarizecompare(
        features,
        ['left', 'right'],
        [(0, 1)],
    )
    assert names == ['left<right', 'left==right', 'left>right']
    assert np.array(comparisons).shape == (3, 4)
    assert sorted(mutex[0]) == [0, 1, 2]


def test_all_public_binarization_strategies_are_reusable():
    values = np.array([0.0, 0.1, 0.4, 0.7, 0.9, 1.0])
    labels = np.array([0, 0, 1, 1, 0, 0])

    for parameters in (
        {'method': 'equaldivisions', 'divisions': 3},
        {'method': 'equaldistribution', 'divisions': 3},
        {'method': 'minimumdifferentiated'},
        {'method': 'equaldivisions', 'divisions': 3, 'binarymode': False},
    ):
        fitted, transformed, _ = lad.binarize(
            values,
            'value',
            labels,
            **parameters,
        )
        reapplied = lad.binarizer(values, **fitted)
        assert np.array_equal(np.asarray(transformed), np.asarray(reapplied))


def test_removed_legacy_module_is_not_importable():
    assert importlib.util.find_spec('lad._legacy_2019') is None


def test_discretizing_transformer_round_trip():
    features = np.array([[0.0], [0.2], [0.8], [1.0]])
    labels = np.array([0, 0, 1, 1])
    transformer = lad.DiscretizingTransformer(
        binarizer_params={'method': 'equaldivisions', 'divisions': 2},
        random_state=0,
        feat_names=['value'],
    )
    transformed = transformer.fit_transform(features, labels)
    assert transformed.shape[0] == features.shape[0]

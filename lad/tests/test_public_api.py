import numpy as np

import lad


def test_complete_public_api_is_exported():
    expected = {
        'LADClassifier',
        'DiscretizingTransformer',
        'FeatureGroup',
        'BooleanEquationClassifier',
        'plot_confusion_matrix',
    }
    assert expected.issubset(set(lad.__all__))


def test_legacy_binarization_helpers_remain_available():
    expected = {
        'binarizer', 'binarize', 'binarizeall', 'postbinarize', 'binarizecompare'
    }
    assert expected.issubset(set(dir(lad.LADClassifier)))

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

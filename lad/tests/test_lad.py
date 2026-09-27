import pytest
import numpy as np

from sklearn.datasets import load_iris
from lad import DiscretizingTransformer, LADClassifier


@pytest.fixture
def data():
    return load_iris(return_X_y=True)

def test_template_classifier(data):
    X, y = data
    clf = LADClassifier()
    clf._testpaper()

    clf.fit(X, y)
    assert hasattr(clf, 'classes_')
    assert hasattr(clf, 'booleqs_')

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)


def test_small_boolean_problem_is_learned_deterministically():
    features = np.array([
        [0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0],
        [0.1, 0.2], [0.2, 0.9], [0.8, 0.1], [0.9, 0.8],
    ])
    labels = np.array([0, 0, 0, 1, 0, 0, 0, 1])
    model = LADClassifier(degree=2, maxcombs=10, random_state=0)
    model.fit(features, labels)
    assert model.predict(features).tolist() == labels.tolist()


def test_interval_bookkeeping_sentinel_is_not_exposed_as_a_rule():
    features = np.zeros((8, 2), dtype=float)
    labels = np.array([0, 1] * 4)
    model = LADClassifier(
        degree=2,
        maxcombs=2,
        threshold_pct=1,
        minmatch_pct=0.1,
        random_state=0,
    )

    model.fit(features, labels)

    assert all(equations == [] for _, _, equations in model.booleqs_)


def test_interval_search_honors_fit_deadline_and_returns_valid_partial_model():
    random = np.random.RandomState(11)
    features = random.normal(size=(80, 12))
    labels = np.resize(np.array([0, 1, 2, 3]), len(features))
    model = LADClassifier(
        degree=4,
        maxcombs=100,
        threshold_pct=0.7,
        minmatch_pct=0.001,
        random_state=0,
        fit_time_limit_seconds=1e-9,
    )

    model.fit(features, labels)

    assert model.fit_timed_out_
    assert model.fit_elapsed_seconds_ >= 0
    assert model.predict(features).shape == labels.shape


def test_level_binarization_emits_nested_threshold_features():
    values = np.array([0.0, 1.0, 2.0, 3.0])
    cut_points = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]

    levels = DiscretizingTransformer._binarizer(
        values, cut_points, binarymode=True, interval=False
    )

    assert np.asarray(levels).tolist() == [
        [False, True, True, True],
        [False, False, True, True],
    ]


def test_equal_distribution_cutpoints_follow_sample_weights():
    features = np.arange(6.0).reshape(-1, 1)
    transformer = DiscretizingTransformer(
        binarizer_params={
            'method': 'equaldistribution', 'divisions': 2,
            'binarymode': True, 'interval': False,
        }
    )
    transformer.fit(features, sample_weight=[1, 1, 1, 1, 1, 20])
    assert transformer.binarizer_values_[0]['cut_points'][1][0] == 5.0
    assert transformer.transform(features).shape == (6, 1)


def test_weighted_interval_precision_and_default_class():
    features = np.array([[False], [False], [True], [True]])
    labels = np.array([0, 1, 0, 1])
    weights = np.array([1.0, 10.0, 10.0, 1.0])
    model = LADClassifier(
        degree=1, random=False, threshold_pct=0.7,
        minmatch_pct=0.01, random_state=0,
    ).fit(features, labels, sample_weight=weights)

    assert model.predict(features).tolist() == [1, 1, 0, 0]
    assert model.sample_weight_sum_ == 22.0
    assert model.effective_sample_size_ == pytest.approx(22 ** 2 / 202)


def test_multioutput_fit_applies_the_same_row_weights_to_each_cutpoint():
    features = np.array([[False], [False], [True], [True]])
    labels = np.column_stack((np.array([0, 1, 0, 1]),
                              np.array([1, 0, 1, 0])))
    weights = np.array([1.0, 10.0, 10.0, 1.0])
    model = LADClassifier(
        degree=1, random=False, threshold_pct=0.7,
        minmatch_pct=0.01, random_state=0,
    ).fit(features, labels, sample_weight=weights)

    assert model.predict(features).tolist() == [[1, 0], [1, 0], [0, 1], [0, 1]]


def test_fit_rejects_invalid_sample_weights():
    features = np.array([[0.0], [1.0]])
    labels = np.array([0, 1])
    for weights in ([1.0], [1.0, -1.0], [1.0, float('nan')]):
        with pytest.raises(ValueError, match='sample_weight'):
            LADClassifier().fit(features, labels, sample_weight=weights)
    with pytest.raises(ValueError, match='weights are zero'):
        LADClassifier().fit(features, labels, sample_weight=[0.0, 0.0])


def test_weighted_alexe_hammer_matches_integer_replication_oracle():
    features = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
    )
    labels = np.array([0, 1, 1, 0])
    weights = np.array([2, 3, 1, 4])
    parameters = dict(
        degree=2, random=False, threshold_pct=0.55,
        minmatch_pct=0.1, random_state=0,
        binarizer_params={'method': 'equaldivisions', 'divisions': 2,
                          'binarymode': True, 'interval': False},
    )
    weighted = LADClassifier(**parameters).fit(
        features, labels, sample_weight=weights.astype(float)
    )
    repeated = LADClassifier(**parameters).fit(
        np.repeat(features, weights, axis=0), np.repeat(labels, weights),
    )
    assert weighted.default_class_ == repeated.default_class_
    assert weighted.booleqs_ == repeated.booleqs_
    np.testing.assert_array_equal(weighted.predict(features), repeated.predict(features))


def test_pattern_cap_bounds_noisy_weighted_search():
    features = np.tile(
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float),
        (8, 1),
    )
    labels = np.tile([0, 1, 1, 0], 8)
    model = LADClassifier(
        degree=2, random=False, threshold_pct=0.55,
        minmatch_pct=0.01, max_patterns_per_class=2,
        binarizer_params={
            'method': 'equaldivisions', 'divisions': 2,
            'binarymode': True, 'interval': False,
        },
    ).fit(features, labels, sample_weight=np.linspace(0.5, 1.0, len(labels)))
    assert all(len(equations) <= 2 for _, _, equations in model.booleqs_)


def test_precision_lift_uses_each_class_base_rate():
    features = np.array([[0.0]] * 4 + [[1.0]] * 6)
    labels = np.array([1, 1, 0, 0] + [0] * 6)
    parameters = dict(
        degree=1, random=False, threshold_pct=0.55, minmatch_pct=0,
        binarizer_params={
            'method': 'equaldivisions', 'divisions': 2,
            'binarymode': True, 'interval': False,
        },
    )
    absolute = LADClassifier(**parameters).fit(features, labels)
    lifted = LADClassifier(
        **parameters, minimum_precision_lift=0.05
    ).fit(features, labels)
    positive_rules = lambda model: next(
        equations for _, target, equations in model.booleqs_ if target == 1
    )
    assert positive_rules(absolute) == []
    assert positive_rules(lifted)
    assert lifted.class_precision_thresholds_[0]["0"] == pytest.approx(0.85)
    assert lifted.class_precision_thresholds_[0]["1"] == pytest.approx(0.25)

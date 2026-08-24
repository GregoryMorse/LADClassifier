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

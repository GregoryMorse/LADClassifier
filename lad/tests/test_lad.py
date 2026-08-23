import pytest
import numpy as np

from sklearn.datasets import load_iris
from lad import LADClassifier


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

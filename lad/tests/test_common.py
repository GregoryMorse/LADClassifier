import pytest

from sklearn.utils.estimator_checks import check_estimator

from lad import LADClassifier


@pytest.mark.parametrize(
    "Estimator", [LADClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)

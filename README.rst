LADClassifier
=============

``LADClassifier`` implements Logical Analysis of Data (LAD) for Python's
scikit-learn ecosystem. LAD constructs Boolean patterns from discretized
features, producing rules that can be inspected and explained rather than
only opaque class scores.

This release reconciles the original public project with the newer classifier
used by the trader project. It retains the 2019 public binarization helpers
while exposing the newer transformer, feature grouping, multiclass, and
Boolean-equation APIs from one canonical package.

Public API
----------

The ``lad`` package exports:

* ``LADClassifier``
* ``DiscretizingTransformer``
* ``FeatureGroup``
* ``BooleanEquationClassifier``
* ``plot_confusion_matrix``

The original ``LADClassifier.binarizer``, ``binarize``, ``binarizeall``,
``postbinarize``, and ``binarizecompare`` helpers remain available for
existing callers.

Installation
------------

Clone the repository and install it in editable mode while developing::

    git clone https://github.com/GregoryMorse/LADClassifier.git
    cd LADClassifier
    python -m pip install -e .

Numba acceleration and plotting are optional::

    python -m pip install -e ".[accelerate,plot]"

Run the tests with::

    python -m pip install -e ".[tests]"
    python -m pytest

Quick start
-----------

::

    import numpy as np
    from lad import LADClassifier

    X = np.array([
        [0.0, 0.1],
        [0.2, 0.1],
        [0.8, 0.9],
        [1.0, 0.8],
    ])
    y = np.array([0, 0, 1, 1])

    classifier = LADClassifier(random_state=0)
    classifier.fit(X, y)
    predictions = classifier.predict(X)

For the longer walkthrough, see `the quick-start guide
<doc/quick_start.rst>`_. Bug reports and contributions are welcome through
the `GitHub repository <https://github.com/GregoryMorse/LADClassifier>`_.

License
-------

LADClassifier is distributed under the MIT License.

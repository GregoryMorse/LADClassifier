LADClassifier
=============

.. image:: doc/_static/ladclassifier-logo.svg
   :alt: LADClassifier - logical patterns, explainable decisions
   :width: 720
   :align: center

``LADClassifier`` implements Logical Analysis of Data (LAD) for Python's
scikit-learn ecosystem. LAD constructs Boolean patterns from discretized
features, producing rules that can be inspected and explained rather than
only opaque class scores.

This release reconciles the original public project with the newer classifier
used by the trader project. It exposes the binarizer, transformer, feature
grouping, multiclass, and Boolean-equation APIs from one canonical package.

Public API
----------

The ``lad`` package exports:

* ``LADClassifier``
* ``DiscretizingTransformer``
* ``FeatureGroup``
* ``BooleanEquationClassifier``
* ``plot_confusion_matrix``

Feature binarization now lives in the public ``lad.binarization`` module. Its
``binarizer``, ``binarize``, ``binarizeall``, ``postbinarize``, and
``binarizecompare`` functions are also exported by ``lad``. Compatibility
aliases remain on ``LADClassifier`` for existing callers.

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

Algorithm validation and performance
------------------------------------

The interval-enumeration core follows Algorithms 1--3 from Sorin Alexe and
Peter L. Hammer, `Accelerated algorithm for pattern detection in logical
analysis of data <https://doi.org/10.1016/j.dam.2005.03.032>`_. Automated
tests reproduce all 20 prevalence matrices in the paper's worked example and
also compare every interval of independent multidimensional examples with a
brute-force oracle.

See `paper validation <doc/algorithm.rst>`_ for the implementation mapping and
scope, and `performance guidance <doc/performance.rst>`_ before selecting a
degree or an exhaustive feature search. A reproducible benchmark is provided
in ``benchmarks/benchmark_lad.py``.

License
-------

LADClassifier is distributed under the MIT License.

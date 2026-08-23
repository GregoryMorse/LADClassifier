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

Selectable methods
------------------

``pattern_method`` selects ``alexe_hammer``, ``chambon_ppc2_prime``,
``chambon_ppc2_strong``, or ``hammer_ilp``. The PPC2 and Hammer ILP variants
generate pure patterns and therefore require ``threshold_pct=1``.
``hammer_ilp`` exactly linearizes the maximum-pattern formulation of Hammer
and Bonates (2006), then can solve their minimum-cardinality pattern-cover
model. Install ``LADClassifier[ilp]`` for PuLP; ``ilp_solver='auto'`` prefers
Gurobi, then in-process HiGHS, with bundled CBC as the final fallback.
``degree_strategy`` independently
selects a fixed bound or the experimental ``gardy_2022`` Model-M1 bound. The
latter uses only training-group counts and retains the paper's uniformity and
independence assumptions.

Algorithm validation and performance
------------------------------------

The interval-enumeration core follows Algorithms 1--3 from Sorin Alexe and
Peter L. Hammer, `Accelerated algorithm for pattern detection in logical
analysis of data <https://doi.org/10.1016/j.dam.2005.03.032>`_. Automated
tests reproduce all 20 prevalence matrices in the paper's worked example and
also compare every interval of independent multidimensional examples with a
brute-force oracle.

The PPC2 engine follows Arthur Chambon et al., `Accelerated Algorithm for
Computation of All Prime Patterns in Logical Analysis of Data
<https://doi.org/10.5220/0007389702100220>`_, and is compared with an independent
exhaustive pattern oracle. The optional degree bound follows Danièle Gardy et
al., `A Computational Model for Logical Analysis of Data
<https://arxiv.org/abs/2207.05664>`_.

See `paper validation <doc/algorithm.rst>`_ for the implementation mapping and
scope, and `performance guidance <doc/performance.rst>`_ before selecting a
degree or an exhaustive feature search. A reproducible benchmark is provided
in ``benchmarks/benchmark_lad.py``.

License
-------

LADClassifier is distributed under the MIT License.

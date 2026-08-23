.. _user_guide:

User guide
==========

Logical Analysis of Data represents a class with conjunctions of simple
conditions. A learned rule can therefore be inspected as a small set of
feature/value restrictions rather than only as an opaque score.

Training flow
-------------

``LADClassifier.fit`` performs four stages:

1. Validate the scikit-learn input and discover output classes.
2. Discretize numeric features into finite values.
3. Enumerate bounded intervals for selected feature projections.
4. Retain patterns meeting the requested precision and minimum-coverage
   criteria, then remove dominated patterns.

At prediction time, every retained class rule set is evaluated. The
highest-scoring matching class wins; an observation matching no rule uses the
most prevalent training class. This makes the residual policy explicit and
prevents class ordering from creating an arbitrary default.

Key controls
------------

``degree``
    Maximum number of discretized features in one projection. Degrees 2 and 3
    are the practical starting points for broad signal sets.

``random`` and ``maxcombs``
    With more available features than ``degree``, random search bounds the
    number of feature projections considered. Set ``random=False`` only when
    exhaustive coverage is tractable and required.

``threshold_pct``
    Minimum pattern precision. ``1.0`` admits only patterns with no
    counterexamples in the training sample.

``minmatch_pct``
    Minimum fraction of training observations covered by a pattern.

``binarizer_params``
    A dictionary shared by all features or a list of per-feature dictionaries.
    Supported methods include equal divisions, equal distribution, minimum
    differentiated ranges, and the transformer's learned grouping mode.

Evaluation discipline
---------------------

Pattern fidelity does not imply predictive validity. For time-series use,
fit and tune only on past observations, retain a genuinely unseen time window,
and account for fees, spread, slippage, latency, class imbalance, and regime
change. Report per-class precision and coverage in addition to aggregate
accuracy.

See :doc:`algorithm` for the paper-to-code audit and :doc:`performance` for
the scaling limits of exhaustive pattern search.

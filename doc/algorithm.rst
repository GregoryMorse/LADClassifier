Paper validation
================

Reference
---------

The interval-enumeration core is based on:

    Sorin Alexe and Peter L. Hammer, *Accelerated algorithm for pattern
    detection in logical analysis of data*, Discrete Applied Mathematics
    154(7), 1050--1063 (2006),
    `doi:10.1016/j.dam.2005.03.032
    <https://doi.org/10.1016/j.dam.2005.03.032>`_.

The implementation is a faithful vectorized translation of the paper's
interval-prevalence engine, with the following direct mapping:

.. list-table:: Paper-to-code mapping
   :header-rows: 1
   :widths: 18 35 47

   * - Paper
     - Implementation
     - Validation
   * - Algorithm 1
     - ``lad._intervals.upper_prevalence``
     - Reverse cumulative sums along every selected axis; checked against a
       brute-force interval oracle.
   * - Algorithm 2
     - ``lad._intervals.extended_gray_code``
     - Visits every mixed-radix basis once, with adjacent bases differing in
       exactly one coordinate.
   * - Algorithm 3
     - ``lad._intervals.update_prevalence`` and ``prevalence_matrices``
     - Each incremental matrix is compared with prevalence recomputed from
       scratch for every basis of a multidimensional example.
   * - Worked example
     - ``LADClassifier._testpaper``
     - Reproduces all 20 prevalence matrices tabulated by the authors.

Audit result
------------

The older test contained an independent, correct Algorithm 2, but production
used a modified copy that did not reverse the later transition directions.
For two binary dimensions it visited only three of four bases and could omit
a mixed quadrant. The production classifier now calls the same canonical
implementation used by the paper test. A classifier-level regression protects
the formerly missing quadrant.

Scope of faithfulness
---------------------

The claim above is deliberately specific. Algorithms 1--3 efficiently
enumerate interval prevalence for a fixed discrete projection. The surrounding
Python classifier adds discretization, random or exhaustive feature-subset
selection, precision and coverage thresholds, multiclass handling, dominance
filtering, and scikit-learn integration. Those engineering and modeling layers
are not presented as line-for-line translations of the paper.

The automated checks establish algorithmic equivalence on the paper's worked
case plus independent finite oracles; they are much stronger than a single
end-to-end accuracy assertion, but they are not a formal proof for every
possible input.

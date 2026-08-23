Performance and scaling
=======================

What the paper guarantees
-------------------------

For a fixed feature projection, the Alexe--Hammer procedure obtains the first
prevalence matrix with cumulative sums, then updates it while walking adjacent
interval bases. Its work is linear in the generated interval representation
for a fixed, small number of attributes. The implementation keeps those hot
matrix operations in NumPy's compiled loops and updates prevalence in place.

What still grows combinatorially
--------------------------------

The classifier must also choose projections. Exhaustive search of ``F``
discretized features at degree ``d`` considers ``C(F, d)`` subsets before the
interval work inside each subset. More bins increase each subset's interval
space as well. No implementation of Algorithms 1--3 removes that outer
combinatorial cost.

PPC2 avoids enumerating every literal term but can still emit exponentially
many prime patterns; its degree bound is therefore a hard safety control, not
only a regularizer. The Gardy 2022 option can propose a smaller bound cheaply,
but its probabilistic assumptions must be validated on the actual data and it
does not change worst-case complexity.

The Hammer ILP engine solves an exact linearization for every distinct anchor.
Its formulation is polynomial in the training rows and discretized features,
but integer-programming runtime remains instance dependent. Each anchor uses a
single lexicographic maximum-coverage/minimum-degree solve; ``minimum_cover``
adds one set-cover solve per class. The anchor cap is an explicit approximation,
not an exactness-preserving optimization. Solver work occurs only in training;
inference remains Boolean rule matching.

Before building each anchor model, duplicate same-class mismatch sets are
combined into weighted coverage variables. Opposite-class mismatch sets are
deduplicated and any constraint implied by a subset constraint is removed.
Both are exact presolve reductions.

For large signal collections:

* start with degree 2 or 3;
* use a deterministic ``random_state`` and a bounded ``maxcombs`` for routine
  nightly searches;
* reserve ``random=False`` for small audit cases or deliberate exhaustive
  studies;
* record runtime, feature count, bin counts, patterns retained, precision, and
  coverage for every build;
* parallelize independent stocks, sectors, or validation folds outside a
  single estimator after profiling establishes the real bottleneck.

Benchmarking
------------

Run the included synthetic benchmark from the repository root::

    python benchmarks/benchmark_lad.py --samples 2000 --features 32 \
        --degree 3 --maxcombs 100 --repeat 3

Select a later method explicitly when profiling it::

    python benchmarks/benchmark_lad.py --samples 240 --features 30 \
        --degree 4 --pattern-method chambon_ppc2_prime --repeat 3

Profile an ILP backend through the same command::

    python benchmarks/benchmark_lad.py --samples 120 --features 16 \
        --degree 3 --pattern-method hammer_ilp --ilp-solver cbc --repeat 1

As a calibration point on the development machine, a single exact run with 120
rows, 16 binary features, degree 3, all anchors, and minimum-cover selection
took about 8.3 seconds with HiGHS. A 24-row, 8-feature instance took about
0.44 seconds with HiGHS and 2.11 seconds with external-process CBC. These are
synthetic measurements, not scaling guarantees; anchor ILPs vary sharply in
difficulty and the 1,600-row trading folds must be profiled before selecting
this engine for a nightly universe-wide build.

Use ``--exhaustive`` only for a deliberately small feature count. The command
reports median and individual fit times as JSON so results can be archived and
compared across commits and machines. It intentionally has no universal pass
or fail threshold: production capacity must be measured on the actual signal
matrix and target hardware.

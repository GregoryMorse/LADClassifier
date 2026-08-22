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

Use ``--exhaustive`` only for a deliberately small feature count. The command
reports median and individual fit times as JSON so results can be archived and
compared across commits and machines. It intentionally has no universal pass
or fail threshold: production capacity must be measured on the actual signal
matrix and target hardware.

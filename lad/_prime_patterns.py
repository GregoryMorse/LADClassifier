"""Prime-pattern generation from Chambon et al. (2019), PPC2.

The paper builds the non-dominated solutions of a constraint matrix for each
positive observation, then transforms every solution into a prime pattern.
This implementation keeps that construction explicit and degree bounded.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


Pattern = tuple[tuple[int, int], ...]


def non_dominated_hitting_sets(
    constraints: Iterable[Iterable[int]],
    max_degree: int,
) -> tuple[frozenset[int], ...]:
    """Return every inclusion-minimal hitting set up to ``max_degree``.

    This is the NDS construction used by PPC2.  Dominated candidates are
    removed after every constraint, which is equivalent to retaining the
    paper's non-dominated solution set and avoids carrying supersets forward.
    """

    if max_degree < 1:
        raise ValueError("max_degree must be positive")
    normalized_sets = {
        frozenset(int(value) for value in constraint) for constraint in constraints
    }
    if any(not constraint for constraint in normalized_sets):
        return tuple()
    normalized = tuple(
        sorted(
            (sum(1 << value for value in constraint) for constraint in normalized_sets),
            key=lambda constraint: (constraint.bit_count(), constraint),
        )
    )

    # Integer bitsets make the paper's set operations constant-time machine
    # operations for the feature counts used by LAD. Two newly extended NDS
    # solutions cannot dominate one another: their parents were already
    # non-dominated and both parents are disjoint from the current constraint.
    solutions: set[int] = {0}
    for constraint in normalized:
        already_satisfied = {
            solution for solution in solutions if solution & constraint
        }
        extended: set[int] = set()
        for solution in solutions - already_satisfied:
            remaining = constraint
            while remaining:
                attribute = remaining & -remaining
                remaining ^= attribute
                candidate = solution | attribute
                if candidate.bit_count() > max_degree:
                    continue
                subset = candidate
                dominated = False
                while subset:
                    if subset in already_satisfied:
                        dominated = True
                        break
                    subset = (subset - 1) & candidate
                if not dominated:
                    extended.add(candidate)
        solutions = already_satisfied | extended
        if not solutions:
            return tuple()
    decoded = (
        frozenset(
            attribute
            for attribute in range(solution.bit_length())
            if solution & (1 << attribute)
        )
        for solution in solutions
    )
    return tuple(
        sorted(decoded, key=lambda value: (len(value), tuple(sorted(value))))
    )


def prime_patterns(
    positive: np.ndarray,
    negative: np.ndarray,
    max_degree: int,
    *,
    strong_only: bool = False,
) -> tuple[Pattern, ...]:
    """Compute all degree-bounded PPC2 prime patterns for one-vs-rest data.

    The 2019 paper states the algorithm for Boolean attributes.  Its constraint
    construction extends directly to finite-domain discretized attributes:
    a literal is equality with the positive observation's value, and a selected
    attribute excludes a negative observation exactly when their values differ.
    """

    positive = np.asarray(positive)
    negative = np.asarray(negative)
    if positive.ndim != 2 or negative.ndim != 2:
        raise ValueError("positive and negative observations must be two-dimensional")
    if positive.shape[1] != negative.shape[1]:
        raise ValueError("positive and negative observations need equal feature counts")
    def is_discrete(values: np.ndarray) -> bool:
        return np.issubdtype(values.dtype, np.integer) or np.issubdtype(
            values.dtype, np.bool_
        )

    if not (is_discrete(positive) and is_discrete(negative)):
        raise ValueError("PPC2 observations must contain discretized integer values")
    if max_degree < 1:
        raise ValueError("max_degree must be positive")
    if len(positive) == 0 or len(negative) == 0:
        return tuple()

    patterns: set[Pattern] = set()
    for observation in positive:
        constraints = [
            np.flatnonzero(observation != other).tolist() for other in negative
        ]
        for solution in non_dominated_hitting_sets(constraints, max_degree):
            pattern = tuple(
                (attribute, int(observation[attribute]))
                for attribute in sorted(solution)
            )
            patterns.add(pattern)

    ordered = tuple(sorted(patterns, key=lambda value: (len(value), value)))
    if not strong_only or not ordered:
        return ordered

    covers = {
        pattern: frozenset(
            np.flatnonzero(
                np.logical_and.reduce(
                    [positive[:, feature] == value for feature, value in pattern]
                )
            ).tolist()
        )
        for pattern in ordered
    }
    return tuple(
        pattern
        for pattern in ordered
        if not any(
            covers[pattern] < covers[other]
            for other in ordered
            if other != pattern
        )
    )

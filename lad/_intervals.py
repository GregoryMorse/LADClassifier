"""Faithful interval-prevalence primitives from Alexe and Hammer (2006)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np


def upper_prevalence(
    distribution: np.ndarray,
    axes: Sequence[int] | None = None,
) -> np.ndarray:
    """Construct the paper's initial prevalence matrix, Pi(V0).

    Algorithm 1 is a reverse cumulative sum along every coordinate. The
    returned matrix is independent of the input and uses NumPy's compiled
    cumulative-sum loops rather than Python element-by-element iteration.
    """
    prevalence = np.array(distribution, copy=True)
    selected_axes = range(prevalence.ndim) if axes is None else axes
    for axis in selected_axes:
        reversed_axis = np.flip(prevalence, axis=axis)
        np.cumsum(reversed_axis, axis=axis, out=reversed_axis)
    return prevalence


def extended_gray_code(
    maxima: Sequence[int] | np.ndarray,
) -> list[tuple[np.ndarray, int, np.ndarray]]:
    """Generate Algorithm 2's mixed-radix extended Gray code.

    Each tuple contains the basis, the coordinate changed to reach it, and
    the transition vector after that change. Consecutive bases differ by one
    on exactly one coordinate, and every basis is emitted exactly once.
    """
    limits = np.asarray(maxima, dtype=np.intp)
    if limits.ndim != 1 or np.any(limits < 0):
        raise ValueError("maxima must be a one-dimensional non-negative vector")

    basis = limits.copy()
    transition = np.full(len(limits), -1, dtype=np.int8)
    changed_axis = 0
    codes: list[tuple[np.ndarray, int, np.ndarray]] = []
    while True:
        codes.append((basis.copy(), changed_axis, transition.copy()))
        candidates = np.flatnonzero(
            (basis + transition >= 0) & (basis + transition <= limits)
        )
        if len(candidates) == 0:
            return codes
        changed_axis = int(candidates[-1])
        basis[changed_axis] += transition[changed_axis]
        transition[changed_axis + 1:] *= -1


def update_prevalence(
    prevalence: np.ndarray,
    current_value: int,
    next_value: int,
    axis: int,
    direction: int,
) -> np.ndarray:
    """Apply equations (23)-(24) in place for one Gray-code transition."""
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    if next_value != current_value + direction:
        raise ValueError("basis transition must move exactly one coordinate")

    view = np.swapaxes(prevalence, 0, axis)
    if direction == 1:
        view[current_value + 1:] -= view[current_value]
        view[:current_value + 1] += view[next_value]
    else:
        view[:current_value] -= view[current_value]
        view[current_value:] += view[next_value]
    return prevalence


def prevalence_matrices(
    distribution: np.ndarray,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield every basis and prevalence matrix using Algorithms 1-3."""
    limits = np.subtract(distribution.shape, 1)
    codes = extended_gray_code(limits)
    prevalence = upper_prevalence(distribution)
    yield codes[0][0].copy(), prevalence.copy()
    for previous, current in zip(codes, codes[1:]):
        axis = current[1]
        direction = int(current[2][axis])
        update_prevalence(
            prevalence,
            int(previous[0][axis]),
            int(current[0][axis]),
            axis,
            direction,
        )
        yield current[0].copy(), prevalence.copy()

"""Hammer and Bonates (2006) maximum-pattern ILP models.

The paper formulates a maximum pattern around an observation as a polynomial
set-covering problem.  This module uses the paper's exact linearization: one
binary variable selects each literal and one binary variable records whether
each same-class observation remains covered.  A second, optional set-covering
ILP selects a minimum-cardinality model from the generated patterns.

PuLP is deliberately imported lazily so the historical and PPC2 engines do
not require an integer-programming installation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np


Pattern = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class MaximumPatternSolution:
    """One optimal observation-anchored pattern and its training coverage."""

    pattern: Pattern
    positive_coverage: int
    solver: str


@dataclass(frozen=True)
class HammerPatternModel:
    """Generated maximum patterns plus model-selection diagnostics."""

    patterns: tuple[Pattern, ...]
    candidate_patterns: tuple[Pattern, ...]
    solver: str
    unique_anchor_count: int
    feasible_anchor_count: int
    covered_positive_count: int


@dataclass(frozen=True)
class _SolverFactory:
    name: str
    pulp: Any
    time_limit_seconds: float | None
    relative_gap: float
    threads: int | None

    def create(self):
        if self.name == "gurobi":
            options: dict[str, Any] = {}
            if self.threads is not None:
                options["Threads"] = self.threads
            return self.pulp.GUROBI(
                msg=False,
                timeLimit=self.time_limit_seconds,
                gapRel=self.relative_gap,
                envOptions={"OutputFlag": 0},
                manageEnv=True,
                **options,
            )
        if self.name == "highs":
            return self.pulp.HiGHS(
                msg=False,
                timeLimit=self.time_limit_seconds,
                gapRel=self.relative_gap,
                threads=self.threads,
            )
        packaged_solver = getattr(self.pulp, "PULP_CBC_CMD", None)
        bundled_cbc = getattr(packaged_solver, "pulp_cbc_path", None)
        return self.pulp.COIN_CMD(
            msg=False,
            timeLimit=self.time_limit_seconds,
            gapRel=self.relative_gap,
            threads=self.threads,
            path=bundled_cbc,
        )


def _solver_factory(
    solver: str,
    *,
    time_limit_seconds: float | None,
    relative_gap: float,
    threads: int | None,
) -> _SolverFactory:
    try:
        import pulp
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "install LADClassifier[ilp] to use the Hammer ILP engine"
        ) from exc

    selected = str(solver).casefold()
    if selected not in {"auto", "gurobi", "highs", "cbc"}:
        raise ValueError("ILP solver must be auto, gurobi, highs, or cbc")
    if time_limit_seconds is not None and time_limit_seconds <= 0:
        raise ValueError("ILP time limit must be positive or None")
    if not 0 <= relative_gap <= 1:
        raise ValueError("ILP relative gap must be between 0 and 1")
    if threads is not None and threads < 1:
        raise ValueError("ILP thread count must be positive or None")

    if selected in {"auto", "gurobi"}:
        candidate = _SolverFactory(
            "gurobi", pulp, time_limit_seconds, relative_gap, threads
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            available = bool(candidate.create().available())
        if available:
            return candidate
        if selected == "gurobi":
            raise RuntimeError("Gurobi was requested but is unavailable or unlicensed")

    if selected in {"auto", "highs"}:
        candidate = _SolverFactory(
            "highs", pulp, time_limit_seconds, relative_gap, threads
        )
        if candidate.create().available():
            return candidate
        if selected == "highs":
            raise RuntimeError("HiGHS was requested but is unavailable")

    candidate = _SolverFactory(
        "cbc", pulp, time_limit_seconds, relative_gap, threads
    )
    if not candidate.create().available():  # pragma: no cover - packaged CBC exists
        raise RuntimeError("PuLP's CBC solver is unavailable")
    return candidate


def _solve(problem: Any, factory: _SolverFactory) -> str:
    solver = factory.create()
    try:
        status_code = problem.solve(solver)
    finally:
        close = getattr(solver, "close", None)
        if close is not None:
            close()
    return str(factory.pulp.LpStatus[status_code])


def _validate_observations(
    positive: np.ndarray,
    negative: np.ndarray,
    max_degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    positive = np.asarray(positive)
    negative = np.asarray(negative)
    if positive.ndim != 2 or negative.ndim != 2:
        raise ValueError("positive and negative observations must be two-dimensional")
    if positive.shape[1] != negative.shape[1]:
        raise ValueError("positive and negative observations need equal feature counts")
    if max_degree < 1:
        raise ValueError("max_degree must be positive")

    def is_discrete(values: np.ndarray) -> bool:
        return np.issubdtype(values.dtype, np.integer) or np.issubdtype(
            values.dtype, np.bool_
        )

    if not (is_discrete(positive) and is_discrete(negative)):
        raise ValueError("Hammer ILP observations must contain discretized integer values")
    return positive, negative


def _pattern_cover(values: np.ndarray, pattern: Pattern) -> np.ndarray:
    if not pattern:
        return np.ones(len(values), dtype=bool)
    return np.logical_and.reduce(
        [values[:, feature] == value for feature, value in pattern]
    )


def _maximum_pattern_with_factory(
    positive: np.ndarray,
    negative: np.ndarray,
    observation: np.ndarray,
    max_degree: int,
    *,
    robustness: int,
    factory: _SolverFactory,
) -> MaximumPatternSolution | None:
    pulp = factory.pulp
    feature_count = positive.shape[1]
    maximum_degree = min(max_degree, feature_count)
    if robustness < 1:
        raise ValueError("ILP robustness must be positive")
    if robustness > maximum_degree:
        return None

    negative_difference_sets = {
        frozenset(np.flatnonzero(row != observation).tolist())
        for row in negative
    }
    if any(
        len(difference) < robustness
        for difference in negative_difference_sets
    ):
        return None
    # If A is a subset of B, enforcing sum(y[j], j in A) >= d also enforces
    # the constraint for B. Keep only inclusion-minimal mismatch sets.
    negative_differences = []
    for difference in sorted(
        negative_difference_sets,
        key=lambda values: (len(values), tuple(sorted(values))),
    ):
        if not any(existing <= difference for existing in negative_differences):
            negative_differences.append(difference)

    problem = pulp.LpProblem("hammer_maximum_pattern", pulp.LpMaximize)
    selected = [
        problem.add_variable(f"literal_{feature}", cat=pulp.LpBinary)
        for feature in range(feature_count)
    ]
    problem += pulp.lpSum(selected) <= maximum_degree, "degree_bound"
    for row, differences in enumerate(negative_differences):
        problem += (
            pulp.lpSum(selected[feature] for feature in differences) >= robustness,
            f"exclude_negative_{row}",
        )

    positive_differences = Counter(
        tuple(np.flatnonzero(sample != observation).tolist())
        for sample in positive
    )
    weighted_covered = []
    for row, (differences, multiplicity) in enumerate(
        sorted(positive_differences.items(), key=lambda item: item[0])
    ):
        variable = problem.add_variable(
            f"covered_positive_{row}", cat=pulp.LpBinary
        )
        weighted_covered.append(multiplicity * variable)
        if not differences:
            problem += variable == 1, f"cover_anchor_equivalent_{row}"
            continue
        for feature in differences:
            problem += (
                variable <= 1 - selected[feature],
                f"cover_upper_{row}_{feature}",
            )
        problem += (
            variable >= 1 - pulp.lpSum(selected[feature] for feature in differences),
            f"cover_lower_{row}",
        )

    coverage_objective = pulp.lpSum(weighted_covered)
    degree_objective = pulp.lpSum(selected)
    # One extra covered observation is worth more than the largest possible
    # degree penalty. This is an exact lexicographic objective with modest
    # integer coefficients: maximize prevalence first, then minimize literals.
    coverage_weight = feature_count + 1
    problem.setObjective(
        coverage_weight * coverage_objective - degree_objective
    )
    status = _solve(problem, factory)
    if status == "Infeasible":
        return None
    if status != "Optimal":
        raise RuntimeError(
            f"maximum-pattern ILP did not prove an optimum with {factory.name}: {status}"
        )
    pattern = tuple(
        (feature, int(observation[feature]))
        for feature, variable in enumerate(selected)
        if float(variable.value()) > 0.5
    )
    positive_coverage = int(np.sum(_pattern_cover(positive, pattern)))
    linearized_coverage = int(round(float(pulp.value(coverage_objective))))
    if positive_coverage != linearized_coverage:
        raise AssertionError("linearized maximum-pattern coverage is inconsistent")
    if np.any(_pattern_cover(negative, pattern)):
        raise AssertionError("maximum-pattern ILP returned an impure pattern")
    return MaximumPatternSolution(pattern, positive_coverage, factory.name)


def maximum_pattern(
    positive: np.ndarray,
    negative: np.ndarray,
    observation: np.ndarray,
    max_degree: int,
    *,
    solver: str = "auto",
    time_limit_seconds: float | None = None,
    relative_gap: float = 0,
    robustness: int = 1,
    threads: int | None = 1,
) -> MaximumPatternSolution | None:
    """Solve the paper's exact linearized maximum ``omega``-pattern model."""

    positive, negative = _validate_observations(
        positive, negative, max_degree
    )
    observation = np.asarray(observation)
    if observation.shape != (positive.shape[1],):
        raise ValueError("observation must have one value per feature")
    if not np.any(np.all(positive == observation, axis=1)):
        raise ValueError("observation must be a member of the positive class")
    if len(positive) == 0 or len(negative) == 0:
        return None
    factory = _solver_factory(
        solver,
        time_limit_seconds=time_limit_seconds,
        relative_gap=relative_gap,
        threads=threads,
    )
    return _maximum_pattern_with_factory(
        positive,
        negative,
        observation,
        max_degree,
        robustness=robustness,
        factory=factory,
    )


def _minimum_pattern_cover_with_factory(
    positive: np.ndarray,
    patterns: tuple[Pattern, ...],
    *,
    coverage: int,
    factory: _SolverFactory,
) -> tuple[Pattern, ...]:
    if coverage < 1:
        raise ValueError("model coverage must be positive")
    if not patterns:
        return tuple()
    cover_matrix = np.column_stack(
        [_pattern_cover(positive, pattern) for pattern in patterns]
    )
    coverable = np.flatnonzero(np.any(cover_matrix, axis=1))
    if any(int(np.sum(cover_matrix[row])) < coverage for row in coverable):
        raise RuntimeError(
            "candidate patterns cannot satisfy the configured model coverage"
        )

    pulp = factory.pulp
    problem = pulp.LpProblem("hammer_minimum_pattern_model", pulp.LpMinimize)
    selected = [
        problem.add_variable(f"pattern_{index}", cat=pulp.LpBinary)
        for index in range(len(patterns))
    ]
    for row in coverable:
        problem += (
            pulp.lpSum(
                selected[index]
                for index in np.flatnonzero(cover_matrix[row]).tolist()
            )
            >= coverage,
            f"cover_positive_{row}",
        )
    problem.setObjective(pulp.lpSum(selected))
    status = _solve(problem, factory)
    if status != "Optimal":
        raise RuntimeError(
            f"minimum-pattern model did not prove an optimum with {factory.name}: {status}"
        )
    result = tuple(
        pattern
        for pattern, variable in zip(patterns, selected)
        if float(variable.value()) > 0.5
    )
    if coverable.size:
        selected_cover = np.column_stack(
            [_pattern_cover(positive, pattern) for pattern in result]
        )
        if np.any(np.sum(selected_cover[coverable], axis=1) < coverage):
            raise AssertionError("minimum-pattern model lost required coverage")
    return result


def minimum_pattern_cover(
    positive: np.ndarray,
    patterns: tuple[Pattern, ...],
    *,
    solver: str = "auto",
    time_limit_seconds: float | None = None,
    relative_gap: float = 0,
    coverage: int = 1,
    threads: int | None = 1,
) -> tuple[Pattern, ...]:
    """Select the paper's minimum-cardinality cover of candidate patterns."""

    positive = np.asarray(positive)
    if positive.ndim != 2:
        raise ValueError("positive observations must be two-dimensional")
    factory = _solver_factory(
        solver,
        time_limit_seconds=time_limit_seconds,
        relative_gap=relative_gap,
        threads=threads,
    )
    return _minimum_pattern_cover_with_factory(
        positive, tuple(patterns), coverage=coverage, factory=factory
    )


def hammer_maximum_patterns(
    positive: np.ndarray,
    negative: np.ndarray,
    max_degree: int,
    *,
    solver: str = "auto",
    time_limit_seconds: float | None = None,
    relative_gap: float = 0,
    robustness: int = 1,
    model_selection: str = "minimum_cover",
    model_coverage: int = 1,
    max_anchors: int = 0,
    threads: int | None = 1,
    min_positive_coverage: int = 1,
) -> HammerPatternModel:
    """Generate one maximum pattern per distinct anchor, then form a model.

    ``max_anchors=0`` preserves the paper's complete observation-anchored
    construction.  A positive value is a deterministic computational cap and
    is therefore an approximation that should be tuned only on development
    data.
    """

    positive, negative = _validate_observations(
        positive, negative, max_degree
    )
    if model_selection not in {"complete", "minimum_cover"}:
        raise ValueError("ILP model selection must be complete or minimum_cover")
    if max_anchors < 0:
        raise ValueError("max_anchors cannot be negative")
    if min_positive_coverage < 1:
        raise ValueError("min_positive_coverage must be positive")
    factory = _solver_factory(
        solver,
        time_limit_seconds=time_limit_seconds,
        relative_gap=relative_gap,
        threads=threads,
    )
    if len(positive) == 0 or len(negative) == 0:
        return HammerPatternModel(tuple(), tuple(), factory.name, 0, 0, 0)

    anchors = []
    seen = set()
    for observation in positive:
        key = tuple(int(value) for value in observation)
        if key in seen:
            continue
        seen.add(key)
        anchors.append(observation)
        if max_anchors and len(anchors) >= max_anchors:
            break

    candidate_solutions: dict[Pattern, int] = {}
    feasible_anchors = 0
    for observation in anchors:
        solution = _maximum_pattern_with_factory(
            positive,
            negative,
            observation,
            max_degree,
            robustness=robustness,
            factory=factory,
        )
        if solution is None:
            continue
        feasible_anchors += 1
        if solution.positive_coverage >= min_positive_coverage:
            candidate_solutions[solution.pattern] = solution.positive_coverage

    candidates = tuple(
        sorted(candidate_solutions, key=lambda pattern: (len(pattern), pattern))
    )
    if model_selection == "minimum_cover":
        patterns = _minimum_pattern_cover_with_factory(
            positive,
            candidates,
            coverage=model_coverage,
            factory=factory,
        )
    else:
        patterns = candidates
    covered_count = int(
        np.sum(
            np.any(
                np.column_stack(
                    [_pattern_cover(positive, pattern) for pattern in patterns]
                ),
                axis=1,
            )
        )
    ) if patterns else 0
    return HammerPatternModel(
        patterns,
        candidates,
        factory.name,
        len(anchors),
        feasible_anchors,
        covered_count,
    )

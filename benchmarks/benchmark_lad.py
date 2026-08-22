"""Reproducible fit-time benchmark for LADClassifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lad import LADClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=2_000)
    parser.add_argument('--features', type=int, default=32)
    parser.add_argument('--degree', type=int, default=3)
    parser.add_argument('--maxcombs', type=int, default=100)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument(
        '--pattern-method',
        choices=(
            'alexe_hammer',
            'chambon_ppc2_prime',
            'chambon_ppc2_strong',
            'hammer_ilp',
        ),
        default='alexe_hammer',
    )
    parser.add_argument(
        '--ilp-solver',
        choices=('auto', 'gurobi', 'highs', 'cbc'),
        default='auto',
    )
    parser.add_argument(
        '--ilp-model-selection',
        choices=('complete', 'minimum_cover'),
        default='minimum_cover',
    )
    parser.add_argument('--ilp-max-anchors', type=int, default=0)
    parser.add_argument('--ilp-time-limit-seconds', type=int, default=30)
    parser.add_argument(
        '--degree-strategy',
        choices=('fixed', 'gardy_2022'),
        default='fixed',
    )
    parser.add_argument(
        '--exhaustive',
        action='store_true',
        help='search every feature combination instead of bounding the search',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 4 or args.features < args.degree or args.degree < 1:
        raise SystemExit('require samples >= 4 and features >= degree >= 1')
    if args.maxcombs < 1 or args.repeat < 1:
        raise SystemExit('require maxcombs >= 1 and repeat >= 1')

    generator = np.random.default_rng(args.seed)
    features = generator.integers(
        0,
        2,
        size=(args.samples, args.features),
        dtype=np.int8,
    ).astype(bool)
    target = features[:, 0].copy()
    if args.degree >= 2:
        target &= features[:, 1]
    if args.degree >= 3:
        target |= features[:, 2] & ~features[:, 0]

    durations = []
    pattern_counts = []
    for repeat in range(args.repeat):
        classifier = LADClassifier(
            degree=args.degree,
            random=not args.exhaustive,
            maxcombs=args.maxcombs,
            random_state=args.seed + repeat,
            threshold_pct=(
                0.9 if args.pattern_method == 'alexe_hammer' else 1
            ),
            pattern_method=args.pattern_method,
            degree_strategy=args.degree_strategy,
            ilp_solver=args.ilp_solver,
            ilp_model_selection=args.ilp_model_selection,
            ilp_max_anchors=args.ilp_max_anchors,
            ilp_time_limit_seconds=args.ilp_time_limit_seconds,
        )
        started = time.perf_counter()
        classifier.fit(features, target)
        durations.append(time.perf_counter() - started)
        pattern_counts.append(
            sum(len(class_rules[2]) for class_rules in classifier.booleqs_)
        )

    print(json.dumps({
        'samples': args.samples,
        'features': args.features,
        'degree': args.degree,
        'search': 'exhaustive' if args.exhaustive else 'bounded-random',
        'pattern_method': args.pattern_method,
        'degree_strategy': args.degree_strategy,
        'maxcombs': args.maxcombs,
        'ilp_solver': args.ilp_solver,
        'ilp_model_selection': args.ilp_model_selection,
        'ilp_max_anchors': args.ilp_max_anchors,
        'ilp_time_limit_seconds': args.ilp_time_limit_seconds,
        'repeat': args.repeat,
        'seconds': durations,
        'median_seconds': statistics.median(durations),
        'patterns': pattern_counts,
    }, sort_keys=True))


if __name__ == '__main__':
    main()

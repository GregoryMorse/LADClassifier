"""Public and reusable LAD feature-binarization helpers."""

import numpy as np


def binarizer(data, method='minimumdifferentiated', divisions=10,
              mn=-1, mx=-1, splitpoints=None, binarymode=True,
              interval=True):
    """Apply previously computed LAD divisions to one feature."""
    data = np.asarray(data)
    splitpoints = [] if splitpoints is None else splitpoints
    if method == 'equaldivisions':
        dist = (mx - mn) / divisions
        if dist == 0 or divisions == 1:
            return [np.ones(len(data), dtype=np.bool_)]
        if binarymode:
            return [
                ((data >= mn + dist * j) if j != 0 else True)
                & (
                    (data < mn + dist * (j + 1))
                    if j != divisions - 1 and interval else True
                )
                for j in range(
                    0 if interval or divisions == 2 else 1,
                    1 if divisions == 2 else divisions,
                )
            ]
        conditions = np.zeros(len(data), dtype=np.uint32)
        for j in range(
            0 if interval or divisions == 2 else 1,
            1 if divisions == 2 else divisions,
        ):
            conditions[
                ((data >= mn + dist * j) if j != 0 else True)
                & (
                    (data < mn + dist * (j + 1))
                    if j != divisions - 1 and interval else True
                )
            ] = j
        return conditions

    if len(splitpoints) <= 1:
        return [np.ones(len(data), dtype=np.bool_)]
    if binarymode:
        return [
            ((data >= splitpoints[j][0]) if j != 0 else True)
            & (
                (data < splitpoints[j][1])
                if j != len(splitpoints) - 1 and interval else True
            )
            for j in range(
                0 if interval or divisions == 2 else 1,
                1 if len(splitpoints) == 2 else len(splitpoints),
            )
        ]
    conditions = np.zeros(len(data), dtype=np.uint32)
    for j in range(
        0 if interval or divisions == 2 else 1,
        1 if len(splitpoints) == 2 else len(splitpoints),
    ):
        conditions[
            ((data >= splitpoints[j][0]) if j != 0 else True)
            & (
                (data < splitpoints[j][1])
                if j != len(splitpoints) - 1 and interval else True
            )
        ] = j
    return conditions


def binarize(data, name, y, method='minimumdifferentiated', divisions=10,
             binarymode=True, interval=True):
    """Compute LAD divisions, transformed values, and readable names."""
    if method == 'equaldivisions':
        mn, mx = min(data), max(data)
        dist = (mx - mn) / divisions
        binvals = {
            'method': method,
            'divisions': divisions,
            'mn': mn,
            'mx': mx,
            'binarymode': binarymode,
            'interval': interval,
        }
        featnames = [
            name
            + ('>=' + str(round(mn + dist * j, 2)) if j != 0 else '')
            + (
                '<' + str(round(mn + dist * (j + 1), 2))
                if j != divisions - 1 else ''
            )
            for j in range(divisions)
        ]
    elif method == 'equaldistribution':
        size, sorted_data = len(data), np.sort(data)
        divs = [
            (
                sorted_data[int(size * j / divisions)],
                sorted_data[
                    int(size * (j + 1) / divisions)
                    - (1 if j == divisions - 1 else 0)
                ],
            )
            for j in range(divisions)
        ]
        binvals = {
            'method': method,
            'divisions': len(divs),
            'splitpoints': divs,
            'binarymode': binarymode,
            'interval': interval,
        }
        featnames = [
            name
            + ('>=' + str(round(divs[j][0], 2)) if j != 0 else '')
            + (
                '<' + str(round(divs[j][1], 2))
                if j != len(divs) - 1 else ''
            )
            for j in range(len(divs))
        ]
    elif method == 'minimumdifferentiated':
        sorted_values = list(zip(data, y))
        sorted_values.sort()
        divs, featnames = [], []
        lastval, index = sorted_values[0], 1
        while index <= len(sorted_values):
            nextval = (
                sorted_values[index]
                if index != len(sorted_values) else lastval
            )
            while index < len(sorted_values) - 1:
                if sorted_values[index][0] != sorted_values[index + 1][0]:
                    break
                if sorted_values[index][1] != sorted_values[index + 1][1]:
                    nextval = (nextval[0], None)
                index += 1
            if (
                index == len(sorted_values) and len(divs) != 0
                or lastval[1] is None
                or nextval[1] != lastval[1]
                and nextval[0] != lastval[0]
            ):
                divs.append((lastval[0], nextval[0]))
                featnames.append(
                    name
                    + (
                        '>=' + str(round(lastval[0], 2))
                        if len(divs) != 1 else ''
                    )
                    + (
                        '<' + str(round(nextval[0], 2))
                        if index != len(sorted_values) else ''
                    )
                )
                lastval = nextval
            index += 1
        binvals = {
            'method': method,
            'divisions': len(divs),
            'splitpoints': divs,
            'binarymode': binarymode,
            'interval': interval,
        }
    else:
        raise ValueError('Unknown binarization method: ' + str(method))

    if binvals['binarymode']:
        featnames = [['!' + feature, feature] for feature in featnames]
    return binvals, binarizer(data, **binvals), featnames


def binarizeall(X, y, feature_names=None, binarizer_params=None):
    """Binarize every feature while retaining reusable parameters."""
    feature_names = (
        ['Feature' + str(index + 1) for index in range(X.shape[1])]
        if feature_names is None else feature_names
    )
    conditions, names, values, bounds = [], [], [], []
    for index in range(X.shape[1]):
        if X[:, index].dtype.type is bool or X[:, index].dtype.type is np.bool_:
            conditions.append(X[:, index])
            names.append(['!' + feature_names[index], feature_names[index]])
            values.append(None)
            bounds.append(2)
            continue
        params = (
            {}
            if binarizer_params is None
            else (
                binarizer_params[index]
                if type(binarizer_params) is list else binarizer_params
            )
        )
        binvals, converted, converted_names = binarize(
            X[:, index], feature_names[index], y, **params
        )
        values.append(binvals)
        if binvals['binarymode']:
            conditions.extend(converted)
            names.extend(converted_names)
            bounds.extend([2] * len(converted))
        else:
            conditions.append(converted)
            names.append(converted_names)
            bounds.append(binvals['divisions'])
    return np.array(conditions).transpose(), names, values, bounds


def postbinarize(X, binarizer_values):
    """Apply fitted public binarizer parameters to new samples."""
    conditions = []
    for index in range(X.shape[1]):
        if binarizer_values[index] is None:
            conditions.append(X[:, index])
        elif binarizer_values[index]['binarymode']:
            conditions.extend(
                binarizer(
                    X[:, index], **binarizer_values[index]
                )
            )
        else:
            conditions.append(
                binarizer(
                    X[:, index], **binarizer_values[index]
                )
            )
    return np.array(conditions).transpose()


def binarizecompare(X, feature_names, featcomp,
                    operations=('lt', 'eq', 'gt')):
    """Create Boolean comparison features and mutual-exclusion groups."""
    feature_names = (
        ['Feature' + str(index + 1) for index in range(X.shape[1])]
        if feature_names is None else feature_names
    )
    comparisons = {
        'lt': (np.less, '<'),
        'eq': (np.equal, '=='),
        'gt': (np.greater, '>'),
        'lte': (np.less_equal, '<='),
        'neq': (np.not_equal, '!='),
        'gte': (np.greater_equal, '>='),
    }
    mutual_groups = [
        {'lt', 'eq', 'gt'}, {'lt', 'gte'}, {'eq', 'neq'}, {'gt', 'lte'}
    ]
    conditions, names, mutex = [], [], []
    operation_set = set(operations)
    mutex_offsets = []
    for group in mutual_groups:
        overlap = operation_set & group
        if len(overlap) >= 2:
            mutex_offsets.append([operations.index(value) for value in overlap])
    for left, right in featcomp:
        for operation in operations:
            function, symbol = comparisons[operation]
            conditions.append(function(X[:, left], X[:, right]))
            names.append(
                feature_names[left] + symbol + feature_names[right]
            )
        start = len(conditions) - len(operations)
        mutex.extend(
            [[start + offset for offset in group] for group in mutex_offsets]
        )
    return conditions, names, mutex

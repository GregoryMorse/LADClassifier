Quick start
===========

Install the project in editable mode while developing::

    python -m pip install -e .

Fit and inspect a classifier::

    import numpy as np
    from lad import LADClassifier

    X = np.array([
        [0.0, 0.1],
        [0.2, 0.1],
        [0.8, 0.9],
        [1.0, 0.8],
    ])
    y = np.array([0, 0, 1, 1])

    classifier = LADClassifier(
        degree=2,
        maxcombs=100,
        random_state=0,
    ).fit(X, y)

    predictions = classifier.predict(X)
    rules = classifier.format_booleqs()

The classifier discretizes numeric features during ``fit`` and stores the
learned transform in ``discretizer_``. Pass ``feature_names`` when readable
rule output matters.

Public binarizer
----------------

The same reusable transform helpers are available independently::

    from lad import binarizeall, postbinarize

    X_binary, names, parameters, bounds = binarizeall(
        X,
        y,
        feature_names=['momentum', 'volume'],
        binarizer_params={'method': 'equaldivisions', 'divisions': 4},
    )
    later_binary = postbinarize(X, parameters)

The functions live in ``lad.binarization``. Aliases on ``LADClassifier`` remain
available for compatibility with earlier releases.

"""
This module contains the LAD classifier implementation
"""
import numba
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, MultiOutputMixin
from sklearn.utils.validation import check_is_fitted #check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import check_array
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from scipy.sparse import issparse

import matplotlib.pyplot as plt
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
"""
import cProfile
cProfile.run('import lad; lad.test_lad()', 'ladstats', sort='tottime')
import pstats
p = pstats.Stats('ladstats')
p.strip_dirs().sort_stats('tottime').print_stats()
"""
class LADClassifier(ClassifierMixin, MultiOutputMixin, BaseEstimator):
    """ Logical Analysis of Data Classifier which includes a binarizer.

    For more information regarding how to use the LAD classifier, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    degree : int, default=4
        Specifies the maximum degree of features to use for pattern finding.
    random : bool, default=True
        Specifies to use a random search of features, otherwise exhaustive
        combinations are tried.  If the degree is greater than or equal to
        the number of features, a random search will not be used regardless
        of this parameter.
    maxcombs : int, default=2000
        For a random search, the maximum number of combinations to try before
        recomputing the rows remaining and checking if convergence is occurring.
    threshold_pct : float, default=0.9
        The minimum precision of a pattern for it to be considered.
    minmatch_pct : float, default=0.001
        The minimum percentage of all samples which must be found covered by a
        pattern for it to be considered.
    feature_names : list, default=None
        The list of feature names corresponding to the features which will be
        used in the fit function call so the binarizer can generate meaningful
        pattern names or Boolean features can have their negation indicated.
        This is optional.
    binarizer_params : list, default=None
        The parameters passed to the binarizer specifying its method and
        division strategy.  The binarizer methods can also be used outside the
        model prior to classification.
    penalty_value : int, default=None
        Optional penalty value which will penalize precision based on the number
        of true values found with exponential decay.  The value must be greater
        than 1 and the higher, the more exponential decay will occur.  The
        default value has no penalty.
    random_state : int, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance
        used by np.random.

    Attributes
    ----------
    n_outputs_ : int
        The number of output columns.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    n_classes_ : int
        The number of unique classes seen at :meth:`fit`.
    self.featnames_ : list
        The feature names computed for each feature if feature_names was provided.
    self.binarizer_values_ : list
        The binarizer parameters computed for each feature.
    self.bounds_ : list
        The shape of each dimension of the features after binarization.
    """
    def __init__(self, degree=4, random=True, maxcombs=100, threshold_pct=1,
                 minmatch_pct=0.001, feature_names=None, binarizer_params=None,
                 penalty_value=None, random_state=None):
        self.degree = degree
        self.random = random
        self.maxcombs = maxcombs
        self.threshold_pct = threshold_pct
        self.minmatch_pct = minmatch_pct
        self.feature_names = feature_names
        self.binarizer_params = binarizer_params
        self.penalty_value = penalty_value #10000000
        self.random_state = random_state
        #self.mutual_exclusions = mutual_exclusions
        self._estimator_type = 'classifier' #needed for stratified k-folds in GridSearchCV
    #def _get_tags(self): return {'poor_score':True,'multioutput':True}
    """Binarizer for LAD classifier.
    This routine allows for convienent binarization of a single feature prior
    to LAD classification preventing binarization from influencing
    the algorithm.  It is recommended to use binarize instead of this routine,
    unless the parameters such as the minimum and maximum or cut points are already known.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        The training input samples.
    method : string, default='minimumdifferentiated'
        Either 'minimumdifferentiated', 'equaldivisons'
        or 'equaldistribution'.
    divisions : int, default=10
        The number of divisions for 'equaldivisions' method.
        It should be greater than or equal to 2.
    mn : float or int, default=-1
        The minimum for the division boundary when using 'equaldivisions'.
    mx : float or int, default=-1
        The maximum for the division boundary when using 'equaldivisions'.
    splitpoints : list, default=[]
        The list of cut points as 2-tuples for start of interval and end of interval,
        and ideally should be continuous without any gaps.  The first and last interval,
        are computed in a one sided manner.  For 'equaldistribution' and
        'minimumdifferentiated' methods only.  If only 2 cut points, then a single
        binary feature will be used as it is sufficient for this special case.
    binarymode : bool, default=True
        This is True if opting for binary features only, otherwise if False, then the
        features will be discretized as numbers from 0 to the number of cut points minus 1
        except in the case of only 2 cut points where there is no difference for this parameter.
    interval : bool, default=True
        This is True if non-overlapping mutually exclusive intervals should be used,
        otherwise False for levels which use only monotonically increasing inequalities.
        In the case of only 2 cut points, this parameter makes no difference.

    Returns
    -------
    conditions : list
        Returns list of binarized sub-features for the provided feature which can be
        converted to a numpy array and transposed if they will be used in the LADClassifier.
    """
    def binarizer(data, method='minimumdifferentiated', divisions=10,
                  mn=-1, mx=-1, splitpoints=[], binarymode=True, interval=True):
        #2 cut points uses a special reduction to binary because of mutual exclusivity of the values in each division
        if method == 'equaldivisions':
            dist = (mx - mn) / divisions
            if dist == 0 or divisions == 1:
                cond = [np.ones(len(data), dtype=np.bool_)]
            elif binarymode:
                cond = [((data >= mn + dist * j) if j != 0 else True) &
                     ((data < mn + dist * (j+1)) if j != divisions-1 and interval else True) for j in range(0 if interval or divisions==2 else 1, 1 if divisions==2 else divisions)]
            else:
                cond = np.zeros(len(data), dtype=np.uint32)
                for j in range(0 if interval or divisions==2 else 1, 1 if divisions==2 else divisions):
                    cond[((data >= mn + dist * j) if j != 0 else True) &
                         ((data < mn + dist * (j+1)) if j != divisions-1 and interval else True)] = j
                    #cond[~(((data >= mn + dist * j) if j != 0 else True) &
                    #     ((data < mn + dist * (j+1)) if j != divisions-1 else True))] = j + divisions
        else:
            if len(splitpoints) <= 1:
                cond = [np.ones(len(data), dtype=np.bool_)]
            elif binarymode:
                cond = [((data >= splitpoints[j][0]) if j != 0 else True) &
                         ((data < splitpoints[j][1]) if j != len(splitpoints)-1 and interval else True) for j in range(0 if interval or divisions==2 else 1, 1 if len(splitpoints)==2 else len(splitpoints))]
            else:
                cond = np.zeros(len(data), dtype=np.uint32)
                for j in range(0 if interval or divisions==2 else 1, 1 if len(splitpoints)==2 else len(splitpoints)):
                    cond[((data >= splitpoints[j][0]) if j != 0 else True) &
                         ((data < splitpoints[j][1]) if j != len(splitpoints)-1 and interval else True)] = j
                    #cond[~(((data >= splitpoints[j][0]) if j != 0 else True) &
                    #     ((data < splitpoints[j][1]) if j != len(splitpoints)-1 else True))] = j + len(splitpoints)
        return cond
    #equal divisions, equal distribution, minimum differentiated ranges in output
    #['equaldivisions', 'equaldistribution', 'minimumdifferentiated']

    """Automatic Binarizer for LAD classifier.
    This routine allows for convienent binarization of data prior
    to LAD classification preventing binarization from influencing
    the algorithm.  This routine guides the process by computing the
    method parameters before binarizing with the binarizer function,
    as well as providing readable feature names.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        The training input samples.
    name : string
        The feature name.
    y : array-like, shape (n_samples,)
        The target values. An array of int.
    method : string, default='minimumdifferentiated'
        Either 'minimumdifferentiated', 'equaldivisons'
        or 'equaldistribution'.
    divisions : int, default=10
        The number of divisions for 'equaldivisions' and 'equaldistribution' methods.
        It should be greater than or equal to 2.
    binarymode : bool, default=True
        This is True if opting for binary features only, otherwise if False, then the
        features will be discretized as numbers from 0 to the number of cut points minus 1
        except in the case of only 2 cut points where there is no difference for this parameter.
    interval : bool, default=True
        This is True if non-overlapping mutually exclusive intervals should be used,
        otherwise False for levels which use only monotonically increasing inequalities.
        In the case of only 2 cut points, this parameter makes no difference.

    Returns
    -------
    (binarizer_values, conditions, feature_names) : (list, list, list)
        Returns 3 lists, the first with the binarizer parameter for the given feature.
        The second is the binarized features which can be converted to a numpy array and transposed
        if they will be used in the LADClassifier.
        The third is the readable list of feature names corresponding to each generated feature,
        based on the name provided.
    """
    def binarize(data, name, y, method='minimumdifferentiated', divisions=10, binarymode=True, interval=True):
        if method == 'equaldivisions':
            mn, mx = min(data), max(data)
            dist = (mx - mn) / divisions
            binvals = {'method':method, 'divisions':divisions, 'mn':mn, 'mx':mx, 'binarymode':binarymode}
            featnames = [name + ('>=' + str(round(mn + dist * j, 2)) if j != 0 else '') +
                     ('<' + str(round(mn + dist * (j+1), 2)) if j != divisions-1 else '')
                     for j in range(divisions)]
        elif method == 'equaldistribution': #need to handle splits on equivalence groups, right now redundant or duplicate values possible
            sz, sorted = len(data), np.sort(data)
            divs = [(sorted[int(sz * j / divisions)], sorted[int(sz * (j+1) / divisions)-(1 if j == divisions-1 else 0)])
                    for j in range(divisions)]
            binvals = {'method':method, 'divisions': len(divs), 'splitpoints':divs, 'binarymode':binarymode}
            featnames = [name + ('>=' + str(round(divs[j][0], 2)) if j != 0 else '') +
                     ('<' + str(round(divs[j][1], 2)) if j != len(divs)-1 else '')
                     for j in range(len(divs))]
        elif method == 'minimumdifferentiated':
            sorted = list(zip(data, y))
            sorted.sort()
            divs, featnames = [], []
            #cannot ignore duplicate values or could wrongly collapse to single division
            lastval, x = sorted[0], 1
            while x <= len(sorted):
                nextval = sorted[x] if x != len(sorted) else lastval
                while x < len(sorted) - 1:
                    if sorted[x][0] != sorted[x+1][0]: break
                    if sorted[x][1] != sorted[x+1][1]: nextval = (nextval[0], None)
                    x += 1
                if x == len(sorted) and len(divs) != 0 or lastval[1] is None or nextval[1] != lastval[1] and nextval[0] != lastval[0]:
                    divs.append((lastval[0], nextval[0]))
                    #condarr.append((data >= lastval[0]) & (data <= sorted[x][0]))
                    featnames.append(name + (('>=' + str(round(lastval[0], 2))) if len(divs) != 1 else '') +
                                     ('<' + str(round(nextval[0], 2)) if x != len(sorted) else ''))
                    lastval = nextval
                x += 1
            binvals = {'method':method, 'divisions':len(divs), 'splitpoints':divs, 'binarymode':binarymode}
        if binvals['binarymode']: featnames = [['!' + x, x] for x in featnames] #for True/False values that were binarized this way, False will come first then True due to 0/1 ordering
        #else: featnames.extend(['!' + x for x in featnames])
        return binvals, LADClassifier.binarizer(data, **binvals), featnames
    """All data binarizer for LAD classifier.
    This routine allows for convienent binarization of all data prior
    to LAD classification preventing binarization from influencing
    the algorithm.  This routine guides the process by computing the
    method parameters before binarizing with the binarizer function.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values. An array of int.
    feature_names : list of string, default=None
        The feature names for each feature in X.  If None is passed,
        it will automatically label features 1...N as 'Feature1', 'Feature2', ..., 'FeatureN'.
    binarizer_params : dict, default=None
        The parameters which will be passed to binarize, containing any parameters from:
        'method', 'divisions', 'binarymode', 'interval'.

    Returns
    -------
    (X_new, feature_names, binarizer_values, bounds) : (array, list, list, list)
        Returns an array and 3 lists, the first with the binarized features as a numpy array
        and already transposed for use in the LADClassifier.  The second is the readable list
        of feature names corresponding to each generated feature, based on the names provided.
        into the binarizer parameter for the given feature.  The third is the binarizer
        parameters for the given features.  The last is the shape of all of the generated features.
    """
    def binarizeall(X, y, feature_names=None, binarizer_params=None):
        feature_names = ['Feature' + str(x+1) for x in range(X.shape[1])] if feature_names is None else feature_names
        condarr, featnames, binvals, bounds = [], [], [], []
        for i in range(X.shape[1]):
            if X[:,i].dtype.type is bool or X[:,i].dtype.type is np.bool_:
                condarr.append(X[:,i]), featnames.append(['!' + feature_names[i], feature_names[i]]), binvals.append(None), bounds.append(2)
            else:
                binparams = {} if binarizer_params is None else (binarizer_params[i] if type(binarizer_params) is list else binarizer_params)
                vals, conds, feats = LADClassifier.binarize(X[:,i], feature_names[i], y, **binparams)
                binvals.append(vals)
                if vals['binarymode']:
                    condarr.extend(conds), featnames.extend(feats), bounds.extend([2] * len(conds))
                else:
                    condarr.append(conds), featnames.append(feats), bounds.append(vals['divisions']) #* 2
                #mutex.append(np.arange(len(condarr)-len(conds), len(condarr)))
        return np.array(condarr).transpose(), featnames, binvals, bounds
    """All data post-binarizer for LAD classifier.
    This routine allows for convienent post-binarization of data
    which has not yet been binarized, but previous representative data
    has been binarized and its parameters will be used to binarize this data.
    Typically useful when binarizing the training data, then post-binarizing the test data.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The training input samples.
    binarizer_values : dict, default=None
        The parameters which will be passed to binarizer, containing any parameters from:
        'method', 'divisions', 'mn', 'mx', 'splitpoints', 'binarymode', 'interval'.

    Returns
    -------
    X_new : array
        Returns an array with the post-binarized features as a numpy array
        and already transposed for use in the LADClassifier.
    """
    def postbinarize(X, binarizer_values):
        condarr = []
        for i in range(X.shape[1]):
            if binarizer_values[i] is None:
                condarr.append(X[:,i])
            elif binarizer_values[i]['binarymode']:
                condarr.extend(LADClassifier.binarizer(X[:,i], **binarizer_values[i]))
            else:
                condarr.append(LADClassifier.binarizer(X[:,i], **binarizer_values[i]))
        return np.array(condarr).transpose()
    #['lt', 'eq', 'gt', 'lte', 'neq', 'gte']
    def binarizecompare(X, feature_names, featcomp, operations=['lt', 'eq', 'gt']):
        feature_names = ['Feature' + str(x+1) for x in range(X.shape[1])] if feature_names is None else feature_names
        compdict = {'lt':(np.less, '<'), 'eq':(np.equal, '=='), 'gt':(np.greater, '>'),
                    'lte':(np.less_equal, '<='), 'neq':(np.not_equal, '!='), 'gte':(np.greater_equal, '>=')}
        mutgroups = [{'lt', 'eq', 'gt'}, {'lt', 'gte'}, {'eq', 'neq'}, {'gt', 'lte'}] #{'lt', 'gt'}, {'lt', 'eq'}, {'gt', 'eq'}
        condarr, featnames, mutex = [], [], []
        muts, so = [], set(operations)
        for i in mutgroups:
            s = so & i
            if len(s) >= 2: muts.append([operations.index(x) for x in s])
        for i in featcomp:
            for c in operations:
                condarr.append(compdict[c][0](X[:,i[0]], X[:,i[1]]))
                featnames.append(featnames[i[0]] + compdict[c][1] + featnames[i[1]])
            mutex.extend([[len(condarr) - len(c) + x for x in y] for y in muts])
        return condarr, featnames, mutex
    """Paper test with simple and general but inefficient code as a proof of concept.
    It shows that the paper example works with a series of assertions for its 3 algorithms.
    Paper: https://www.sciencedirect.com/science/article/pii/S0166218X05003161    
    """
    def _testpaper():
        
        #paper example
        def calc_PI_V0(X, n=None):
            if n is None: n = range(len(X.shape))
            H = np.array(X)
            def sum_func(a):
                for j in range(len(a) - 1-1, -1, -1):
                    a[j] = a[j] + a[j+1]
                return a
            for i in n:
                #Ni, Nk = tuple([slice(None, None, None)] * i), tuple([slice(None, None, None)] * (len(H.shape)-1 - i-1))
                #for j in range(H.shape[i]-1-1, -1, -1): H[Ni + np.s_[j,] + Nk] += H[Ni + np.s_[j+1,] + Nk]
                H = np.apply_along_axis(sum_func, i, H)
            return H
        def extended_gray_code(K):
            codes, istar, V, T = [], 0, np.array(K), np.repeat(-1, len(K))
            while True:
                codes.append((np.array(V), istar, np.array(T)))
                VT = V + T
                S = np.nonzero((VT >= 0) & (VT <= K))[0]
                if len(S) == 0: break
                istar = np.max(S)
                V[istar] = V[istar] + T[istar]
                T[istar+1:] = -T[istar+1:]
                #print(V)
            return codes
        def calc_PI_VI(Vistar, Vprimeistar, istar, Tistar, PiVi):
            PiVprime = np.array(PiVi) #np.zeros(X.shape, dtype=np.uint32)
            def diff_func(a):
                a[Vistar+1:] -= a[Vistar]
                a[:Vistar+1] += a[Vprimeistar]
                return a
            def diff_funcneg(a):
                a[:Vistar] -= a[Vistar]
                a[Vistar:] += a[Vprimeistar]
                return a
            return np.apply_along_axis(diff_func if Tistar == 1 else diff_funcneg, istar, PiVprime)
        #paper has columns as axis 0 and rows as axis 1
        Xpaperpts = [tuple(np.flip(x)) for x in [(0, 4), (1, 2), (1, 3), (2, 2), (2, 3), (3, 0)]]
        Xpapercalc = np.zeros(np.flip((4, 5), 0), dtype=np.uint32)
        Xpapercalc[tuple(np.array(Xpaperpts).T)] += 1
        Xpaper = np.flip(np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 1]]), 0) #paper has axis 0 flipped
        assert(np.array_equal(Xpaper, Xpapercalc))
        #Xpaperneg = np.subtract(1, Xpaper)
        M1paper = np.flip(np.array([[1, 0, 0, 0], [2, 2, 1, 0], [2, 2, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1]]), 0)
        #paper has error showing last row with [6, 4, 3, 1] unlike its first example which was correct
        PiV0paper = np.flip(np.array([[1, 0, 0, 0], [3, 2, 1, 0], [5, 4, 2, 0], [5, 4, 2, 0], [6, 5, 3, 1]]), 0)
        PiV1paper = np.flip(np.array([[3, 2, 1, 0], [2, 2, 1, 0], [4, 4, 2, 0], [4, 4, 2, 0], [5, 5, 3, 1]]), 0)
        #row major order vs column major order (paper)
        #0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19
        #0  7  8  15 16 17 14 9  6  1  2  5  10 13 18 19 12 11 4  3
        PiVipaper = [PiV0paper,
          np.flip(np.array([[1, 0, 0, 0], [3, 2, 1, 1], [5, 4, 2, 2], [5, 4, 2, 2], [5, 4, 2, 3]]), 0),
          np.flip(np.array([[1, 0, 0, 0], [2, 1, 2, 2], [3, 2, 4, 4], [3, 2, 4, 4], [3, 2, 4, 5]]), 0),
          np.flip(np.array([[1, 1, 1, 1], [1, 2, 3, 3], [1, 3, 5, 5], [1, 3, 5, 5], [1, 3, 5, 6]]), 0),
          
          np.flip(np.array([[1, 2, 3, 3], [0, 1, 2, 2], [0, 2, 4, 4], [0, 2, 4, 4], [0, 2, 4, 5]]), 0),
          np.flip(np.array([[2, 1, 2, 2], [1, 1, 2, 2], [2, 2, 4, 4], [2, 2, 4, 4], [2, 2, 4, 5]]), 0),
          np.flip(np.array([[3, 2, 1, 1], [2, 2, 1, 1], [4, 4, 2, 2], [4, 4, 2, 2], [4, 4, 2, 3]]), 0),
          PiV1paper,
          
          np.flip(np.array([[5, 4, 2, 0], [4, 4, 2, 0], [2, 2, 1, 0], [2, 2, 1, 0], [3, 3, 2, 1]]), 0),
          np.flip(np.array([[5, 4, 2, 2], [4, 4, 2, 2], [2, 2, 1, 1], [2, 2, 1, 1], [2, 2, 1, 2]]), 0),
          np.flip(np.array([[3, 2, 4, 4], [2, 2, 4, 4], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 3]]), 0),
          np.flip(np.array([[1, 3, 5, 5], [0, 2, 4, 4], [0, 1, 2, 2], [0, 1, 2, 2], [0, 1, 2, 3]]), 0),
          
          np.flip(np.array([[1, 3, 5, 5], [0, 2, 4, 4], [0, 1, 2, 2], [0, 0, 0, 0], [0, 0, 0, 1]]), 0),
          np.flip(np.array([[3, 2, 4, 4], [2, 2, 4, 4], [1, 1, 2, 2], [0, 0, 0, 0], [0, 0, 0, 1]]), 0),
          np.flip(np.array([[5, 4, 2, 2], [4, 4, 2, 2], [2, 2, 1, 1], [0, 0, 0, 0], [0, 0, 0, 1]]), 0),
          np.flip(np.array([[5, 4, 2, 0], [4, 4, 2, 0], [2, 2, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1]]), 0),

          np.flip(np.array([[6, 5, 3, 1], [5, 5, 3, 1], [3, 3, 2, 1], [1, 1, 1, 1], [1, 1, 1, 1]]), 0),
          np.flip(np.array([[5, 4, 2, 3], [4, 4, 2, 3], [2, 2, 1, 2], [0, 0, 0, 1], [0, 0, 0, 1]]), 0),
          np.flip(np.array([[3, 2, 4, 5], [2, 2, 4, 5], [1, 1, 2, 3], [0, 0, 0, 1], [0, 0, 0, 1]]), 0),
          np.flip(np.array([[1, 3, 5, 6], [0, 2, 4, 5], [0, 1, 2, 3], [0, 0, 0, 1], [0, 0, 0, 1]]), 0)
          ]
        PiVipaperC = [PiVipaper[x] for x in [0, 7, 8, 15, 16, 17, 14, 9, 6, 1, 2, 5, 10, 13, 18, 19, 12, 11, 4, 3]]
        assert(np.array_equal(calc_PI_V0(Xpaper, [1]), M1paper))
        assert(np.array_equal(calc_PI_V0(Xpaper), PiV0paper))
        assert(np.array_equal(calc_PI_VI(4, 3, 0, -1, calc_PI_V0(Xpaper)), PiV1paper)) #extended_gray_code(np.subtract(Xpaper.shape, 1))[:2]
        gcodes = extended_gray_code(np.subtract(Xpaper.shape, 1))
        gcodes = [(np.flip(x[0]), 1-x[1], np.flip(x[2])) for x in extended_gray_code(np.flip(np.subtract(Xpaper.shape, 1), 0))]
        PiV0 = calc_PI_V0(Xpaper)
        PiVi = [PiV0]
        for i in range(1, len(gcodes)):
            istar = gcodes[i][1]
            PiVi.append(calc_PI_VI(gcodes[i-1][0][istar], gcodes[i][0][istar], istar, gcodes[i][2][istar], PiVi[-1]))
            #print(PiVi[-1])
        assert(np.all([np.array_equal(x, PiVi[i]) for i, x in enumerate(PiVipaperC)]))
    """Plots learning examples of algorithm and the Iris dataset.

    Parameters
    ----------
    curdir : string
        The output directory for PNG and SVG images generated.
    """
    def _make_learn_plots(curdir):
        import matplotlib.pyplot as plt
        import os
        plt.clf()
        plt.subplot(111)
        plt.xlim(0, 4)
        plt.ylim(0, 5)
        plt.gca().set_xticks(np.arange(5))
        plt.annotate('', (4, 0), (0, 0), arrowprops=dict(arrowstyle="-|>", shrinkA=0, shrinkB=0, color='red'))
        plt.annotate('', (0, 0), (0, 1), arrowprops=dict(arrowstyle="-|>", shrinkA=0, shrinkB=0, color='red'))
        plt.annotate('', (0, 1), (4, 1), arrowprops=dict(arrowstyle="-|>", shrinkA=0, shrinkB=0, color='red'))
        plt.annotate('', (4, 1), (4, 2), arrowprops=dict(arrowstyle="-|>", shrinkA=0, shrinkB=0, color='red'))
        plt.annotate('', (4, 2), (0, 2), arrowprops=dict(arrowstyle="-|>", shrinkA=0, shrinkB=0, color='red'))
        plt.annotate('', (0, 2), (0, 3), arrowprops=dict(arrowstyle="-|>", shrinkA=0, shrinkB=0, color='red'))
        plt.annotate('', (0, 3), (4, 3), arrowprops=dict(arrowstyle="-|>", shrinkA=0, shrinkB=0, color='red'))
        plt.annotate('', (4, 3), (4, 4), arrowprops=dict(arrowstyle="-|>", shrinkA=0, shrinkB=0, color='red'))
        plt.annotate('', (4, 4), (0, 4), arrowprops=dict(arrowstyle="-|>", shrinkA=0, shrinkB=0, color='red'))
        plt.annotate('', (0, 4), (0, 5), arrowprops=dict(arrowstyle="-|>", shrinkA=0, shrinkB=0, color='red'))
        plt.annotate('', (0, 5), (4, 5), arrowprops=dict(arrowstyle="-|>", shrinkA=0, shrinkB=0, color='red'))
        plt.grid(True)
        plt.title('Demonstration of extended Gray code iteration strategy')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.tight_layout()
        plt.savefig(os.path.join(curdir, 'data', 'gcodes.svg'), format='svg')#, bbox_inches = extent, pad_inches = 0)
        plt.savefig(os.path.join(curdir, 'data', 'gcodes.png'), format='png')#, bbox_inches = extent, pad_inches = 0)       
        plt.gcf()
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import confusion_matrix
        from sklearn import datasets
        bec = LADClassifier(maxcombs=500)
        iris = datasets.load_iris()
        bec.feature_names = iris.feature_names
        #condarr, featnames, binvals, mutex = bec.binarizeall(iris.data, iris.target, iris.feature_names)
        params = [{'degree':[4], 'threshold_pct':[1]}]
        clf = GridSearchCV(bec, params, cv=5, iid=True, error_score='raise', verbose=100)
        with np.printoptions(precision=2, suppress=True):
            mdl = clf.fit(iris.data, iris.target)
            o = clf.predict(iris.data) #it is already refitted with the best model
            cm = confusion_matrix(iris.target, o)
            print(clf.best_score_, cm, clf.best_estimator_.format_booleqs())
            plot_confusion_matrix(iris.target, o, classes=iris.target_names, cmap=plt.cm.Blues, normalize=False)
    def fit(self, X, y):
        """LAD classifer implementation of a fitting function.
        It first binarizes the data if necessary, then finds patterns
        until full sample coverage or convergence is determined not possible.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        #binarization comes first
        #X, y = check_X_y(X, y)
        #self.classes_ = unique_labels(y)
        #print(X.shape[1])
        X = check_array(X, dtype=[np.float_, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64, np.bool_], accept_sparse=False) #"csc")
        y = check_array(y, ensure_2d=False, dtype=None)
        if issparse(X):
            X.sort_indices()
            if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
                raise ValueError("No support for np.int64 index based "
                                 "sparse matrices")
        if len(y) != X.shape[0]:
            raise ValueError("Number of labels=%d "
                             "does not match number of samples=%d"
                             % (len(y), X.shape[0]))
        check_classification_targets(y)
        condarr, self.featnames_, self.binarizer_values_, self.bounds_ = LADClassifier.binarizeall(X, y, self.feature_names, self.binarizer_params)
        #self.mutex_ = self.mutual_exclusions[:] + self.mutex_
        #self.mutex_ = {y:x for x in self.mutex_ for y in x}
        self.outtype_ = y.dtype
        if y.ndim == 1:
            self.n_outputs_ = 1
            self.classes_ = np.unique(y)
            self.n_classes_ = self.classes_.shape[0]
            self.booleqs_ = self._fit(condarr, y, self.classes_)
        else:
            self.n_outputs_ = y.shape[1]
            self.classes_ = []
            self.n_classes_ = []
            self.booleqs_ = []
            for k in range(self.n_outputs_):
                classes_k = np.unique(y[:, k])
                self.classes_.append(classes_k)
                self.n_classes_.append(classes_k.shape[0])
                self.booleqs_.append({}) #(prefer positive, positive patterns, negative patterns)
            for k in range(self.n_outputs_):
                self.booleqs_[k] = self._fit(condarr, y[:, k], self.classes_[k])
        return self
    def _fit(self, X, y, classes): #in DNF, if want CNF, can negate X and y per DeMorgan's law?
        #print(X.shape[1])
        vals, origsz = [], len(X)
        for k in classes:
            vals.append(X[y == k,:])
        minmatch = int(len(y) * self.minmatch_pct)
        import itertools
        def prec_penalty(precision, featpct): #reduce precision based on number of features
            a = self.penalty_value
            return (a ** (1-featpct) - 1) / (a-1) * precision #exponential between 0 and 1: (a^x-1)/(a-1) where higher a has higher decay                
        #@numba.njit
        def sum_func(a):
            for j in range(len(a) - 1-1, -1, -1):
                a[j] = a[j] + a[j+1]
            return a
        @numba.njit
        def calc_PI_V0_axis(H, Ni, Nk, s_):
            for ii in np.ndindex(Ni):
                for kk in np.ndindex(Nk):
                    a = H[ii + s_ + kk]
                    for j in range(len(a) - 1-1, -1, -1):
                        a[j] = a[j] + a[j+1]
        def calc_PI_V0(X): #construct PI(V_0)
            for i in range(X.ndim):
                #calc_PI_V0_axis(X, X.shape[:i], X.shape[i+1:], np.s_[:,])
                #Ni, Nk = tuple([slice(None, None, None)] * i), tuple([slice(None, None, None)] * (len(X.shape)-1 - i-1))
                #for j in range(X.shape[i]-1-1, -1, -1): X[Ni + np.s_[j,] + Nk] += X[Ni + np.s_[j+1,] + Nk]
                view = np.flip(X.swapaxes(0, i), 0)
                np.cumsum(view, 0, out=view)
                #X = np.apply_along_axis(sum_func, i, X)
            return X
        def mod_gray_code(K):
            codes, istar, V, T = [], 0, np.array(K), np.repeat(-1, len(K))
            while True:
                codes.append((np.array(V), istar, np.array(T)))
                VT = V + T
                S = np.nonzero((VT >= 0) & (VT <= K))[0]
                if len(S) == 0: break
                istar = np.max(S)
                V[istar] = V[istar] + T[istar]
                #T[istar+1:] = -T[istar+1:]
                #print(V)
            return codes
        def extended_gray_code(K):
            codes, istar, V, T = [], 0, np.array(K), np.repeat(-1, len(K))
            while True:
                codes.append((np.array(V), istar, np.array(T)))
                VT = V + T
                S = np.nonzero((VT >= 0) & (VT <= K))[0]
                if len(S) == 0: break
                istar = np.max(S)
                V[istar] = V[istar] + T[istar]
                T[istar+1:] = -T[istar+1:]
                #print(V)
            return codes
        #@numba.njit
        def diff_func(a, Vistar, Vprimeistar):
            a[Vistar+1:] -= a[Vistar]
            a[:Vistar+1] += a[Vprimeistar]
            return a
        #@numba.njit
        def diff_funcneg(a, Vistar, Vprimeistar):
            a[:Vistar] -= a[Vistar]
            a[Vistar:] += a[Vprimeistar]
            return a
        """
        preidxs = []
        for i in range(degree):
            newidx = numba.typed.List()
            Ni, Nk = tuple([2] * i), tuple([2] * (degree-1 - i-1))
            for ii in np.ndindex(Ni):
                for kk in np.ndindex(Nk):
                    newidx.append(ii + np.s_[:,] + kk)
            preidxs.append(newidx)
        """
        @numba.njit #(parallel=True)
        def calc_PI_VI_axis(H, idxs, Tistar, Vistar, Vprimeistar): #Ni, Nk, s_,
            if Tistar == 1:
                #for ii in np.ndindex(Ni):
                #    for kk in np.ndindex(Nk):
                #        a = H[ii + s_ + kk]
                for x in range(len(idxs)):
                    a = H[idxs[x]]
                    a[Vistar+1:] -= a[Vistar]
                    a[:Vistar+1] += a[Vprimeistar]
            else:
                for x in range(len(idxs)):
                    a = H[idxs[x]]
                    a[:Vistar] -= a[Vistar]
                    a[Vistar:] += a[Vprimeistar]
        def calc_PI_VI(Vistar, Vprimeistar, istar, Tistar, PiVi):
            #calc_PI_VI_axis(PiVi, preidxs[istar], Tistar, np.int64(Vistar), Vprimeistar)
            #calc_PI_VI_axis(PiVi, PiVi.shape[:istar], PiVi.shape[istar+1:], np.s_[:,], Tistar, np.int64(Vistar), Vprimeistar)
            #return PiVi
            #return np.apply_along_axis(diff_func if Tistar == 1 else diff_funcneg, istar, PiVi, Vistar, Vprimeistar)
            #Ni, Nk = tuple([slice(None, None, None)] * istar), tuple([slice(None, None, None)] * (len(PiVi.shape)-1 - istar-1))
            view = np.swapaxes(PiVi, 0, istar)
            if Tistar == 1:
                #optimized for binary shaped dimensions
                #if Vistar != 1: PiVi[Ni + np.s_[1,] + Nk] -= PiVi[Ni + np.s_[Vistar,] + Nk]
                #PiVi[Ni + np.s_[0,] + Nk] += PiVi[Ni + np.s_[Vprimeistar,] + Nk]
                #if Vistar == 1: PiVi[Ni + np.s_[Vistar,] + Nk] += PiVi[Ni + np.s_[Vprimeistar,] + Nk]
                #for j in range(Vistar+1, PiVi.shape[istar]): PiVi[Ni + np.s_[j,] + Nk] -= PiVi[Ni + np.s_[Vistar,] + Nk]
                #for j in range(Vistar+1): PiVi[Ni + np.s_[j,] + Nk] += PiVi[Ni + np.s_[Vprimeistar,] + Nk]
                view[Vistar+1:] -= view[Vistar]
                view[:Vistar+1] += view[Vprimeistar]
            else:
                #if Vistar != 0: PiVi[Ni + np.s_[0,] + Nk] -= PiVi[Ni + np.s_[Vistar,] + Nk]
                #if Vistar == 0: PiVi[Ni + np.s_[Vistar,] + Nk] += PiVi[Ni + np.s_[Vprimeistar,] + Nk]
                #PiVi[Ni + np.s_[1,] + Nk] += PiVi[Ni + np.s_[Vprimeistar,] + Nk]
                #for j in range(Vistar): PiVi[Ni + np.s_[j,] + Nk] -= PiVi[Ni + np.s_[Vistar,] + Nk]
                #for j in range(Vistar, PiVi.shape[istar]): PiVi[Ni + np.s_[j,] + Nk] += PiVi[Ni + np.s_[Vprimeistar,] + Nk]
                view[:Vistar] -= view[Vistar]
                view[Vistar:] += view[Vprimeistar]
            return PiVi
        @numba.njit
        def subpat(a, b): #for two sorted lists, -1 no relation, 0 if a contains b, 1 if b contains a, 2 if a==b
            i1, i2 = len(a) - 1, len(b) - 1
            l1, l2 = i1, i2
            while i1 != -1 and i2 != -1: #could use binary search here, hardly matters for normally small degrees
                if a[i1] < b[i2]:
                    while i2 != -1 and b[i2] > a[i1]: i2 -= 1
                else:
                    while i1 != -1 and a[i1] > b[i2]: i1 -= 1
                if i1 == -1 or i2 == -1: return -1
                while i1 != -1 and i2 != -1 and a[i1] == b[i2]: i1, i2 = i1 - 1, i2 - 1
            if i1 == -1:
                if i2 != -1: return 1
                if l1 == l2: return 2
                return 0 if l1 > l2 else 1
            elif i2 == -1: return 0
            return -1
        #assert(np.all([subpat([], []) == 2, subpat([1], [0]) == -1, subpat([1], [1]) == 2, subpat([1], [1, 2]) == 1,
        #       subpat([1, 2], [1]) == 0, subpat([2], [1, 2]) == 1, subpat([1, 2], [2]) == 0, subpat([1, 3], [1, 2, 3]) == 1, subpat([1, 2, 3], [1, 3]) == 0]))
        #n chooses degree projections to consider
        def add_permute(cmb, counts, tot, pats):
            for k in range(len(counts)):
                sigmaI = counts[k] / tot
                if not self.penalty_value is None:
                    sigmaI = prec_penalty(sigmaI, counts[k] / origsz)
                if sigmaI >= self.threshold_pct and counts[k] >= minmatch:
                    do_add_permute(cmb, sigmaI, counts[k], pats[k]) #positive pattern
        def do_add_permute(cmb, sigmaI, num, pats):
            #if len(pats) > maxconds * 2: del pats[:-maxconds]
            #for x in range(len(cmb)):
            #    if (~x if x < 0 else x) in self.mutex_: #verify a mutually exclusive value is not added unnecessarily
            #        if np.any([tuple(np.sort(np.array([*cmb[:x], y if x < 0 else ~y, *cmb[x+1:]], copy=True, dtype=np.int32))) in patset for y in self.mutex_[x]]): return
            cmb.sort() #sort with tuples uses element 0 then 1, etc
            found = tuple(cmb)
            #found = tuple(np.sort(np.array(cmb, copy=True, dtype=np.int32))) #careful as np.int32 maintains a reference
            if found in patset: return
            patset.add(found)
            #if sigmaI > pats[-1][0]:
            #    preds = self._predict(X, [found])
            #    cm = conf_mat(y, preds)
            #    print(found, sigmaI, cm)
            doadd, delidxs = True, set()
            #print(found, pats[-1])
            #operator.itemgetter faster than indexing
            for q, pq in enumerate(pats):
                bq = pq[2]
                if len(bq) == 0: continue
                s = subpat(found, bq)
                if s == -1: continue
                if s == 1 and sigmaI >= pq[0]:
                    delidxs.add(q)
                    continue
                if sigmaI > pq[0]: continue
                doadd = False
                break
            if len(delidxs) != 0: #pats = [x for i, x in enumerate(pats) if i not in delidxs]
                pats = list(np.delete(np.array(pats), list(delidxs), axis=0)) #converts tuple to array
            if doadd:
                lo, hi = 0, len(pats)
                while lo < hi:
                    mid = (lo+hi)//2
                    if sigmaI < pats[mid][0]: hi = mid
                    else: lo = mid+1
                pats.insert(lo, (sigmaI, num, found))            
        def calc_permute(comb, pats):
            #deg = len(comb)
            #gcodes = extended_gray_code(tuple([1] * deg))
            #if deg == 1: #fast route for single features
            #    pos, neg = np.sum(r), np.sum(rn)
            #    tot = pos + neg
            #    if tot == 0: return
            #    add_permute(comb, pos / tot, pos, neg, pats)
            #else:
            bounds = tuple([self.bounds_[x] for x in comb])
            if bounds in gcodesdict: gcodes = gcodesdict[bounds]
            else:
                gcodes = mod_gray_code(tuple([x - 1 for x in bounds])) #extended_gray_code(tuple([x - 1 for x in bounds]))
                gcodesdict[bounds] = gcodes
            #print(bounds)
            PiV0s = []
            for v in vals:
                M = np.zeros(bounds, dtype=np.uint32)
                r = v[:,comb]
                #for x in r.astype(np.uint32): M[tuple(x)] += 1
                np.add.at(M, tuple(r.T.astype(np.uint32)), 1)
                PiV0s.append(calc_PI_V0(M)) #start at tuple([1] * deg)
            #idx = tuple([1] * deg)
            b = gcodes[0][0] #initial PI_V0 index
            #np.arange(np.prod(bounds)).reshape(bounds)
            num_gcodes, lenV, lenPiV0s = len(gcodes), len(b), len(PiV0s)
            cmb = np.array(comb)
            for gcode in range(num_gcodes):
                #i = np.ndindex(bounds)
                curgcode = gcodes[gcode]
                V = curgcode[0]
                idxs = np.ix_(*[[0, x] if y else [x] for x,y in zip(V, V==b)])
                counts = np.array([PiV0[idxs] for PiV0 in PiV0s])
                tots = np.sum(counts, 0)
                #tots[tots != 0]
                origidxs = np.moveaxis(np.array(np.unravel_index(np.ravel_multi_index(idxs, bounds), bounds)), 0, -1)
                for k in range(len(counts)):
                    sigmaI = np.zeros(counts[k].shape)
                    np.divide(counts[k], tots, where=tots!=0, out=sigmaI)
                    if not self.penalty_value is None:
                        sigmaI = prec_penalty(sigmaI, counts[k] / origsz)
                    for i in np.argwhere((sigmaI >= self.threshold_pct) & (counts[k] >= minmatch)):
                        ti = tuple(i)
                        #print(i, bounds, origidxs.shape, sigmaI.shape, counts[k].shape, origidxs[ti], V, cmb, np.argwhere((sigmaI >= self.threshold_pct) & (counts[k] >= minmatch)))
                        same = V == origidxs[ti]
                        if np.sum(same) == 0: continue
                        ccmb = list(zip(cmb[same], V[same]))
                        do_add_permute(ccmb, sigmaI[ti], counts[k][ti], pats[k]) #positive pattern
                """
                i = [tuple(V)]
                p = np.nonzero(V == b)[0] # (V == b) | (V == 0)
                for d in range(1, min(len(p)+1, lenV)): #len(V) is only 1 permutation where its the zero-length combination
                    for c in itertools.combinations(p, d):
                        Vnew = np.array(V)
                        Vnew[list(c)] = 0
                        i.append(tuple(Vnew)) #np.where([q in c for q in range(len(V))], 0, V)
                for idx in i: #if mutually exclusive, the not values of interval are useless, only need to check where V and idx intersect
                    #if the entire interval, skip as its a zero-length combination
                    #if not np.any(idx == V): continue
                    counts = np.array([PiV0[idx] for PiV0 in PiV0s])
                    tot = np.sum(counts)
                    #for k in range(len(vals)):
                        #preds = self._predict(vals[k], [cmb])
                        #print(np.sum(preds), counts[k], cmb, V, idx, PiV0s[k])
                        #assert(np.sum(preds) == counts[k])
                    #cm = conf_mat(y, preds)
                    #assert(cm[1,1] == pos and cm[0,1] == PiV0neg[idx])
                    if tot != 0:
                        cmb = np.array(comb)
                        #cmb[V == 0] = ~cmb[V == 0]
                        same = V == idx
                        cmb = list(zip(cmb[same], V[same]))
                        #cmb = [(cmb[x], Vidx[x]) for x in range(len(comb)) if Vidx[x] == idx[x]] #interval from idx[x] to V[x]
                        add_permute(cmb, counts, tot, pats)
                """
                if gcode == num_gcodes - 1: break
                nextgcode = gcodes[gcode+1]
                istar = nextgcode[1]
                Vistar, Vprimeistar, Tistar = V[istar], nextgcode[0][istar], nextgcode[2][istar]
                for k in range(lenPiV0s):
                    PiV0s[k] = calc_PI_VI(Vistar, Vprimeistar, istar, Tistar, PiV0s[k])
        gcodesdict, degree = {}, min(self.degree, X.shape[1])
        if np.all(X.shape == 2):
            gcodesdict[tuple([2] * degree)] = mod_gray_code(tuple([1] * degree)) #extended_gray_code(tuple([1] * degree))
        permute = np.arange(X.shape[1], dtype=np.int32)
        patset = set()
        pats = [] #numba.typed.List()
        for k in range(len(vals)):
            pats.append([(np.float64(0), np.uint32(0), np.array([], dtype=np.int32))])
        if self.random and X.shape[1] > degree:
            if self.random_state is None:
                rnd = np.random
            elif type(self.random_state) is np.random.RandomState:
                rnd = self.random_state
            else:
                rnd = np.random.RandomState(self.random_state)
            for _ in range(self.maxcombs):
                rnd.shuffle(permute)
                for i in range(0, len(permute), degree):
                    calc_permute(permute[i:i+degree], pats)
            cur, c = [len(vals[k]) for k in range(len(vals))], 0
            #cantfind = set()
            print(cur)
            while True:
                pmt, rem, falsethresh = [], [], []
                for k in range(len(vals)):
                    if cur[k] == 0: continue
                    if len(pats[k]) == 1:
                        remaining = np.nonzero(y == classes[k])[0]
                        falsethresh.append(False)
                    else:
                        preds = self._predict(X, [x[2] for x in pats[k]])
                        falsepos = np.sum(preds & (y != classes[k]))
                        falsethresh.append(falsepos > len(vals[k]) * (1 - self.threshold_pct))
                        remaining = np.nonzero(~preds & (y == classes[k]))[0]
                    rem.append(len(remaining))
                    pmt.extend([np.nonzero(X[remaining[x]])[0] for x in range(len(remaining))])
                if all([x == 0 for x in rem]) or all(falsethresh): break
                permute = np.array(list(set(np.concatenate(pmt))))
                if any([rem[x] != cur[x] for x in range(len(rem))]):
                    cur = rem
                    print(pats, cur, c, len(permute)) #, confusion_matrix(y, preds), confusion_matrix(y, negpreds))
                #if cur != 0:
                #    permute = np.nonzero(X[remaining[0]])[0]
                #else:
                #    permute = np.nonzero(X[remainingneg[0]])[0]
                curlens, startc = [len(p) for p in pats], c
                while all([curlens[k] == len(pats[k]) for k in range(len(pats))]) and startc + self.maxcombs > c:
                    rnd.shuffle(permute)
                    for i in range(0, len(permute), degree):
                        calc_permute(permute[i:i+degree], pats)
                    c += 1
                if startc + self.maxcombs <= c: break
                #    cantfind.add(remaining[0] if cur != 0 else remainingneg[0])
        else:
            for comb in itertools.combinations(permute, degree):
                calc_permute(comb, pats)
        #print(pats, negpats)
        finaleqs = []
        for k in range(len(pats)):
            eqs = [x[2] for x in pats[k]]
            preds = self._predict(X, eqs)
            finaleqs.append((f1_score(y == classes[k], preds, pos_label=True), classes[k], eqs))
        #print(minmatch, accuracy_score(y, preds), accuracy_score(y, negpreds), pats, negpats)
        #print(preferpos, cm, mcc(cm), cmneg, mcc(cmneg))
        finaleqs.sort(reverse=True)
        print(finaleqs)
        return finaleqs
    def _predict(self, X, eqs):
        sz = len(X)
        out = np.zeros(sz, dtype=bool)
        for y in eqs:
            if len(y) == 0: continue
            col = np.ones(sz, dtype=bool)
            for z in y:
                #col = np.logical_and(col, np.logical_not(X[:,~z]))
                col = np.logical_and(col, X[:,z[0]] == z[1])
            out = np.logical_or(out, col)
        return out
    def predict(self, X):
        """ LAD classifier implementation of prediction.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the last
            matching pattern found during fit.  The final label
            is computed as all remaining samples which did not
            yet receive a label.
        """
        check_is_fitted(self, attributes='booleqs_')
        X = check_array(X, dtype=np.float32, accept_sparse="csr")
        if issparse(X) and (X.indices.dtype != np.intc or
                            X.indptr.dtype != np.intc):
            raise ValueError("No support for np.int64 index based "
                             "sparse matrices")
        if X.shape[1] != len(self.binarizer_values_):
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is %s and "
                             "input n_features is %s "
                             % (len(self.binarizer_values_), X.shape[1]))
        X_ = LADClassifier.postbinarize(X, self.binarizer_values_)
        if type(self.n_classes_) is list:
            out = np.zeros((len(X), len(self.booleqs_)), dtype=self.outtype_)
            for n, booleqs in enumerate(self.booleqs_):
                cumpreds = np.zeros(len(X), dtype=np.bool_)
                for k in booleqs[-2::-1]:
                    preds = self._predict(X_, k[2])
                    out[preds, n] = k[1]
                    cumpreds = np.logical_or(preds, cumpreds)
                out[~cumpreds, n] = booleqs[-1][1]
        else:
            out = np.zeros(len(X), dtype=self.outtype_)
            cumpreds = np.zeros(len(X), dtype=np.bool_)
            for k in self.booleqs_[-2::-1]:
                preds = self._predict(X_, k[2])
                out[preds] = k[1]
                cumpreds = np.logical_or(preds, cumpreds)
            out[~cumpreds] = self.booleqs_[-1][1]
        return out
    def score(self, X, y, sample_weight=None):
        preds = self.predict(X)
        #from sklearn.metrics import confusion_matrix #, matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score #with adjusted=True
        print(confusion_matrix(y, preds))
        #cm = confusion_matrix(y, preds)
        #print(cm)
        #print(extended_conf_mat(cm, partial=True), precision(cm), fbetascore(cm, 1), fbetascore(cm, 1), mcc(cm), informedness(cm), kappa(cm))
        #return mcc(cm)
        return accuracy_score(y, preds, sample_weight=sample_weight)
    def format_booleqs(self):
        check_is_fitted(self, attributes='booleqs_')
        def do_format_booleqs(booleqs):
            return {k[1]:(k[0], [[self.featnames_[x[0]][x[1]] for x in y] for y in k[2]]) for k in booleqs}
        if type(self.n_classes_) is list:
            return [do_format_booleqs(x) for x in self.booleqs_]
        else:
            return do_format_booleqs(self.booleqs_)
    def get_params(self, deep=True):
        return {'degree': self.degree, 'random': self.random,
                'maxcombs': self.maxcombs, 'threshold_pct': self.threshold_pct,
                'minmatch_pct': self.minmatch_pct, 'feature_names': self.feature_names,
                'binarizer_params':self.binarizer_params, 'random_state':self.random_state}
                #'mutual_exclusions':self.mutual_exclusions}
    def set_params(self, **params):
        if 'degree' in params: self.degree = params['degree']
        if 'random' in params: self.random = params['random']
        if 'maxcombs' in params: self.maxcombs = params['maxcombs']
        if 'threshold_pct' in params: self.threshold_pct = params['threshold_pct']
        if 'minmatch_pct' in params: self.minmatch_pct = params['minmatch_pct']
        if 'feature_names' in params: self.feature_names = params['feature_names']
        if 'binarizer_params' in params: self.binarizer_params = params['binarizer_params']
        if 'random_state' in params: self.random_state = params['random_state']
        #if 'mutual_exclusions' in params: self.mutual_exclusions = params['mutual_exclusions']
        return self

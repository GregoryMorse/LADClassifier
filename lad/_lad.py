"""
This module contains the LAD classifier implementation
"""
"""Logical Analysis of Data classifier (primary production model candidate).

This is the complete pre-refactor LAD implementation, relocated without
algorithmic changes so it can be calibrated before modernization.
"""

try:
    import numba
except ImportError:  # Numba is an acceleration, not a semantic requirement.
    class _NumbaFallback:
        @staticmethod
        def njit(function=None, **_kwargs):
            if function is not None:
                return function

            def decorator(inner):
                return inner

            return decorator

    numba = _NumbaFallback()
import numpy as np
import settrie
from sklearn.base import BaseEstimator, ClassifierMixin, MultiOutputMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted #check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.multiclass import check_classification_targets
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import check_array
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from scipy.sparse import issparse

try:
    import matplotlib.pyplot as plt
except ImportError:  # Plotting is optional for classifier training/inference.
    class _MissingPlot:
        class cm:
            Blues = None

        def __getattr__(self, _name):
            raise RuntimeError("install the 'plot' dependency group to create LAD plots")

    plt = _MissingPlot()
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

class FeatureGroup(): #dask delayed can do this as well?
    def __init__(self, data, callback, params, namecb = None):
        self.data = data
        self.callback = callback
        self.params = params
        self.namecb = str if namecb is None else namecb
    def get_names(self):
        return [self.namecb(x) for x in self.params]
    def get_name(self, index):
        return self.namecb(self.params[index])
    @property
    def ndim(self): return 2
    @property
    def shape(self): return (len(self.data), len(self.params))
    def __getitem__(self, key):
        if type(key) is tuple:
            l_key = len(key)
            if l_key >= 1 and l_key <= 2:
                #if type(key[0]) is int or np.issubdtype(type(key[0]), np.integer): d = self.data.iloc[key[0]:key[0]+1]
                #elif type(key[0]) is slice or type(key[0]) is np.ndarray:
                d = self.data.iloc[key[0]]
                #else: raise KeyError()
            else: raise IndexError()
            if l_key == 2:
                #if type(key[1]) is int or np.issubdtype(type(key[1]), np.integer): s = [self.params[key[1]]]
                #elif type(key[1]) is slice or type(key[1]) is np.ndarray:
                s = self.params[key[1]]
                #else: raise KeyError()
                if not type(s) is type(self.params):
                    return np.array(self.callback(d, s)).T
                else:
                    return np.array([self.callback(d, p) for p in s]).T
            else:
                return np.array([self.callback(d, p) for p in self.params]).T
        #elif type(key) is int or np.issubdtype(type(key), np.integer):
        #    d = self.data.iloc[key:key+1]
        #    return np.array([self.callback(d, p) for p in self.params]).T
        #elif type(key) is slice or type(key) is np.ndarray:
        else:
            d = self.data.iloc[key]
            return np.array([self.callback(d, p) for p in self.params]).T
        #else:
        #    raise KeyError()
    def __len__(self):
        return len(self.data)

class DiscretizingTransformer(BaseEstimator, TransformerMixin):
    """ An example transformer that returns the element-wise square root.

    For more information regarding how to build your own transformer, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
    """
    def __init__(self, binarizer_params=None, random_state=None, feat_names=None):
        self.binarizer_params = binarizer_params
        self.random_state = random_state
        self.feat_names = feat_names

    """Binarizer for LAD classifier.
    This routine allows for convienent binarization of a single feature prior
    to LAD classification preventing binarization from influencing
    the algorithm.  It is recommended to use binarize instead of this routine,
    unless the parameters such as the minimum and maximum or cut points are already known.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        A single training input feature.
    cut_points : list
        The list of cut points as 2-tuples for start of interval and end of interval,
        and ideally should be continuous without any gaps.  The first and last interval,
        are computed in a one sided manner.  For 'equaldistribution' and
        'minimumdifferentiated' methods only.  If only 2 cut points, then a single
        binary feature will be used as it is sufficient for this special case.
    binarymode : bool
        This is True if opting for binary features only, otherwise if False, then the
        features will be discretized as numbers from 0 to the number of cut points minus 1
        except in the case of only 2 cut points where there is no difference for this parameter.
    interval : bool
        This is True if non-overlapping mutually exclusive intervals should be used,
        otherwise False for levels which use only monotonically increasing inequalities.
        In the case of only 2 cut points, this parameter makes no difference.

    Returns
    -------
    conditions : list
        Returns list of binarized sub-features for the provided feature which can be
        converted to a numpy array and transposed if they will be used in the LADClassifier.
    """
    def _binarizer(data, cut_points, binarymode, interval):
        #2 cut points uses a special reduction to binary because of mutual exclusivity of the values in each division
        lcp = len(cut_points)
        if lcp <= 1:
            return [np.zeros(len(data), dtype=np.bool_)] if binarymode else np.zeros(len(data), dtype=np.bool_)
        cond = np.searchsorted([x[0] for x in cut_points[1:]], data, side='right') #np.digitize()
        if not binarymode: return cond
        condbin = (cond, np.arange(len(data)))
        cond = np.zeros((len(cut_points), len(data)), dtype=np.bool_)
        cond[condbin] = 1
        """
        condold = [((data >= cut_points[j][0]) if j != 0 else True) &
                   ((data < cut_points[j][1]) if j != lcp-1 and interval else True) for j in range(lcp)]
        assert(np.array_equal(cond, condold))
        else:
            condold = np.zeros(len(data), dtype=np.uint32)
            for j in range(1, lcp):
                condold[((data >= cut_points[j][0]) if j != 0 else True) &
                     ((data < cut_points[j][1]) if j != lcp-1 and interval else True)] = j
                #cond[~(((data >= cut_points[j][0]) if j != 0 else True) &
                #     ((data < cut_points[j][1]) if j != lcp-1 else True))] = j + lcp
            print(cond, condold, data, cut_points)
        """
        return cond #[list(x) for x in list(cond)]
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
        A single training input feature.
    y : array-like, shape (n_samples,)
        The target values. An array of int.  Only needed for method='minimumdifferentiated'.
    method : string, default=None
        Either 'minimumdifferentiated', 'equaldivisons'
        or 'equaldistribution'.  When None, defaults to 'minimumdifferentiated'.
    divisions : int, default=None
        The number of divisions for 'equaldivisions' and 'equaldistribution' methods.
        To be useful, it should be greater than or equal to 2.  When None, defaults to 10.

    Returns
    -------
    cut_points : list
        Returns a list, containing 2-tuples with the cutpoints.
    """
    def _binarize(data, y, method=None, divisions=None):
        if method is None: method = 'minimumdifferentiated'
        if divisions is None: divisions = 10
        if method == 'equaldivisions':
            mn, mx = np.min(data), np.max(data)
            #ranges = np.linspace(mn, mx, divisions+1)
            #ranges[0] = -np.inf
            #ranges[-1] = np.inf
            #best = DiscretizingTransformer._hist_np_laxis(X, bins=max_bins, range=(mn, mx), include_low_high=True)
            #divs = {ranges, best}
            dist = (mx - mn) / divisions if divisions != 0 else 0
            if dist == 0:
                divs = list()
            else:
                divs = [(mn + dist * j, mn + dist * (j+1)) for j in range(divisions)]
        elif method == 'equaldistribution': #need to handle splits on equivalence groups, right now redundant or duplicate values possible
            sz, sorted = len(data), np.sort(data)
            divs = [(sorted[int(sz * j / divisions)], sorted[int(sz * (j+1) / divisions)-(1 if j == divisions-1 else 0)])
                    for j in range(divisions)]
            #divs = list(set(divs)) #remove duplicates
            fdivs = list()
            for x in range(divisions-1):
                if divs[x][0] != divs[x+1][0]: fdivs.append(divs[x])
            if divisions > 0: fdivs.append(divs[-1])
            divs = fdivs
        elif method == 'minimumdifferentiated':
            sorted = list(zip(data, y)) #must use float for setting np.nan as a unique placeholder value
            sorted.sort()
            """
            divsold, featnames = list(), list()
            #cannot ignore duplicate values or could wrongly collapse to over-broad divisions
            lastval, x = sorted[0], 1
            while x <= len(sorted):
                nextval = sorted[x] if x != len(sorted) else lastval
                while x < len(sorted) - 1:
                    if sorted[x][0] != sorted[x+1][0]: break
                    if sorted[x][1] != sorted[x+1][1]: nextval = (nextval[0], None)
                    x += 1
                if x == len(sorted) and len(divsold) != 0 or lastval[1] is None or nextval[1] != lastval[1] and nextval[0] != lastval[0]:
                    divsold.append((lastval[0], nextval[0]))
                    #condarr.append((data >= lastval[0]) & (data <= sorted[x][0]))
                    lastval = nextval
                x += 1
            """
            # Keep feature values numeric while allowing arbitrary sklearn-compatible
            # class labels.  Building a homogeneous two-column ndarray coerces the
            # values to strings when ``y`` contains strings, which makes np.diff fail.
            values = np.asarray([row[0] for row in sorted])
            labels = np.asarray([row[1] for row in sorted], dtype=object)
            label_changed = np.zeros(len(labels), dtype=bool)
            label_changed[1:] = labels[1:] != labels[:-1]
            #unq, idx, cts = np.unique(values, return_index=True, return_counts=True)
            #arr[(diffs[:,0]==0) & (diffs[:,1]!=0), 1]
            #for x in unq[cts != 1]:
            #    y = arr[arr[:,0]==x,1]
            #    if y[0] != y: arr[idx[x],1] = np.nan
            divs = np.unique(values[label_changed])
            if len(divs) != 0:
                divs = [(divs[x], divs[x+(1 if x != len(divs)-1 else 0)]) for x in range(len(divs))]
                if divs[0][0] != values[0]: divs = [(values[0], divs[0][0]), *divs]
            #assert(np.array_equal(divs, divsold))
        return divs
    def _binarize_outputgroups(data, y, n_classes, classes, ymap, random_state, max_bins=100, n_splits=5):
        mn, mx = np.min(data), np.max(data)
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = list(kfold.split(data, y))
        train_hist, test_hist, train_counts, test_counts, train_test = [], [], [], [], []
        for (train, test) in folds: train_test.append((data[train], y[train], data[test], y[test]))
        for hist_bins in range(n_classes, max_bins):
            train_hist.append(np.zeros((n_classes, n_splits, 1, hist_bins)))
            train_counts.append(np.zeros((n_classes, n_splits)))
            test_hist.append(np.zeros((n_classes, n_splits, 1, hist_bins)))
            test_counts.append(np.zeros((n_classes, n_splits)))
            for i, (X_train, y_train, X_test, y_test) in enumerate(train_test):
                DiscretizingTransformer._fit_group(X_train, y_train, n_classes, classes, ymap, hist_bins, train_hist, train_counts, i, (mn, mx))
                DiscretizingTransformer._fit_group(X_test, y_test, n_classes, classes, ymap, hist_bins, test_hist, test_counts, i, (mn, mx))
        _train_score, _test_score, _smooth_test_score, best_bins, _best_feat, best = DiscretizingTransformer._compute(train_hist, train_counts, test_hist, test_counts)
        ranges = np.linspace(hist_range[0], hist_range[1], best_bins+n_classes+1)
        ranges[0] = -np.inf
        ranges[-1] = np.inf
        #cut_points = [[] for _ in range(n_classes)]
        return {'ranges':ranges, 'best':best, 'best_bins':best_bins}
    def _binarize_bestoutputgroups(X, y, n_classes, classes, ymap, random_state, feat_names, max_bins=100, hist_range=(-1.0, 1.0), n_splits=5, feature_batch_size=200):
        #from findiff import FinDiff
        #dx = 1 #1 day interval
        #d_dx = FinDiff(0, dx, 1, acc=7) #acc=3 #for 5-point stencil, currenly uses +/-1 day only
        #d2_dx2 = FinDiff(0, dx, 2, acc=7)

        #mn, mx = np.min(X), np.max(X)
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = list(kfold.split(X, y))
        train_score, test_score, smooth_test_score, score = [], [], [], 0
        #hist_ranges = np.zeros((n_features, 2))
        for f in range(0, X.shape[1], feature_batch_size):
            train_hist, test_hist, train_counts, test_counts, train_test = [], [], [], [], []
            Xcur = X[:,f:f+feature_batch_size]
            #s = np.argsort(Xcur, axis=0)
            #hist_ranges[f:f+feature_batch_size] = Xcur[s[[0, -1]], np.arange(Xcur.shape[1])] #+/-len(s)//max_bins
            for (train, test) in folds: train_test.append((Xcur[train,:], y[train], Xcur[test,:], y[test]))
            cur_features = train_test[0][0].shape[1]
            print(f, X.shape)
            for hist_bins in range(n_classes, max_bins):
                train_hist.append(np.zeros((n_classes, n_splits, cur_features, hist_bins)))
                train_counts.append(np.zeros((n_classes, n_splits)))
                test_hist.append(np.zeros((n_classes, n_splits, cur_features, hist_bins)))
                test_counts.append(np.zeros((n_classes, n_splits)))
                for i, (X_train, y_train, X_test, y_test) in enumerate(train_test):
                    DiscretizingTransformer._fit_group(X_train, y_train, n_classes, classes, ymap, hist_bins, train_hist, train_counts, i, hist_range)
                    DiscretizingTransformer._fit_group(X_test, y_test, n_classes, classes, ymap, hist_bins, test_hist, test_counts, i, hist_range)
            cur_train_score, cur_test_score, cur_smooth_test_score, cur_best_bins, cur_best_feat, cur_best = DiscretizingTransformer._compute(train_hist, train_counts, test_hist, test_counts)
            cur_score = cur_smooth_test_score[cur_best_bins - (7-1)//2, cur_best_feat]
            if cur_score > score:
                best_bins, best_feat, best, score = cur_best_bins, cur_best_feat + f, cur_best, cur_score
            train_score.append(cur_train_score), test_score.append(cur_test_score)
            smooth_test_score.append(cur_smooth_test_score)
        train_score = np.concatenate(train_score, axis=1)
        test_score = np.concatenate(test_score, axis=1)
        smooth_test_score = np.concatenate(smooth_test_score, axis=1)
        best_test_scores = smooth_test_score[np.argmax(smooth_test_score, axis=0),np.arange(smooth_test_score.shape[1])]
        for i in np.argsort(best_test_scores)[::-1][:200]:
            print(feat_names[i], np.max(best_test_scores[i]))
        #print(train_score, test_score)
        ranges = np.linspace(hist_range[0], hist_range[1], best_bins+n_classes+1)
        ranges[0] = -np.inf
        ranges[-1] = np.inf
        cut_points = [[] for _ in range(n_classes)]
        for i, x in enumerate(best):
            if cut_points[x] and cut_points[x][-1][1] == ranges[i]:
                cut_points[x][-1] = (cut_points[x][-1][0], ranges[i + 1])
            else: cut_points[x].append((ranges[i], ranges[i + 1]))
        if test_score[best_bins, best_feat] >= 0.25:
            #maxacc = 0
            #for i in range(n_features):
            i = best_feat
            curl = smooth_test_score[:,i] #smooth the signal
            #print(curl)
            #if len(curl) == 0: continue
            #m = np.argmax(curl)
            #if curl[m] > maxacc:
                #maxacc = curl[m]
            m = best_bins
            maxacc = curl[m - (7-1)//2]
            plt.gcf()
            plt.title(('Feature ' + str(i) if feat_names is None else feat_names[i]) + ' OverallAcc: {} AvgAcc: {} Bins: {}'.format(round(maxacc, 3), round(2 * (maxacc-1) / 4 + 1, 3), self.n_classes_+m))
            plt.plot(np.arange(n_classes+(7-1)//2, n_classes+(7-1)//2+len(curl)), curl, label='Test')
            curltrain = DiscretizingTransformer._moving_average(train_score[:,i], 7)
            plt.plot(np.arange(n_classes+(7-1)//2, n_classes+(7-1)//2+len(curltrain)), curltrain, label='Train')
            plt.ylabel('Accuracy')
            plt.xlabel('Histogram Bins')
            plt.legend()
            #plt.plot(d_dx(np.array(l))[4:-4])
            #plt.plot(d_dx(np.array(ltrain))[4:-4])
            plt.show()
        return {'ranges':ranges,'best':best,'best_bins':best_bins,'best_feat':best_feat,'best_test_scores':best_test_scores}

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
        The target values. An array of int.  Only needed for method='minimumdifferentiated'.
    feature_names : list of string, default=None
        The feature names for each feature in X.  If None is passed,
        it will automatically label features 1...N as 'Feature1', 'Feature2', ..., 'FeatureN'.
    binarizer_params : dict, default=None
        The parameters which will be passed to binarize and then binarizer, containing any parameters from:
        'method', 'divisions', 'binarymode', 'interval'.

    Returns
    -------
    (X_new, binarizer_values, bounds) : (array, list, list)
        Returns an array and 2 lists, the first with the binarized features as a numpy array
        and already transposed for use in the LADClassifier.  The second is the binarizer
        parameters for the given features.  The last is the shape of all of the generated features.
    """
    def _binarizeall(X, y, n_classes, classes, ymap, random_state, binarizer_params=None):
        #method='bestoutputgroups', max_bins=100, hist_range=(-1.0, 1.0), n_splits=5, feature_batch_size=200
        #method='outputgroups', max_bins=100, hist_range=(-1.0, 1.0), n_splits=5
        #method='histogram', max_bins=100, hist_range=(-1.0, 1.0)
        #method='equaldivisions', 'equaldistribution', 'minimumdifferentiated', divisions=None
        binvals = list()
        for i in range(X.shape[1]):
            data = X[:,i]
            if data.dtype.type is bool or data.dtype.type is np.bool_:
                binvals.append(None)
            else:
                binparams = dict() if binarizer_params is None else (binarizer_params[i] if type(binarizer_params) is list else binarizer_params)
                method = binparams['method'] if 'method' in binparams else None
                if method == 'outputgroups':
                    vals = DiscretizingTransformer._binarize_outputgroups(data, y, n_classes, classes, ymap, random_state,
                        binparams['max_bins'] if 'max_bins' in binparams else None, binparams['n_splits'] if 'n_splits' in binparams else None)
                else:
                    vals = DiscretizingTransformer._binarize(data, y, method=method, divisions=binparams['divisions'] if 'divisions' in binparams else None)
                    bval = {'cut_points': vals, 'binarymode': binparams['binarymode'] if 'binarymode' in binparams else True, 'interval': binparams['interval'] if 'interval' in binparams else True}
                binvals.append(bval)
        return binvals
    """Binarized feature name generator for LAD classifier.
    This routine returns a human readable representation of the binarized version
    for each feature that was previously binarized based on its cut points.

    Parameters
    ----------
    binarizer_values : dict, default=None
        The parameters which will be passed to binarize, containing any parameters from:
        'method', 'divisions', 'binarymode', 'interval'.
    feature_names : list of string, default=None
        The feature names for each feature in X.  If None is passed,
        it will automatically label features 1...N as 'Feature1', 'Feature2', ..., 'FeatureN'.

    Returns
    -------
    feature_names : list
        Returns a list of strings with feature names corresponding to each
        generated feature, based on the names provided or generated.
    """
    def _binarizer_feat_names(binarizer_values, feature_names=None):
        sz = len(binarizer_values)
        feature_names = ['Feature' + str(x+1) for x in range(sz)] if feature_names is None else feature_names
        featnames = list()
        for i in range(sz):
            name = feature_names[i]
            if binarizer_values[i] is None:
                featnames.append(['!' + name, name])
            else:
                cut_points = binarizer_values[i]['cut_points']
                interval = binarizer_values[i]['interval']
                lcp = len(cut_points)
                if lcp <= 1:
                    feats = [name]
                else:
                    feats = [name + ('>=' + str(round(cut_points[j][0], 2)) if j != 0 else '') +
                             ('<' + str(round(cut_points[j][1], 2)) if j != lcp-1 and interval else '')
                             for j in range(0 if interval or lcp==2 else 1, 1 if lcp==2 else lcp)]
                if binarizer_values[i]['binarymode']:
                    featnames.extend([['!' + x, x] for x in feats]) #for True/False values that were binarized this way, False will come first then True due to 0/1 ordering
                else:
                    featnames.append(feats)
        return featnames

    def binarizer_bounds(self, X):
        bounds = list()
        for i in range(X.shape[1]):
            if self.binarizer_values_[i] is None:
                bounds.append(2)
            elif self.binarizer_values_[i]['binarymode']:
                bounds.extend([2] * len(self.binarizer_values_[i]['cut_points']))
            else:
                bounds.append(max(1, len(self.binarizer_values[i]['cut_points']))) #* 2
            #mutex.append(np.arange(len(condarr)-len(conds), len(condarr)))
        return bounds

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
    def _postbinarize(X, binarizer_values):
        condarr = list()
        for i in range(X.shape[1]):
            if binarizer_values[i] is None:
                condarr.append(X[:,i])
            #elif binarizer_values[i]['binarymode']:
            #    condarr.extend(LADClassifier.binarizer(X[:,i], **binarizer_values[i]))
            else:
                converted = DiscretizingTransformer._binarizer(X[:,i], **binarizer_values[i])
                if binarizer_values[i]['binarymode']:
                    condarr.extend(converted)
                else:
                    condarr.append(converted)
        return np.array(condarr).transpose()
    #['lt', 'eq', 'gt', 'lte', 'neq', 'gte']
    def _binarizecompare(X, feature_names, featcomp, operations=['lt', 'eq', 'gt']):
        feature_names = ['Feature' + str(x+1) for x in range(X.shape[1])] if feature_names is None else feature_names
        compdict = {'lt':(np.less, '<'), 'eq':(np.equal, '=='), 'gt':(np.greater, '>'),
                    'lte':(np.less_equal, '<='), 'neq':(np.not_equal, '!='), 'gte':(np.greater_equal, '>=')}
        mutgroups = [{'lt', 'eq', 'gt'}, {'lt', 'gte'}, {'eq', 'neq'}, {'gt', 'lte'}] #{'lt', 'gt'}, {'lt', 'eq'}, {'gt', 'eq'}
        condarr, featnames, mutex = list(), list(), list()
        muts, so = list(), set(operations)
        for i in mutgroups:
            s = so & i
            if len(s) >= 2: muts.append([operations.index(x) for x in s])
        for i in featcomp:
            for c in operations:
                condarr.append(compdict[c][0](X[:,i[0]], X[:,i[1]]))
                featnames.append(featnames[i[0]] + compdict[c][1] + featnames[i[1]])
            mutex.extend([[len(condarr) - len(c) + x for x in y] for y in muts])
        return condarr, featnames, mutex

    def _hist_laxis(data, n_bins, range_limits, include_low_high=False):
        # Setup bins and determine the bin location for each element for the bins
        R = range_limits
        N = data.shape[-1]
        bins = np.linspace(R[0],R[1],n_bins+1)
        data2D = data.reshape(-1,N)
        idx = np.searchsorted(bins, data2D,'right')-1

        # We need to use bincount to get bin based counts. To have unique IDs for
        # each row and not get confused by the ones from other rows, we need to
        # offset each row by a scale (using row length for this).
        if include_low_high: idx[idx==-1], idx[idx==n_bins] = 0, n_bins-1
        # Some elements would be off limits, so get a mask for those
        else: bad_mask = (idx==-1) | (idx==n_bins)
        scaled_idx = n_bins*np.arange(data2D.shape[0])[:,None] + idx

        # Set the bad ones to be last possible index+1 : n_bins*data2D.shape[0]
        limit = n_bins*data2D.shape[0]
        if not include_low_high: scaled_idx[bad_mask] = limit

        # Get the counts and reshape to multi-dim
        counts = np.bincount(scaled_idx.reshape(-1),minlength=limit+1)[:-1]
        counts.shape = data.shape[:-1] + (n_bins,)
        return counts

    _range = range
    def _hist_np_laxis(a, bins=10, range=None, weights=None, include_low_high=False):
        # Initialize empty histogram
        N = a.shape[-1]
        data2D = a.reshape(-1,N)
        limit = bins*data2D.shape[0]
        # gh-10322 means that type resolution rules are dependent on array
        # shapes. To avoid this causing problems, we pick a type now and stick
        # with it throughout.
        bin_type = np.result_type(range[0], range[1], a)
        if np.issubdtype(bin_type, np.integer):
            bin_type = np.result_type(bin_type, float)
        bin_edges = np.linspace(range[0],range[1],bins+1, endpoint=True, dtype=bin_type)
        # Histogram is an integer or a float array depending on the weights.
        if weights is None:
            ntype = np.dtype(np.intp)
        else:
            ntype = weights.dtype
        n = np.zeros(limit, ntype)
        # Pre-compute histogram scaling factor
        norm = bins / (range[1] - range[0])
        # We set a block size, as this allows us to iterate over chunks when
        # computing histograms, to minimize memory usage.
        BLOCK = 65536
        # We iterate over blocks here for two reasons: the first is that for
        # large arrays, it is actually faster (for example for a 10^8 array it
        # is 2x as fast) and it results in a memory footprint 3x lower in the
        # limit of large arrays.
        for i in DiscretizingTransformer._range(0, data2D.shape[0], BLOCK):
            tmp_a = data2D[i:i+BLOCK]
            block_size = tmp_a.shape[0]
            if weights is None:
                tmp_w = None
            else:
                tmp_w = weights[i:i + BLOCK]
            if include_low_high:
                tmp_a[tmp_a < range[0]] = range[0]
                tmp_a[tmp_a > range[1]] = range[1]
            else:
                # Only include values in the right range
                keep = (tmp_a >= range[0])
                keep &= (tmp_a <= range[1])
                if not np.logical_and.reduce(np.logical_and.reduce(keep)):
                    tmp_a = tmp_a[keep]
                    if tmp_w is not None:
                        tmp_w = tmp_w[keep]
            # This cast ensures no type promotions occur below, which gh-10322
            # make unpredictable. Getting it wrong leads to precision errors
            # like gh-8123.
            tmp_a = tmp_a.astype(bin_edges.dtype, copy=False)

            # Compute the bin indices, and for values that lie exactly on
            # last_edge we need to subtract one
            f_indices = (tmp_a - range[0]) * norm
            indices = f_indices.astype(np.intp)
            indices[indices == bins] -= 1

            # The index computation is not guaranteed to give exactly
            # consistent results within ~1 ULP of the bin edges.
            decrement = tmp_a < bin_edges[indices]
            indices[decrement] -= 1
            # The last bin includes the right edge. The other bins do not.
            increment = ((tmp_a >= bin_edges[indices + 1])
                         & (indices != bins - 1))
            indices[increment] += 1

            if include_low_high:
                indices = (bins*np.arange(i, i+block_size)[:,None] + indices).reshape(-1)
            else:
                indices = ((bins*np.arange(i, i+block_size)[:,None] * keep)[keep].reshape(indices.shape) + indices).reshape(-1)
            #indices = scaled_idx.reshape(-1)
            # We now compute the histogram using bincount
            if ntype.kind == 'c':
                n.real += np.bincount(indices, weights=tmp_w.real,
                                      minlength=limit)
                n.imag += np.bincount(indices, weights=tmp_w.imag,
                                      minlength=limit)
            else:
                n += np.bincount(indices, weights=tmp_w,
                                 minlength=limit).astype(ntype)
        n.shape = a.shape[:-1] + (bins,)
        return n

    def _moving_average(a, n=3, axis=None):
        #pd.Series(a).rolling(window=n, center=True).mean().dropna()
        #pd.Series(a).rolling(window=n, center=True, min_periods=1).mean()
        ret = np.cumsum(a, dtype=float, axis=axis)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def _fit_group(X, y, n_classes, classes, ymap, num_bins, out_hist, out_counts, fold, hist_range):
        for k in classes:
            vals = X[y == k,:]
            #hist = np.array([np.histogram(vals[:,x], bins=ranges)[0] for x in range(vals.shape[1])])
            #h = DiscretizingTransformer.hist_laxis(vals.T, n_bins=num_bins, range_limits=(hist_range[0], hist_range[1]), include_low_high=True)
            h = DiscretizingTransformer._hist_np_laxis(vals.T, bins=num_bins, range=(hist_range[0], hist_range[1]), include_low_high=True)
            #shape of h is (n_features_, num_bins)
            #print(h, h.shape, hist, hist.shape)
            #assert(np.array_equal(np.array(h), hist))
            idx = ymap[k]
            out_counts[num_bins-n_classes][idx][fold] += vals.shape[0]
            out_hist[num_bins-n_classes][idx][fold] += h
    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        if not type(X) is FeatureGroup:
            X = check_array(X, accept_sparse=True)

        self.n_features_ = X.shape[1]
        if not y is None:
            self.classes_ = np.unique(y)
            self.n_classes_ = self.classes_.shape[0]
            #if not hasattr(self, 'ymap_'):
            self.ymap_ = dict()
            for i, k in enumerate(self.classes_):
                self.ymap_[k] = i
            self.binarizer_values_ = DiscretizingTransformer._binarizeall(
                X, y, self.n_classes_, self.classes_, self.ymap_, self.random_state,
                self.binarizer_params,
            )
        else:
            self.binarizer_values_ = DiscretizingTransformer._binarizeall(
                X, y, None, None, None, self.random_state, self.binarizer_params,
            )
        # Return the transformer
        return self
    #def partial_fit(self, X, y=None): pass
    def _compute(train_hist, train_counts, test_hist, test_counts):
        train_score, test_score = [], []
        for i in range(len(train_hist)):
            ha = train_hist[i]
            #shape of ha is (self.n_classes_, self.n_splits, self.n_features_, bin_number)
            #print(ha.shape)
            totaltrues = np.sum(ha, axis=0)[np.newaxis,...]
            #shape of totaltrues is (1, self.n_splits, self.n_features_, bin_number)
            #print(totaltrues, totaltrues.shape)
            out = np.zeros(ha.shape)
            b = np.argmax(np.divide(ha, totaltrues, out=out, where=totaltrues != 0), axis=0)
            #print(b.shape, b)
            #shape of b is (self.n_splits, self.n_features_, bin_number)
            #best.append(b) #tie always goes to minimal index though
            #out = np.zeros(ha.shape)
            #print(np.divide(ha, totaltrues, out=out, where=totaltrues != 0), self.best_)
            totals = np.sum(train_counts[i], axis=0)[:,np.newaxis] #shape is (self.n_classes_, self.n_splits)
            #print(totals, totals.shape)
            #print(tuple([x.reshape(-1) for x in np.indices(b.shape)]))
            best_vals = ha[(b.reshape(-1),) + tuple([x.reshape(-1) for x in np.indices(b.shape)])].reshape(b.shape)
            #best_vals.shape == b.shape
            #print(ha, b, best_vals)
            best_totals = np.sum(np.sum(best_vals, axis=2) / totals, axis=0) / self.n_splits #shape is (self.n_features_,)
            train_score.append(best_totals)

            ha = test_hist[i]
            totals = np.sum(test_counts[i], axis=0)[:,np.newaxis]
            best_vals = ha[(b.reshape(-1),) + tuple([x.reshape(-1) for x in np.indices(b.shape)])].reshape(b.shape)
            best_totals = np.sum(np.sum(best_vals, axis=2) / totals, axis=0) / self.n_splits #shape is (self.n_features_,)
            test_score.append(best_totals)
            #print(train_score[-1].shape, test_score[-1].shape, train_score, test_score)
        train_score, test_score = np.array(train_score), np.array(test_score) #shape is (self.max_bins-self.n_classes_, self.n_features)
        smooth_test_score = DiscretizingTransformer._moving_average(test_score, 7, 0)
        #print(test_score.shape, smooth_test_score.shape)
        best_idx = np.unravel_index(np.argmax(smooth_test_score), smooth_test_score.shape)
        best_bins, best_feat = best_idx[0] + (7-1)//2, best_idx[1]
        #can choose any train-test pair to recompute the final best bin counts
        ha = train_hist[best_bins][:,0,best_feat,:] + test_hist[best_bins][:,0,best_feat,:] #shape is (self.n_classes_, bin_number)
        totaltrues = np.sum(ha, axis=0)[np.newaxis,...]
        out = np.zeros(ha.shape)
        best = np.argmax(np.divide(ha, totaltrues, out=out, where=totaltrues != 0), axis=0)
        #print(self.best_bins_, self.best_feat_, self.best_.shape)
        return train_score, test_score, smooth_test_score, best_bins, best_feat, best
    def _score_final(self, X, y):
        preds = self._transform_final(X)
        res, total = 0, len(y) #np.zeros(self.n_features_)
        for k in self.classes_:
            vals = preds[y == k]
            truetot = np.sum(vals == self.ymap_[k]) #, axis=1
            #print(truetot)
            res += truetot
            #falsepos = np.sum(preds[y != k] == self.ymap_[k])
            #totfalse += falsepos
            #trueneg = np.sum(preds[y != k] != self.ymap_[k])
            #resacc += truetot + trueneg
            #print(preds, total, truetot, trueneg, falsepos, (truetot + trueneg) / total)
        acc = res / total
        #print(res, total, acc, np.mean(acc))
        #avgacc = resacc / total / len(ha)
        #print(total, totfalse, res, avgacc, acc, (2 * acc + 1) / len(ha)+1)
        #assert(avgacc == (2 * acc + 1) / len(ha)+1) #floating point accuracy not enough
        return acc
    def score(self, X, y=None, sample_weight=None): #only for 'bestoutputgroups'
        #if no y was provided, curhist_ is the output...

        #total = sum([x[0] for x in self.curhist_])
        #for i in range(len(ha)):
            #print(np.sum(ha[i, best==i]), np.sum(best==i), np.sum(ha[i, best==i]) / total)
        #print(totaltrues, best, ha.shape, ha[best, np.arange(self.histbins)], total, np.sum(ha[best, np.arange(self.histbins)]) / total)
        #return np.sum(ha[best, np.arange(self.histbins)]) / total
        # Check is fit had been called
        check_is_fitted(self)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')

        # Input validation
        if not type(X) is FeatureGroup:
            X = check_array(X, accept_sparse=True)
        #if not hasattr(self, 'ymap_'): return 0
        #if not hasattr(self, 'best_'): self._compute()

        return self._score_final(X, y)

    def _transform_final(self, X):
        cond = np.searchsorted(self.ranges_[1:-1], X[:,self.best_feat_], side='right')
        #print(self.ranges_[1:-1].shape, np.unique(cond), self.best_feat_, 2+self.best_bins_)
        #print(cond, cond.shape, self.best_, self.best_.shape, self.ranges_[1:-1])
        #shape of best is (# bins, self.n_features_)
        #print(np.add(cond.T, np.arange(0, self.best_.shape[0] * self.n_features_, self.best_.shape[0])).T.reshape(-1))
        #preds = self.best_.T.reshape(-1)[np.add(cond, np.arange(0, self.best_.shape[0] * self.n_features_, self.best_.shape[0])).T.reshape(-1)].reshape(cond.shape[1], cond.shape[0])
        preds = self.best_[cond]
        return preds.T
    def transform(self, X):
        """ A reference implementation of a transform function.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        # Check is fit had been called
        check_is_fitted(self)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')

        # Input validation
        if not type(X) is FeatureGroup:
            X = check_array(X, accept_sparse=True)
        #if not hasattr(self, 'ymap_'): return #discretize based on equal divisions/distributions
        #if not hasattr(self, 'best_'): self._compute()
        if isinstance(self.binarizer_params, dict) and self.binarizer_params.get('method') == 'bestoutputgroups':
            return self._transform_final(X).T
        return DiscretizingTransformer._postbinarize(X, self.binarizer_values_)

    def get_params(self, deep=True):
        return {'binarizer_params':self.binarizer_params,
                'random_state':self.random_state, 'feat_names':self.feat_names}
    def set_params(self, **params):
        if 'binarizer_params' in params: self.binarizer_params = params['binarizer_params']
        if 'random_state' in params: self.random_state = params['random_state']
        if 'feat_names' in params: self.feat_names = params['feat_names']
        return self

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
        model prior to classification.  It can be a list of dictionaries.
        If None, defaults will be used. If a dictionary, then the same paremeter
        will be used for all features.
        Dictionaries include any of 'method', 'divisions', 'binarymode', 'interval'.
        It is the same as passed to binarizeall.
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
        The binarizer parameters computed for each feature as a list of dictionaries.
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

    """Paper test with simple and general but inefficient code as a proof of concept.
    It shows that the paper example works with a series of assertions for its 3 algorithms.
    Paper: https://www.sciencedirect.com/science/article/pii/S0166218X05003161
    """
    @staticmethod
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
            codes, istar, V, T = list(), 0, np.array(K), np.repeat(-1, len(K))
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
    @staticmethod
    def _make_learn_plots(curdir):
        import matplotlib.pyplot as plt
        import os
        from trading_system.config import get_settings
        output_dir = get_settings().output_dir
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
        plt.savefig(os.path.join(output_dir, 'gcodes.svg'), format='svg')#, bbox_inches = extent, pad_inches = 0)
        plt.savefig(os.path.join(output_dir, 'gcodes.png'), format='png')#, bbox_inches = extent, pad_inches = 0)
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
            clf.fit(iris.data, iris.target)
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
        if y is None:
            raise ValueError(
                "This LADClassifier estimator requires y to be passed, "
                "but the target y is None"
            )
        #binarization comes first
        #X, y = check_X_y(X, y)
        #self.classes_ = unique_labels(y)
        #print(X.shape[1])
        X = check_array(X, dtype=[np.float64, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.bool_], accept_sparse=False) #"csc")
        y = check_array(y, ensure_2d=False, dtype=None)
        self.n_features_in_ = X.shape[1]
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
        #self.mutex_ = self.mutual_exclusions[:] + self.mutex_
        #self.mutex_ = {y:x for x in self.mutex_ for y in x}
        self.outtype_ = y.dtype
        if y.ndim == 1:
            self.n_outputs_ = 1
            self.classes_, idxs = np.unique(y, return_inverse=True)
            self.n_classes_ = self.classes_.shape[0]
            if self.n_classes_ < 2:
                raise ValueError("LADClassifier cannot fit data with only 1 class")
            self.discretizer_ = DiscretizingTransformer(
                self.binarizer_params, self.random_state, self.feature_names
            )
            condarr = self.discretizer_.fit_transform(X, y)
            self.binarizer_values_ = self.discretizer_.binarizer_values_
            self.bounds_ = self.discretizer_.binarizer_bounds(X)
            self.featnames_ = DiscretizingTransformer._binarizer_feat_names(
                self.binarizer_values_, self.feature_names
            )
            #condarr, self.binarizer_values_, self.bounds_ = LADClassifier.binarizeall(X, idxs, self.binarizer_params)
            #self.featnames_ = LADClassifier.binarizer_feat_names(self.binarizer_values_, self.feature_names)
            self.booleqs_ = self._fit(condarr, y, self.classes_, self.bounds_)
        else:
            self.n_outputs_ = y.shape[1]
            self.classes_ = list()
            self.n_classes_ = list()
            self.booleqs_ = list()
            self.discretizer_ = list()
            self.featnames_ = list()
            self.binarizer_values_ = list()
            self.bounds_ = list()
            for k in range(self.n_outputs_):
                classes_k, idxs = np.unique(y[:, k], return_inverse=True)
                if classes_k.shape[0] < 2:
                    raise ValueError("LADClassifier cannot fit an output with only 1 class")
                self.classes_.append(classes_k)
                self.n_classes_.append(classes_k.shape[0])
                self.discretizer_.append(DiscretizingTransformer(
                    self.binarizer_params, self.random_state, self.feature_names
                ))
                condarr = self.discretizer_[k].fit_transform(X, y[:, k])
                binarizer_values = self.discretizer_[k].binarizer_values_
                self.binarizer_values_.append(binarizer_values)
                self.bounds_.append(self.discretizer_[k].binarizer_bounds(X))
                self.featnames_.append(DiscretizingTransformer._binarizer_feat_names(
                    binarizer_values, self.feature_names
                ))
                #condarr, binarizer_values, bounds = LADClassifier.binarizeall(X, idxs, self.binarizer_params)
                #self.binarizer_values_.append(binarizer_values), self.bounds_.append(bounds)
                #self.featnames_.append(LADClassifier.binarizer_feat_names(binarizer_values, self.feature_names))
                self.booleqs_.append(dict()) #(prefer positive, positive patterns, negative patterns)
            for k in range(self.n_outputs_):
                self.booleqs_[k] = self._fit(condarr, y[:, k], self.classes_[k], self.bounds_[k])
        return self
    def _fit(self, X, y, classes, curbounds): #in DNF, if want CNF, can negate X and y per DeMorgan's law?
        #print(X.shape[1])
        vals, origsz = list(), len(X)
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
            codes, istar, V, T = list(), 0, np.array(K), np.repeat(-1, len(K))
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
            codes, istar, V, T = list(), 0, np.array(K), np.repeat(-1, len(K))
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
        preidxs = list()
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
        #assert(np.all([subpat(list(), list()) == 2, subpat([1], [0]) == -1, subpat([1], [1]) == 2, subpat([1], [1, 2]) == 1,
        #       subpat([1, 2], [1]) == 0, subpat([2], [1, 2]) == 1, subpat([1, 2], [2]) == 0, subpat([1, 3], [1, 2, 3]) == 1, subpat([1, 2, 3], [1, 3]) == 0]))
        #n chooses degree projections to consider
        def add_permute(cmb, counts, tot, pats, pattrie):
            for k in range(len(counts)):
                sigmaI = counts[k] / tot
                if not self.penalty_value is None:
                    sigmaI = prec_penalty(sigmaI, counts[k] / origsz)
                if sigmaI >= self.threshold_pct and counts[k] >= minmatch:
                    do_add_permute(cmb, sigmaI, counts[k], pats[k], pattrie[k]) #positive pattern
        def do_add_permute(cmb, sigmaI, num, pats, pattrie):
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
            """
            for q in pattrie.itersupersets(found):
                tq = tuple(sorted(q))
                if sigmaI >= pats[tq][0]:
                    pattrie.remove(q)
                    pats.pop(tq)
            for q in pattrie.itersubsets(found):
                if sigmaI <= pats[tuple(sorted(q))][0]: doadd = False
            if doadd:
                pattrie.add(found)
                pats[found] = (sigmaI, num)
            """
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
            if delidxs:
                pats[:] = [pattern for index, pattern in enumerate(pats) if index not in delidxs]
            if doadd:
                lo, hi = 0, len(pats)
                while lo < hi:
                    mid = (lo+hi)//2
                    if sigmaI < pats[mid][0]: hi = mid
                    else: lo = mid+1
                pats.insert(lo, (sigmaI, num, found))
        def calc_permute(comb, pats, pattrie):
            #deg = len(comb)
            #gcodes = extended_gray_code(tuple([1] * deg))
            #if deg == 1: #fast route for single features
            #    pos, neg = np.sum(r), np.sum(rn)
            #    tot = pos + neg
            #    if tot == 0: return
            #    add_permute(comb, pos / tot, pos, neg, pats)
            #else:
            bounds = tuple([curbounds[x] for x in comb])
            if bounds in gcodesdict: gcodes = gcodesdict[bounds]
            else:
                gcodes = mod_gray_code(tuple([x - 1 for x in bounds])) #extended_gray_code(tuple([x - 1 for x in bounds]))
                gcodesdict[bounds] = gcodes
            #print(bounds)
            PiV0s = list()
            for v in vals:
                M = np.zeros(bounds, dtype=np.uint32)
                r = v[:,comb]
                #for x in r.astype(np.uint32): M[tuple(x)] += 1
                np.add.at(M, tuple(r.T.astype(np.uint32)), 1)
                PiV0s.append(calc_PI_V0(M)) #start at tuple([1] * deg)
            #idx = tuple([1] * deg)
            b = gcodes[0][0] #initial PI_V0 index
            #np.arange(np.prod(bounds)).reshape(bounds)
            num_gcodes, lenPiV0s = len(gcodes), len(PiV0s)
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
                        do_add_permute(ccmb, sigmaI[ti], counts[k][ti], pats[k], pattrie[k]) #positive pattern
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
        gcodesdict, degree = dict(), min(self.degree, X.shape[1])
        if np.all(X.shape == 2):
            gcodesdict[tuple([2] * degree)] = mod_gray_code(tuple([1] * degree)) #extended_gray_code(tuple([1] * degree))
        permute = np.arange(X.shape[1], dtype=np.int32)
        patset = set()
        pattrie = list()
        pats = list() #numba.typed.List()
        for k in range(len(vals)):
            pats.append([(np.float64(0), np.uint32(0), np.array(list(), dtype=np.int32))])
            #pats.append(dict())
            pattrie.append(settrie.SetTrie())
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
                    calc_permute(permute[i:i+degree], pats, pattrie)
            cur, c = [len(vals[k]) for k in range(len(vals))], 0
            #cantfind = set()
            print(cur)
            while True:
                pmt, rem, falsethresh = list(), list(), list()
                for k in range(len(vals)):
                    if cur[k] == 0:
                        remaining = list()
                        falsethresh.append(False)
                    elif len(pats[k]) == 1:
                        remaining = np.nonzero(y == classes[k])[0]
                        falsethresh.append(False)
                    else:
                        #preds = self._predict(X, [x for x in pats[k].keys()])
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
                    print(cur, c, len(permute)) #, pats, confusion_matrix(y, preds), confusion_matrix(y, negpreds))
                #if cur != 0:
                #    permute = np.nonzero(X[remaining[0]])[0]
                #else:
                #    permute = np.nonzero(X[remainingneg[0]])[0]
                curlens, startc = [len(p) for p in pats], c
                while all([curlens[k] == len(pats[k]) for k in range(len(pats))]) and startc + self.maxcombs > c:
                    rnd.shuffle(permute)
                    for i in range(0, len(permute), degree):
                        calc_permute(permute[i:i+degree], pats, pattrie)
                    c += 1
                if startc + self.maxcombs <= c: break
                #    cantfind.add(remaining[0] if cur != 0 else remainingneg[0])
        else:
            for comb in itertools.combinations(permute, degree):
                calc_permute(comb, pats, pattrie)
        #print(pats, negpats)
        finaleqs = list()
        for k in range(len(pats)):
            #eqs = [x for x in pats[k].keys()]
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
        check_is_fitted(self) #, attributes='booleqs_')
        X = check_array(X, dtype=[np.float64, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.bool_], accept_sparse=False) #accept_sparse="csr"
        if issparse(X) and (X.indices.dtype != np.intc or
                            X.indptr.dtype != np.intc):
            raise ValueError("No support for np.int64 index based "
                             "sparse matrices")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {self.__class__.__name__} "
                f"is expecting {self.n_features_in_} features as input"
            )
        if type(self.n_classes_) is list:
            out = np.zeros((len(X), len(self.booleqs_)), dtype=self.outtype_)
            for n, booleqs in enumerate(self.booleqs_):
                X_ = self.discretizer_[n].transform(X)
                #X_ = LADClassifier.postbinarize(X, self.binarizer_values_[n])
                cumpreds = np.zeros(len(X), dtype=np.bool_)
                for k in booleqs[-2::-1]:
                    preds = self._predict(X_, k[2])
                    out[preds, n] = k[1]
                    cumpreds = np.logical_or(preds, cumpreds)
                out[~cumpreds, n] = booleqs[-1][1]
        else:
            X_ = self.discretizer_.transform(X)
            #X_ = LADClassifier.postbinarize(X, self.binarizer_values_)
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
        check_is_fitted(self) #, attributes='booleqs_')
        def do_format_booleqs(booleqs, featnames):
            return {k[1]:(k[0], [[featnames[x[0]][x[1]] for x in y] for y in k[2]]) for k in booleqs}
        if type(self.n_classes_) is list:
            return [do_format_booleqs(x, self.featnames_[n]) for n, x in enumerate(self.booleqs_)]
        else:
            return do_format_booleqs(self.booleqs_, self.featnames_)
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
def test_lad():
    LADClassifier._testpaper()
    #LADClassifier._make_learn_plots(curdir)
    from sklearn.utils.estimator_checks import check_estimator, check_classifiers_train, check_supervised_y_2d, check_classifiers_one_label
    #check_classifiers_train(LADClassifier.__name__, LADClassifier())
    #check_supervised_y_2d(LADClassifier.__name__, LADClassifier())
    #check_classifiers_one_label(LADClassifier.__name__, LADClassifier())
    check_estimator(LADClassifier())
class BooleanEquationClassifier:
    def __init__(self, isPositive=None, beta=0.01, minmatch_pct=0.001, onesided=False):
        self.isPositive = isPositive
        self.beta = beta
        self.minmatch_pct = minmatch_pct
        self.onesided = onesided
        self._estimator_type = 'classifier' #needed for stratified k-folds in GridSearchCV
        self.booleqs_ = list()
        self.scores_ = list()
    def fit(self, X, y):
        if self.isPositive is None:
            p, n = self._fit(X, y, True), self._fit(X, y, False)
            self.scores_, self.booleqs_ = (p[0], n[0]), (p[1], n[1])
        else:
            self.scores_, self.booleqs_ = self._fit(X, y, self.isPositive)
        return self
    def _fit(self, X, y, pos):
        def sorted_tuple(l):
            l.sort()
            return tuple(l)
        def insort_right(a, x, lo=0, hi=None, key=None):
            lo = bisect_right(a, x, lo, hi, key)
            a.insert(lo, x)
        def bisect_right(a, x, lo=0, hi=None, key=None):
            if lo < 0:
                raise ValueError('lo must be non-negative')
            if hi is None: hi = len(a)
            while lo < hi:
                mid = (lo+hi)//2
                if x < a[mid] if key is None else key(x) < key(a[mid]): hi = mid
                else: lo = mid+1
            return lo
        #need more research into entropy of binary matrices
        #https://stats.stackexchange.com/questions/17109/measuring-entropy-information-patterns-of-a-2d-binary-matrix
        def calc_shannon_bit_ent(conds, classvals): #row entropies
            ents, tot = list(), conds.shape[1]
            for i in range(len(conds) + 1):
                counts = sum_bits(conds[:,i] if i != len(conds) else classvals) / len(conds)
                se = 0.0
                se -= counts * np.log(counts) / np.log(2)
                se -= (1 - counts) * np.log(1 - counts) / np.log(2)
                ents.append(se)
            return ents
        def prec_penalty(precision, featpct): #reduce precision based on number of features
            a = 10000000
            return (a ** (1-featpct) - 1) / (a-1) * precision #exponential between 0 and 1: (a^x-1)/(a-1) where higher a has higher decay
        #import nimfa
        #bmf = nimfa.Bmf(X, seed='random_vcol', rank=50, max_iter=15000, n_run=10, initialize_only=True, lambda_w=1.5, lambda_h=1.5)
        #column=[a11b11+a12b21+a1nbn1, a21b11+a22b21+a2nbn1...]
        #colmn2=[a11b12+a12b22+a1nbn2, a21b12+a22b22+a2nbn2...]
        #column&colmn2=a11b11a11b12+a11b11a12b22+a11b11a1nbn2+a12b21a11b12+a12b21a12b22+a12b21a1nbn2+a1nbn1a11b12+a1nbn1a12b22+a1nbn1a1nbn2
        #print(bmf.estimate_rank())
        #fit = bmf()
        #a, b = fit.basis().A.round().astype(np.bool_), fit.coef(None).A.round().astype(np.bool_)
        #print(fit.basis(), fit.basis().shape, np.sum(a), fit.coef(None), fit.coef(None).shape, np.sum(b), np.sum(a.dot(b)), np.sum(X))
        #trading targets are 66% (weak), 75% (decent), 80% (strong) and 90% (very strong)
        #transpose = np.transpose(np.array(X))

        cv = boolarr_tobytes(y)
        andfunc = np.bitwise_and if pos else np.bitwise_or
        orfunc = np.bitwise_or if pos else np.bitwise_and
        trueposidx = (1,1) if pos else (0,0)
        truenegidx = (0,0) if pos else (1,1)
        falseposidx = (0,1) if pos else (1,0)
        falsenegidx = (1,0) if pos else (0,1)
        #np.flip(cm) would also work to reverse positives and negatives
        #if not self.isPositive: transpose, y = ~transpose, ~y
        #transpose = np.array(list(map(list, zip(*X))))
        sz, origpos = len(cv), sum_bits(cv)
        origposrate, origneg = origpos / sz, sz - origpos
        minmatch, confsn = round(sz * self.minmatch_pct), list()
        for x in range(X.shape[1]): #initial values
            tp = boolarr_tobytes(X[:,x])
            #tp = X[:,x]
            cm = conf_mat_packed(cv, tp, sz)
            #assert(np.array_equal(cm, conf_mat(y, X[:,x])))
            #if (not np.array_equal(confusion_matrix(y, tp), cm)):
            #    print((confusion_matrix(y, tp), cm))
            #assert(np.array_equal(confusion_matrix(y, tp), cm))
            #true negatives: cm[0,0], false negatives: cm[1,0], true positives: cm[1,1], false positives: cm[0,1]
            #if no true positives, discard the value as it does not contribute
            #if true positives less than minimum percentage, discard as too weak
            #if no true negatives and any false positives, discard the value as it contributes no information
            #if the false positive rate is too high could be discarded as not useful
            #specifically if total positives minus false negatives or true negatives is not greater than minimum percentage, discard as too weak
            #classification data must have at least the minimum percentage of true negatives and positives or there is little reason to do such a procedure
            if cm[trueposidx] >= minmatch and cm[truenegidx] >= minmatch and (cm[truenegidx] != 0 or cm[falseposidx] == 0):
                posrate = precision(cm, not pos)
                for y in range(len(confsn)): #O(n^2) too similar check
                    if sum_bits(np.bitwise_xor(tp, confsn[y][2])) <= sz * 0.1: #similarity check
                        if posrate > confsn[y][0]: confsn[y] = (posrate, (x,), tp)
                        break #xnor is equality
                else:
                    confsn.append((posrate, (x,), tp))
        if len(confsn) == 0:
            return 0, list()
        #mx = max(confsn, key = lambda y: y[0])
        mxpos = max(confsn, key = lambda y: y[0])
        #mxneg = max(confsn, key = lambda y: y[2])
        print(X.shape[1], len(confsn), sz, pos, self.beta, self.minmatch_pct,
              round(self.minmatch_pct * sz / 0.8 / X.shape[1], 3), minmatch, round(origposrate, 2),
              #mx[:2], featnamesclose[mx[1][0]], mxneg[:2], featnamesclose[mxneg[1][0]],
              mxpos[:2], featnamesclose[mxpos[1][0]])
        confsn.sort(key = lambda y: y[0]) #need ascending order for bisect
        def process_and(curx, cury, l):
            val = andfunc(curx[2], cury[2])
            #cm = conf_mat(y, val)
            cm = conf_mat_packed(cv, val, sz)
            #if cm[1,1] >= sz * 0.01: #1% of data at least matching
            #need formula for best based on rate and total number
            #obviously highest rate possible, but for maximum total, so can accept lower rate if more true positive values detected
            #if totalrate >= 0.6:
            #true positives and false positives will decrease, while true negatives and false negatives will increase
            #need false positives to decrease more than true positives decreases
            #the only things to compare are the positive rate, total rate, and the number of matches
            #the only thing to terminate based on for and conditions is when its too specific as anything else could be still a possibility
            if cm[trueposidx] >= minmatch: #posrate > curx[1] and (cm[0,0]+cm[1,1] - (curxcm[0,0]+curxcm[1,1])) >= minmatch and (
                    #cm[falsenegidx] > cm[falseposidx] * (1 - min(0, posrate - origposrate) / (1 - origposrate))): #really this depends on sz/X.shape[1]
                return (prec_penalty(precision(cm, not pos), len(l) / origsz), l, val) #need to apply a penalty increasing based on len(l) / origsz
            return None
        searched, origsz, confsnor = set(), len(confsn), confsn[:]
        maxands, maxcheck = 700, 5
        revlookup = {v[1][-1]: i for i, v in enumerate(confsn)}
        #posfilter = lambda x: x[1] >= (x[3][falsenegidx] + x[3][trueposidx] + minmatch) / sz
        #if len(l) == 3:
        #    idx1 = next(i for i,v in enumerate(confsn) if v[4][0] == x[4][0])
        #    idx2 = next(i for i,v in enumerate(confsn) if v[4][0] == x[4][1])
        #    if process_and(confsn[idx1], confsn, y, (idx1, confsn[y][4][-1])) is None and process_and(confsn[idx2], confsn, y, (idx2, confsn[y][4][-1])) is None:
        #        print('Transitivity failed', l, idx1, idx2, y)
        def add_max(base, ybase, currentmx):
            if base[0] > currentmx[0][0]:
                if len(currentmx) > maxcheck * 2: del currentmx[:-maxcheck]
                idx = next((i for i, q in enumerate(currentmx) if q[1][1] == ybase[1]), None)
                if idx is None:
                    insort_right(currentmx, (base[0], ybase), key=lambda q: q[0])
                else: currentmx[idx] = (base[0], ybase)
            if base[0] > confsnor[-maxands][0]:
                if len(confsnor) > maxands * 2: del confsnor[:-maxands] #deque with popleft() wont be efficient for sorted insertion
                #if the match is too similar to an already present match, just take the better of the two
                doadd = True
                for q in range(1, maxands+1):
                    if sum_bits(np.bitwise_xor(base[2], confsnor[-q][2])) <= minmatch:
                        doadd = base[0] > confsnor[-q][0]
                        if doadd: del confsnor[-q]
                        break
                if doadd:
                    if base[0] > confsnor[-1][0]:
                        confsnor.append(base)
                        #preds = self._predict(X, pos, [base[1]])
                        #cm = conf_mat(y, preds)
                        print(len(searched), confsnor[-maxands][:2], base[:2], [featnamesclose[q] for q in base[1]], conf_mat_packed(cv, base[2], sz))
                    else: insort_right(confsnor, base, key=lambda q: q[0])
        def recurse_and(x): #unfortunately filter(filter(a & b) & c) does not imply filter(a & c) | filter(b & c) and still yet none of the 3 combinations could still find with filter(a & b & c)
            currentmx, oldmax = [(x[0], x)], confsnor[-maxands][1]
            if len(x[1]) > 0: print(x[1], confsnor[-maxands][0])
            for y in range(origsz): #the very first set need only be done from x+1
                cury = confsn[y]
                if cury[1][-1] in x[1]: continue
                l = sorted_tuple([*x[1],cury[1][-1]])
                if l in searched: continue
                searched.add(l)
                ybase = cury if len(x[1]) == 0 else process_and(x, cury, l)
                if ybase is None: continue
                yset = set([revlookup[q] for q in ybase[1]])
                arr = np.array(list(i for i in range(origsz) if i not in yset))
                add_max(ybase, ybase, currentmx)
                for z in range(300):
                    np.random.shuffle(arr)
                    errcount, base = 0, ybase
                    for q in arr:
                        cury = confsn[q]
                        l = sorted_tuple([*base[1],cury[1][-1]])
                        ret = process_and(base, cury, l)
                        if not ret is None:
                            base = ret
                            add_max(base, ybase, currentmx)
                        elif errcount > 5: break
                        else: errcount += 1
                        #for q in range(1, maxands):
                        #    if confsnor[-q-1][0] > confsnor[-q][0]: print('not sorted', q, confsnor[-q-1][:2], confsnor[-q][:2])
            currentmx = currentmx[-maxcheck:] #conserve memory
            if oldmax != confsnor[-maxands][1]:
                for y in currentmx:
                    if y[1][1] != x[1]: recurse_and(y[1])
        #for x in range(origsz):
        recurse_and((0, list(), list())) #confsn[x], range(confsn[x][1][0]+1, origsz))
        #    if x % int(origsz * 0.1) == 0: print(x, confsnor[-maxands][:2], confsnor[-1][:2], conf_mat_packed(cv, confsnor[-maxands][2], sz))
        #    if confsnor[-maxands][0] >= 0.7: break
        confsn = confsnor[-maxands:]
        #confsn = list(filter(posfilter, confsn)) #must be greater than positive rate at least number of positives plus minimum size, not too many predicted positives
        #confsn.sort(key = lambda x: x[0]) #weakest to strongest
        print(len(confsnor), confsnor[-maxands][:2], confsnor[-1][:2], conf_mat_packed(cv, confsnor[-maxands][2], sz), conf_mat_packed(cv, confsnor[-1][2], sz))
        #find best logical or of values which computes certain mimimum amount of negative values
        confsnor = [(x[0] if self.onesided else mcc(conf_mat_packed(cv, x[2], sz)), (y,), x[2]) for y, x in enumerate(confsn)]
        confsn = [x[1] for x in confsn] #free memory
        if len(confsnor) == 0:
            return 0, list()
        def process_or(curx, cury, l):
            val = orfunc(curx[2], cury[2])
            #cm = conf_mat(y, val)
            cm = conf_mat_packed(cv, val, sz)
            #minimize the false negatives
            #if cm[0,0] >= sz * 0.01:
            #the only terminating condition here is that its too general - too many true and false positives, or too few true positives
            if cm[truenegidx] >= minmatch:#(totalrate, posrate)[1 if self.onesided else 0] >= curx[0] and (cm[0,0]+cm[1,1] - (curxcm[0,0]+curxcm[1,1])) >= minmatch: #if posrate >= 0.7:# and (cm[0,0] + cm[1,1] >= sz * 0.5): # and negrate >= confsn[x][2]
                return (fbetascore(cm, self.beta, not pos) if self.onesided else mcc(cm), l, val)
            return None
        #a & b == c === a == c & b == c | b==c & a | a==c & b === a == c & b == c | a | c != b | c, a | b == c === ~(a | b) == ~c === ~a & ~b == ~c
        #[x & y == z for x in [False, True] for y in [False, True] for z in [False, True]]
        #[(x == z) & (y == z) | (y == z) & x | (x == z) & y for x in [False, True] for y in [False, True] for z in [False, True]]
        #[(x == z) & (y == z) | ((x | z) != (y | z)) for x in [False, True] for y in [False, True] for z in [False, True]]
        #xor and equality are just negation of each other, but or operation needed also
        #best reduction is to use a variation of a random walk similar to Monte Carlo Tree Search (MCTS) to eliminate the factorial or Markov Chain Monte Carlo (MCMC)
        #np.random.seed(seed) #to allow deterministic behavior
        def check_max(base, ybase, currentmx, curmx):
            if base[0] > currentmx[0][0]:
                if len(currentmx) > maxcheck * 2: del currentmx[:-maxcheck]
                idx = next((i for i, q in enumerate(currentmx) if q[1][1] == ybase[1]), None)
                if idx is None:
                    insort_right(currentmx, (base[0], ybase), key=lambda q: q[0])
                else: currentmx[idx] = (base[0], ybase)
            if base[0] > curmx[0]:
                curmx = base
                #preds = self._predict(X, pos, [confsn[y] for y in curmx[1]])
                #cm = conf_mat(y, preds)
                print(len(searched), curmx[:2], conf_mat_packed(cv, curmx[2], sz))
            return curmx
        def recurse_or(x):
            nonlocal curmx
            currentmx, oldmax = [(x[0], x)], curmx[1]
            if len(x[1]) > 0: print(x[1])
            for y in range(origsz):
                cury = confsnor[y]
                if cury[1][-1] in x[1]: continue
                l = sorted_tuple([*x[1],cury[1][-1]])
                if l in searched: continue
                searched.add(l)
                ybase = cury if len(x[1]) == 0 else process_or(x, cury, l)
                if ybase is None: continue
                yset = set(ybase[1]) #do not need reverse lookup since ybase[1][-1] == index if no sorting
                arr = np.array(list(i for i in range(origsz) if i not in yset))
                curmx = check_max(ybase, ybase, currentmx, curmx)
                for z in range(10):
                    np.random.shuffle(arr)
                    errcount, base = 0, ybase
                    for q in arr:
                        cury = confsnor[q]
                        l = sorted_tuple([*base[1],cury[1][-1]])
                        ret = process_or(base, cury, l)
                        if not ret is None:
                            base = ret
                            curmx = check_max(base, ybase, currentmx, curmx)
                        elif errcount > 3: break
                        else: errcount += 1
            currentmx = currentmx[-maxcheck:] #conserve memory
            for y in currentmx:
                if y[1][1] != x[1] and oldmax != curmx[1]: recurse_or(y[1])
        curmx, searched, origsz = confsnor[0], set(), len(confsnor)
        recurse_or((0, list(), list()))
        #one idea is a subtractive algorithm which keeps removing the worst value but expensive computationally due to or combination
        #while True:
        #    vals = list()
        #    for x in range(origsz):
        #        for y in range(origsz):
        #            if x == y: continue
        #            if len(vals) >= x: vals.append(confsnor[y][5])
        #            elif len(vals[x]) == origsz-1: vals[x] = process_or(vals[x], confsnor, y)
        #            else: vals[x] = orfunc(vals[x], confsnor[y][5])
        #preds = self._predict(X, pos, [confsn[y] for y in curmx[1]])
        #cm = conf_mat(y, preds)
        cm = conf_mat_packed(cv, curmx[2], sz)
        print(curmx[:2], [[featnamesclose[x] for x in confsn[y]] for y in curmx[1]], cm, extended_conf_mat(cm, partial=True), precision(cm, not self.isPositive), fbetascore(cm, self.beta, not self.isPositive), fbetascore(cm, self.beta, self.isPositive), mcc(cm), informedness(cm), kappa(cm))
        return curmx[0], [confsn[y] for y in curmx[1]]
    def predict(self, X):
        if self.isPositive is None:
            #if True False -> True
            #if True True -> Indeterminate -> False #bias towards negative when uncertain
            #if False True -> False
            #if False False -> Indeterminate -> False
            #out = self._predict(X, True, self.booleqs_[0]) & ~self._predict(X, False, self.booleqs_[1])
            out = self._predict(X, self.scores_[0] >= self.scores_[1], self.booleqs_[0 if self.scores_[0] >= self.scores_[1] else 1])
        else:
            out = self._predict(X, self.isPositive, self.booleqs_)
        return out
    def _predict(self, X, pos, eqs):
        #transpose = np.transpose(np.array(X))
        #transpose = np.array(list(map(list, zip(*X))))
        sz = len(X)
        andfunc = np.logical_and if pos else np.logical_or
        orfunc = np.logical_or if pos else np.logical_and
        initzeros = np.zeros if pos else np.ones
        initones = np.ones if pos else np.zeros
        out = initzeros(sz, dtype=bool)
        for y in eqs:
            if len(y) == 0: continue
            col = initones(sz, dtype=bool)
            for z in y:
                col = andfunc(col, X[:,z])
            out = orfunc(out, col)
        return out
    def score(self, X, y):
        if self.isPositive is None:
            preds = self._predict(X, True, self.booleqs_[0])
            cm = conf_mat(y, preds)
            print(self.scores_[0], extended_conf_mat(cm, partial=True))
            preds = self._predict(X, False, self.booleqs_[1])
            cm = conf_mat(y, preds)
            print(self.scores_[1], extended_conf_mat(cm, partial=True))
        preds = self.predict(X)
        cm = conf_mat(y, preds)
        print(extended_conf_mat(cm, partial=True), precision(cm, not self.isPositive), fbetascore(cm, self.beta, not self.isPositive), fbetascore(cm, self.beta, self.isPositive), mcc(cm), informedness(cm), kappa(cm))
        return fbetascore(cm, self.beta, not self.isPositive) if self.onesided else mcc(cm)
    def get_params(self, deep=True):
        return {'beta':self.beta, 'minmatch_pct':self.minmatch_pct,
                'isPositive':self.isPositive, 'onesided':self.onesided}
    def set_params(self, **params):
        if 'beta' in params: self.beta = params['beta']
        if 'minmatch_pct' in params: self.minmatch_pct = params['minmatch_pct']
        if 'isPositive' in params: self.isPositive = params['isPositive']
        if 'onesided' in params: self.onesided = params['onesided']
        return self


# Preserve the public preprocessing helpers shipped by LADClassifier 0.0.1.
# The production classifier now delegates fitting to DiscretizingTransformer,
# but existing callers still use these methods directly.
from ._legacy_2019 import LADClassifier as _LegacyLADClassifier2019

for _legacy_helper_name in (
    "binarizer",
    "binarize",
    "binarizeall",
    "postbinarize",
    "binarizecompare",
):
    setattr(
        LADClassifier,
        _legacy_helper_name,
        staticmethod(getattr(_LegacyLADClassifier2019, _legacy_helper_name)),
    )

del _legacy_helper_name
del _LegacyLADClassifier2019

import numba
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, MultiOutputMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import check_array
from sklearn.metrics import accuracy_score
from scipy.sparse import issparse
"""
import cProfile
cProfile.run('import lad; lad.test_lad()', 'ladstats', sort='tottime')
import pstats
p = pstats.Stats('ladstats')
p.strip_dirs().sort_stats('tottime').print_stats()
"""
class LADClassifier(ClassifierMixin, MultiOutputMixin, BaseEstimator):
    def __init__(self, degree=4, random=True, maxcombs=2000, threshold_pct=0.9,
                 minmatch_pct=0.001, feature_names=None, binarizer_params=None,
                 remaining_label=None, mutual_exclusions=[]):
        self.degree = degree
        self.random = random
        self.maxcombs = maxcombs
        self.threshold_pct = threshold_pct
        self.minmatch_pct = minmatch_pct
        self.feature_names = feature_names
        self.binarizer_params = binarizer_params
        self.remaining_label = remaining_label #priority order also could add, could be based on score or user defined order
        #self.mutual_exclusions = mutual_exclusions
        self._estimator_type = 'classifier' #needed for stratified k-folds in GridSearchCV
    #def _get_tags(self): return {'poor_score':True,'multioutput':True}
    def binarizer(self, data, method='minimumdifferentiated', divisions=10,
                  mn=-1, mx=-1, splitpoints=[], binarymode=True):
        if method == 'equaldivisions':
            dist = (mx - mn) / divisions
            if binarymode:
                cond = [((data >= splitpoints[j][0]) if j != 0 else True) &
                     ((data < splitpoints[j][1]) if j != len(splitpoints)-1 else True) for j in range(divisions)]
            else:
                cond = np.zeros(len(data), dtype=np.uint32)                
                for j in range(divisions):
                    cond[((data >= mn + dist * j) if j != 0 else True) &
                         ((data < mn + dist * (j+1)) if j != divisions-1 else True)] = j
        else:
            if binarymode:
                cond = [((data >= splitpoints[j][0]) if j != 0 else True) &
                         ((data < splitpoints[j][1]) if j != len(splitpoints)-1 else True) for j in range(len(splitpoints))]
            else:
                cond = np.zeros(len(data), dtype=np.uint32)
                for j in range(len(splitpoints)):
                    cond[((data >= splitpoints[j][0]) if j != 0 else True) &
                         ((data < splitpoints[j][1]) if j != len(splitpoints)-1 else True)] = j
        return cond
    #equal divisions, equal distribution, minimum differentiated ranges in output
    #['equaldivisions', 'equaldistribution', 'minimumdifferentiated']
    def binarize(self, data, name, y, method='minimumdifferentiated', divisions=10, binarymode=True):
        if method == 'equaldivisions':
            mn, mx = min(data), max(data)
            dist = (mx - mn) / divisions
            binvals = {'method':method, 'divisions':divisions, 'mn':mn, 'mx':mx, 'binarymode':binarymode}
            featnames = [name + ('>=' + str(round(mn + dist * j, 2)) if j != 0 else '') +
                     ('<' + str(round(mn + dist * (j+1), 2)) if j != divisions-1 else '')
                     for j in range(divisions)]
        elif method == 'equaldistribution': #need to handle splits on equivalence groups, right now redundant or duplicate values possible
            sz, sorted = len(data), np.sort(data)
            divs = [(sorted[int(sz * j / divisions)], sorted[int(sz * (j+1) / divisions)-(1 if j == len(divisions)-1 else 0)])
                    for j in range(divisions)]
            binvals = {'method':method, 'divisions': len(divs), 'splitpoints':divs, 'binarymode':binarymode}
            featnames = [name + ('>=' + str(round(divs[j][0], 2)) if j != 0 else '') +
                     ('<' + str(round(divs[j][1], 2)) if j != len(divs)-1 else '')
                     for j in range(len(divs))]
        elif method == 'minimumdifferentiated':
            sorted = list(zip(data, y))
            sorted.sort()
            divs, featnames, base = [], [], 0
            #cannot ignore duplicate values or could wrongly collapse to single division
            x = -1
            while x < len(sorted):
                eqvcls, y = False, x + 1
                while y < len(sorted) - 1:
                    if sorted[y][0] != sorted[y+1][0]: break
                    if sorted[y][1] != sorted[y+1][1]: eqvcls = True
                    y += 1
                if x == len(sorted)-1 or eqvcls or x != -1 and sorted[x][1] != sorted[y][1] and sorted[x][0] != sorted[y][0]:
                    divs.append((sorted[base][0], sorted[y if x != len(sorted)-1 else x][0]))
                    #condarr.append((data >= sorted[base][0]) & (data <= sorted[x][0]))
                    featnames.append(name + (('>=' + str(round(sorted[base][0], 2))) if len(divs) != 1 else '') +
                                     ('<' + str(round(sorted[y][0], 2)) if x != len(sorted)-1 else ''))
                    base = y
                x = y
            binvals = {'method':method, 'divisions':len(divs), 'splitpoints':divs, 'binarymode':binarymode}
        if binvals['binarymode']: featnames = [['!' + x, x] for x in featnames]
        return binvals, self.binarizer(data, **binvals), featnames
    def binarizeall(self, X, y, feature_names, binarizer_params):
        feature_names = ['Feature' + str(x+1) for x in range(X.shape[1])] if feature_names is None else feature_names
        condarr, featnames, binvals, bounds = [], [], [], []
        for i in range(X.shape[1]):
            if X[:,i].dtype is np.dtype(np.bool_):
                condarr.append(X[:,i]), featnames.append(['!' + feature_names[i], feature_names[i]]), binvals.append(None), bounds.append(2)
            else:
                binparams = {} if binarizer_params is None else (binarizer_params[i] if type(binarizer_params) is list else binarizer_params)
                vals, conds, feats = self.binarize(X[:,i], feature_names[i], y, **binparams)
                binvals.append(vals)
                if vals['binarymode']:
                    condarr.extend(conds), featnames.extend(feats), bounds.extend([2] * len(conds))
                else:
                    condarr.append(conds), featnames.append(feats), bounds.append(vals['divisions'])
                #mutex.append(np.arange(len(condarr)-len(conds), len(condarr)))
        return np.array(condarr).transpose(), featnames, binvals, bounds
    def postbinarize(self, X, binarizer_values):
        condarr = []
        for i in range(X.shape[1]):
            if binarizer_values[i] is None:
                condarr.append(X[:,i])
            elif binarizer_values[i]['binarymode']:
                condarr.extend(self.binarizer(X[:,i], **binarizer_values[i]))
            else:
                condarr.append(self.binarizer(X[:,i], **binarizer_values[i]))
        return np.array(condarr).transpose()
    #['lt', 'eq', 'gt', 'lte', 'neq', 'gte']
    def binarizecompare(self, X, feature_names, featcomp, operations=['lt', 'eq', 'gt']):
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
    def _testpaper(self):
        #https://www.sciencedirect.com/science/article/pii/S0166218X05003161
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
    def fit(self, X, y):
        #binarization
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
        condarr, self.featnames_, self.binarizer_values_, self.bounds_ = self.binarizeall(X, y, self.feature_names, self.binarizer_params)
        #self.mutex_ = self.mutual_exclusions[:] + self.mutex_
        #self.mutex_ = {y:x for x in self.mutex_ for y in x}
        self.outtype_ = y.dtype
        if y.ndim == 1:
            self.n_outputs_ = 1
            self.classes_ = np.unique(y)
            self.n_classes_ = self.classes_.shape[0]
            remaining_label = self.classes_[-1] if self.remaining_label is None else self.remaining_label
            self.booleqs_ = {} #(prefer positive, positive patterns, negative patterns)
            for c in self.classes_:
                if c == remaining_label:
                    self.booleqs_[c] = ()
                    continue
                self.booleqs_[c] = self._fit(condarr, y == c)
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
                remaining_label = self.classes_[k][-1] if self.remaining_label is None else (self.remaining_label[k] if type(self.remaining_label) is list else self.remaining_label)
                for c in self.classes_[k]:
                    if c == remaining_label:
                        self.booleqs_[k][c] = ()
                        continue
                    self.booleqs_[k][c] = self._fit(condarr, y[:, k] == c)
        return self
    def _fit(self, X, y): #in DNF, if want CNF, can negate X and y per DeMorgan's law?
        noty = np.logical_not(y)
        posvals, negvals = X[y,:], X[noty,:]
        minmatch = int(len(y) * self.minmatch_pct)
        import itertools
        def prec_penalty(precision, featpct): #reduce precision based on number of features
            a = 10000000
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
                view = X.swapaxes(0, i)
                view = np.cumsum(view[::-1], axis=0)[::-1]
                X = view.swapaxes(0, i)               
                #X = np.apply_along_axis(sum_func, i, X)
            return X
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
        def add_permute(cmb, sigmaI, pos, neg, pats, negpats):
            if sigmaI <= 1 - self.threshold_pct and neg >= minmatch: do_add_permute(cmb, 1 - sigmaI, neg, negpats) #negative pattern
            if sigmaI >= self.threshold_pct and pos >= minmatch: do_add_permute(cmb, sigmaI, pos, pats) #positive pattern
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
        def calc_permute(comb, pats, negpats):
            #deg = len(comb)
            #gcodes = extended_gray_code(tuple([1] * deg))
            r, rn = posvals[:,comb], negvals[:,comb]
            #if deg == 1: #fast route for single features
            #    pos, neg = np.sum(r), np.sum(rn)
            #    tot = pos + neg
            #    if tot == 0: return
            #    add_permute(comb, pos / tot, pos, neg, pats, negpats)
            #else:
            bounds = tuple([self.bounds_[x] for x in comb])
            if bounds in gcodesdict: gcodes = gcodesdict[bounds]
            else:
                gcodes = extended_gray_code(tuple([x - 1 for x in bounds]))
                gcodesdict[bounds] = gcodes
            #print(bounds)
            M, Mneg = np.zeros(bounds, dtype=np.uint32), np.zeros(bounds, dtype=np.uint32)
            #for x in r.astype(np.uint32): M[tuple(x)] += 1
            np.add.at(M, tuple(r.T.astype(np.uint32)), 1)
            #for x in rn.astype(np.uint32): Mneg[tuple(x)] += 1
            np.add.at(Mneg, tuple(rn.T.astype(np.uint32)), 1)
            PiV0, PiV0neg = calc_PI_V0(M), calc_PI_V0(Mneg) #start at tuple([1] * deg)
            #idx = tuple([1] * deg)
            b = gcodes[0][0] #initial PI_V0 index
            for gcode in range(len(gcodes)):
                #i = np.ndindex(bounds)
                curgcode = gcodes[gcode]
                V = curgcode[0]
                i = [tuple(V)]
                p = np.nonzero(V == b)[0]
                for d in range(1, min(len(p)+1, len(V))): #len(V) is only 1 permutation where its the zero-length combination
                    for c in itertools.combinations(p, d):
                        Vnew = np.array(V)
                        Vnew[list(c)] = 0
                        i.append(tuple(Vnew)) #np.where([q in c for q in range(len(V))], 0, V)
                for idx in i: #if mutually exclusive, the not values of interval are useless, only need to check where V and idx intersect
                    #if the entire interval, skip as its a zero-length combination
                    #if not np.any(idx == V): continue
                    cmb = np.array(comb)
                    #cmb[V == 0] = ~cmb[V == 0]
                    same = V == idx
                    cmb = list(zip(cmb[same], V[same]))
                    #cmb = [(cmb[x], Vidx[x]) for x in range(len(comb)) if Vidx[x] == idx[x]] #interval from idx[x] to V[x]
                    pos, neg = PiV0[idx], PiV0neg[idx]
                    tot = pos + neg
                    #preds = self._predict(X, [cmb[np.array(idx == V, dtype=np.bool_)]])
                    #cm = conf_mat(y, preds)
                    #assert(cm[1,1] == pos and cm[0,1] == PiV0neg[idx])
                    if tot != 0:
                        add_permute(cmb, pos / tot, pos, neg, pats, negpats)
                if gcode == len(gcodes) - 1: break
                nextgcode = gcodes[gcode+1]
                istar = nextgcode[1]
                Vistar, Vprimeistar, Tistar = V[istar], nextgcode[0][istar], nextgcode[2][istar]
                PiV0 = calc_PI_VI(Vistar, Vprimeistar, istar, Tistar, PiV0)
                PiV0neg = calc_PI_VI(Vistar, Vprimeistar, istar, Tistar, PiV0neg)
        gcodesdict, degree = {}, min(self.degree, X.shape[1])
        if np.all(X.shape == 2):
            gcodesdict[tuple([2] * degree)] = extended_gray_code(tuple([1] * degree))
        permute = np.arange(X.shape[1], dtype=np.int32)
        patset = set()
        pats, negpats = [], [] #numba.typed.List()
        pats.append((np.float64(0), np.uint32(0), np.array([], dtype=np.int32)))
        negpats.append((np.float64(0), np.uint32(0), np.array([], dtype=np.int32)))
        if self.random and X.shape[1] > degree:
            for _ in range(self.maxcombs):
                np.random.shuffle(permute)
                for i in range(0, len(permute), degree):
                    calc_permute(permute[i:i+degree], pats, negpats)
            cur, curneg, c = len(posvals), len(negvals), 0
            #cantfind = set()
            print(cur, curneg)
            while True:
                if len(pats) == 1: permute = np.nonzero(posvals[0])[0]
                else:
                    if cur != 0:
                        preds = self._predict(X, [x[2] for x in pats])
                        remaining = np.nonzero(~preds & y)[0] #[x for x in np.nonzero(~preds & y)[0] if x not in cantfind]
                    preds = ~self._predict(X, [x[2] for x in negpats])
                    remainingneg = np.nonzero(preds & ~y)[0] #[x for x in np.nonzero(preds & ~y)[0] if x not in cantfind]
                    if len(remaining) == 0 and len(remainingneg) == 0: break
                    permute = np.array(list(set(np.concatenate([np.nonzero(X[remaining[x]])[0] for x in range(len(remaining))] + [np.nonzero(X[remainingneg[x]])[0] for x in range(len(remainingneg))]))))
                    if len(remaining) != cur or len(remainingneg) != curneg:
                        cur, curneg = len(remaining), len(remainingneg)
                        print(cur, curneg, c, len(permute))
                    #if cur != 0:
                    #    permute = np.nonzero(X[remaining[0]])[0]
                    #else:
                    #    permute = np.nonzero(X[remainingneg[0]])[0]
                curlen, curneglen, startc = len(pats), len(negpats), c
                while curlen == len(pats) and curneglen == len(negpats) and startc + self.maxcombs > c:
                    np.random.shuffle(permute)
                    for i in range(0, len(permute), degree):
                        calc_permute(permute[i:i+degree], pats, negpats)
                    c += 1
                if startc + self.maxcombs <= c: break
                #    cantfind.add(remaining[0] if cur != 0 else remainingneg[0])
        else:
            for comb in itertools.combinations(permute, degree):
                calc_permute(comb, pats, negpats)
        #print(pats, negpats)
        eqs = ([x[2] for x in pats], [x[2] for x in negpats])
        preds = self._predict(X, eqs[0])
        #cm = conf_mat(y, preds)
        negpreds = ~self._predict(X, eqs[1])
        #cmneg = conf_mat(y, negpreds)
        preferpos = accuracy_score(y, preds) >= accuracy_score(y, negpreds)
        #print(minmatch, accuracy_score(y, preds), accuracy_score(y, negpreds), pats, negpats)
        #print(preferpos, cm, mcc(cm), cmneg, mcc(cmneg))
        return (preferpos, eqs[0], eqs[1])
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
        X_ = self.postbinarize(X, self.binarizer_values_)
        if type(self.booleqs_) is list:
            out = np.zeros((len(X), len(self.booleqs_)), dtype=self.outtype_)
            for n, booleqs in enumerate(self.booleqs_):
                cumpreds = np.zeros(len(X), dtype=np.bool_)
                for k, v in booleqs.items():
                    if len(v) == 0:
                        remain = k
                        continue
                    elif v[0]:
                        preds = self._predict(X_, v[1])
                    else:
                        preds = ~self._predict(X_, v[2])
                    out[preds == True, n] = k
                    cumpreds = np.logical_or(preds, cumpreds)
                out[cumpreds == False, n] = remain
        else:
            out = np.zeros(len(X), dtype=self.outtype_)
            cumpreds = np.zeros(len(X), dtype=np.bool_)
            for k, v in self.booleqs_.items():
                if len(v) == 0:
                    remain = k
                    continue
                elif v[0]:
                    preds = self._predict(X_, v[1])
                else:
                    preds = ~self._predict(X_, v[2])
                out[preds == True] = k
                cumpreds = np.logical_or(preds, cumpreds)
            out[cumpreds == False] = remain
        return out
    def score(self, X, y, sample_weight=None):
        preds = self.predict(X)
        #from sklearn.metrics import confusion_matrix #, matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score #with adjusted=True
        #cm = confusion_matrix(y, preds)
        #print(cm)
        #print(extended_conf_mat(cm, partial=True), precision(cm), fbetascore(cm, 1), fbetascore(cm, 1), mcc(cm), informedness(cm), kappa(cm))
        #return mcc(cm)
        return accuracy_score(y, preds, sample_weight=sample_weight)
    def format_booleqs(self):
        check_is_fitted(self, attributes='booleqs_')
        def do_format_booleqs(booleqs):
            return {k:(v[0], [[self.featnames_[x[0]][x[1]] for x in y] for y in v[1]],
                    [[self.featnames_[x[0]][x[1]] for x in y] for y in v[2]]) if len(v) != 0 else () for k, v in booleqs.items()}
        if type(self.booleqs_) is list:
            return [do_format_booleqs(x) for x in self.booleqs_]
        else:
            return do_format_booleqs(self.booleqs_)
    def get_params(self, deep=True):
        return {'degree': self.degree, 'random': self.random,
                'maxcombs': self.maxcombs, 'threshold_pct': self.threshold_pct,
                'minmatch_pct': self.minmatch_pct, 'feature_names': self.feature_names,
                'binarizer_params':self.binarizer_params, 'remaining_label': self.remaining_label}
                #'mutual_exclusions':self.mutual_exclusions}
    def set_params(self, **params):
        if 'degree' in params: self.degree = params['degree']
        if 'random' in params: self.random = params['random']
        if 'maxcombs' in params: self.maxcombs = params['maxcombs']
        if 'threshold_pct' in params: self.threshold_pct = params['threshold_pct']
        if 'minmatch_pct' in params: self.minmatch_pct = params['minmatch_pct']
        if 'feature_names' in params: self.feature_names = params['feature_names']
        if 'binarizer_params' in params: self.binarizer_params = params['binarizer_params']
        if 'remaining_label' in params: self.remaining_label = params['remaining_label']
        #if 'mutual_exclusions' in params: self.mutual_exclusions = params['mutual_exclusions']
        return self

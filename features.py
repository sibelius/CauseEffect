import numpy as np
from sklearn.base import BaseEstimator
from scipy.special import psi
from scipy.stats.stats import pearsonr
from igci import igci
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
import math
import rpy2.robjects as robjects

r = robjects.r

class FeatureMapper:
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        for feature_name, column_names, extractor in self.features:
            extractor.fit(X[column_names], y)

    def transform(self, X):
        extracted = []
        for feature_name, column_names, extractor in self.features:
            fea = extractor.transform(X[column_names])
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        if len(extracted) > 1:
            return np.concatenate(extracted, axis=1)
        else:
            return extracted[0]

    def fit_transform(self, X, y=None):
        extracted = []
        for feature_name, column_names, extractor in self.features:
            print feature_name
            fea = extractor.fit_transform(X[column_names], y)
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        if len(extracted) > 1:
            return np.concatenate(extracted, axis=1)
        else:
            return extracted[0]
	def get_params(self, deep=True):
		return dict(features=self.features)

def identity(x):
    return x

def count_unique(x):
    return len(set(x))

def normalized_entropy(x):
    x = (x - np.mean(x)) / np.std(x)
    x = np.sort(x)

    hx = 0.0;
    for i in range(len(x)-1):
        delta = x[i+1] - x[i];
        if delta != 0:
            hx += np.log(np.abs(delta));
    hx = hx / (len(x) - 1) + psi(len(x)) - psi(1);

    return hx

def entropy_difference(x, y):
    return normalized_entropy(x) - normalized_entropy(y)

def correlation(x, y):
    return pearsonr(x, y)[0]

def correlation_magnitude(x, y):
    return abs(correlation(x, y))

# igci uniform + entropy
def igci11(x,y):
    return igci(x,y,1,1)

# igci uniform + integral approximation
def igci12(x,y):
    return igci(x,y,1,2)
# igci Gaussian + entropy
def igci21(x,y):
    return igci(x,y,2,1)

# igci Gaussian + integral approximation
def igci22(x,y):
	return igci(x,y,2,2)

# igci baseado no tamanho da amostra
def n_igci11(x,y):
	return len(x) * igci(x,y,1,1)

def n_igci12(x,y):
	return len(x) * igci(x,y,1,2)

def n_igci21(x,y):
	return len(x) * igci(x,y,2,1)

def n_igci22(x,y):
	return len(x) * igci(x,y,2,2)


# Media
def media(x):
	return np.mean(x)

# Mediana
def mediana(x):
	return np.median(x)

# Intervalo
def intervalo(x):
	return np.ptp(x)

# Desvio Padrao
def desvio_padrao(x):
	return np.std(x)

# Variancia
def variancia(x):
	return np.var(x)

############################################
# CauseEffect code
def normalize(x):
    x = (x - np.mean(x)) / np.std(x)
    x = np.sort(x)
    return x

def quadregress(x,y):

    ux = set(x)
    uy = set(y)

    nx = len(ux)
    ny = len(uy)
    if( nx < 30 or ny < 30 ):
        return 0.5
    ab = doquadregress(x,y)
    ba = doquadregress(y,x)
    if(  ab > ba ):
        return ab
    else:
        return -ba

def doquadregress(x,y):
    x = x.tolist()
    y = y.tolist()
    rx = robjects.FloatVector(x)
    ry = robjects.FloatVector(y)
    robjects.globalenv["rx"] = rx
    robjects.globalenv["ry"] = ry
    lm = r.lm("ry ~ rx + I(rx^2)")
    rs = r.summary(lm).rx2("adj.r.squared")[0]
    return rs


def polyregress(x,y):

    ux = set(x)
    uy = set(y)

    nx = len(ux)
    ny = len(uy)
    if( nx < 30 or ny < 30 ):
        return 0.5
    ab = dopolyregress(x,y)
    ba = dopolyregress(y,x)
    if(  ab > ba ):
        return ab
    else:
        return -ba

def dopolyregress(x,y):
    x = x.tolist()
    y = y.tolist()
    rx = robjects.FloatVector(x)
    ry = robjects.FloatVector(y)
    robjects.globalenv["rx"] = rx
    robjects.globalenv["ry"] = ry
    lm = r.lm("ry ~ rx + I(rx^2) + I(rx^3)")
    rs = r.summary(lm).rx2("adj.r.squared")[0]
    return rs


def sineregress(x,y):

    ux = set(x)
    uy = set(y)

    nx = len(ux)
    ny = len(uy)
    if( nx < 30 or ny < 30 ):
        return 0.5
    ab = dosineregress(x,y)
    ba = dosineregress(y,x)
    if(  ab > ba ):
        return ab
    else:
        return -ba

def dosineregress(x,y):
    x = x.tolist()
    y = y.tolist()
    rx = robjects.FloatVector(x)
    ry = robjects.FloatVector(y)
    robjects.globalenv["rx"] = rx
    robjects.globalenv["ry"] = ry
    lm = r.lm("ry ~ rx + I(sin(rx))")
    rs = r.summary(lm).rx2("adj.r.squared")[0]
    return rs

def linregress(x,y):
    ux = set(x)
    uy = set(y)

    nx = len(ux)
    ny = len(uy)
    if( nx > 30 or ny > 30 ):
        return 0.5
    gradient, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    return r_value**2

def dist(x):

    ax = np.array(x)

    ax.sort()
    x=ax
    n = len(x)
    x1 = x[range(n-1)]
    x2 = x[range(1,n)]
    diff = x2-x1
    r = diff[ int(n/2 )]
    if r == 0:
        r = (max(x)  - min(x))/len(x)

    return r


def RFscore(x,y):
    r = range(len(x))

    np.random.shuffle(r)
    x = x[r]
    y = y[r]
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)

    x = np.array(x, ndmin=2)
    y = np.array(y, ndmin=2)
    x = x.T
    y = y.T


    xy = RFscore_one(x,y)
    yx = RFscore_one(y,x)
    return xy - yx

def RFscore_one(x,y):
    folds=3
    r = range(len(x))

    np.random.shuffle(r)
    x = x[r]
    y = y[r]
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)

    x = np.array(x, ndmin=2)
    y = np.array(y, ndmin=2)

    x = x.T
    y = y.T

    rf = RandomForestRegressor(n_estimators=50, verbose=0,n_jobs=1,min_samples_split=10,compute_importances=True,random_state=1)
    fit = rf.fit(x,y)

    s = fit.score(x,y)

    cv = cross_validation.KFold(len(x), n_folds=folds, indices=False)
    score = 0
    median = dist(y)
    for traincv, testcv in cv:
        fit = rf.fit(x[traincv], y[traincv])
        score += fit.score(x[testcv], y[testcv])

    score /= folds
    score /= median
    return score





def diffcor_one(x,y):

    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    ax = np.array(x)
    ay = np.array(y)

    order = ax.argsort()
    x = ax[order[::1]]
    y = ay[order[::1]]

    n = len(x)
    y1 = y[range(n-1)]
    y2 = y[range(1,n)]
    diff = y2-y1
    x1 = x[range(n-1)]
    return correlation_magnitude(x1,diff)

def diffcor(x,y):
    a = diffcor_one(x,y)
    b = diffcor_one(y,x)
    return a-b


def categoric_cond_prob(x,y):
    ux = set(x)
    uy = set(y)

    nx = len(ux)
    ny = len(uy)
    if( nx > 30 or ny > 30 ):
        return 0.5
    counts = np.zeros((nx,ny))
    ix ={}
    k=0
    for thisx in sorted(list(ux)):
        ix[thisx] = k
        k = k + 1

    iy ={}
    k=0
    for thisy in sorted(list(uy)):
        iy[thisy] = k
        k = k + 1

    for i in range(len(x)):
        counts[ ix[ x[i] ], iy[ y[i]]] += 1

    return likelihood(counts, len(x))



def categoric(x,y):
    ux = set(x)
    uy = set(y)

    nx = len(ux)
    ny = len(uy)
    if( nx > 30 or ny > 30 ):
        return 0.5
    counts = np.zeros((nx,ny))
    ix ={}
    k=0
    for thisx in sorted(list(ux)):
        ix[thisx] = k
        k = k + 1

    iy ={}
    k=0
    for thisy in sorted(list(uy)):
        iy[thisy] = k
        k = k + 1

    for i in range(len(x)):
        counts[ ix[ x[i] ], iy[ y[i]]] += 1

    return prob_indep(counts, len(x))

def slope_entropy_rev(x, y):
    return slope_entropy(x,y)

def slope_entropy(x, y):
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    ax = np.array(x)
    ay = np.array(y)

    order = ax.argsort()
    x = ax[order[::1]]
    y = ay[order[::1]]

    hx = 0.0;
    for i in range(len(x)-1):
        deltax = x[i+1] - x[i]
        deltay = y[i+1] - y[i]
        if( (deltax != 0) and (deltay != 0) ):
            hx += np.log(np.abs(deltay/deltax))
    hx = hx / (len(x) - 1)

    return hx

def slope_entropy_diff(x,y):
    return slope_entropy(x,y) - slope_entropy(x,y)

def midpoints(data):
    data = np.sort(list(set(data)))
    n = len(data)
    mid=[0] * (1 + n)
    mid[0] = data[0] - (data[1] - data[0])/2.0
    for i in range(1, n):
        mid[i] = data[i] - (data[i] - data[i-1]) / 2.0
    mid[n] = data[n-1] + (data[n-1] - data[n-2])/2.0
    return mid

def kernel(x,y, z):
    ux = count_unique(x)
    uy = count_unique(y)
    if( ux < 20 ):
        return 2
    if( uy < 20 ):
        return 3
    midx = midpoints(x)
    midy = midpoints(y)
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    # Perform a kernel density estimate on the data:
    shash = str(ux) + "-" + str(uy) + "-" + str(len(x)) + "-" + str(len(y)) + "-" + str(xmax) + "-" + str(ymax)
    if FeatureMapper.kernel_cache != None and FeatureMapper.kernel_cache.has_key(shash):
        kernel = FeatureMapper.kernel_cache.get(shash)
    else:
        values = np.vstack([x, y])
        kernel = stats.gaussian_kde(values)
        FeatureMapper.kernel_cache[shash] = kernel

    plot="no"
    area = (xmax - xmin) * (ymax-ymin) / (30 * 30 )

    if FeatureMapper.z_cache != None and FeatureMapper.z_cache.has_key(shash):
        Z = FeatureMapper.z_cache.get(shash)
    else:
        X, Y = np.mgrid[xmin:xmax:30j, ymin:ymax:30j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kernel(positions).T, X.shape)
        FeatureMapper.z_cache[shash] = Z

    prob = prob_indep(Z, len(x), area, 30)
    print "Feature:indep " + z + " " + str(prob)
    if plot == "yes":
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
        ax.plot(x,y, 'k.', markersize=2)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
    return prob

def cond_prob(x, y, z):

    ux = count_unique(x)
    uy = count_unique(y)
    if( ux < 20 ):
        return 0.5
    if( uy < 20 ):
        return 0.5
    midx = midpoints(x)
    midy = midpoints(y)
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    # Perform a kernel density estimate on the data:
    shash = str(ux) + "-" + str(uy) + "-" + str(len(x)) + "-" + str(len(y)) + "-" + str(xmax) + "-" + str(ymax)
    if FeatureMapper.kernel_cache != None and FeatureMapper.kernel_cache.has_key(shash):
        kernel = FeatureMapper.kernel_cache.get(shash)
    else:
        values = np.vstack([x, y])
        kernel = stats.gaussian_kde(values)
        FeatureMapper.kernel_cache[shash] = kernel

    plot="no"
    area = (xmax - xmin) * (ymax-ymin) / (30 * 30 )

    if FeatureMapper.z_cache != None and FeatureMapper.z_cache.has_key(shash):
        Z = FeatureMapper.z_cache.get(shash)
    else:
        X, Y = np.mgrid[xmin:xmax:30j, ymin:ymax:30j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kernel(positions).T, X.shape)
        FeatureMapper.z_cache[shash] = Z

    prob = likelihood(Z, len(x), area, 30)

    print "Feature:cond_prob " + z + " " + str(prob)
    return prob


def log_upsilon(n, counts):
    k=0
    result=0
    for index, c in np.ndenumerate(counts):
        result = result + math.lgamma(c+1)
        k = k + 1

    result = result + math.lgamma(k) - math.lgamma(k+n)
    return(result)



def likelihood(Z, n):


    rowcounts=np.sum(Z, axis=0)
    colcounts=np.sum(Z, axis=1)
    bycell=log_upsilon(n, Z)
    byrow=log_upsilon(n, rowcounts)
    bycol=log_upsilon(n, colcounts)

    likelihoodA = bycol
    for i in range(Z.shape[1]):
        counts = Z[:,i]
        thisn = np.sum(counts)
        likelihoodA = likelihoodA + log_upsilon(thisn, counts)

    likelihoodB = byrow
    for i in range(Z.shape[0]):
        counts = Z[i,:]
        thisn = np.sum(counts)
        likelihoodB = likelihoodB + log_upsilon(thisn, counts)

    try:
        ratio = math.exp(likelihoodA - likelihoodB)
    except OverflowError:
        return 0
    r = (1/( 1+ratio))
    return r




def prob_indep(Z, n):
    rowcounts=np.sum(Z, axis=0)
    colcounts=np.sum(Z, axis=1)
    bycell=log_upsilon(n, Z)
    byrow=log_upsilon(n, rowcounts)
    bycol=log_upsilon(n, colcounts)

    logratio = bycell - byrow - bycol
    try:
        ratio = math.exp(logratio)
    except OverflowError:
        return 0

    prior = 0.5
    prob = 1/( 1 + ((1-prior) / prior) * ratio   )
    return prob























#def hsic(x,y):
#	octave = op.Oct2Py()
#	octave.put('x', x)
#	octave.put('y', y)
#	octave.run('h = hsic(x\', y\');');
#	h = octave.get('h')
#	return h

class SimpleTransform(BaseEstimator):
    def __init__(self, transformer=identity):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return np.array([self.transformer(x) for x in X], ndmin=2).T

class MultiColumnTransform(BaseEstimator):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return np.array([self.transformer(*x[1]) for x in X.iterrows()], ndmin=2).T

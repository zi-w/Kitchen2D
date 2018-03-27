# Author: Zi Wang
import numpy as np
import scipy.optimize
from sklearn.utils import shuffle
import cPickle as pickle
from sklearn.metrics import confusion_matrix
import os
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import truncnorm
from functools import partial
EPS = 1e-4


def get_pour_context(expid):
    '''
    Returns the context of pouring in the experiment with ID expid.
    '''
    _, _, c_pour = get_xx_yy(expid, 'gp_lse', exp='pour')
    pour_to_w, pour_to_h, pour_from_w, pour_from_h = c_pour
    return pour_to_w, pour_to_h, pour_from_w, pour_from_h

def get_scoop_context(expid):
    '''
    Returns the context of scooping in the experiment with ID expid.
    '''
    _, _, c_scoop = get_xx_yy(expid, 'gp_lse', exp='scoop')
    scoop_w, scoop_h = c_scoop
    return scoop_w, scoop_h

def get_xx_yy(expid, method, exp='pour'):
    '''
    Returns the training data {xx, yy} and the context c of an experiment.
    Args:
        expid: experiment ID.
        method: training method (e.g. 'gp_lse', 'nn_classification', 'nn_regression', 'random').
        exp: experimented action (e.g. 'scoop', 'pour', 'push').
    '''
    dirnm = 'data/'
    fnm_prefix = os.path.join(dirnm, exp)
    initx, inity = pickle.load(open('{}_init_data_{}.pk'.format(fnm_prefix, expid)))
    fnm = '{}_{}_{}.pk'.format(fnm_prefix, method, expid)
    xx, yy, c = pickle.load(open(fnm, 'rb'))
    xx = np.vstack((initx, xx))
    yy = np.hstack((inity, yy))
    return xx, yy, c

def get_func_from_exp(exp):
    '''
    Returns the function func associated with exp.
    Args: 
        exp: experimented action ('scoop' | 'pour').
    '''
    if exp == 'pour':
        from kitchen2d.pour import Pour
        func = Pour()
    elif exp == 'scoop':
        from kitchen2d.scoop import Scoop
        func = Scoop()
    return func

def get_learner_from_method(method, initx, inity, func):
    '''
    Returns an active learner.
    Args:
        method: learning method, including 
            'nn_classification': a classification neural network 
                based learning algorithm that queries the input that has 
                the largest output.
            'nn_regression': a regression neural network based 
                learning algorithm that queries the input that has 
                the largest output.
            'gp_best_prob': a Gaussian process based learning algorithm
                that queries the input that has the highest probability of 
                having a positive function value.
            'gp_lse': a Gaussian process based learning algorithm called
                straddle algorithm. See B. Bryan, R. C. Nichol, C. R. Genovese, 
                J. Schneider, C. J. Miller, and L. Wasserman, "Active learning for 
                identifying function threshold boundaries," in NIPS, 2006.
            'random': an algorithm that query uniformly random samples.
        initx: initial x data 
        inity: initial y data
        func: a scoring function; e.g. Pour in kitchen2d/pour.py.
    '''
    if method is 'nn_classification':
        from active_learners.active_nn import ActiveNN
        active_learner = ActiveNN(func, initx, inity, 'classification')
    elif method is 'nn_regression':
        from active_learners.active_nn import ActiveNN
        active_learner = ActiveNN(func, initx, inity, 'regression')
    elif method is 'gp_best_prob':
        from active_learners.active_gp import ActiveGP
        active_learner = ActiveGP(func, initx, inity, 'best_prob')
    elif method is 'gp_lse':
        from active_learners.active_gp import ActiveGP
        active_learner = ActiveGP(func, initx, inity, 'lse')
    elif method is 'random':
        from active_learners.active_learner import RandomSampler
        active_learner = RandomSampler(func)
    return active_learner

def process_gp_sample(expid, flag_lk=False, is_adaptive=True, 
                      task_lengthscale=None, exp='pour', betalambda=0.95):
    '''
    Returns the GP learned by level set estimation and the context in an experiment.
    Args:
        expid: experiment ID.
        flag_lk: True if learning kernels along the way, otherwise False. This only 
        applies to diverse sampling.
        is_adaptive: False if sampling diversely; True if sampling adaptively from 
        the feasible region; None if doing rejection sampling with uniform proposal 
        distribution.
    '''
    xx, yy, c = get_xx_yy(expid, 'gp_lse', exp=exp)
    func = get_func_from_exp(exp)
    from active_learners.active_gp import ActiveGP
    gp = ActiveGP(func, xx, yy, 'lse', flag_lk=flag_lk, is_adaptive=is_adaptive,
        task_lengthscale=task_lengthscale, betalambda=betalambda)
    gp.retrain()
    return gp, c

def diversity(xx, active_dim):
    '''
    Returns the diversity of the list xx, with active dimensions active_dim.
    Diversity is measured by log |K/0.01 + I|, where K is the squared 
    exponential gram matrix on xx, with length scale 1.
    '''
    n = len(xx)
    xx = xx[:, active_dim]
    l = np.ones(xx.shape[1]) 
    K = se_kernel(xx, xx, l)
    return np.log(scipy.linalg.det(K/0.01+np.eye(n))) #
def check_close(x, xx):
    '''
    Check if x is close to any item in xx.
    '''
    dist = np.array([np.linalg.norm(x- xp) for xp in xx])
    i = dist.argmin()
    print xx[i]
    print 'dist=', dist[i]

def sample_tgmm(center, scale, n, xmin, xmax):
    '''
    Sample from a truncated Gaussian mixture model (TGMM).
    Returns the samples and their importance weight.
    Args:
        center: center of TGMM.
        scale: scale of TGMM.
        n: number of samples.
        xmin: smallest values of the truncated interval. 
        xmax: largest values of the truncated interval. 
    '''
    dx = len(xmin)
    slen = len(center)
    rd_centers = np.random.choice(slen, (n))
    ta = (xmin - center[rd_centers]) / scale
    tb = (xmax - center[rd_centers]) / scale
    x_samples_gmm = truncnorm.rvs(ta, tb, loc=center[rd_centers], scale=scale)
    
    ta = (xmin - center) / scale
    tb = (xmax - center) / scale
    def truncpdf(j,i):
        return truncnorm.pdf(x_samples_gmm[:,j], ta[i][j], tb[i][j], center[i][j], scale[j])
    
    prob = [np.prod(map(partial(truncpdf, i=i), range(dx)), axis=0) for i in range(slen)]
    prob = np.sum(prob, axis=0) / slen
    np.clip(prob, EPS, 1/EPS)
    return x_samples_gmm, 1./prob


def grid_around_point(p, grange, n, x_range):
    '''
    Returns a list of the points on the grid around point p.
    Args:
        p: the point around which the grid is generated
        grange: a dx vector, each element denotes half length of the grid on dimension d
        n: the number of points on each dimension
    '''
    dx = len(p)
    if not hasattr(n, "__len__"):
        n = [n]*dx
    xmin = [max(p[d] - grange[d], x_range[0, d]) for d in range(dx)]
    xmax = [min(p[d] + grange[d], x_range[1, d]) for d in range(dx)]
    mg = np.meshgrid(*[np.linspace(xmin[d], xmax[d], n[d]) for d in range(dx)])
    grids = map(np.ravel, mg)
    return np.array(grids).T


def grid(n, x_range):
    '''
    p is the point around which the grid is generated
    grange is a dx vector, each element denotes half length of the grid on dimension d
    n is the number of points on each dimension
    '''
    dx = x_range.shape[1]
    if not hasattr(n, "__len__"):
        n = [n]*dx
    xmin, xmax = x_range
    mg = np.meshgrid(*[np.linspace(xmin[d], xmax[d], n[d]) for d in range(dx)])
    grids = map(np.ravel, mg)
    return np.array(grids).T


def global_minimize(f, fg, x_range, n, guesses, callback=None):
    dx = x_range.shape[1]
    tx = np.random.uniform(x_range[0], x_range[1], (n, dx))
    tx = np.vstack((tx, guesses))
    ty = f(tx)
    x0 = tx[ty.argmin()]  # 2d array 1*dx
    if fg is None:
        res = minimize(f, x0, bounds=x_range.T, method='L-BFGS-B', callback=None)
        x_star, y_star = res.x, res.fun
        return x_star, y_star
    else:
        x_star, y_star, _ = scipy.optimize.fmin_l_bfgs_b(
            fg, x0=x0, bounds=x_range.T, maxiter=100, callback=None)
        return x_star, y_star


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def l2_squared(X, X2, l):
    '''
    l2 distance between each pair of items from X, X2, rescaled by l.
    '''
    if X.ndim == 1:
        X = X[None, :]
    if X2.ndim == 1:
        X2 = X2[None, :]
    X = X*l
    X2 = X2*l
    X1sq = np.sum(np.square(X), 1)
    X2sq = np.sum(np.square(X2), 1)
    r2 = -2.*np.dot(X, X2.T) + X1sq[:, None] + X2sq[None, :]
    r2 = np.clip(r2, 0, np.inf)
    return r2

def argmax_min_dist(X, X2, l=None):
    if l is None:
        l = np.ones(X.shape[1])
    r2 = l2_squared(X, X2, l)
    r2 = r2.min(axis=1)
    return r2.argmax()

def se_kernel(X, X2, l):
    '''
    Squared exponential kernel, with inverse lengthscale l.
    '''
    dist = l2_squared(X, X2, l)
    return np.exp(-0.5*dist)

def matern52(X, X2, l):
    '''
    Matern52 kernel with inverse lengthscale l.
    '''
    import GPy as gpy
    kern = gpy.kern.Matern52(X.shape[1], lengthscale=1./l, ARD=True)
    return kern.K(X, X2)

def argmax_condvar(X, X2, l=None):
    '''
    Returns the argmax of conditional variance on X2. The candidates are X.
    l is the inverse length scale of a squared exponential kenel.
    '''
    if l is None:
        l = np.ones(X.shape[1])
    kxx2 = se_kernel(X, X2, l)
    kx2x2 = se_kernel(X2, X2, l) + np.eye(X2.shape[0])*0.01
    factor = scipy.linalg.cholesky(kx2x2)
    negvar = (kxx2 * scipy.linalg.cho_solve((factor, False), kxx2.T).T).sum(axis=1, keepdims=1)
    return negvar.argmin()

def important_d(s, X, l):
    '''
    Returns the most important dimension given that the last sample is s and the samples before
    s is X. l is the inverse length scale of a squared exponential kenel.
    '''
    dx = X.shape[1]
    importance = np.zeros(dx)
    kxx = se_kernel(X, X, l) + np.eye(X.shape[0])*0.01
    factor = scipy.linalg.cholesky(kxx)
    for d in range(dx):
        l2=np.zeros(l.shape)
        l2[d] = l[d]
        ksx = se_kernel(s, X, l2)
        importance[d] = ksx.dot(scipy.linalg.cho_solve((factor, False), ksx.T))
    
    return importance.argmin()

def regression_acc(y_true, y_pred):
    label = y_true > 0
    pred = y_pred > 0
    tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
    acc = (tn + tp)*1.0 / len(label)
    fpr = fp*1.0/(tn + fp)
    fnr = fn*1.0/(tp + fn)
    return acc, fpr, fnr


def gen_data(func, N, parallel=False):
    '''
    Generate N data points on function func.
    Use multiprocessing if parallel is True; otherwise False.
    '''
    X = np.random.uniform(
        func.x_range[0], func.x_range[1], (N, func.x_range.shape[1]))
    if parallel:
        from multiprocessing import Pool
        import multiprocessing
        cpu_n = multiprocessing.cpu_count()
        p = Pool(cpu_n)
        y = np.array(p.map(func, X))
    else:
        y = np.array(map(func, X))
    return X, y

def gen_context(func, N=1):
    '''
    Generate N random contexts associated with function func. 
    '''
    xmin = func.x_range[0, func.context_idx]
    xmax = func.x_range[1, func.context_idx]
    if N == 1:
        return np.random.uniform(xmin, xmax)
    else:
        return np.random.uniform(xmin, xmax, (N, len(func.context_idx)))


def find_closest_positive_context_param(context, xx, yy, param_idx, context_idx):
    '''
    Find the closest data point (in terms of context distance) that has a positive label.
    Args:
        context: current context
        xx: training inputs
        yy: training outpus
        param_idx: index of parameters in an input
        context_idx: index of contexts in an input
    '''
    if yy.ndim == 2:
        yy = np.squeeze(yy)
    positive_idx = yy > 0
    if np.sum(positive_idx) == 0:
        return xx[0, param_idx], xx[0, context_idx]
    xx = xx[positive_idx]
    yy = yy[positive_idx]
    distances = np.linalg.norm(xx[:, context_idx] - context, axis=1)
    idx = distances.argmin()
    return xx[idx, param_idx], xx[idx, context_idx]

def gen_biased_data(func, pos_ratio, N):
    '''
    Generate N data points on function func, with pos_ratio percentage of the 
    data points to have a positive label.
    '''
    pos = []
    neg = []
    i = 0
    while len(pos) < pos_ratio * N or len(neg) < N - pos_ratio * N:
        x = np.random.uniform(func.x_range[0], func.x_range[1])
        y = func(x)
        if y > 0:
            if len(pos) < pos_ratio * N:
                pos.append(np.hstack((x, y)))
        elif len(neg) < N - pos_ratio * N:
            neg.append(np.hstack((x, y)))
    xy = np.vstack((pos, neg))
    xy = shuffle(xy)
    return xy[:, :-1], xy[:, -1]

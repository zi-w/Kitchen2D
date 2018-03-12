# Author: Zi Wang
import numpy as np
import scipy.optimize
from sklearn.utils import shuffle
import cPickle as pickle
from sklearn.metrics import confusion_matrix
import os
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import norm
import GPy as gpy
from active_gp import ActiveGP
from scipy.stats import truncnorm
from functools import partial
EPS = 1e-4

def diversity(xx, active_dim):
    n = len(xx)
    xx = xx[:, active_dim]
    l = np.ones(xx.shape[1]) 
    K = se_kernel(xx, xx, l)
    return np.log(scipy.linalg.det(K/0.01+np.eye(n))) #
def check_close(x, xx):
    dist = np.array([np.linalg.norm(x- xp) for xp in xx])
    i = dist.argmin()
    print xx[i]
    print 'dist=', dist[i]

def sample_gmm(center, scale, n, xmin, xmax):
    dx = len(xmin)
    slen = len(center)
    rd_centers = np.random.choice(slen, (n))
    #x_samples_gmm = np.random.normal(x_samples[rd_centers], scale)
    #x_samples_gmm = np.clip(x_samples_gmm, xmin, xmax)
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
def get_xx_yy(expid, c_i, method, exp='pour'):
    dirnm = 'data/problems/{}_{}/'.format(exp, expid)
    if exp == 'pour':
        fnm_prefix = os.path.join(
            dirnm, 'res_sample_5')
        initx, inity = pickle.load(
            open(os.path.join(dirnm, 'init_data.pk'), 'rb'))
        fnm = '{}_{}_c_{}.pk'.format(fnm_prefix, method, c_i)
        xx, yy, _, _, _, c = pickle.load(open(fnm, 'rb'))
    elif exp == 'push1d':
        fnm_prefix = os.path.join(dirnm, 'res_pos_ratio_{}'.format(0.3))
        initx, inity = pickle.load(
            open(os.path.join(dirnm, 'pos_ratio_0.3.pk'), 'rb'))
        fnm = '{}_{}_c_{}.pk'.format(fnm_prefix, method, c_i)
        xx, yy, _, _, c = pickle.load(open(fnm, 'rb'))
    elif exp == 'push':
        fnm_prefix = os.path.join(
            dirnm, 'res_sample_10_pos_ratio_{}'.format(0.3))
        initx, inity = pickle.load(
            open(os.path.join(dirnm, 'pos_ratio_0.3.pk'), 'rb'))
        fnm = '{}_{}_c_{}.pk'.format(fnm_prefix, method, c_i)
        if c_i > 0:
            fnm = '{}_iters_200_{}_c_{}.pk'.format(fnm_prefix, method, c_i)
        xx, yy, _, _, c = pickle.load(open(fnm, 'rb'))
    elif exp == 'scoop':
        fnm_prefix = os.path.join(
            dirnm, 'res_sample_2')
        initx, inity = pickle.load(
            open(os.path.join(dirnm, 'init_data.pk'), 'rb'))
        fnm = '{}_{}_c_{}.pk'.format(fnm_prefix, method, c_i)
        xx, yy, _, _, _, c = pickle.load(open(fnm, 'rb'))
    xx = np.vstack((initx, xx))
    yy = np.hstack((inity, yy))
    '''
    fnm = '{}_{}_c_{}.pk'.format(fnm_prefix, 'random', c_i)
    initx, inity, _, _, _, c = pickle.load(open(fnm, 'rb'))
    xx = np.vstack((initx, xx))
    yy = np.hstack((inity, yy))
    '''
    return xx,yy,c
def process_gp_sample(expid, c_i, is_adapt=None, is_uniform=True, task_lengthscale=None, exp='pour', betalambda=0.95):
    xx, yy, c = get_xx_yy(expid, c_i, 'gp_lse', exp=exp)
    if exp == 'pour':
        from pour import Pour
        func = Pour()
    elif exp == 'push1d':
        from push import Push1D, Push
        func = Push1D()
    elif exp == 'push':
        from push import Push1D, Push
        func = Push()
    elif exp == 'scoop':
        from scoop import Scoop
        func = Scoop()

    gp = ActiveGP(func, xx, yy, 'lse', is_adapt=is_adapt, is_uniform=is_uniform,
        task_lengthscale=task_lengthscale, betalambda=betalambda)
    gp.retrain()
    return gp, c

def process_nn_sample(expid, c_i, method, exp='pour'):
    from active_nn import ActiveNN
    xx, yy, c = get_xx_yy(expid, c_i, method, exp=exp)
    if exp == 'pour':
        from pour import Pour
        func = Pour()
    elif exp == 'push1d':
        from push import Push1D, Push
        func = Push1D()
    elif exp == 'push':
        from push import Push1D, Push
        func = Push()
        
    if method is 'nn_classification':
        nn = ActiveNN(func, xx, yy, 'classification')
    else:
        nn = ActiveNN(func, xx, yy, 'regression')
    nn.retrain()
    return nn, c
def posa_metric(pos, angle, pos2, angle2):
    d1 = np.linalg.norm(np.array(pos) - np.array(pos2))
    def angle2sincos(angle):
        return np.array([np.sin(angle), np.cos(angle)])
    d2 = np.linalg.norm(angle2sincos(angle) - angle2sincos(angle2))
    return d1 + d2
def grid_around_point(p, grange, n, x_range):
    '''
    p is the point around which the grid is generated
    grange is a dx vector, each element denotes half length of the grid on dimension d
    n is the number of points on each dimension
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
    dist = l2_squared(X, X2, l)
    return np.exp(-0.5*dist)
def matern52(X, X2, l):
    kern = gpy.kern.Matern52(X.shape[1], lengthscale=1./l, ARD=True)
    return kern.K(X, X2)

def argmax_condvar(X, X2, l=None):
    if l is None:
        l = np.ones(X.shape[1])
    kxx2 = se_kernel(X, X2, l) #matern52(X, X2, l)#
    kx2x2 = se_kernel(X2, X2, l) + np.eye(X2.shape[0])*0.01 #matern52(X2, X2, l)
    factor = scipy.linalg.cholesky(kx2x2)
    negvar = (kxx2 * scipy.linalg.cho_solve((factor, False), kxx2.T).T).sum(axis=1, keepdims=1)
    return negvar.argmin()

def argmax_lcb_condvar(X, X2, l=None, alpha=0, beta=0.1):
    if l is None:
        l = np.ones(X.shape[1])
    kxx2 = se_kernel(X, X2, l) #matern52(X, X2, l)#
    kx2x2 = se_kernel(X2, X2, l) + np.eye(X2.shape[0])*0.01 #matern52(X2, X2, l)
    factor = scipy.linalg.cholesky(kx2x2)
    negvar = (kxx2 * scipy.linalg.cho_solve((factor, False), kxx2.T).T).sum(axis=1, keepdims=1)

    return negvar.argmin()

def important_d(s, X, l):
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
    X = np.random.uniform(
        func.x_range[0], func.x_range[1], (N, func.x_range.shape[1]))
    #y = np.array([func(s) for s in X])
    if parallel:
        from multiprocessing import Pool
        import multiprocessing
        cpu_n = multiprocessing.cpu_count()
        p = Pool(cpu_n)
        y = np.array(p.map(func, X))
    else:
        y = np.array(map(func, X))
    return X, y

def gen_data_filterNone(func, N, parallel=False):
    X = np.random.uniform(
        func.x_range[0], func.x_range[1], (N, func.x_range.shape[1]))
    #y = np.array([func(s) for s in X])
    if parallel:
        from multiprocessing import Pool
        import multiprocessing
        cpu_n = multiprocessing.cpu_count()
        p = Pool(cpu_n)
        y = np.array(p.map(func, X))
    else:
        y = np.array(map(func, X))
    return X, y

def gen_context(func, N):
    xmin = func.x_range[0, func.context_idx]
    xmax = func.x_range[1, func.context_idx]
    return np.random.uniform(xmin, xmax, (N, len(func.context_idx)))


def find_closest_positive_context_param(context, xx, yy, param_idx, context_idx):
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

def find_K_closest_positive_context_param(context, K, xx, yy, param_idx, context_idx):
    if yy.ndim == 2:
        yy = np.squeeze(yy)
    positive_idx = yy > 0
    xx = xx[positive_idx]
    yy = yy[positive_idx]
    distances = np.linalg.norm(xx[:, context_idx] - context, axis=1)
    idx = distances.argsort()
    #import pdb; pdb.set_trace()
    return xx[idx[:K]][:, param_idx]

def gen_biased_data(func, pos_ratio, n_data):
    pos = []
    neg = []
    i = 0
    while len(pos) < pos_ratio * n_data or len(neg) < n_data - pos_ratio * n_data:
        x = np.random.uniform(func.x_range[0], func.x_range[1])
        y = func(x)
        if y > 0:
            if len(pos) < pos_ratio * n_data:
                pos.append(np.hstack((x, y)))
        elif len(neg) < n_data - pos_ratio * n_data:
            neg.append(np.hstack((x, y)))
    xy = np.vstack((pos, neg))
    xy = shuffle(xy)
    return xy[:, :-1], xy[:, -1]

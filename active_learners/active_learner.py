from __future__ import print_function, division
import numpy as np
import scipy.optimize
from sklearn.utils import shuffle
import cPickle as pickle
from sklearn.metrics import confusion_matrix
import os
import GPy as gpy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import norm
import time


class ActiveLearner(object):
    def __init__(self):
        pass

    def query(self, context):
        pass

    def sample(self, context, N):
        pass

    def retrain(self, x, y):
        pass


class RandomSampler(ActiveLearner):
    def __init__(self, func):
        self.func = func
        self.name = 'random'

    def query(self, context):
        xmin = self.func.x_range[0, self.func.param_idx]
        xmax = self.func.x_range[1, self.func.param_idx]
        x_star = np.random.uniform(xmin, xmax)
        return np.hstack((x_star, context))

    def sample(self, context, N=1):
        return self.query(context)
    def reset_sample(self):
        pass


def test_activelearner(active_learner, func, contexts, fnm_prefix, num_sample, iters):
    n_c = 1#len(contexts)
    for c_i in range(n_c):
        fnm = '{}_{}_c_{}.pk'.format(fnm_prefix, active_learner.name, c_i)
        evalfnm = '{}_{}_c_{}_eval.pk'.format(fnm_prefix, active_learner.name, c_i)
        c = contexts[c_i]
        xq, yq = None, None
        xx = np.zeros((iters, func.x_range.shape[1]))
        yy = np.zeros(iters)
        sampled_xx = []
        sampled_yy = []
        sampled_xx_ts = []
        sampled_xx_ts2 = []
        i_start = -1

        tot_sample_time = 0
        for i in range(i_start+1, iters):
            active_learner.retrain(xq, yq)
            start = time.time()
            sampled_xx.append(active_learner.sample(c))
            sampled_yy.append(func(sampled_xx[-1]))
            active_learner.reset_sample()
            #sampled_xx_ts.append(active_learner.sample_MH(c, num_sample))
            tot_sample_time += time.time() - start
            xq = active_learner.query(c)
            yq = func(xq)
            xx[i] = xq
            yy[i] = yq
            print('i={}, xq={}, yq={}'.format(i, xq, yq))

            pickle.dump((xx, yy, sampled_xx, sampled_xx_ts, i, c), open(fnm, 'wb'))
            pickle.dump((sampled_xx, sampled_yy), open(evalfnm, 'wb'))
        print('total sample time = {}'.format(tot_sample_time))

import numpy as np
import GPy as gpy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import norm
import helper
from active_learner import ActiveLearner
import scipy
import time
import logging

class ActiveGP(ActiveLearner):
    def __init__(self, func, initx, inity, query_type, fixmodel=True, is_adapt=False, is_uniform=False, task_lengthscale=None, betalambda=0.999, sample_time_limit=200):
        if inity.ndim == 1:
            inity = inity[:, None]
        self.xx = initx
        self.yy = inity
        assert(initx.ndim == 2 and inity.ndim == 2)
        self.func = func
        self.query_type = query_type
        self.name = 'gp_{}'.format(query_type)
        self.init_len = len(initx)
        self.sampled_xx = []
        self.good_samples = []
        if task_lengthscale is None:
            self.task_lengthscale = func.task_lengthscale[func.param_idx]
        else:
            self.task_lengthscale = task_lengthscale
        self.beta = 3.24
        self.fixmodel = fixmodel
        self.is_adapt = is_adapt
        self.is_uniform = is_uniform
        self.betalambda = betalambda
        self.sample_time_limit = sample_time_limit
    def query_best_prob(self, context):
        x0, x0context = helper.find_closest_positive_context_param(
            context, self.xx, self.yy, self.func.param_idx, self.func.context_idx)
        self.model = self.model

        def ac_f(x):
            if x.ndim == 1:
                x = x[None, :]
            x = np.hstack((x, np.tile(context, (x.shape[0], 1))))
            mu, var = self.model.predict(x)
            return (-mu)/np.sqrt(var)

        def ac_fg(x):
            if x.ndim == 1:
                x = x[None, :]
            x = np.hstack((x, np.tile(context, (x.shape[0], 1))))
            mu, var = self.model.predict(x)
            dmdx, dvdx = self.model.predictive_gradients(x)
            dmdx = dmdx[0, :, 0]
            dvdx = dvdx[0, :]
            f = (-mu)/np.sqrt(var)
            g = (-np.sqrt(var)*dmdx - 0.5*(-mu)*dvdx/np.sqrt(var)) / var
            return f[0, 0], g[0, self.func.param_idx]

        x0 = np.vstack((x0, self.xx[np.squeeze(self.yy)>0][:,self.func.param_idx]))
        x_star, y_star = helper.global_minimize(
            ac_f, ac_fg, self.func.x_range[:, self.func.param_idx], 10000, x0)
        print 'best beta=', -y_star
        self.best_beta = -y_star
        self.beta = norm.ppf(self.betalambda*norm.cdf(self.best_beta))
        if self.fixmodel:
            if self.beta > self.best_beta:
                raise ValueError('beta cannot be larger than best beta')
        return np.hstack((x_star, context))

    def gen_adaptive_samples(self, context, n=10000, m=50):
        # rejection sampling
        def ac_f(x):
            if x.ndim == 1:
                x = x[None, :]
            x = np.hstack((x, np.tile(context, (x.shape[0], 1))))
            mu, var = self.model.predict(x)
            return np.squeeze((mu)/np.sqrt(var))
        #if self.is_adapt is None:
        #    m = 50 ############ ATTENTION: THIS IS JUST FOR TIMING
        dx = len(self.func.param_idx)
        #import pdb; pdb.set_trace()
        good_samples = np.zeros((0, dx))
        prob = np.zeros(0)
        t_start = time.time()
        sampled_cnt = 0

        xmin = self.func.x_range[0, self.func.param_idx]
        xmax = self.func.x_range[1, self.func.param_idx]
        prob_unif_unit = np.prod(xmax - xmin)
        x_samples = self.xx[np.squeeze(self.yy)>0][:,self.func.param_idx]
        if len(self.sampled_xx) == 0:
            best_x = self.query_best_prob(context)
            x_samples = np.vstack((x_samples, best_x[self.func.param_idx]))
        else:
            x_samples = np.vstack((x_samples, self.sampled_xx[:, self.func.param_idx]))
        assert(len(x_samples) > 0)
        good_inds = ac_f(x_samples) > self.beta
        good_samples = np.vstack((x_samples[good_inds], good_samples))
        prob = np.hstack(( np.ones(len(good_samples)), prob ))
        scale = np.array(self.model.kern.lengthscale[self.func.param_idx]) * 1.
        flag = True
        while flag or len(good_samples) <= m:
            flag = False # make sure it samples at least once
            if time.time() - t_start > self.sample_time_limit:
                print('elapsed sampling time = {}, sampling iterations = {}'.format(time.time() - t_start), sampled_cnt)
                raise ValueError('not enough good samples')
            sampled_cnt += 1

            x_samples_unif = np.random.uniform(xmin, xmax, (n, dx))
            prob_unif = np.ones(n) * prob_unif_unit
            good_inds = ac_f(x_samples_unif) > self.beta
            x_samples_unif = x_samples_unif[good_inds]
            prob_unif = prob_unif[good_inds]
            good_samples = np.vstack((x_samples_unif, good_samples))
            prob = np.hstack((prob_unif, prob))

            if len(x_samples) > 0 and self.is_adapt is not None:
                x_samples_gmm, prob_gmm = helper.sample_gmm(x_samples, scale, n, xmin, xmax)
                good_inds = ac_f(x_samples_gmm) > self.beta
                x_samples_gmm = x_samples_gmm[good_inds]
                prob_gmm = prob_gmm[good_inds]
                good_samples = np.vstack((x_samples_gmm, good_samples))
                prob = np.hstack((prob_gmm, prob))
                if len(x_samples_gmm) > n/2.:
                    scale *= 2
                elif len(x_samples_gmm) < n/10.:
                    scale *= 0.5
            if len(good_samples) < m:
                x_samples = good_samples
            else:
                x_samples_inds = np.random.choice(np.arange(len(good_samples)), size=m, replace=False, p=prob/np.sum(prob))
                x_samples = good_samples[x_samples_inds]


        print('number of good samples = {}'.format(len(good_samples)))
        self.good_samples = good_samples
        return x_samples
    def reset_sample(self):
        self.sampled_xx = []
        self.good_samples = []

    def sample_uniform(self, context):
        if len(self.sampled_xx) == 0:
            xx = self.gen_adaptive_samples(context)
            self.unif_samples = np.hstack((xx, np.tile(context, (xx.shape[0], 1))))
            self.sampled_xx = np.array([self.unif_samples[0]])
        else:
            if len(self.unif_samples) == 0: #### ATTENTION THIS IS JUST FOR TIMING
                xx = self.gen_adaptive_samples(context)
                self.unif_samples = np.hstack((xx, np.tile(context, (xx.shape[0], 1))))

            new_s = self.unif_samples[0]

            self.sampled_xx = np.vstack((self.sampled_xx, new_s))
        
        self.unif_samples = np.delete(self.unif_samples, (0), axis=0)
        return self.sampled_xx[-1]
    def sample(self, context):
        if self.is_uniform:
            return self.sample_uniform(context)

        if len(self.sampled_xx) == 0:
            self.sampled_xx = np.array([self.query_best_prob(context)])
        else:
            # update task lengthscale
            if self.is_adapt and len(self.sampled_xx) >= 2:
                d = helper.important_d(self.sampled_xx[-1, self.func.param_idx], self.sampled_xx[:-1, self.func.param_idx], self.task_lengthscale)
                self.task_lengthscale[d] *= 0.7

            if len(self.good_samples) < 50:
                self.gen_adaptive_samples(context)
            sid = helper.argmax_condvar(self.good_samples, self.sampled_xx[:, self.func.param_idx], self.task_lengthscale)
            new_s = np.hstack((self.good_samples[sid], context))
            self.good_samples = np.delete(self.good_samples, (sid), axis=0)
            self.sampled_xx = np.vstack((self.sampled_xx, new_s))
        return self.sampled_xx[-1]
   
    def query(self, context):
        if self.query_type is 'best_prob':
            return self.query_best_prob(context)
        elif self.query_type is 'lse':
            return self.query_lse(context)

    def retrain(self, newx=None, newy=None):
        if newx is not None and newy is not None:
            self.xx = np.vstack((self.xx, newx))
            self.yy = np.vstack((self.yy, newy))
        lengthscale = (self.func.x_range[1] - self.func.x_range[0]) * 0.05
        k = gpy.kern.Matern52(self.func.x_range.shape[1], ARD=True, lengthscale=lengthscale)
        #k = gpy.kern.Matern52(self.func.x_range.shape[1], ARD=False, lengthscale=0.1)
        self.model = gpy.models.GPRegression(self.xx, self.yy, k)
        for i in range(self.func.dx):
            self.model.kern.lengthscale[i:i+1].constrain_bounded(self.func.lengthscale_bound[0][i],
                self.func.lengthscale_bound[1][i], warning=False)
        self.model['.*variance'].constrain_bounded(1e-1,2., warning=False)
        #gpm.likelihood.variance = 0.1 # need tuning
        self.model['Gaussian_noise.variance'].constrain_bounded(1e-4,0.01, warning=False)
        #gpm.likelihood.variance.constrain_fixed()
        
        self.model.optimize(messages=False)
        print self.model
        #self.task_lengthscale = np.array(1./self.model.kern.lengthscale[self.func.param_idx])

    def query_lse(self, context):
        x0, x0context = helper.find_closest_positive_context_param(
            context, self.xx, self.yy, self.func.param_idx, self.func.context_idx)
        self.model = self.model

        def ac_f(x):
            if x.ndim == 1:
                x = x[None, :]
            x = np.hstack((x, np.tile(context, (x.shape[0], 1))))
            mu, var = self.model.predict(x)
            return -1.96*np.sqrt(var) + np.abs(mu)
        x_star, _ = helper.global_minimize(
            ac_f, None, self.func.x_range[:, self.func.param_idx], 10000, x0)
        
        return np.hstack((x_star, context))

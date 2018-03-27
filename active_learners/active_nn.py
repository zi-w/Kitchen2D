from __future__ import print_function, division
import numpy as np
import scipy.optimize

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import adam
from active_learner import ActiveLearner
import keras.backend as kb
from sklearn.utils import shuffle
import cPickle as pickle
from sklearn.metrics import confusion_matrix
import os
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import norm
import helper
import time


def get_classification_model(input_dim, nodes_per_layer):
    model = Sequential()
    model.add(Dense(nodes_per_layer, input_dim=input_dim, activation='tanh'))
    model.add(Dropout(.2))
    model.add(Dense(nodes_per_layer, activation='tanh'))
    model.add(Dropout(.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(adam(lr=0.0005), loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def get_regression_model(input_dim, nodes_per_layer):
    model = Sequential()
    model.add(Dense(nodes_per_layer, input_dim=input_dim, activation='tanh'))
    model.add(Dropout(.2))
    model.add(Dense(nodes_per_layer, activation='tanh'))
    model.add(Dropout(.2))
    model.add(Dense(1, activation='linear'))
    model.compile(adam(lr=0.0005), loss='mse')
    return model


class ActiveNN(ActiveLearner):
    def __init__(self, func, initx, inity, model_type, epochs=1000, use_sample_weights=0):
        if model_type is 'classification':
            print('ActiveNN using classification')
            inity = (inity > 0).astype(float)
            model = get_classification_model(func.x_range.shape[1], 10)
        else:
            print('ActiveNN using regression')
            model = get_regression_model(func.x_range.shape[1], 10)
        self.model_type = model_type
        self.xx = initx
        self.yy = inity
        self.func = func
        self.model = model
        self.use_sample_weights = use_sample_weights
        self.name = 'nn_{}'.format(model_type)
        self.epochs = epochs
        self.beta = 0.5

    def metric(self):
        pred = self.model.predict(self.xx)
        if self.model_type is 'classification':
            pred = pred > 0.5
        else:
            pred = pred > 0
        label = self.yy > 0
        try:
            tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
            acc = (tn + tp)*1.0 / len(label)
            print('accuracy = {}, fpr = {}, fnr = {}'.format(
                acc, fp*1.0/(tn + fp), fn*1.0/(tp + fn)))
        except:
            print('error in metric')

    def query(self, context):
        x0, x0context = helper.find_closest_positive_context_param(
            context, self.xx, self.yy, self.func.param_idx, self.func.context_idx)
        g = kb.gradients(self.model.outputs[0], self.model.inputs)
        gfn = kb.function(self.model.inputs, g)

        def fn(param):
            x = np.hstack((param, np.tile(context, (param.shape[0], 1))))
            return -self.model.predict(x).astype(np.float64)

        def fgfn(param):
            x = np.hstack((param, context))
            return -self.model.predict(np.array([x]))[0].astype(np.float64), \
                   -gfn([np.array([x])])[0][0,
                                            self.func.param_idx].astype(np.float64)
        x_range = self.func.x_range
        guesses = helper.grid_around_point(
            x0, 0.5*(x_range[1]-x_range[0]), 5, x_range)
        x_star, y_star = helper.global_minimize(
            fn, fgfn, x_range[:, self.func.param_idx], 10000, guesses)
        print('x_star={}, y_star={}'.format(x_star, y_star))
        return np.hstack((x_star, context))
    def reset_sample(self):
        pass
    def sample_adaptive(self, context, N=10):
        xx = self.gen_adaptive_samples(context, m=N)
        return np.hstack((xx, np.tile(context, (xx.shape[0], 1))))
    def gen_adaptive_samples(self, context, n=10000, m=50):
        def ac_f(param):
            x = np.hstack((param, np.tile(context, (param.shape[0], 1))))
            ret = self.model.predict(x).astype(np.float64)
            if self.model_type is not 'classification':
                ret = helper.sigmoid(ret)
            return np.squeeze(ret)

        dx = len(self.func.param_idx)
        good_samples = np.zeros((0, dx))
        prob = np.zeros(0)
        t_start = time.time()
        

        xmin = self.func.x_range[0, self.func.param_idx]
        xmax = self.func.x_range[1, self.func.param_idx]
        prob_unif_unit = np.prod(xmax - xmin)
        x_samples = self.xx[np.squeeze(self.yy)>0][:,self.func.param_idx]
        x_samples = np.vstack((x_samples, self.sample(context)[:, self.func.param_idx]))

        good_inds = ac_f(x_samples) > self.beta
        if len(x_samples) == 1 and good_inds == False:
            raise ValueError('no good samples to start with')
        if len(x_samples) == 1:
            good_samples = x_samples
        else:
            good_samples = np.vstack((x_samples[good_inds], good_samples))
        prob = np.hstack(( np.ones(len(good_samples)), prob ))
        scale = np.array(self.func.lengthscale_bound[1][self.func.param_idx]) * 1.
        flag = True
        while flag or len(good_samples) <= m:
            flag = False # make sure it sample at least once
            if time.time() - t_start > 60:
                break
            x_samples_unif = np.random.uniform(xmin, xmax, (n, dx))
            prob_unif = np.ones(n) * prob_unif_unit
            good_inds = ac_f(x_samples_unif) > self.beta
            x_samples_unif = x_samples_unif[good_inds]
            prob_unif = prob_unif[good_inds]
            good_samples = np.vstack((x_samples_unif, good_samples))
            prob = np.hstack((prob_unif, prob))

            if len(x_samples) > 0:
                x_samples_gmm, prob_gmm = helper.sample_tgmm(x_samples, scale, n, xmin, xmax)
                good_inds = ac_f(x_samples_gmm) > self.beta
                x_samples_gmm = x_samples_gmm[good_inds]
                prob_gmm = prob_gmm[good_inds]
                good_samples = np.vstack((x_samples_gmm, good_samples))
                prob = np.hstack((prob_gmm, prob))
                if len(x_samples_gmm) > n/2.:
                    scale *= 2
                    print('tune up scale')
                elif len(x_samples_gmm) < n/10.:
                    scale *= 0.5
                    print('tune down scale')
                print('gmm good={}'.format(len(x_samples_gmm)))
            print('unif good={}'.format(len(x_samples_unif)))
            if len(good_samples) < m:
                x_samples = good_samples
            else:

                x_samples_inds = np.random.choice(np.arange(len(good_samples)), size=m, replace=False, p=prob/np.sum(prob))
                x_samples = good_samples[x_samples_inds]


        print('good samples len = {}'.format(len(good_samples)))
        self.good_samples = good_samples
        return x_samples
    def get_sample_weight(self):
        n_tot = self.xx.shape[0]

        if not self.use_sample_weights:
            sample_weight = np.ones(n_tot)
            return sample_weight

        pos_idx = self.yy > 0
        n_pos = np.sum(pos_idx)
        print('number of positive datapoints = {}'.format(n_pos))
        sample_weight = np.ones(n_tot)
        sample_weight[pos_idx] = (n_tot - n_pos)*1.0 / n_pos
        return sample_weight

    def retrain(self, newx=None, newy=None, initial_epoch=0):
        if newx is not None and newy is not None:
            self.xx = np.vstack((self.xx, newx))
            if self.model_type is 'classification':
                newy = newy > 0
            self.yy = np.hstack((self.yy, newy))
        sample_weight = self.get_sample_weight()
        self.xx, self.yy, sample_weight = shuffle(
            self.xx, self.yy, sample_weight)
        self.model.fit(self.xx, self.yy, epochs=initial_epoch+self.epochs, batch_size=100,
                       validation_split=0.1, sample_weight=sample_weight, initial_epoch=initial_epoch, verbose=0)
        self.metric()


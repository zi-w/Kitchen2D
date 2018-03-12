#!/usr/bin/env python
# Copyright (c) 2017 Zi Wang
from push_world import b2WorldInterface, construct_scene, end_effector, simulate_push
import sys
import numpy as np
import cPickle as pickle
import os


class Push(object):
    def __init__(self, use_gui=False):
        self.x_range = np.array(
            [[-5., -5., 0., 0., -5., -5.], [5., 5., 2*np.pi, 2., 5., 5.]])
        self.lengthscale_bound = np.array([np.ones(6)*0.1, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
        self.context_idx = [4, 5]
        self.param_idx = [0, 1, 2, 3]
        self.dx = len(self.x_range[0])
        self.world = b2WorldInterface(use_gui)
        oshape, osize, ofriction, odensity, bfriction = 'circle', 1, 0.01, 0.05, 0.01
        robot_shape, robot_size = 'rectangle', (0.3, 1)

        thing_init_location = (0, 0)

        self.object, self.base = construct_scene(
            500, 500, self.world, oshape, osize, ofriction, odensity, bfriction, thing_init_location)
        self.robot = end_effector(
            self.world, (5, 5), self.base, 0, robot_shape, robot_size)
        self.initx, self.init_opos = None, None

    def simulate(self, robot_pos, robot_angle, simulate_time):
        self.robot.set_pos(robot_pos, robot_angle)
        self.object.position = (0, 0)
        xvel = 10. * np.cos(robot_angle)
        yvel = 10. * np.sin(robot_angle)
        opos = simulate_push(self.world, self.object, self.robot,
                             self.base, xvel, yvel, int(simulate_time*100))
        return opos

    def simulate_x(self, x):
        return self.simulate(x[:2], x[2], x[3])

    def __call__(self, x):
        opos = self.simulate_x(x[self.param_idx])
        return 2. - np.linalg.norm(np.array(x[self.context_idx]) - opos)


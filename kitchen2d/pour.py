#!/usr/bin/env python
# Copyright (c) 2017 Zi Wang
import kitchen_stuff as ks
from kitchen_stuff import Kitchen2D
from gripper import Gripper
import sys
import numpy as np
import cPickle as pickle
import os
import time
settings = {
    0: {
        'do_gui': False,
        'sink_w': 4.,
        'sink_h': 5.,
        'sink_d': 1.,
        'sink_pos_x': 50,
        'left_table_width': 100.,
        'right_table_width': 100.,
        'faucet_h': 8.,
        'faucet_w': 5.,
        'faucet_d': 0.5,
        'planning': False,
        'save_fig': False
    }
}

class Pour(object):
    def __init__(self):
        #grasp_ratio, relative_pos_x, relative_pos_y, dangle, cw1, ch1, cw2, ch2
        self.x_range = np.array(
            [[0., -10., 1., np.pi/2, 4., 4., 3., 3.], 
            [1., 10., 10., np.pi, 5., 5., 4.5, 5.]])
            #[1., 10., 10., np.pi, 8., 5., 4.5, 5.]]) this is the upper bound used in the paper.
        self.lengthscale_bound = np.array([np.ones(8)*0.1, [0.15, 0.5, 0.5, 0.2, 0.5, 0.5, 0.5, 0.5]])
        self.context_idx = [4, 5, 6, 7]
        self.param_idx = [0, 1, 2, 3]
        self.dx = len(self.x_range[0])
        self.task_lengthscale = np.ones(8)*10
        self.do_gui = False
    def check_legal(self, x):
        grasp_ratio, rel_x, rel_y, dangle, cw1, ch1, cw2, ch2 = x
        dangle *= np.sign(rel_x)
        settings[0]['do_gui'] = self.do_gui
        kitchen = Kitchen2D(**settings[0])
        gripper = Gripper(kitchen, (5,8), 0)
        cup1 = ks.make_cup(kitchen, (0,0), 0, cw1, ch1, 0.5)
        cup2 = ks.make_cup(kitchen, (-15,0), 0, cw2, ch2, 0.5)
        gripper.set_grasped(cup2, grasp_ratio, (-15,0), 0)
        gripper.set_position((rel_x, rel_y), 0)
        if not kitchen.planning:
            g2 = gripper.simulate_itself()
            _, collision = g2.check_path_collision((rel_x, rel_y), 0, (rel_x, rel_y), dangle)

            if collision:
                return False
        self.kitchen = kitchen
        self.gripper = gripper
        self.cup1 = cup1
        self.cup2 = cup2
        return True
    def sampled_x(self, n):
        i = 0
        while i < n:
            x = np.random.uniform(self.x_range[0], self.x_range[1])
            legal = self.check_legal(x)
            if legal:
                i += 1
                yield x

    def __call__(self, x, image_name=None):
        if not self.check_legal(x):
            return -1.
        grasp_ratio, rel_x, rel_y, dangle, cw1, ch1, cw2, ch2 = x
        dangle *= np.sign(rel_x)
        if self.kitchen.planning:
            self.gripper.close()
            dpos = self.cup1.position + (rel_x, rel_y)
            self.gripper.set_position(dpos, dangle)
            self.kitchen.image_name = image_name
            self.kitchen.step()
            return
        self.kitchen.gen_liquid_in_cup(self.cup2, 500)
        self.gripper.compute_post_grasp_mass()
        self.gripper.close(timeout=0.1)
        self.gripper.check_grasp(self.cup2)
        success, score = self.gripper.pour(self.cup1, (rel_x, rel_y), dangle, exact_control=False, p_range=cw1/2)
        return np.exp(2*(score*10 - 9.5)) - 1.
        


if __name__ == '__main__':
    func = Pour()
    N = 10
    samples = func.sampled_x(N)
    x = list(samples)
    for xx in x:
        start = time.time()
        print func(xx)
        print time.time() - start
    
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


class Scoop(object):
    def __init__(self):
        #grasp_ratio, relative_pos_x, relative_pos_y, dangle, cw1, ch1
        self.x_range = np.array(
            [[0., 0., 0., 0., 0., 0., 0., 5., 4.], 
            [1., 1., 1., 1., 1., 1.5, 1., 10., 8.]])
        self.lengthscale_bound = np.array([np.ones(9)*0.01, [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 2., 2.,]])
        self.context_idx = [7, 8]
        self.param_idx = [0, 1, 2, 3, 4, 5, 6]
        self.dx = len(self.x_range[0])
        self.task_lengthscale = np.ones(9)
        self.do_gui = False
    def check_legal(self, x):
        rel_x1, rel_y1, rel_x2, rel_y2, rel_x3, rel_y3, grasp_ratio, cw1, ch1 = x
        settings[0]['do_gui'] = self.do_gui
        kitchen = Kitchen2D(**settings[0])
        gripper = Gripper(kitchen, (5,8), 0)
        cup = ks.make_cup(kitchen, (0,0), 0, cw1, ch1, 0.5)
        spoon = ks.make_spoon(kitchen, (5,10), 0, 0.2, 3, 1.)
        gripper.set_grasped(spoon, grasp_ratio, (5,10), 0)
        dposa1, dposa2 = gripper.get_scoop_init_end_pose(cup, (rel_x1, rel_y1), (rel_x3, rel_y3))
        gripper.set_grasped(spoon, grasp_ratio, dposa1[:2], dposa1[2])
        g2 = gripper.simulate_itself()
        collision = g2.check_point_collision(dposa1[:2], dposa1[2])

        if collision:
            return False
        collision = g2.check_point_collision(dposa2[:2], dposa2[2])
        
        if collision:
            return False
        self.kitchen = kitchen
        self.gripper = gripper
        self.cup = cup
        self.spoon = spoon
        return True
    def sampled_x(self, n):
        i = 0
        while i < n:
            x = np.random.uniform(self.x_range[0], self.x_range[1])
            legal = self.check_legal(x)
            if legal:
                i += 1
                yield x

    def __call__(self, x):
        if not self.check_legal(x):
            return -1.
        rel_x1, rel_y1, rel_x2, rel_y2, rel_x3, rel_y3, grasp_ratio, cw1, ch1 = x
        self.kitchen.gen_liquid_in_cup(self.cup, 1000)
        self.gripper.compute_post_grasp_mass()
        self.gripper.close(timeout=0.1)
        self.gripper.check_grasp(self.spoon)
        success, score = self.gripper.scoop(self.cup, (rel_x1, rel_y1), (rel_x2, rel_y2), (rel_x3, rel_y3))
        print 'score=',score
        return 5. * (score - 0.2)
        

if __name__ == '__main__':
    func = Scoop()
    N = 10
    samples = func.sampled_x(N)
    x = list(samples)
    for xx in x:
        start = time.time()
        print func(xx)
        print time.time() - start
    
from kitchen_predicates import *
import sys
from ss.model.functions import Predicate, rename_functions, initialize, TotalCost, Increase
from ss.model.problem import Problem
from ss.model.operators import Action, Axiom
from ss.model.streams import GenStream, CondGen, Stream, FnStream
from ss.algorithms.dual_focused import dual_focused
from ss.utils import INF

import numpy as np
import time
import random

import kitchen2d.kitchen_stuff as ks
from kitchen2d.gripper import Gripper
from kitchen_tasks.kitchen_utils import scale_cost, FreeTrajectory, HoldingTrajectory, \
    PushTrajectory, Command, ClosedTrajectory
import active_learners.helper as helper

import place_holders as ph
from kitchen_tasks.kitchen_actions import get_actions
from kitchen_tasks.kitchen_axioms import get_axioms
from kitchen_tasks.kitchen_streams import get_streams

##################################################

def create_problem(initial_poses, initial_atoms, goal_literals,
                   make_kitchen, make_body,
                   expid_pour, expid_scoop,
                   do_motion=True, do_collisions=True, 
                   do_grasp=True, buffer_distance=0.):
    '''
    Create a STRIPStream problem.
    Args:
        initial_poses: a dictionary mapping from the object names to their poses.
        initial_atoms: initial state of the system, expressed by a list of literals.
        goal_literals: list of goal conjunctive literals.
        make_kitchen: function to create kitchen.
        make_body: function to create Box2D bodies.
        expid_pour: experiment ID for learning to pour.
        expid_scoop: experiment ID for learning to scoop.
        do_motion: whether to check feasibility of motion planning when planning.
        do_collisions: whether to check collisions when planning.
        do_grasp: whether to check grasp feasibility when planning.
        buffer_distance: the buffer on obstacle objects when doing motion planning.
    '''
    ph.initial_poses = initial_poses
    ph.make_kitchen = make_kitchen
    ph.make_body = make_body
    ph.gripperInitPos = initial_poses['gripper']
    ph.cupInitPos = initial_poses['cup']
    ph.expid_pour = expid_pour
    ph.expid_scoop = expid_scoop

    ph.do_motion = do_motion
    ph.do_collisions = do_collisions
    ph.do_grasp = do_grasp
    ph.buffer_distance = buffer_distance

    if not do_motion:
        print 'Warning! Disabled motion planning'
    if not do_collisions:
        print 'Warning! Movable object collisions are disabled'

    ##################################################

    # Objective is the numeric function to minimize
    objective = TotalCost()
    
    initial_atoms += [
        # Fluent function
        initialize(TotalCost(), 0) # Maintains the total plan cost
        ]

    # Ensure the list of initial atoms is complete
    for name, pose in initial_poses.items():
        if 'gripper' in name:
            initial_atoms += [IsGripper(name)]
        if 'cup' in name:
            initial_atoms += [IsCup(name)]
        if 'spoon' in name:
            initial_atoms += [IsSpoon(name)]
            initial_atoms += [IsStirrer(name)]
        if 'stirrer' in name:
            initial_atoms += [IsStirrer(name)]
        if 'block' in name:
            initial_atoms += [IsBlock(name)]
        initial_atoms += [IsPose(name, pose), AtPose(name, pose), TableSupport(pose)]

    actions = get_actions()
    axioms = get_axioms()
    streams = get_streams()

    return Problem(initial_atoms, goal_literals, actions, axioms, streams, objective=objective)

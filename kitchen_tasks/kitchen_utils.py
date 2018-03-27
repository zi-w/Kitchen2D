import sys
sys.path.append('pddlstream/')
import numpy as np
from ss.model.problem import get_length, get_cost

import math

import kitchen2d.kitchen_stuff as ks
from kitchen2d.kitchen_stuff import Kitchen2D
from kitchen2d.gripper import Gripper
from functools import partial


SCALE_COST = 100.


def scale_cost(cost):
    # Unfortunately, FastDownward only supports nonnegative, integer functions
    # This function scales all costs, so decimals can be factored into the cost
    return int(math.ceil(SCALE_COST * cost))

class FreeTrajectory(object):
    def __init__(self, gripper, path):
        self.gripper = gripper
        self.path = tuple(path)
    def reverse(self):
        return self.__class__(self.gripper, self.path[::-1])
    def __repr__(self):
        return 'ft({},{})'.format(self.gripper, len(self.path))

class ClosedTrajectory(object):
    def __init__(self, gripper, path):
        self.gripper = gripper
        self.path = tuple(path)
    def reverse(self):
        return self.__class__(self.gripper, self.path[::-1])
    def __repr__(self):
        return 'ct({},{})'.format(self.gripper, len(self.path))

class HoldingTrajectory(object):
    def __init__(self, gripper, body, pos_ratio, path):
        self.gripper = gripper
        self.body = body
        self.pos_ratio = pos_ratio
        self.path = tuple(path)
    def reverse(self):
        return self.__class__(self.gripper, self.body, self.pos_ratio, self.path[::-1])
    def __repr__(self):
        return 'ht({},{},{},{})'.format(self.gripper, self.body, self.pos_ratio, len(self.path))

class PushTrajectory(object):
    def __init__(self, gripper, body, init_pose, path):
        self.gripper = gripper
        self.body = body
        self.init_pose = init_pose
        self.path = tuple(path)
    def reverse(self):
        raise NotImplementedError()
    def __repr__(self):
        return 'pt({},{},{},{})'.format(self.gripper, self.body, self.init_pose, len(self.path))


class Command(object):
    def __init__(self, trajectories, parameters=tuple()):
        self.trajectories = tuple(trajectories)
        self.parameters = tuple(parameters)
    def reverse(self):
        return self.__class__(traj.reverse() for traj in reversed(self.trajectories))
    def __repr__(self):
        return 'c{}'.format(id(self) % 100)


def print_plan(plan, evaluations):
    if plan is None:
        print '\nFailed to find a plan'
    else:
        print '\nFound a plan'
        print 'Length:', get_length(plan, evaluations)
        print 'Cost:', get_cost(plan, evaluations) / SCALE_COST
        for i, (action, args) in enumerate(plan):
            print i, action, list(args)

def step_command(gripper, body_from_name, command, downsample=10): # 1 | 10 | 50
    for traj in command.trajectories:
        if isinstance(traj, FreeTrajectory):
            gripper.release()
            gripper.set_open()
        elif isinstance(traj, ClosedTrajectory):
            gripper.release()
            gripper.set_closed()
        elif isinstance(traj, HoldingTrajectory):
            body = body_from_name[traj.body]
            posa = traj.path[1]
            gripper.set_grasped(body, traj.pos_ratio, posa[:2], posa[2])
        elif isinstance(traj, PushTrajectory):
            body = body_from_name[traj.body]
            gripper.set_closed()
            body.position = traj.init_pose[:2]
            gripper.attach(body)
        else:
            raise ValueError(traj)
        print traj
        for q in traj.path[1::downsample]:
            gripper.set_position(q[:2], q[2])
            gripper.b2w.draw()
            raw_input('Continue?')

def execute_plan_command(plan, kitchen_params, gripperInitPos, cupInitPos, cupSize, kettleInitPos, kettleSize, **kwargs):
    kitchen = Kitchen2D(planning=False, **kitchen_params)
    kitchen.enable_gui()
    gripper = Gripper(kitchen, init_pos=gripperInitPos[:2], init_angle=gripperInitPos[2])
    cup = ks.make_cup(kitchen, cupInitPos[:2], cupInitPos[2], *cupSize)
    kettle = ks.make_cup(kitchen, kettleInitPos[:2], kettleInitPos[2], *kettleSize)

    pause = False
    def execute_command(command):
        for traj in command.trajectories:
            print traj
            if pause:
                raw_input('Execute?')
            if isinstance(traj, FreeTrajectory):
                if gripper.attached:
                    gripper.open()
                    gripper.release()
            else:
                if not gripper.attached:
                    gripper.close()
                    gripper.attach(cup)
            gripper.apply_control(traj.path, maxspeed=2.5)

    raw_input('Begin?')
    for i, (action, args) in enumerate(plan):
        print i, action, args
        if action.name == 'fill':
            pass
        elif action.name == 'place':
            execute_command(args[-1].reverse())
        elif action.name == 'pour-gp':
            kp, p2 = args[-2:]
            rel_pose = np.array(p2) - np.array(kp)
            poured, score = gripper.pour(kettle, rel_pose[:2], rel_pose[2], 0.45, 200)
        else:
            execute_command(args[-1])


def set_pose(body, pose):
    if isinstance(body, Gripper):
        body.set_position(pose[:2], pose[2])
    else:
        body.position = pose[:2]
        body.angle = pose[2]

def step_plan(plan, kitchen, body_from_name):
    for action, args in plan:
        if action.name in ('fill',):
            pass
        elif action.name == 'pour-gp':
            grip, p1, cup, g, _, _, command = args
            step_command(body_from_name['gripper'], body_from_name, command)
        elif action.name in ('place', 'stack'):
            step_command(body_from_name['gripper'], body_from_name, args[-1].reverse())
        else:
            step_command(body_from_name['gripper'], body_from_name, args[-1])


def execute_plan(plan, kitchen, body_from_name):
    finished = True
    score = -1
    collision_buffer = 0.5 
    for i, (action, args) in enumerate(plan):
        print i, action, args
        gripper = body_from_name[args[0]]
        if action.name == 'move':
            _, _, pos2, control = args
            traj = control.trajectories[0]
            if isinstance(traj, ClosedTrajectory):
                gripper.close()
            elif isinstance(traj, FreeTrajectory):
                gripper.open()
            finished = gripper.find_path(pos2[:2], pos2[2], collision_buffer=collision_buffer)
        elif action.name == 'move-holding':
            _, _, _, _, pos2, _ = args
            finished = gripper.find_path(pos2[:2], pos2[2], collision_buffer=collision_buffer, motion_angle=pos2[2])
        elif action.name == 'pick':
            _, dpos_up, cup_name, _, _, control = args
            gripper.open()
            dpos = control.trajectories[0].path[-1]
            gripper.apply_lowlevel_control(dpos[:2], dpos[2])
            gripper.close()
            print 'Check grasp:', gripper.check_grasp(body_from_name[cup_name])
            gripper.apply_lowlevel_control(dpos_up[:2], dpos_up[2])
        elif action.name in ('place', 'stack'):
            dpos_up, control = args[1], args[-1]
            dpos = control.trajectories[0].path[-1]
            gripper.apply_lowlevel_control(dpos[:2], dpos[2])
            gripper.open()
            gripper.apply_lowlevel_control(dpos_up[:2], dpos_up[2])
        elif action.name == 'fill':
            finished = gripper.get_liquid_from_faucet(10)
        elif action.name == 'pour-gp':
            _, p1, _, _, kettle_name, kp, command = args
            poured, score = gripper.pour(body_from_name[kettle_name], *command.parameters)
            gripper.apply_control([p1])
        elif action.name == 'scoop':
            _, _, _, spoon_name, _, cup_name, _, control = args
            print gripper.scoop(body_from_name[cup_name], *control.parameters)
        elif action.name == 'dump':
            #GRIPPER, POSE, CUP, GRASP, KETTLE, POSE2, CONTROL
            _, _, _, spoon_name, _, cup_name, _, control = args
            print gripper.dump(body_from_name[cup_name], *control.parameters)
        elif action.name == 'stir':
            #GRIPPER, POSE, CUP, GRASP, KETTLE, POSE2, CONTROL
            _, _, stir_name, grasp, cup_name, _, control = args
            gripper.apply_control(control.trajectories[0].path)
            for i in xrange(5):
                for traj in control.trajectories[1:-1]:
                    gripper.apply_control(traj.path)
            gripper.apply_control(control.trajectories[-1].path)
        elif action.name == 'push':
            # GRIPPER, POSE, POSE2, BLOCK, POSE3, POSE4, CONTROL
            _, _, _, block_name, _, _, control = args
            gripper.close()
            for traj in control.trajectories:
                gripper.apply_control(traj.path)
        else:
            raise ValueError(action.name)
        if not finished:
            raise RuntimeError('Failed to plan a path')
    return finished, score


def make_kitchen(planning, kitchen_params, poses, sizes):
    '''
    Create a kitchen object of class Kitchen2D. 
    We add the holders for spoons and stirrers as static 
    Box2D objects in the kitchen.
    '''
    kitchen = Kitchen2D(planning=planning, **kitchen_params)  

    # create holders for spoons and stirrers
    width_eps = 0.05
    for name, pose in poses.items():
        _, holder_h, holder_d = sizes['holder']
        if 'spoon' in name:
            holder_w = 2*holder_d + sizes[name][2] + width_eps
            pose_x = poses[name][0] - holder_d - width_eps/2
        elif 'stirrer' in name:
            holder_w = 2*holder_d + 2*sizes[name][2] + width_eps
            pose_x = poses[name][0] - holder_w/2
        else:
            continue
        
        holder = ks.make_static_cup(kitchen, (pose_x, 0), 0, 
            holder_w, holder_h, holder_d)

    return kitchen

def kitchen_creation_methods(expid_pour, expid_scoop, kitchen_params, poses, sizes):
    '''
    Returns the make_kitchen method and make_body method customized for the current 
    setting of the kitchen.
    '''
    make_kitchen_ = partial(make_kitchen, kitchen_params=kitchen_params, 
        poses=poses, sizes=sizes)

    sizes = update_sizes(expid_pour, expid_scoop, sizes)

    make_body_ = partial(ks.make_body, args=sizes)
    
    return make_kitchen_, make_body_

def update_sizes(expid_pour, expid_scoop, sizes):
    '''
    Update the sizes dictionary according to the experiment IDs used in learning.
    '''
    import active_learners.helper as helper
    pour_to_w, pour_to_h, pour_from_w, pour_from_h = helper.get_pour_context(expid_pour)
    scoop_w, scoop_h = helper.get_scoop_context(expid_scoop)
    additional_args = {
        'cup': (pour_to_w, pour_to_h, sizes['holder'][2]),
        'cream_cup': (pour_from_w, pour_from_h, sizes['holder'][2]),
        'sugar_cup': (scoop_w, scoop_h, sizes['holder'][2])
    }
    sizes.update(additional_args)
    return sizes

def initialize_objects_in_kitchen(kitchen, make_body, initial_poses, initial_atoms):
    '''
    Create and initialize the poses of objects in a kitchen object.
    Returns a dictionary that maps from an object name to its corresponding Box2D object. 
    '''
    from kitchen_tasks.kitchen_predicates import *
    kitchen.enable_gui()
    body_from_name = {}
    for name, pose in initial_poses.items():
        body_from_name[name] = make_body(kitchen, name, pose)
        if 'sugar_cup' in name and HasSugar(name) in initial_atoms:
            kitchen.gen_liquid_in_cup(body_from_name[name], 1000, userData='sugar')
        elif 'cream_cup' in name and HasCream(name) in initial_atoms:
            kitchen.gen_liquid_in_cup(body_from_name[name], 100, userData='cream')
    for atom in initial_atoms:
        if atom.head.function is Grasped:
            name, grasp = atom.head.args
            gripper = body_from_name['gripper']
            gripper.set_grasped(body_from_name[name], grasp, 
                initial_poses['gripper'][:2], initial_poses['gripper'][2])
    kitchen.draw()
    return body_from_name
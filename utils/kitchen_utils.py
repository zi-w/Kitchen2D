import numpy as np
from ss.model.problem import get_length, get_cost

import math

import kitchen_stuff as ks
from kitchen_stuff import Kitchen2D, Gripper

SCALE_COST = 100.


def scale_cost(cost):
    # Unfortunately, FastDownward only supports nonnegative, integer functions
    # This function scales all costs, so decimals can be factored into the cost
    return int(math.ceil(SCALE_COST * cost))

# TODO(caelan): combine these
# TODO(caelan): close/open gripper actions

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
        self.init_pose = init_pose # Relative pose?
        self.path = tuple(path)
    def reverse(self):
        raise NotImplementedError()
    def __repr__(self):
        return 'pt({},{},{},{})'.format(self.gripper, self.body, self.init_pose, len(self.path))

# TODO(caelan): execute and step methods

class Command(object):
    def __init__(self, trajectories, parameters=tuple()):
        # TODO(caelan): name the parameters
        self.trajectories = tuple(trajectories)
        self.parameters = tuple(parameters)
    def reverse(self):
        return self.__class__(traj.reverse() for traj in reversed(self.trajectories))
    def __repr__(self):
        return 'c{}'.format(id(self) % 100)


def print_plan(plan, evaluations):
    # print '\nEvaluations:'
    # dump_evaluations(evaluations)
    if plan is None:
        print '\nFailed to find a plan'
        # return
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
            gripper.set_closed(None)
        elif isinstance(traj, HoldingTrajectory):
            #body = gripper.b2w.get_body(traj.body)
            body = body_from_name[traj.body]
            gripper.set_closed(body)
            gripper.attach(body)
        elif isinstance(traj, PushTrajectory):
            body = body_from_name[traj.body]
            gripper.set_closed(None)
            body.position = traj.init_pose[:2]
            gripper.attach(body)
        else:
            raise ValueError(traj)
        print traj
        for q in traj.path[1::downsample]:
            gripper.set_position(q[:2], q[2])
            #gripper.b2w.step()
            gripper.b2w.draw() # TODO(caelan): something is strange with grasping...
            raw_input('Continue?')

def step_plan(plan, KITCHEN_PARAMS, gripperInitPos, cupInitPos, cupSize, kettleInitPos, kettleSize, **kwargs):
    # TODO(caelan): planning=False or planning=True?
    kitchen = Kitchen2D(planning=True, **KITCHEN_PARAMS)
    kitchen.enable_gui()
    gripper = Gripper(kitchen, init_pos=gripperInitPos[:2], init_angle=gripperInitPos[2])
    body_from_name = {
        'cup': ks.make_cup(kitchen, cupInitPos[:2], cupInitPos[2], *cupSize),
        'kettle': ks.make_cup(kitchen, kettleInitPos[:2], kettleInitPos[2], *kettleSize),
    }

    kitchen.draw()
    raw_input('Start?')
    for action, args in plan:
        if action.name in ('fill', 'pour-gp'):
            pass
        elif action.name == 'place':
            step_command(gripper, body_from_name, args[-1].reverse())
        else:
            step_command(gripper, body_from_name, args[-1])


def execute_plan(plan, KITCHEN_PARAMS, gripperInitPos, cupInitPos, cupSize, kettleInitPos, kettleSize, **kwargs):
    kitchen = Kitchen2D(planning=False, **KITCHEN_PARAMS)
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
                if gripper.grasped:
                    gripper.open()
                    gripper.release()
            else:
                if not gripper.grasped:
                    gripper.close()
                    gripper.attach(cup)
            gripper.apply_control(traj.path, maxspeed=2.5)
            # TODO: add extract control bit after this?
            # TODO: simulate control between waypoints on trajectory?

    raw_input('Begin?')
    for i, (action, args) in enumerate(plan):
        print i, action, args
        if action.name == 'fill':
            #time.sleep(1.0)
            pass # TODO(caelan): stay in place
        elif action.name == 'place':
            execute_command(args[-1].reverse())
        elif action.name == 'pour-gp':
            #execute_command(args[-1])
            kp, p2 = args[-2:]
            rel_pose = np.array(p2) - np.array(kp)
            poured, score = gripper.pour(kettle, rel_pose[:2], rel_pose[2], 0.45, 200)
        else:
            # ValueError: A value in x_new is above the interpolation range.
            execute_command(args[-1])


def set_pose(body, pose):
    if isinstance(body, Gripper):
        body.set_position(pose[:2], pose[2])
    else:
        body.position = pose[:2]
        body.angle = pose[2]
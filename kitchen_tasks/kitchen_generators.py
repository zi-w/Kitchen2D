# Define generator functions
import sys
sys.path.append('pddlstream/')
import place_holders as ph
import numpy as np

import kitchen2d.kitchen_stuff as ks
from kitchen2d.gripper import Gripper
from kitchen_tasks.kitchen_utils import FreeTrajectory, HoldingTrajectory, \
    PushTrajectory, Command, ClosedTrajectory
import active_learners.helper as helper
from ss.model.streams import CondGen
from ss.model.bounds import SharedOutputSet

def test_collision(command, body_name, pose):
    '''
    Returns True if the trajectory associated by command is colliding 
    with a body of name body_name and pose pose. Otherwise False.
    '''
    if not ph.do_collisions:
        return False
    for traj in command.trajectories:
        if body_name == traj.gripper or (isinstance(traj, HoldingTrajectory) and (body_name == traj.body)):
            continue

        kitchen = ks.b2WorldInterface(False)
        init_pose = ph.gripperInitPos
        gripper = Gripper(kitchen, init_pos=init_pose[:2], init_angle=init_pose[2])
        if isinstance(traj, FreeTrajectory):
            gripper.release()
            gripper.set_open()
        elif isinstance(traj, PushTrajectory):
            body = ph.make_body(kitchen, traj.body, traj.init_pose)
            gripper.set_closed()
            gripper.attach(body)
        elif isinstance(traj, ClosedTrajectory):
            gripper.release()
            gripper.set_closed()
        else:
            body = ph.make_body(kitchen, traj.body, ph.cupInitPos)
            gripper.set_grasped(body, traj.pos_ratio, init_pose[:2], init_pose[2])

        ph.make_body(kitchen, body_name, pose)
        g2 = gripper.simulate_itself()

        for q in traj.path:
            if g2.check_point_collision(q[:2], q[2]):
                print 'Collision: {}'.format(body_name)
                return True
    return False
def genGrasp(obj_name):
    yield (np.random.uniform(0, 1), )

def genGraspControl(grip_name, cup_name, pose2, grasp):
    kitchen = ph.make_kitchen(planning=True)
    gripper = ph.make_body(kitchen, grip_name, ph.gripperInitPos)
    gripper.open()

    cup = ph.make_body(kitchen, cup_name, pose2)
    down_pos, up_pos = gripper.get_grasp_poses(cup, grasp)
    up_pos = np.array(down_pos) + np.array([0, 5])

    ori = 0
    up_pose = (up_pos[0], up_pos[1], ori)
    down_pose = (down_pos[0], down_pos[1], ori)
    if not ph.do_grasp:
        traj1 = FreeTrajectory(grip_name, (up_pose, down_pose))
        traj2 = HoldingTrajectory(grip_name, cup_name, grasp, (down_pose, up_pose))
        return (up_pose, Command([traj1, traj2]))

    traj1 = FreeTrajectory(grip_name, (up_pose, down_pose))
    gripper.set_position(down_pose[:2], down_pose[2])

    gripper.close()
    if not gripper.check_grasp(cup): return

    traj2 = HoldingTrajectory(grip_name, cup_name, grasp, (down_pose, up_pose))

    command = Command([traj1, traj2])
    return (up_pose, command)

def genFaucet(grip_name, cup_name, grasp):
    kitchen = ph.make_kitchen(planning=True)
    gripper = ph.make_body(kitchen, grip_name, ph.gripperInitPos)
    gripper.open()
    cup = ph.make_body(kitchen, cup_name, ph.cupInitPos)
    if not gripper.set_grasped(cup, grasp, ph.gripperInitPos[:2], ph.gripperInitPos[2]):
        return
    pose = gripper.get_liquid_from_faucet_pos()
    ori = 0
    pose = (pose[0], pose[1], ori)
    return (pose,)

def genPour(grip_name, cup_name, kettle_name, pose2):
    if kettle_name != 'cup' or cup_name != 'cream_cup':
        return 
    if not np.isclose(pose2[2], 0, atol=0.05):
        return
    kitchen = ph.make_kitchen(planning=True)
    gripper = Gripper(kitchen, init_pos=ph.gripperInitPos[:2], init_angle=ph.gripperInitPos[2])
    cup = ph.make_body(kitchen, cup_name, ph.cupInitPos)
    kettle = ph.make_body(kitchen, kettle_name, pose2)

    #####################GP START##############################
    gp, c = helper.process_gp_sample(ph.expid_pour, ph.FLAG_LK, ph.IS_ADAPTIVE,
                                     task_lengthscale=ph.TASK_LENGTH_SCALE)
    while True:
        grasp, rel_x, rel_y, dangle, _, _, _, _ = gp.sample(c) 
        #####################GP END##############################
        dangle *= np.sign(rel_x)
        rel_pose = (rel_x, rel_y, dangle)
        pour_pose = tuple(np.array(pose2) + np.array(rel_pose))
        initial_pose = tuple(np.array(pose2) + (rel_x, rel_y, 0))
        if not gripper.set_grasped(cup, grasp, initial_pose[:2], initial_pose[2]):
            return 
        g2 = gripper.simulate_itself()
        if g2.check_point_collision(initial_pose[:2], initial_pose[2]):
            continue
        command = Command([], parameters=[rel_pose[:2], rel_pose[2]])
        yield (grasp, initial_pose, command)

class MotionGen(CondGen):
    use_context = True
    def generate(self, context=None):
        self.enumerated = True # Prevents from being resampled
        grip_name, pose1, pose2 = self.inputs
        if not ph.do_motion:
            path = (pose1, pose2)
            command = Command([ClosedTrajectory(grip_name, path)])
            return [(command,)]

        kitchen = ph.make_kitchen(planning=True)
        if self.use_context and (context is not None):
            for atom in context.conditions:
                _, name, pose = atom.head.args
                if name not in [grip_name] and not isinstance(pose, SharedOutputSet):
                    ph.make_body(kitchen, name, pose)

        gripper = Gripper(kitchen, init_pos=pose1[:2], init_angle=pose1[2])
        gripper.set_closed()

        path = gripper.plan_path(pose2[:2], pose2[2], collision_buffer=ph.buffer_distance)
        if path is None:
            return []
        command = Command([ClosedTrajectory(grip_name, path)])
        return [(command,)]

def genScoop(grip_name, spoon_name, kettle_name, pose2):
    if kettle_name != 'sugar_cup':
        return 
    kitchen = ph.make_kitchen(planning=True)
    gripper = ph.make_body(kitchen, grip_name, ph.gripperInitPos)
    spoon = ph.make_body(kitchen, spoon_name, ph.cupInitPos)
    kettle = ph.make_body(kitchen, kettle_name, pose2)

    #####################GP START##############################
    gp, c = helper.process_gp_sample(ph.expid_scoop, flag_lk=ph.FLAG_LK, is_adaptive=ph.IS_ADAPTIVE,
                                     task_lengthscale=ph.TASK_LENGTH_SCALE, betalambda=0.998, exp='scoop')
    while True:
        rel_x1, rel_y1, rel_x2, rel_y2, rel_x3, rel_y3, grasp, _, _ = gp.sample(c)
        rel_pos1 = (rel_x1, rel_y1); rel_pos2 = (rel_x2, rel_y2); rel_pos3 = (rel_x3, rel_y3)
        ######################GP END#############################

        gripper.set_grasped(spoon, grasp, ph.gripperInitPos[:2], ph.gripperInitPos[2])
        scoop_pose, end_pose = map(tuple, gripper.get_scoop_init_end_pose(kettle, rel_pos1, rel_pos3))
        init_pose = tuple(np.array(scoop_pose) + np.array([0, 5, 0])) # 7.5

        traj1 = HoldingTrajectory(grip_name, spoon_name, grasp, (init_pose, scoop_pose))
        traj2 = HoldingTrajectory(grip_name, spoon_name, grasp, (scoop_pose, end_pose))
        command = Command([traj1, traj2], [rel_pos1, rel_pos2, rel_pos3])
        yield (init_pose, end_pose, grasp, command)

class MotionHoldingGen(CondGen):
    use_context = True
    def generate(self, context=None):
        self.enumerated = True # Prevents from being resampled
        grip_name, pose1, cup_name, grasp, pose2 = self.inputs
        if not ph.do_motion:
            path = (pose1, pose2)
            command = Command([HoldingTrajectory(grip_name, cup_name, grasp, path)])
            return [(command,)]

        kitchen = ph.make_kitchen(planning=True)
        if self.use_context and (context is not None):
            for atom in context.conditions:
                _, name, pose = atom.head.args
                if name not in [grip_name, cup_name] and not isinstance(pose, SharedOutputSet):
                    ph.make_body(kitchen, name, pose)
        gripper = Gripper(kitchen, init_pos=pose1[:2], init_angle=pose1[2])
        cup = ph.make_body(kitchen, cup_name, ph.cupInitPos)
        
        gripper.set_grasped(cup, grasp, pose1[:2], pose1[2])
        motion_angle = min(pose1[2], pose2[2])
        path = gripper.plan_path(pose2[:2], pose2[2], 
                                 motion_angle=motion_angle, collision_buffer=ph.buffer_distance)
        if path is None:
            return []
        command = Command([HoldingTrajectory(grip_name, cup_name, grasp, path)])
        return [(command,)]

def genDump(grip_name, spoon_name, grasp, kettle_name, pose2):
    kitchen = ph.make_kitchen(planning=True)
    gripper = ph.make_body(kitchen, grip_name, ph.gripperInitPos)
    spoon = ph.make_body(kitchen, spoon_name, ph.cupInitPos)
    kettle = ph.make_body(kitchen, kettle_name, pose2)

    gripper.set_grasped(spoon, grasp, ph.gripperInitPos[:2], ph.gripperInitPos[2])
    rel_pos_x = 0.8
    init_pose, dump_pose = map(tuple, gripper.get_dump_init_end_pose(kettle, rel_pos_x))

    traj1 = HoldingTrajectory(grip_name, spoon_name, grasp, (init_pose, dump_pose))
    return (init_pose, dump_pose, Command([traj1], parameters=[rel_pos_x]))

def genStir(grip_name, stir_name, grasp, kettle_name, pose2):
    kitchen = ph.make_kitchen(planning=True)
    gripper = ph.make_body(kitchen, grip_name, ph.gripperInitPos)
    stirrer = ph.make_body(kitchen, stir_name, ph.cupInitPos)
    kettle = ph.make_body(kitchen, kettle_name, pose2)

    gripper.set_grasped(stirrer, grasp, ph.gripperInitPos[:2], ph.gripperInitPos[2])
    init_pose, stir_pose = map(tuple, gripper.get_stir_init_end_pose(kettle, (0, 0), (1, 0)))
    up_pose = tuple(np.array(init_pose) + np.array([0, 4, 0]))

    traj1 = HoldingTrajectory(grip_name, stir_name, grasp, (up_pose, init_pose))
    traj2 = HoldingTrajectory(grip_name, stir_name, grasp, (init_pose, stir_pose))
    return (up_pose, Command([traj1, traj2, traj2.reverse(), traj1.reverse()]))

def genPush(grip_name, block_name, pose3, pose4):
    assert pose3[1:] == pose4[1:]
    kitchen = ph.make_kitchen(planning=True)
    gripper = ph.make_body(kitchen, grip_name, ph.gripperInitPos)
    block = ph.make_body(kitchen, block_name, pose3)

    push_start, push_end = gripper.get_push_pos(block, pose4[0])
    push_start, push_end = tuple(np.append(push_start, 0)), tuple(np.append(push_end, 0))

    rel_up = np.array([0, 4, 0])
    up_start = tuple(push_start + rel_up)
    up_end = tuple(push_end + rel_up)

    traj1 = ClosedTrajectory(grip_name, [up_start, push_start])
    traj2 = PushTrajectory(grip_name, block_name, pose3, [push_start, push_end])
    traj3 = ClosedTrajectory(grip_name, [push_end, up_end])
    command = Command([traj1, traj2, traj3])
    return (up_start, up_end, command)

def stackGen(cup_name, block_name, bottom_pose):
    kitchen = ph.make_kitchen(planning=True)
    block = ph.make_body(kitchen, block_name, bottom_pose)
    upper = np.array(ks.get_body_vertices(block)).max(axis=0)
    top_pose = (bottom_pose[0], upper[1] + ks.EPS/2., 0)
    return (top_pose,)

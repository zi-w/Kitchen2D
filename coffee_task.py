import sys
from ss.model.functions import Predicate, rename_functions, initialize, TotalCost, Increase
from ss.model.problem import Problem
from ss.model.operators import Action, Axiom
from ss.model.streams import GenStream, CondGen, Stream, FnStream
from ss.algorithms.dual_focused import dual_focused
from ss.utils import INF

import numpy as np
import sys
import cProfile
import pstats
import time
import random

import kitchen2d.kitchen_stuff as ks
from kitchen2d.kitchen_stuff import Gripper, Kitchen2D
from utils.kitchen_utils import scale_cost, FreeTrajectory, HoldingTrajectory, PushTrajectory, Command, print_plan, \
    set_pose, step_command, ClosedTrajectory
import utils.helper as helper

##################################################

IS_UNIFORM = False
IS_ADAPT = False
TASK_LENGTH_SCALE = None
MAX_EXP_ID = 50

holder_d = 0.5
stirrer_w = holder_d
block_goal = (-25, 0, 0)

KITCHEN_PARAMS = {
    'do_gui': False,
    'sink_w': 10.,
    'sink_h': 5.,
    'sink_d': 1.,
    'sink_pos_x': 20.,
    'left_table_width': 55.,
    'right_table_width': 5.,
    #'faucet_h': 12., # TODO(caelan): collisions when 12?
    'faucet_h': 15.,
    'faucet_w': 5.,
    'faucet_d': 0.5,
    'obstacles': [],
    #'obstacles': [((-ks.SCREEN_WIDTH/2, 10), (5, 10)), ((ks.SCREEN_WIDTH/2, 10), (5, 10))]
    # TODO(caelan): obstacles for holding the tools
}

EXP_CONFIGS = [
    {
        'expid_pour': 0,
        'c_i_pour': 0,
        'expid_scoop': 0,
        'c_i_scoop': 0,
        'gripper_x': 0,
        'cup_x': 7.5,
        'sugar_x': -7.5,
        'cream_x': 15,
        'spoon_x': 2.5, # -5 | 15 | -10
        'stir_x': 20, # 0
        'block_x': -20,
    },
    {
        'cup_x': -6.381114728205619,
        'gripper_x': 0,
        'c_i_pour': 0,
        'cream_x': -0.605213846078481,
        'expid_scoop': 12,
        'spoon_x': 4.997754557358444,
        'block_x': -20,
        'c_i_scoop': 0,
        'expid_pour': 39,
        'stir_x': -11.987259132334422,
        'sugar_x': 15.0
    },
    {
        'cup_x': 11.51532580864416,
        'gripper_x': 0,
        'c_i_pour': 0,
        'cream_x': -9.388623747066761,
        'expid_scoop': 8,
        'spoon_x': -15.431875022911019,
        'block_x': -22.5,
        'c_i_scoop': 0,
        'expid_pour': 18,
        'stir_x': 17.39794403202508,
        'sugar_x': 1.0
    }
]

##################################################

def create_problem(initial_poses, make_kitchen, make_body,
                   expid_pour, c_i_pour, expid_scoop, c_i_scoop,
                   do_motion=True, do_collisions=True, do_grasp=True):

    gripperInitPos = initial_poses['gripper']
    cupInitPos = initial_poses['cup']

    if not do_motion:
        print 'Warning! Disabled motion planning'
    if not do_collisions:
        print 'Warning! Movable object collisions are disabled'

    ##################################################

    # Objective is the numeric function to minimize
    objective = TotalCost()

    # Parameter names (for predicate, action, axiom, and stream declarations)
    # Parameters are strings with '?' as the prefix
    GRIPPER = '?gripper'
    CUP = '?cup'
    KETTLE = '?kettle'
    BLOCK = '?block'
    POSE = '?end_pose'
    POSE2 = '?pose2'
    POSE3 = '?pose3'
    POSE4 = '?pose4'
    GRASP = '?grasp'
    CONTROL = '?control'

    ##################################################

    # Static predicates (predicates that do not change over time)
    IsGripper = Predicate([GRIPPER])
    IsCup = Predicate([CUP])
    IsStirrer = Predicate([KETTLE])
    IsSpoon = Predicate([KETTLE])
    IsBlock = Predicate([CUP])
    IsPourable = Predicate([CUP])
    #IsGraspable = Predicate([CUP])

    IsPose = Predicate([CUP, POSE])
    IsGrasp = Predicate([CUP, GRASP])
    IsControl = Predicate([CONTROL])

    CanGrasp = Predicate([GRIPPER, POSE, CUP, POSE2, GRASP, CONTROL])
    BelowFaucet = Predicate([GRIPPER, POSE, CUP, GRASP])
    CanPour = Predicate([GRIPPER, POSE, CUP, GRASP, KETTLE, POSE2, CONTROL])
    Motion = Predicate([GRIPPER, POSE, POSE2, CONTROL])
    MotionH = Predicate([GRIPPER, POSE, CUP, GRASP, POSE2, CONTROL])

    CanScoop = Predicate([GRIPPER, POSE, POSE2, CUP, GRASP, KETTLE, POSE3, CONTROL])
    CanDump = Predicate([GRIPPER, POSE, CUP, GRASP, KETTLE, POSE2, CONTROL])
    CanStir = Predicate([GRIPPER, POSE, CUP, GRASP, KETTLE, POSE2, CONTROL])
    Push = Predicate([GRIPPER, POSE, POSE2, CUP, POSE3, POSE4, CONTROL])

    Stackable = Predicate([CUP, BLOCK])
    BlockSupport = Predicate([CUP, POSE, BLOCK, POSE2]) # [POSE, POSE2]
    Clear = Predicate([BLOCK])
    TableSupport = Predicate([POSE])
    #Cool = Predicate([POSE])
    #Hot = Predicate([POSE])

    # Fluent predicates (predicates that change over time)
    AtPose = Predicate([CUP, POSE])
    Grasped = Predicate([CUP, GRASP])
    Empty = Predicate([GRIPPER])
    CanMove = Predicate([GRIPPER])
    HasCoffee = Predicate([CUP])
    HasSugar = Predicate([CUP])
    HasCream = Predicate([CUP])
    Mixed = Predicate([CUP])
    Scooped = Predicate([CUP])

    # Derived predicates (predicates updated by axioms)
    Unsafe = Predicate([CONTROL])
    Holding = Predicate([CUP])
    On = Predicate([CUP, BLOCK])

    ##################################################

    AIR_POSE = (0, 100, 0)

    def set_open(gripper, pose):
        set_pose(gripper, AIR_POSE)
        gripper.open()
        set_pose(gripper, pose)
        return True

    def set_holding_simulate(gripper, pose, body, grasp):
        set_pose(gripper, AIR_POSE)
        gripper.open()
        init_pose = initial_poses[body.name]
        set_pose(body, init_pose)  # Setting at initial end_pose to prevent gravity from pulling down
        gripper.set_position(gripper.gripper_from_body(body, grasp), 0)
        gripper.close()
        if not gripper.check_grasp(body):
            return False
        set_pose(gripper, pose)
        return True

    def set_holding_instant(gripper, pose, body, grasp):
        gripper.release(); gripper.set_open()
        set_pose(gripper, pose) # Resetting in case moved on previous iteration
        set_pose(body, AIR_POSE)
        body.position = gripper.body_from_gripper(body, grasp)
        #gripper.close() # TODO: cannot do this when the object is in the air
        gripper.set_closed(body); gripper.attach(body)
        #return gripper.check_grasp(body)
        return True

    ##################################################

    def test_collision(command, body_name, pose):
        if not do_collisions:
            return False
        for traj in command.trajectories:
            if body_name == traj.gripper or (isinstance(traj, HoldingTrajectory) and (body_name == traj.body)):
                continue

            kitchen = make_kitchen(planning=True)
            init_pose = gripperInitPos
            gripper = Gripper(kitchen, init_pos=init_pose[:2], init_angle=init_pose[2])
            if isinstance(traj, FreeTrajectory):
                gripper.release()
                gripper.set_open()
            else:
                body = make_body(kitchen, traj.body, cupInitPos)
                body.position = gripper.body_from_gripper(body, traj.pos_ratio)
                gripper.set_closed(body)
                gripper.attach(body)
            make_body(kitchen, body_name, pose)
            for q in traj.path:
                if gripper.check_point_collision(q[:2], q[2]):
                    print 'Collision: {}'.format(body_name)
                    return True
        # TODO: don't really need to fill this in if I don't do eager
        # TODO: would be nice to have for the non move actions
        return False

    # External predicates (boolean functions evaluated by fn)
    Collision = Predicate([CONTROL, GRIPPER, POSE], domain=[IsControl(CONTROL), IsPose(GRIPPER, POSE)],
                          fn=test_collision, bound=False) # Bound is the "optimistic" value

    # This is just used for debugging
    rename_functions(locals())

    ##################################################

    # TODO: prevent double moves
    actions = [
        Action(name='move', param=[GRIPPER, POSE, POSE2, CONTROL],
               pre=[Motion(GRIPPER, POSE, POSE2, CONTROL),
                    Empty(GRIPPER), CanMove(GRIPPER),
                    AtPose(GRIPPER, POSE), ~Unsafe(CONTROL)],
               eff=[AtPose(GRIPPER, POSE2),
                    ~AtPose(GRIPPER, POSE), ~CanMove(GRIPPER),
                    Increase(TotalCost(), scale_cost(1))]),

        Action(name='move-holding', param=[GRIPPER, POSE, CUP, GRASP, POSE2, CONTROL],
               pre=[MotionH(GRIPPER, POSE, CUP, GRASP, POSE2, CONTROL),
                    AtPose(GRIPPER, POSE), Grasped(CUP, GRASP), CanMove(GRIPPER), ~Unsafe(CONTROL)],
               eff=[AtPose(GRIPPER, POSE2),
                    ~AtPose(GRIPPER, POSE), ~CanMove(GRIPPER),
                    Increase(TotalCost(), scale_cost(1))]),

        # TODO: Could add a control and precondition that nothing blocks the filling of the water
        Action(name='fill', param=[GRIPPER, POSE, CUP, GRASP],
               pre=[BelowFaucet(GRIPPER, POSE, CUP, GRASP),
                    AtPose(GRIPPER, POSE), Grasped(CUP, GRASP)],
               eff=[HasCoffee(CUP), CanMove(GRIPPER),
                    Increase(TotalCost(), scale_cost(1))]),

        # TODO: ideally would make the contents of things a variable

        # TODO: Could add a precondition that nothing blocks the pouring of the water
        Action(name='pour-gp', param=[GRIPPER, POSE, CUP, GRASP, KETTLE, POSE2, CONTROL],
               pre=[CanPour(GRIPPER, POSE, CUP, GRASP, KETTLE, POSE2, CONTROL),
                    AtPose(GRIPPER, POSE), Grasped(CUP, GRASP), AtPose(KETTLE, POSE2), HasCream(CUP),
                    HasCoffee(KETTLE)
                    ],
               eff=[HasCream(KETTLE), CanMove(GRIPPER),
                    ~HasCream(CUP),
                    Increase(TotalCost(), scale_cost(1))]),

        Action(name='scoop', param=[GRIPPER, POSE, POSE2, CUP, GRASP, KETTLE, POSE3, CONTROL],
               pre=[CanScoop(GRIPPER, POSE, POSE2, CUP, GRASP, KETTLE, POSE3, CONTROL),
                    AtPose(GRIPPER, POSE), Grasped(CUP, GRASP), AtPose(KETTLE, POSE3), HasSugar(KETTLE)],
               eff=[AtPose(GRIPPER, POSE2), HasSugar(CUP), CanMove(GRIPPER), Scooped(CUP),
                    ~AtPose(GRIPPER, POSE),
                    Increase(TotalCost(), scale_cost(1))]),

        Action(name='dump', param=[GRIPPER, POSE, CUP, GRASP, KETTLE, POSE2, CONTROL],
               pre=[CanDump(GRIPPER, POSE, CUP, GRASP, KETTLE, POSE2, CONTROL),
                    AtPose(GRIPPER, POSE), Grasped(CUP, GRASP), AtPose(KETTLE, POSE2), HasSugar(CUP),
                    HasCoffee(KETTLE)
                    ],
               eff=[HasSugar(KETTLE), CanMove(GRIPPER),
                    ~HasSugar(CUP), ~Scooped(CUP),
                    Increase(TotalCost(), scale_cost(1))]),

        # TODO: conditional effect to mixed whatever is incorporated
        Action(name='stir', param=[GRIPPER, POSE, CUP, GRASP, KETTLE, POSE2, CONTROL],
               pre=[CanStir(GRIPPER, POSE, CUP, GRASP, KETTLE, POSE2, CONTROL),
                    AtPose(GRIPPER, POSE), Grasped(CUP, GRASP), AtPose(KETTLE, POSE2),
                    HasCoffee(KETTLE), HasCream(KETTLE), HasSugar(KETTLE)
                    ],
               eff=[Mixed(KETTLE), CanMove(GRIPPER),
                    Increase(TotalCost(), scale_cost(1))]),

        Action(name='pick', param=[GRIPPER, POSE, CUP, POSE2, GRASP, CONTROL],
               pre=[CanGrasp(GRIPPER, POSE, CUP, POSE2, GRASP, CONTROL), TableSupport(POSE2),
                    AtPose(GRIPPER, POSE), AtPose(CUP, POSE2), Empty(GRIPPER), ~Unsafe(CONTROL)],
               eff=[Grasped(CUP, GRASP), CanMove(GRIPPER),
                    ~AtPose(CUP, POSE2), ~Empty(GRIPPER),
                    Increase(TotalCost(), scale_cost(1))]),

        #Action(name='unstack', param=[GRIPPER, POSE, CUP, POSE2, GRASP, CONTROL],
        #       pre=[CanGrasp(GRIPPER, POSE, CUP, POSE2, GRASP, CONTROL),
        #            AtPose(GRIPPER, POSE), AtPose(CUP, POSE2), Empty(GRIPPER), ~Unsafe(CONTROL)],
        #       eff=[Grasped(CUP, GRASP), CanMove(GRIPPER),
        #            ~AtPose(CUP, POSE2), ~Empty(GRIPPER),
        #            Increase(TotalCost(), scale_cost(1))]),

        # TODO(caelan): precondition that cannot place hot thing directly on table
        Action(name='place', param=[GRIPPER, POSE, CUP, POSE2, GRASP, CONTROL],
               pre=[CanGrasp(GRIPPER, POSE, CUP, POSE2, GRASP, CONTROL), TableSupport(POSE2),
                    AtPose(GRIPPER, POSE), Grasped(CUP, GRASP), ~Scooped(CUP), ~Unsafe(CONTROL)],
               eff=[AtPose(CUP, POSE2), Empty(GRIPPER), CanMove(GRIPPER),
                    ~Grasped(CUP, GRASP),
                    Increase(TotalCost(), scale_cost(1))]),

        Action(name='stack', param=[GRIPPER, POSE, CUP, POSE2, GRASP, BLOCK, POSE3, CONTROL],
               pre=[CanGrasp(GRIPPER, POSE, CUP, POSE2, GRASP, CONTROL), BlockSupport(CUP, POSE2, BLOCK, POSE3),
                    AtPose(GRIPPER, POSE), Grasped(CUP, GRASP), AtPose(BLOCK, POSE3), Clear(BLOCK), ~Unsafe(CONTROL)],
               eff=[AtPose(CUP, POSE2), Empty(GRIPPER), CanMove(GRIPPER),
                    ~Grasped(CUP, GRASP), ~Clear(BLOCK),
                    Increase(TotalCost(), scale_cost(1))]),

        Action(name='push', param=[GRIPPER, POSE, POSE2, BLOCK, POSE3, POSE4, CONTROL],
               pre=[Push(GRIPPER, POSE, POSE2, BLOCK, POSE3, POSE4, CONTROL),
                    AtPose(GRIPPER, POSE), AtPose(BLOCK, POSE3), Empty(GRIPPER), Clear(BLOCK)],
               eff=[AtPose(GRIPPER, POSE2), AtPose(BLOCK, POSE4), CanMove(GRIPPER),
                    ~AtPose(GRIPPER, POSE), ~AtPose(BLOCK, POSE3),
                    Increase(TotalCost(), scale_cost(1))]),
    ]

    # Axioms (inference rules that are automatically applied at every state)
    axioms = [
        #Axiom(param=[CONTROL, CUP, POSE],
        #      pre=[Collision(CONTROL, CUP, POSE),
        #           AtPose(CUP, POSE)],
        #      eff=Unsafe(CONTROL)),
        Axiom(param=[CUP, GRASP],
              pre=[IsGrasp(CUP, GRASP),
                   Grasped(CUP, GRASP)],
              eff=Holding(CUP)),

        Axiom(param=[CUP, POSE, BLOCK, POSE2],
              pre=[BlockSupport(CUP, POSE, BLOCK, POSE2),
                   AtPose(CUP, POSE), AtPose(BLOCK, POSE2)],
              eff=On(CUP, BLOCK)),
    ]

    ##################################################

    def genGrasp(grip_name, cup_name, pose2, grasp):
        kitchen = make_kitchen(planning=True)
        gripper = make_body(kitchen, grip_name, gripperInitPos)
        gripper.open()

        cup = make_body(kitchen, cup_name, pose2)
        down_pos, up_pos = gripper.get_grasp_poses(cup, grasp)
        # TODO(caelan): the up poses are often not very good
        up_pos = np.array(down_pos) + np.array([0, 5])

        ori = 0
        up_pose = (up_pos[0], up_pos[1], ori)
        down_pose = (down_pos[0], down_pos[1], ori)
        #traj = FreeTrajectory(gripper, [up_pose, down_pose, up_pose])

        if not do_grasp:
            traj1 = FreeTrajectory(grip_name, (up_pose, down_pose))
            traj2 = HoldingTrajectory(grip_name, cup_name, grasp, (down_pose, up_pose))
            return (up_pose, Command([traj1, traj2]))

        #gripper.set_position(up_pose[:2], up_pose[2])
        #path1 = gripper.plan_linear(down_pose[:2], down_pose[2]) # plan_path
        #if path1 is None: return
        #traj1 = FreeTrajectory(grip_name, path1)
        traj1 = FreeTrajectory(grip_name, (up_pose, down_pose))
        gripper.set_position(down_pose[:2], down_pose[2])

        gripper.close()
        if not gripper.check_grasp(cup): return

        #path2 = gripper.plan_linear(up_pose[:2], up_pose[2]) # plan_path
        #if path2 is None: return
        #traj2 = HoldingTrajectory(grip_name, cup_name, pos_ratio, path2)
        traj2 = HoldingTrajectory(grip_name, cup_name, grasp, (down_pose, up_pose))

        command = Command([traj1, traj2])
        return (up_pose, command)

    def genFaucet(grip_name, cup_name, grasp):
        kitchen = make_kitchen(planning=True)
        gripper = make_body(kitchen, grip_name, gripperInitPos)
        gripper.open()
        # cupInitPos[2] has to be 0
        cup = make_body(kitchen, cup_name, cupInitPos)
        if not gripper.set_grasped(cup, grasp, gripperInitPos[:2], gripperInitPos[2]):
            return
        # TODO: this seems a little fishy
        pose = gripper.get_water_pos()
        ori = 0
        pose = (pose[0], pose[1], ori)
        return (pose,)

    pourDict = {}
    def genPour(grip_name, cup_name, kettle_name, pose2):
        if not np.isclose(pose2[2], 0, atol=0.05):
            return
        ########################GP BEGIN##########################
        kitchen = make_kitchen(planning=True)
        gripper = Gripper(kitchen, init_pos=gripperInitPos[:2], init_angle=gripperInitPos[2])
        cup = make_body(kitchen, cup_name, cupInitPos)
        kettle = make_body(kitchen, kettle_name, pose2)

        start = time.time()
        gp, c = helper.process_gp_sample(expid_pour, c_i_pour, IS_ADAPT, IS_UNIFORM,
                                         task_lengthscale=TASK_LENGTH_SCALE)
        pourDict['gp_timing'] = time.time() - start
        pourDict['gp'] = gp
        tot = []
        pourDict['timing'] = tot
        while True:
            start = time.time()
            #import pdb; pdb.set_trace()
            grasp, rel_x, rel_y, dangle, _, _, _, _ = gp.sample(c) # _uniform
            tot.append(time.time() - start)
            dangle *= np.sign(rel_x)
            rel_pose = (rel_x, rel_y, dangle)
            pour_pose = tuple(np.array(pose2) + np.array(rel_pose))
            initial_pose = tuple(np.array(pose2) + (rel_x, rel_y, 0))

            if not set_holding_simulate(gripper, initial_pose, cup, grasp):
                continue
            if gripper.check_point_collision(initial_pose[:2], initial_pose[2]):
                continue
            traj = HoldingTrajectory(grip_name, cup_name, grasp, [initial_pose, pour_pose])
            command = Command([traj, traj.reverse()], parameters=[rel_pose[:2], rel_pose[2], 0.45, 200])
            yield (grasp, initial_pose, command)

    class MotionGen(CondGen):
        # TODO(caelan): can buffer these to ensure we don't get too close to obstacles
        use_context = True
        def generate(self, context=None):
            self.enumerated = True # Prevents from being resampled
            grip_name, pose1, pose2 = self.inputs
            if not do_motion:
                path = (pose1, pose2)
                command = Command([FreeTrajectory(grip_name, path)])
                return [(command,)]

            kitchen = make_kitchen(planning=True) # TODO: could also move to top
            if self.use_context and (context is not None):
                for atom in context.conditions:
                    assert(atom.head.function is Collision)
                    _, name, pose = atom.head.args
                    if name not in [grip_name]:
                        make_body(kitchen, name, pose)

            # TODO(caelan): need to be careful that doesn't collide with pouring cup
            buffer_distance = 0.2 # 1.0
            buffered_kitchen = ks.buffer_kitchen(kitchen, None, radius=buffer_distance)
            gripper = Gripper(buffered_kitchen, init_pos=pose1[:2], init_angle=pose1[2])
            set_open(gripper, pose1)
            #buffered_kitchen.enable_gui(); buffered_kitchen.draw(); raw_input('Continue?')

            path = gripper.plan_path(pose2[:2], pose2[2])
            if path is None:
                return []
            command = Command([FreeTrajectory(grip_name, path)])
            return [(command,)]

    def genScoop(grip_name, spoon_name, kettle_name, pose2):
        kitchen = make_kitchen(planning=True)
        gripper = make_body(kitchen, grip_name, gripperInitPos)
        spoon = make_body(kitchen, spoon_name, cupInitPos)
        kettle = make_body(kitchen, kettle_name, pose2)

        #####################GP START##############################
        gp, c = helper.process_gp_sample(expid_pour, c_i_scoop, is_adapt=IS_ADAPT, is_uniform=IS_UNIFORM,
                                         task_lengthscale=TASK_LENGTH_SCALE, betalambda=0.998, exp='scoop')
        while True:
            rel_x1, rel_y1, rel_x2, rel_y2, rel_x3, rel_y3, grasp, _, _ = gp.sample(c)
            rel_pos1 = (rel_x1, rel_y1); rel_pos2 = (rel_x2, rel_y2); rel_pos3 = (rel_x3, rel_y3)
            ######################GP END#############################

            #if not set_holding_simulate(gripper, initial_pose, spoon, grasp):
            #    continue
            #if gripper.check_point_collision(initial_pose[:2], initial_pose[2]):
            #    continue

            set_holding_instant(gripper, gripperInitPos, spoon, grasp)
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
            if not do_motion:
                path = (pose1, pose2)
                command = Command([HoldingTrajectory(grip_name, cup_name, grasp, path)])
                return [(command,)]

            kitchen = make_kitchen(planning=True)
            if self.use_context and (context is not None):
                for atom in context.conditions:
                    assert(atom.head.function is Collision)
                    _, name, pose = atom.head.args
                    if name not in [grip_name, cup_name]:
                        make_body(kitchen, name, pose)
            buffer_distance = 0.2 # 1.0
            buffered_kitchen = ks.buffer_kitchen(kitchen, None, radius=buffer_distance)
            gripper = Gripper(buffered_kitchen, init_pos=pose1[:2], init_angle=pose1[2])
            cup = make_body(buffered_kitchen, cup_name, cupInitPos)

            if not set_holding_instant(gripper, pose1, cup, grasp): # Ground is buffered...
                return []
            #buffered_kitchen.enable_gui(); buffered_kitchen.draw(); raw_input('Continue?')

            path = gripper.plan_path(pose2[:2], pose2[2])
            if path is None:
                return []
            command = Command([HoldingTrajectory(grip_name, cup_name, grasp, path)])
            return [(command,)]

    def genDump(grip_name, spoon_name, grasp, kettle_name, pose2):
        kitchen = make_kitchen(planning=True)
        gripper = make_body(kitchen, grip_name, gripperInitPos)
        spoon = make_body(kitchen, spoon_name, cupInitPos)
        kettle = make_body(kitchen, kettle_name, pose2)

        set_holding_instant(gripper, gripperInitPos, spoon, grasp)
        rel_pos_x = 0.8
        init_pose, dump_pose = map(tuple, gripper.get_dump_init_end_pose(kettle, rel_pos_x))

        traj1 = HoldingTrajectory(grip_name, spoon_name, grasp, (init_pose, dump_pose))
        traj2 = HoldingTrajectory(grip_name, spoon_name, grasp, (dump_pose, init_pose))
        return (init_pose, Command([traj1, traj2], parameters=[rel_pos_x]))

    def genStir(grip_name, stir_name, grasp, kettle_name, pose2):
        kitchen = make_kitchen(planning=True)
        gripper = make_body(kitchen, grip_name, gripperInitPos)
        stirrer = make_body(kitchen, stir_name, cupInitPos)
        kettle = make_body(kitchen, kettle_name, pose2)

        stir_offset = np.array([0, 3, 0])
        set_holding_instant(gripper, gripperInitPos, stirrer, grasp)
        init_pose, stir_pose = map(tuple, gripper.get_stir_init_end_pose(kettle, (0, 0), (1, 0)))
        init_pose = tuple(np.array(init_pose) + stir_offset)
        stir_pose = tuple(np.array(stir_pose) + stir_offset)
        up_pose = tuple(np.array(init_pose) + np.array([0, 4, 0]))

        traj1 = HoldingTrajectory(grip_name, stir_name, grasp, (up_pose, init_pose))
        traj2 = HoldingTrajectory(grip_name, stir_name, grasp, (init_pose, stir_pose))
        return (up_pose, Command([traj1, traj2, traj2.reverse(), traj1.reverse()]))

    def genPush(grip_name, block_name, pose3, pose4):
        assert pose3[1:] == pose4[1:]
        kitchen = make_kitchen(planning=True)
        gripper = make_body(kitchen, grip_name, gripperInitPos)
        block = make_body(kitchen, block_name, pose3)

        #direction = np.sign(pose4[0] - pose3[0])
        #rel_push = np.array([-direction*1, 0, 0])
        #push_start = tuple(np.array(pose3) + rel_push)
        #push_end = tuple(np.array(pose4) + rel_push)

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
        kitchen = make_kitchen(planning=True)
        block = make_body(kitchen, block_name, bottom_pose)
        upper = np.array(ks.get_body_vertices(block)).max(axis=0)
        #cup = make_body(kitchen, cup_name, )
        top_pose = (bottom_pose[0], upper[1], 0)
        return (top_pose,)

    ##################################################

    bound = 'shared' # unqiue | depth | shared

    # Stream declarations
    # inp: a list of input parameters
    # domain: a conjunctive list of atoms indicating valid inputs defined on inp
    # out: a list of output parameters
    # graph: a list of atoms certified by the stream defined on both inp and out
    streams = [
        # Unconditional streams (streams with no input parameters)
        FnStream(name='sample-grasp', inp=[GRIPPER, CUP, POSE2, GRASP],
                  domain=[IsGripper(GRIPPER), IsPose(CUP, POSE2), IsGrasp(CUP, GRASP)],
                  fn=genGrasp, out=[POSE, CONTROL],
                  graph=[CanGrasp(GRIPPER, POSE, CUP, POSE2, GRASP, CONTROL),
                         IsPose(GRIPPER, POSE), IsControl(CONTROL)], bound=bound),

        FnStream(name='sample-stack', inp=[CUP, BLOCK, POSE2],
                  domain=[Stackable(CUP, BLOCK), IsPose(BLOCK, POSE2)],
                  fn=stackGen, out=[POSE],
                  graph=[BlockSupport(CUP, POSE, BLOCK, POSE2),
                         IsPose(CUP, POSE)], bound=bound),

        FnStream(name='sample-fill', inp=[GRIPPER, CUP, GRASP],
                  domain=[IsGripper(GRIPPER), IsCup(CUP), IsGrasp(CUP, GRASP)],
                  fn=genFaucet, out=[POSE],
                  graph=[BelowFaucet(GRIPPER, POSE, CUP, GRASP),
                         IsPose(GRIPPER, POSE)], bound=bound),

        GenStream(name='sample-pour', inp=[GRIPPER, CUP, KETTLE, POSE2],
                  domain=[IsGripper(GRIPPER), IsPourable(CUP), IsCup(KETTLE), IsPose(KETTLE, POSE2)],
                  fn=genPour, out=[GRASP, POSE, CONTROL],
                  graph=[CanPour(GRIPPER, POSE, CUP, GRASP, KETTLE, POSE2, CONTROL),
                         IsPose(GRIPPER, POSE), IsGrasp(CUP, GRASP)], bound=bound), # IsPourControl(CONTROL),

        GenStream(name='sample-scoop', inp=[GRIPPER, CUP, KETTLE, POSE3],
                  domain=[IsGripper(GRIPPER), IsSpoon(CUP), IsCup(KETTLE), IsPose(KETTLE, POSE3)],
                  fn=genScoop, out=[POSE, POSE2, GRASP, CONTROL],
                  graph=[CanScoop(GRIPPER, POSE, POSE2, CUP, GRASP, KETTLE, POSE3, CONTROL),
                         IsPose(GRIPPER, POSE), IsPose(GRIPPER, POSE2), IsGrasp(CUP, GRASP)], bound=bound),

        FnStream(name='sample-dump', inp=[GRIPPER, CUP, GRASP, KETTLE, POSE2],
                  domain=[IsGripper(GRIPPER), IsSpoon(CUP), IsGrasp(CUP, GRASP), IsCup(KETTLE), IsPose(KETTLE, POSE2)],
                  fn=genDump, out=[POSE, CONTROL],
                  graph=[CanDump(GRIPPER, POSE, CUP, GRASP, KETTLE, POSE2, CONTROL),
                         IsPose(GRIPPER, POSE)], bound=bound),

        FnStream(name='sample-stir', inp=[GRIPPER, CUP, GRASP, KETTLE, POSE2],
                  domain=[IsGripper(GRIPPER), IsStirrer(CUP), IsGrasp(CUP, GRASP), IsCup(KETTLE), IsPose(KETTLE, POSE2)],
                  fn=genStir, out=[POSE, CONTROL],
                  graph=[CanStir(GRIPPER, POSE, CUP, GRASP, KETTLE, POSE2, CONTROL),
                         IsPose(GRIPPER, POSE)], bound=bound),

        Stream(name='sample-motion', inp=[GRIPPER, POSE, POSE2],
                  domain=[IsGripper(GRIPPER), IsPose(GRIPPER, POSE), IsPose(GRIPPER, POSE2)],
                  fn=MotionGen, out=[CONTROL],
                  graph=[Motion(GRIPPER, POSE, POSE2, CONTROL), IsControl(CONTROL)], bound=bound),

        Stream(name='sample-motion-h', inp=[GRIPPER, POSE, CUP, GRASP, POSE2],
                  domain=[IsGripper(GRIPPER), IsPose(GRIPPER, POSE), IsGrasp(CUP, GRASP), IsPose(GRIPPER, POSE2)],
                  fn=MotionHoldingGen, out=[CONTROL],
                  graph=[MotionH(GRIPPER, POSE, CUP, GRASP, POSE2, CONTROL), IsControl(CONTROL)], bound=bound),

        FnStream(name='sample-push', inp=[GRIPPER, BLOCK, POSE3, POSE4],
               domain=[IsGripper(GRIPPER), IsBlock(BLOCK), IsPose(BLOCK, POSE3), IsPose(BLOCK, POSE4)],
               fn=genPush, out=[POSE, POSE2, CONTROL],
               graph=[Push(GRIPPER, POSE, POSE2, BLOCK, POSE3, POSE4, CONTROL),
                      IsPose(GRIPPER, POSE), IsPose(GRIPPER, POSE2)], bound=bound),
    ]

    initial_atoms = [
        # Static atoms
        IsGrasp('cup', .5),
        #IsGrasp('spoon', .8),
        IsGrasp('stirrer', .8),
        #IsGrasp('cream_cup', .5),
        IsPose('block', block_goal),

        # Fluent atoms
        Empty('gripper'),
        CanMove('gripper'),
        HasSugar('sugar_cup'),
        HasCream('cream_cup'),
        IsPourable('cream_cup'),

        Stackable('cup', 'block'),
        Clear('block'),

        # Fluent function
        initialize(TotalCost(), 0), # Maintains the total plan cost
    ]
    for name, pose in initial_poses.items():
        if 'gripper' in name:
            initial_atoms += [IsGripper(name)]
        if 'cup' in name:
            initial_atoms += [IsCup(name)]
        if 'spoon' in name:
            initial_atoms += [IsSpoon(name)]
        if 'stirrer' in name:
            initial_atoms += [IsStirrer(name)]
        if 'block' in name:
            initial_atoms += [IsBlock(name)]
        initial_atoms += [IsPose(name, pose), AtPose(name, pose), TableSupport(pose)]

    # Goal literals (list of conjunctive literals)
    goal_literals = [
        #Grasped(cup_name, goal_grasp),
        #HasCoffee(cup_name),
        #HasCoffee(kettle_name),
        #Empty(),
        #AtPose('gripper', gripperInitPos),
        #Holding(spoon_name),
        #HasSugar('spoon'),
        #HasSugar('cup'),
        #HasCream('cup'),
        #HasCoffee('cup'),
        #Holding('cream_cup'),
        #AtPose('cup', initial_poses['cup']),

        #AtPose('spoon', initial_poses['spoon']),
        #AtPose('cup', AIR_POSE),

        AtPose('block', block_goal),
        On('cup', 'block'),
        Mixed('cup'),
        Empty('gripper'),
    ]
    #for name, pose in initial_poses.items():
    #    goal_literals += [AtPose(name, pose)]

    # TODO(caelan): other ideas
    # - Remove cap from cup (pick or push)
    # - Press button on the faucet
    # - Dynamic pushing where something falls
    # - Push coffee into the fountain (how do you remove?)
    # - Stacking
    # - Push battery into the coffee machine
    # - Would be nice to show off other dependencies
    # - Hot must be placed on a coaster
    # - Move out of the way to place

    return Problem(initial_atoms, goal_literals, actions, axioms, streams, objective=objective), pourDict

##################################################

def initialize_kitchen(kitchen, make_body, initial_poses):
    kitchen.enable_gui()
    body_from_name = {}
    for name, pose in initial_poses.items():
        body_from_name[name] = make_body(kitchen, name, pose)
    gripper = body_from_name['gripper']
    kitchen.gen_liquid_in_cup(body_from_name['sugar_cup'], 1000, userData='sugar') # 300
    kitchen.gen_liquid_in_cup(body_from_name['cream_cup'], 100, userData='cream')
    kitchen.draw()
    raw_input('Start?')
    return body_from_name

def step_plan(plan, make_kitchen, make_body, initial_poses):
    kitchen = make_kitchen(planning=True)
    body_from_name = initialize_kitchen(kitchen, make_body, initial_poses)
    #kitchen.simulate(10.0)
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

def execute_plan(plan, make_kitchen, make_body, initial_poses):
    kitchen = make_kitchen(planning=False)
    body_from_name = initialize_kitchen(kitchen, make_body, initial_poses)
    #kitchen.disable_liquid = True

    finished = True
    score = -1
    buffer = 0.5 # TODO(caelan): buffer more on the bottom?
    # TODO(caelan): controllers are still shooting too low
    for i, (action, args) in enumerate(plan):
        print i, action, args
        #raw_input('Continue')
        gripper = body_from_name[args[0]]
        if action.name == 'move':
            _, _, pos2, _ = args
            finished = gripper.find_path(pos2[:2], pos2[2], buffer=buffer)
        elif action.name == 'move-holding':
            _, _, _, _, pos2, _ = args
            finished = gripper.find_path(pos2[:2], pos2[2], buffer=buffer, motion_angle=pos2[2])
        elif action.name == 'pick':
            _, dpos_up, cup_name, _, _, control = args
            dpos = control.trajectories[0].path[-1]
            gripper.apply_lowlevel_control(dpos[:2], dpos[2])
            gripper.close()
            print 'Check grasp:', gripper.check_grasp(body_from_name[cup_name])
            gripper.apply_lowlevel_control(dpos_up[:2], dpos_up[2])
            #finished &= gripper.grasp2(dpos[:2], dpos_up[:2], body_from_name[cup_name])
            #finished &= gripper.find_path(dpos_up[:2], dpos_up[2], buffer=None)
        elif action.name in ('place', 'stack'):
            #_, _, _, pos2, _, _ = args
            #_, dpos_up, _, _, _, control = args
            dpos_up, control = args[1], args[-1]
            dpos = control.trajectories[0].path[-1]
            gripper.apply_lowlevel_control(dpos[:2], dpos[2])
            gripper.open()
            gripper.apply_lowlevel_control(dpos_up[:2], dpos_up[2])
            #finished &= gripper.place(pos2[:2], pos2[2])
            #finished &= gripper.find_path(dpos_up[:2], dpos_up[2], buffer=None)
        elif action.name == 'fill':
            #kitchen.disable_liquid = False
            finished = gripper.get_water(10)
            #kitchen.simulate(5)
            #kitchen.disable_liquid = True
        elif action.name == 'pour-gp':
            _, p1, _, _, kettle_name, kp, command = args
            poured, score = gripper.pour(body_from_name[kettle_name], *command.parameters)
            gripper.apply_control([p1])
        elif action.name == 'scoop':
            _, _, _, spoon_name, _, cup_name, _, control = args
            print gripper.scoop(body_from_name[cup_name], *control.parameters)
        elif action.name == 'dump':
            #GRIPPER, POSE, CUP, GRASP, KETTLE, POSE2, CONTROL
            _, _, spoon_name, _, cup_name, _, control = args
            print gripper.dump(body_from_name[cup_name], *control.parameters)
            # TODO(caelan): should return to start pose
        elif action.name == 'stir':
            #GRIPPER, POSE, CUP, GRASP, KETTLE, POSE2, CONTROL
            _, _, stir_name, grasp, cup_name, _, control = args
            #stir_y = grasp + .2 #.15
            #gripper.stir(body_from_name[cup_name], (0, stir_y), (1, stir_y))
            gripper.apply_control(control.trajectories[0].path)
            for i in xrange(5):
                for traj in control.trajectories[1:-1]:
                    gripper.apply_control(traj.path)
                    #gripper.apply_lowlevel_control(dpos_up[:2], dpos_up[2])
            gripper.apply_control(control.trajectories[-1].path)
        elif action.name == 'push':
            # GRIPPER, POSE, POSE2, BLOCK, POSE3, POSE4, CONTROL
            _, _, _, block_name, _, _, control = args
            #gripper.push(body_from_name[block_name], (1., 0.), -6, 0.5)
            gripper.close()
            for traj in control.trajectories:
                gripper.apply_control(traj.path)
            gripper.open()
        else:
            raise ValueError(action.name)
        if not finished:
            #break
            raise RuntimeError('Failed to plan a path')
    return finished, score

##################################################

def get_pour_dims(expid, c_i):
    _, _, c_pour = helper.get_xx_yy(expid, c_i, 'gp_lse', exp='pour')
    _, _, pour_w, pour_h = c_pour
    return pour_w, pour_h

def get_scoop_dims(expid, c_i):
    _, _, c_scoop = helper.get_xx_yy(expid, c_i, 'gp_lse', exp='scoop')
    scoop_w, scoop_h = c_scoop
    return scoop_w, scoop_h

def main(test_args, expid_pour, c_i_pour, expid_scoop, c_i_scoop, gripper_x, cup_x, sugar_x, cream_x, spoon_x, stir_x, block_x):
    print 'Test Params:', test_args

    initial_poses = {
        'gripper': (gripper_x, 15., 0.),
        'cup': (cup_x, 0., 0.),
        'sugar_cup': (sugar_x, 0., 0.),
        'cream_cup': (cream_x, 0, 0),
        'spoon': (spoon_x, holder_d, 0),
        'stirrer': (stir_x, (3 + holder_d), 0),
        'block': (block_x, 0, 0),
    }

    def make_kitchen(planning=True):
        kitchen = ks.Kitchen2D(planning=planning, liquid_name='coffee', frequency=0.2, # frequency=0.02
                               **KITCHEN_PARAMS)  # TODO: could also move to top
        width_eps = 0.05
        spoon_holder = ks.make_static_cup(kitchen, (spoon_x-(holder_d+width_eps/2), 0), 0,
                                     2+width_eps, 2, holder_d) # (15-2.5/2
        #spoon_holder = ks.make_cup(kitchen, (15, 0), 0, 2.5, 2.5, holder_d)
        stir_holder_w = 2*holder_d + 2*stirrer_w + width_eps
        stir_holder = ks.make_static_cup(kitchen, (stir_x-stir_holder_w/2, 0), 0,
                                     stir_holder_w, stir_holder_w, holder_d)
        #stir_holder = ks.make_cup(kitchen, (20, 0), 0, stir_holder_w, stir_holder_w, holder_d)
        return kitchen

    pour_w, pour_h = get_pour_dims(expid_pour, c_i_pour)
    scoop_w, scoop_h = get_scoop_dims(expid_scoop, c_i_scoop)
    print 'Pour:', (pour_w, pour_h)
    print 'Scoop:', (scoop_w, scoop_h)

    def make_body(kitchen, name, pose):
        if 'gripper' in name:
            body = ks.Gripper(kitchen, init_pos=pose[:2], init_angle=pose[2])
        elif 'sugar_cup' in name:
            body = ks.make_cup(kitchen, pose[:2], pose[2], scoop_w, scoop_h, holder_d)
        elif 'cup' in name:
            body = ks.make_cup(kitchen, pose[:2], pose[2], pour_w, pour_h, holder_d)
        elif 'spoon' in name:
            body = ks.make_good_spoon(kitchen, pose[:2], pose[2], 1, 3, 0.2)
        elif 'stir' in name:
            body = ks.make_stirrer(kitchen, pose[:2], pose[2], 0.2, 5., stirrer_w)
        elif 'block' in name:
            body = ks.make_block(kitchen, pose[:2], pose[2], 5, 1)
        else:
            raise NotImplementedError(name)
        body.name = name
        return body

    problem, _ = create_problem(initial_poses, make_kitchen, make_body,
                   expid_pour, c_i_pour, expid_scoop, c_i_scoop, do_motion=False, do_collisions=False, do_grasp=True)

    kitchen = make_kitchen(planning=False)
    initialize_kitchen(kitchen, make_body, initial_poses)
    kitchen.enable_gui()
    #return

    pr = cProfile.Profile()
    pr.enable()
    plan, evaluations = dual_focused(problem, planner='ff-wastar3', max_time=120, verbose=True, max_planner_time=10,
                                     bind=True, use_context=True)
    pr.disable()
    pstats.Stats(pr).sort_stats('tottime').print_stats(10) # cumtime | tottime
    print_plan(plan, evaluations)
    if plan is None:
        return

    #step_plan(plan, make_kitchen, make_body, initial_poses)
    execute_plan(plan, make_kitchen, make_body, initial_poses)
    raw_input('Done!')

def sample_exp_config():
    params = {
        'c_i_pour': 0,
        'c_i_scoop': 0,
        'block_x': -22.5,
        'gripper_x': 0,
    }

    x_positions = ['sugar_x', 'cup_x', 'cream_x', 'spoon_x', 'stir_x'] #+ ['block_x']
    #min_x = -15; max_x = 15
    min_x = params['block_x'] + 7; max_x = 18

    #for name in x_positions:
    #    params[name] = np.random.uniform(min_x, max_x)
    #return params

    def get_radius(name):
        if name == 'sugar_x':
            return 10/2
        return (ks.OPEN_WIDTH + ks.GRIPPER_WIDTH)/2

    def get_min_distance(name, other):
        if 'sugar_x' in (name, other):
            return 5 + ks.OPEN_WIDTH #+ ks.GRIPPER_WIDTH
        return ks.OPEN_WIDTH + ks.GRIPPER_WIDTH

    def sample_x(name):
        if name == 'sugar_x':
            #return np.random.uniform(-10, 15)
            return np.random.uniform(1, 1)

        return np.random.uniform(min_x, max_x)

    attempts = 0
    while True:
        print 'Attempt:', attempts
        #random.shuffle(x_positions)
        new_params = params.copy()
        new_params['expid_pour'] = random.randint(0, MAX_EXP_ID-1)
        new_params['expid_scoop'] = random.randint(0, MAX_EXP_ID - 1)

        #pour_w, pour_h = get_pour_dims(new_params['expid_pour'], new_params['c_i_pour'])
        #scoop_w, scoop_h = get_scoop_dims(new_params['expid_scoop'], new_params['c_i_scoop'])

        for i, name in enumerate(x_positions):
            for _ in xrange(100):
                x = sample_x(name)
                if not any((other in new_params) and (abs(x - new_params[other]) < get_min_distance(name, other)) for other in x_positions):
                    new_params[name] = x
                    break
            else:
                print 'Fail:', i, name
                new_params = None
                break
        if new_params is not None:
            return new_params
        attempts += 1

if __name__ == '__main__':
    #sys.argv[1]
    num_conf = 0
    exp_params = EXP_CONFIGS[num_conf]
    #exp_params = sample_exp_config()

    main(exp_params, **exp_params)

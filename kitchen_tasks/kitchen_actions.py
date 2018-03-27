from kitchen_predicates import *
from ss.model.operators import Action
from ss.model.functions import TotalCost, Increase
from kitchen_tasks.kitchen_utils import scale_cost

def get_actions():
    # Define actions
    # param: parameters of the action
    # pre: preconditions
    # eff: effects
    actions = [
        Action(name='move', param=[GRIPPER, POSE, POSE2, CONTROL],
               pre=[Motion(GRIPPER, POSE, POSE2, CONTROL),
                    Empty(GRIPPER), CanMove(GRIPPER), 
                    AtPose(GRIPPER, POSE), ~Unsafe(CONTROL)],
               eff=[AtPose(GRIPPER, POSE2),
                    ~AtPose(GRIPPER, POSE), ~CanMove(GRIPPER), # This is to avoid double move
                    Increase(TotalCost(), scale_cost(1))]),

        Action(name='move-holding', param=[GRIPPER, POSE, CUP, GRASP, POSE2, CONTROL],
               pre=[MotionH(GRIPPER, POSE, CUP, GRASP, POSE2, CONTROL), ~Empty(GRIPPER),
                    AtPose(GRIPPER, POSE), Grasped(CUP, GRASP), CanMove(GRIPPER), ~Unsafe(CONTROL)],
               eff=[AtPose(GRIPPER, POSE2),
                    ~AtPose(GRIPPER, POSE), ~CanMove(GRIPPER),
                    Increase(TotalCost(), scale_cost(1))]),

        Action(name='fill', param=[GRIPPER, POSE, CUP, GRASP],
               pre=[BelowFaucet(GRIPPER, POSE, CUP, GRASP),
                    AtPose(GRIPPER, POSE), Grasped(CUP, GRASP)],
               eff=[HasCoffee(CUP), CanMove(GRIPPER),
                    Increase(TotalCost(), scale_cost(1))]),

        # in this example, pour only works for sugar
        Action(name='pour-gp', param=[GRIPPER, POSE, CUP, GRASP, KETTLE, POSE2, CONTROL],
               pre=[CanPour(GRIPPER, POSE, CUP, GRASP, KETTLE, POSE2, CONTROL),
                    AtPose(GRIPPER, POSE), Grasped(CUP, GRASP), AtPose(KETTLE, POSE2), HasCream(CUP),
                    ],
               eff=[HasCream(KETTLE), CanMove(GRIPPER),
                    ~HasCream(CUP),
                    Increase(TotalCost(), scale_cost(1))]),

        # in this example, scoop only works for sugar
        Action(name='scoop', param=[GRIPPER, POSE, POSE2, SPOON, GRASP, KETTLE, POSE3, CONTROL],
               pre=[CanScoop(GRIPPER, POSE, POSE2, SPOON, GRASP, KETTLE, POSE3, CONTROL),
                    AtPose(GRIPPER, POSE), Grasped(SPOON, GRASP), AtPose(KETTLE, POSE3), HasSugar(KETTLE)],
               eff=[AtPose(GRIPPER, POSE2), HasSugar(SPOON), CanMove(GRIPPER), Scooped(SPOON),
                    ~AtPose(GRIPPER, POSE),
                    Increase(TotalCost(), scale_cost(1))]),

        Action(name='dump', param=[GRIPPER, POSE, POSE3, SPOON, GRASP, KETTLE, POSE2, CONTROL],
               pre=[CanDump(GRIPPER, POSE, POSE3, SPOON, GRASP, KETTLE, POSE2, CONTROL),
                    AtPose(GRIPPER, POSE), Grasped(SPOON, GRASP), AtPose(KETTLE, POSE2), HasSugar(SPOON)
                    ],
               eff=[HasSugar(KETTLE), CanMove(GRIPPER),
                    ~HasSugar(SPOON), ~Scooped(SPOON),
                    ~AtPose(GRIPPER, POSE), AtPose(GRIPPER, POSE3),
                    Increase(TotalCost(), scale_cost(1))]),

        Action(name='stir', param=[GRIPPER, POSE, SPOON, GRASP, KETTLE, POSE2, CONTROL],
               pre=[CanStir(GRIPPER, POSE, SPOON, GRASP, KETTLE, POSE2, CONTROL),
                    AtPose(GRIPPER, POSE), Grasped(SPOON, GRASP), AtPose(KETTLE, POSE2),
                    HasCoffee(KETTLE), HasCream(KETTLE), HasSugar(KETTLE)
                    ],
               eff=[Mixed(KETTLE), CanMove(GRIPPER),
                    Increase(TotalCost(), scale_cost(1))]),

        Action(name='pick', param=[GRIPPER, POSE, CUP, POSE2, GRASP, CONTROL],
               pre=[CanGrasp(GRIPPER, POSE, CUP, POSE2, GRASP, CONTROL), TableSupport(POSE2),
                    AtPose(GRIPPER, POSE), AtPose(CUP, POSE2), Empty(GRIPPER)], 
               eff=[Grasped(CUP, GRASP), CanMove(GRIPPER),
                    ~AtPose(CUP, POSE2), ~Empty(GRIPPER),
                    Increase(TotalCost(),
                    scale_cost(1))]),

        Action(name='place', param=[GRIPPER, POSE, CUP, POSE2, GRASP, CONTROL],
               pre=[CanGrasp(GRIPPER, POSE, CUP, POSE2, GRASP, CONTROL), TableSupport(POSE2), 
                    AtPose(GRIPPER, POSE), Grasped(CUP, GRASP), 
                    ~Scooped(CUP)],
               eff=[AtPose(CUP, POSE2), Empty(GRIPPER), CanMove(GRIPPER),
                    ~Grasped(CUP, GRASP),
                    Increase(TotalCost(), scale_cost(1))]),

        Action(name='stack', param=[GRIPPER, POSE, CUP, POSE2, GRASP, BLOCK, POSE3, CONTROL],
               pre=[CanGrasp(GRIPPER, POSE, CUP, POSE2, GRASP, CONTROL), 
                    BlockSupport(CUP, POSE2, BLOCK, POSE3), AtPose(GRIPPER, POSE), 
                    Grasped(CUP, GRASP), AtPose(BLOCK, POSE3), Clear(BLOCK)], 
               eff=[AtPose(CUP, POSE2), Empty(GRIPPER), CanMove(GRIPPER),
                    ~Grasped(CUP, GRASP), ~Clear(BLOCK),
                    Increase(TotalCost(), scale_cost(1))]),

        Action(name='push', param=[GRIPPER, POSE, POSE2, BLOCK, POSE3, POSE4, CONTROL],
               pre=[CanPush(GRIPPER, POSE, POSE2, BLOCK, POSE3, POSE4, CONTROL),
                    AtPose(GRIPPER, POSE), AtPose(BLOCK, POSE3), Empty(GRIPPER), Clear(BLOCK)],
               eff=[AtPose(GRIPPER, POSE2), AtPose(BLOCK, POSE4), CanMove(GRIPPER),
                    ~AtPose(GRIPPER, POSE), ~AtPose(BLOCK, POSE3),
                    Increase(TotalCost(), scale_cost(1))]),
    ]
    
    return actions
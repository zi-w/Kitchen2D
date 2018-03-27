import sys
sys.path.append('pddlstream/')
from kitchen_predicates import *
from ss.model.operators import Axiom

def get_axioms():
    # Axioms (inference rules that are automatically applied at every state)
    axioms = [
        Axiom(param=[CONTROL, CUP, POSE],
              pre=[Collision(CONTROL, CUP, POSE),
                   AtPose(CUP, POSE)],
              eff=Unsafe(CONTROL)),

        Axiom(param=[CUP, GRASP],
              pre=[IsGrasp(CUP, GRASP),
                   Grasped(CUP, GRASP)],
              eff=Holding(CUP)),

        Axiom(param=[CUP, POSE, BLOCK, POSE2],
              pre=[BlockSupport(CUP, POSE, BLOCK, POSE2),
                   AtPose(CUP, POSE), AtPose(BLOCK, POSE2)],
              eff=On(CUP, BLOCK)),
    ]
    return axioms
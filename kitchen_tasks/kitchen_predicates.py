import sys
sys.path.append('pddlstream/')
from ss.model.functions import Predicate, rename_functions

# Parameter names (for predicate, action, axiom, and stream declarations)
# Parameters are strings with '?' as the prefix
GRIPPER = '?gripper'
CUP = '?cup'
SPOON = '?spoon'
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

IsPose = Predicate([CUP, POSE])
IsGrasp = Predicate([CUP, GRASP])
IsControl = Predicate([CONTROL])

CanGrasp = Predicate([GRIPPER, POSE, CUP, POSE2, GRASP, CONTROL])
BelowFaucet = Predicate([GRIPPER, POSE, CUP, GRASP])
CanPour = Predicate([GRIPPER, POSE, CUP, GRASP, KETTLE, POSE2, CONTROL])
Motion = Predicate([GRIPPER, POSE, POSE2, CONTROL])
MotionH = Predicate([GRIPPER, POSE, CUP, GRASP, POSE2, CONTROL])

CanScoop = Predicate([GRIPPER, POSE, POSE2, CUP, GRASP, KETTLE, POSE3, CONTROL])
CanDump = Predicate([GRIPPER, POSE, POSE3, CUP, GRASP, KETTLE, POSE2, CONTROL])
CanStir = Predicate([GRIPPER, POSE, CUP, GRASP, KETTLE, POSE2, CONTROL])
CanPush = Predicate([GRIPPER, POSE, POSE2, CUP, POSE3, POSE4, CONTROL])

Stackable = Predicate([CUP, BLOCK])
BlockSupport = Predicate([CUP, POSE, BLOCK, POSE2]) # [POSE, POSE2]
Clear = Predicate([BLOCK])
TableSupport = Predicate([POSE])

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


# External predicates (boolean functions evaluated by fn)
from kitchen_generators import test_collision
Collision = Predicate([CONTROL, GRIPPER, POSE], domain=[IsControl(CONTROL), IsPose(GRIPPER, POSE)],
                      fn=test_collision, bound=False)
#Collision = Predicate([CONTROL, GRIPPER, POSE])
# This is for showing function names when debugging
rename_functions(locals())
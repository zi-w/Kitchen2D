from kitchen_predicates import *
from kitchen_generators import *
from ss.model.streams import GenStream, Stream, FnStream, TestStream
def get_streams():
    bound = 'shared' # unique | depth | shared

    # Stream declarations
    # inp: a list of input parameters
    # domain: a conjunctive list of atoms indicating valid inputs defined on inp
    # out: a list of output parameters
    # graph: a list of atoms certified by the stream defined on both inp and out
    streams = [
        # Unconditional streams (streams with no input parameters)
        FnStream(name='sample-grasp-ctrl', inp=[GRIPPER, CUP, POSE2, GRASP],
                  domain=[IsGripper(GRIPPER), IsPose(CUP, POSE2), IsGrasp(CUP, GRASP)],
                  fn=genGraspControl, out=[POSE, CONTROL],
                  graph=[CanGrasp(GRIPPER, POSE, CUP, POSE2, GRASP, CONTROL),
                         IsPose(GRIPPER, POSE), IsControl(CONTROL)], bound=bound),

        GenStream(name='sample-grasp-stirrer', inp=[SPOON],
                  domain=[IsStirrer(SPOON)],
                  fn=genGrasp, out=[GRASP],
                  graph=[IsGrasp(SPOON, GRASP)], bound=bound),

        GenStream(name='sample-grasp-cup', inp=[CUP],
                  domain=[IsCup(CUP)],
                  fn=genGrasp, out=[GRASP],
                  graph=[IsGrasp(CUP, GRASP)], bound=bound),

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

        GenStream(name='sample-scoop', inp=[GRIPPER, SPOON, KETTLE, POSE3],
                  domain=[IsGripper(GRIPPER), IsSpoon(SPOON), IsCup(KETTLE), IsPose(KETTLE, POSE3)],
                  fn=genScoop, out=[POSE, POSE2, GRASP, CONTROL],
                  graph=[CanScoop(GRIPPER, POSE, POSE2, SPOON, GRASP, KETTLE, POSE3, CONTROL),
                         IsPose(GRIPPER, POSE), IsPose(GRIPPER, POSE2), IsGrasp(SPOON, GRASP)], bound=bound),

        FnStream(name='sample-dump', inp=[GRIPPER, SPOON, GRASP, KETTLE, POSE2],
                  domain=[IsGripper(GRIPPER), IsSpoon(SPOON), IsGrasp(SPOON, GRASP), IsCup(KETTLE), IsPose(KETTLE, POSE2)],
                  fn=genDump, out=[POSE, POSE3, CONTROL],
                  graph=[CanDump(GRIPPER, POSE, POSE3, SPOON, GRASP, KETTLE, POSE2, CONTROL),
                         IsPose(GRIPPER, POSE), IsPose(GRIPPER, POSE3)], bound=bound),

        FnStream(name='sample-stir', inp=[GRIPPER, SPOON, GRASP, KETTLE, POSE2],
                  domain=[IsGripper(GRIPPER), IsStirrer(SPOON), IsGrasp(SPOON, GRASP), IsCup(KETTLE), IsPose(KETTLE, POSE2)],
                  fn=genStir, out=[POSE, CONTROL],
                  graph=[CanStir(GRIPPER, POSE, SPOON, GRASP, KETTLE, POSE2, CONTROL),
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
               graph=[CanPush(GRIPPER, POSE, POSE2, BLOCK, POSE3, POSE4, CONTROL),
                      IsPose(GRIPPER, POSE), IsPose(GRIPPER, POSE2)], bound=bound)
    ]
    return streams
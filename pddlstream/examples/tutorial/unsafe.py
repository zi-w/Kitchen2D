from ss.model.functions import Predicate, Function, rename_functions, initialize, TotalCost, Increase
from ss.model.problem import Problem
from ss.model.operators import Action, Axiom
from ss.model.streams import Stream, FnStream, TestStream
from ss.algorithms.incremental import incremental, exhaustive
from ss.algorithms.focused import focused
from ss.algorithms.plan_focused import plan_focused
from ss.algorithms.dual_focused import dual_focused
from ss.algorithms.sequence_focused import sequence_focused
from ss.algorithms.hierarchical_focused import hierarchical_focused
from ss.utils import INF

import cProfile
import pstats


def main(n=3, bound='shared'):
    initial_conf = 0
    initial_poses = {'b{}'.format(i): i for i in xrange(n)}

    goal_poses = {'b1': 2}

    Block = Predicate('?b')
    Pose = Predicate('?p')
    Conf = Predicate('?q')

    Kin = Predicate('?q ?p')
    Collision = Predicate('?p1 ?p2', domain=[Pose('?p1'), Pose('?p2')],
                          fn=lambda p1, p2: p1 == p2, bound=False)

    AtConf = Predicate('?q')
    AtPose = Predicate('?b ?p')
    Holding = Predicate('?b')
    HandEmpty = Predicate('')

    Unsafe = Predicate('?p')

    rename_functions(locals())

    streams = [
        Stream(name='placement', inp=[], domain=[],
               fn=lambda: ([(p,)] for p in xrange(n, n + 1)),

               out='?p', graph=[Pose('?p')], bound=bound),
        FnStream(name='ik', inp='?p', domain=[Pose('?p')],
                 fn=lambda p: (p,),
                 out='?q', graph=[Kin('?q', '?p'), Conf('?q')], bound=bound),
    ]
    actions = [
        Action(name='Move', param='?q1 ?q2',
               pre=[Conf('?q1'), Conf('?q2'),
                    AtConf('?q1')],
               eff=[AtConf('?q2'), ~AtConf('?q1'), Increase(TotalCost(), 1)]),
        Action(name='Pick', param='?b ?p ?q',
               pre=[Block('?b'), Kin('?q', '?p'),
                    AtPose('?b', '?p'), HandEmpty(), AtConf('?q')],
               eff=[Holding('?b'), ~AtPose('?b', '?p'), ~HandEmpty(), ]),

        Action(name='Place', param='?b ?p ?q',
               pre=[Block('?b'), Kin('?q', '?p'),
                    Holding('?b'), AtConf('?q'), ~Unsafe('?p')],
               eff=[AtPose('?b', '?p'), HandEmpty(), ~Holding('?b'), ]),

    ]

    axioms = [
        Axiom(param='?p1 ?b2 ?p2',
              pre=[Block('?b2'), Collision('?p1', '?p2'),
                   AtPose('?b2', '?p2')],
              eff=Unsafe('?p1')),
    ]
    initial_atoms = [
        Conf(initial_conf),
        AtConf(initial_conf),
        HandEmpty(),
        initialize(TotalCost(), 0),
    ]
    for b, p in initial_poses.items():
        initial_atoms += [Block(b), Pose(p), AtPose(b, p)]
    for b, p in goal_poses.items():
        initial_atoms += [Pose(p)]

    goal_literals = [AtPose(b, p) for b, p in goal_poses.items()]

    problem = Problem(initial_atoms, goal_literals, actions, axioms, streams,)

    print problem

    pr = cProfile.Profile()
    pr.enable()

    plan, evaluations = dual_focused(
        problem, terminate_cost=INF, verbose=True, effort_weight=1)

    print plan
    pr.disable()
    pstats.Stats(pr).sort_stats('cumtime').print_stats(10)

if __name__ == '__main__':
    main()

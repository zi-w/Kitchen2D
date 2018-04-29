from ss.model.functions import Predicate, rename_functions, initialize, TotalCost, Increase
from ss.model.problem import Problem
from ss.model.operators import Action, Axiom
from ss.model.streams import FnStream, GenStream, Stream, CondGen
from ss.algorithms.incremental import incremental, exhaustive
from ss.algorithms.focused import focused
from ss.algorithms.plan_focused import plan_focused
from ss.algorithms.dual_focused import dual_focused
from ss.utils import INF
from collections import namedtuple
from random import randint, choice
from itertools import count

import cProfile
import pstats


class POSE(object):

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return 'p{}'.format(self.value)


class CONF(object):

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return 'q{}'.format(self.value)


class PoseGen(CondGen):

    def __init__(self, b):

        self.b = b
        self.n = count()
        super(PoseGen, self).__init__()

    def generate(self, context=None):
        print context
        placed = set()
        if context is not None:
            for atom in context.conditions:
                b1, p1, b2, p2 = atom.head.args
                if b1 == self.b:
                    placed.add(p2.value)
                else:
                    placed.add(p1.value)
        p = POSE(self.b, choice(list(set(xrange(6)) - placed)))

        return [(p,)]


def main(n=2, bound='unique'):
    initial_conf = CONF(-1)

    blocks = ['b{}'.format(i) for i in xrange(n)]
    initial_poses = {b: POSE(b, i) for i, b in enumerate(blocks)}

    goal_poses = {'b0': POSE('b0', 1)}

    Block = Predicate('?b')
    IsPose = Predicate('?b ?p')
    Conf = Predicate('?q')

    Kin = Predicate('?b ?q ?p')
    Collision = Predicate('?b ?p ?b2 ?p2', domain=[IsPose('?b', '?p'), IsPose('?b2', '?p2')],
                          fn=lambda b, p, b2, p2: p.value == p2.value, eager=True, bound=False)

    AtConf = Predicate('?q')
    AtPose = Predicate('?b ?p')
    Holding = Predicate('?b')
    HandEmpty = Predicate('')

    Unsafe = Predicate('?b ?p')

    rename_functions(locals())

    streams = [



        Stream(name='placement', inp=['?b'], domain=[Block('?b')],
               fn=PoseGen, out='?p', graph=[IsPose('?b', '?p')], bound=bound),
        FnStream(name='ik', inp='?b ?p', domain=[IsPose('?b', '?p')],
                 fn=lambda b, p: (CONF(p.value),),
                 out='?q', graph=[Kin('?b', '?q', '?p'), Conf('?q')], bound=bound),
    ]
    actions = [
        Action(name='Move', param='?q1 ?q2',
               pre=[Conf('?q1'), Conf('?q2'),
                    AtConf('?q1')],
               eff=[AtConf('?q2'), ~AtConf('?q1'),
                    Increase(TotalCost(), 1)]),
        Action(name='Pick', param='?b ?p ?q',
               pre=[Kin('?b', '?q', '?p'),
                    AtPose('?b', '?p'), HandEmpty(), AtConf('?q')],
               eff=[Holding('?b'), ~AtPose('?b', '?p'), ~HandEmpty(),
                    Increase(TotalCost(), 1)]),
        Action(name='Place', param='?b ?p ?q',
               pre=[Kin('?b', '?q', '?p'),
                    Holding('?b'), AtConf('?q'),
                    ~Unsafe('?b', '?p')],
               eff=[AtPose('?b', '?p'), HandEmpty(), ~Holding('?b'),
                    Increase(TotalCost(), 1)]),
    ]

    axioms = [
        Axiom(param='?b ?p ?b2 ?p2',
              pre=[Collision('?b', '?p', '?b2', '?p2'),
                   AtPose('?b2', '?p2')],
              eff=Unsafe('?b', '?p')),
    ]
    initial_atoms = [
        Conf(initial_conf),
        AtConf(initial_conf),
        HandEmpty(),
        initialize(TotalCost(), 0),
    ]
    for b, p in initial_poses.items():
        initial_atoms += [Block(b), IsPose(b, p), AtPose(b, p)]
    for b, p in goal_poses.items():
        initial_atoms += [IsPose(b, p)]

    goal_literals = [AtPose(b, p) for b, p in goal_poses.items()]

    problem = Problem(initial_atoms, goal_literals, actions,
                      axioms, streams, objective=TotalCost())
    print problem

    pr = cProfile.Profile()
    pr.enable()

    plan, evaluations = dual_focused(problem, terminate_cost=INF, verbose=False, bind=True,
                                     use_context=True, temp_dir='fish/', clean=True)

    print plan
    pr.disable()
    pstats.Stats(pr).sort_stats('cumtime').print_stats(10)

if __name__ == '__main__':
    main()

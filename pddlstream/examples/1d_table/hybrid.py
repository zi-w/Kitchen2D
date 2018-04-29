from ss.model.functions import Predicate, Function, rename_functions, initialize, TotalCost, Increase
from ss.model.problem import Problem
from ss.model.operators import Action, Axiom
from ss.model.streams import Stream, FnStream, ListStream
from ss.algorithms.incremental import incremental, exhaustive
from ss.algorithms.focused import focused
from ss.algorithms.plan_focused import plan_focused
from ss.algorithms.dual_focused import dual_focused
from ss.model.bounds import InputOutputSet
from collections import namedtuple

import cProfile
import pstats


Pose = namedtuple('Pose', ['o', 'x'])
Grasp = namedtuple('Grasp', ['o'])
BConf = namedtuple('BConf', ['x', 'y'])


def main():
    initial_bq = BConf(0, 1)

    initial_poses = {
        'green': [10, 10, 30],
        'table': [10, 20, 30],
    }
    movable = {'green'}

    initial_p_from_name = {}
    class_from_name = {}
    for cl in initial_poses:
        for i, x in enumerate(initial_poses[cl]):
            name = cl + str(i)
            class_from_name[name] = cl
            initial_p_from_name[name] = Pose(name, x)
    print initial_p_from_name
    print class_from_name

    O = '?o'
    S = '?s'
    R = '?r'
    P = '?p'
    P2 = '?p2'
    G = '?g'
    Q = '?q'
    Q2 = '?q2'
    C = '?c'

    base = 'base'
    head = 'head'
    left = 'left'
    right = 'right'

    IsPose = Predicate([O, P])
    IsGrasp = Predicate([O, G])
    IsConf = Predicate([P, Q])
    IsMovable = Predicate([O])
    IsFixed = Predicate([O])

    IsKin = Predicate([R, O, P, G, Q])

    IsSupported = Predicate([P, P2])

    IsArm = Predicate([R])
    IsClass = Predicate([O, C])

    AtPose = Predicate([O, P])
    AtConf = Predicate([R, Q])
    HasGrasp = Predicate([R, O, G])
    HandEmpty = Predicate([R])

    Holding = Predicate([O])
    On = Predicate([O, S])

    base_constant_cost = 1

    def get_circle(value):
        if isinstance(value, BConf):
            return value.x, 0
        if isinstance(value, Pose):
            return value.x, 0
        if isinstance(value, InputOutputSet):
            if value.stream.name == 'IK':
                _, _, p, _ = value.inputs
                x, r = get_circle(p)
                return x, r + 0
            if value.stream.name == 'placement':
                _, _, p = value.inputs
                x, r = get_circle(p)
                return x, r + 0
        raise ValueError(value)

    def distance_bound_fn(q1, q2):
        x1, r1 = get_circle(q1)
        x2, r2 = get_circle(q2)
        return max(abs(x2 - x1) - (r1 + r2), 0) + base_constant_cost

    Distance = Function([Q, Q2], domain=[IsConf(base, Q), IsConf(base, Q2)],
                        fn=lambda q1, q2: abs(
                            q2.x - q1.x) + abs(q2.y - q1.y) + base_constant_cost,
                        bound=distance_bound_fn)

    rename_functions(locals())

    bound = 'shared'
    streams = [
        FnStream(name='grasp', inp=[O], domain=[IsMovable(O)],
                 fn=lambda o: (Grasp(o),),
                 out=[G], graph=[IsGrasp(O, G)], bound=bound),
        ListStream(name='IK', inp=[R, O, P, G], domain=[IsArm(R), IsPose(O, P), IsGrasp(O, G)],
                   fn=lambda r, o, p, g: [(BConf(p.x, +1),)],
                   out=[Q], graph=[IsKin(R, O, P, G, Q), IsConf(base, Q)], bound=bound),
        FnStream(name='placement', inp=[O, S, P],
                 domain=[IsMovable(O), IsFixed(S), IsPose(S, P)],

                 fn=lambda o, s, p: (Pose(o, p.x),),
                 out=[P2], graph=[IsPose(O, P2), IsSupported(P2, P)], bound=bound),
    ]

    actions = [
        Action(name='pick', param=[R, O, P, G, Q],
               pre=[IsKin(R, O, P, G, Q),
                    HandEmpty(R), AtPose(O, P), AtConf(base, Q)],
               eff=[HasGrasp(R, O, G), ~HandEmpty(R)]),

        Action(name='place', param=[R, O, P, G, Q],
               pre=[IsKin(R, O, P, G, Q),
                    HasGrasp(R, O, G), AtConf(base, Q)],
               eff=[HandEmpty(R), AtPose(O, P), ~HasGrasp(R, O, G)]),

        Action(name='move_base', param=[Q, Q2],
               pre=[IsConf(base, Q), IsConf(base, Q2),
                    AtConf(base, Q)],
               eff=[AtConf(base, Q2), ~AtConf(base, Q),
                    Increase(TotalCost(), 1)]),


        Action(name='move_head', param=[Q, Q2],
               pre=[IsConf(head, Q), IsConf(head, Q2),
                    AtConf(head, Q)],
               eff=[AtConf(head, Q2), ~AtConf(head, Q)]),









    ]
    axioms = [
        Axiom(param=[R, O, G],
              pre=[IsArm(R), IsGrasp(O, G),
                   HasGrasp(R, O, G)],
              eff=Holding(O)),
        Axiom(param=[O, P, S, P2],
              pre=[IsPose(O, P), IsPose(S, P2), IsSupported(P, P2),
                   AtPose(O, P), AtPose(S, P2)],
              eff=On(O, S)),
    ]

    initial_atoms = [
        IsArm(left),


        HandEmpty(left),
        HandEmpty(right),
        IsConf(base, initial_bq), AtConf(base, initial_bq),


        initialize(TotalCost(), 0),
    ]

    for n, p in initial_p_from_name.items():
        initial_atoms += [IsPose(n, p), AtPose(n, p)]
        if class_from_name[n] not in movable:
            continue
        for n2, p2 in initial_p_from_name.items():
            if class_from_name[n2] in movable:
                continue
            if p.x == p2.x:
                initial_atoms.append(IsSupported(p, p2))
    for n, cl in class_from_name.items():

        if cl in movable:
            initial_atoms.append(IsMovable(n))
        else:
            initial_atoms.append(IsFixed(n))

    goal_literals = [On('green0', 'table1'), On('green1', 'table1')]

    problem = Problem(initial_atoms, goal_literals, actions,
                      axioms, streams, objective=TotalCost())

    print problem

    pr = cProfile.Profile()
    pr.enable()

    plan, _ = dual_focused(problem)
    pr.disable()
    pstats.Stats(pr).sort_stats('tottime').print_stats(10)

    if plan is None:
        print plan
        return
    print 'Length', len(plan)
    for i, act in enumerate(plan):
        print i, act

if __name__ == '__main__':
    main()

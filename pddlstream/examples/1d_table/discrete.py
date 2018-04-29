from ss.model.functions import Predicate, Function, rename_functions, initialize, TotalCost, Increase
from ss.model.problem import Problem
from ss.model.operators import Action, Axiom
from ss.algorithms.incremental import exhaustive, incremental


def main(n=2, verbose=False):

    Item = Predicate('?b')
    Part = Predicate('?r')
    Class = Predicate('?c')

    IsMovable = Predicate('?i', domain=[Item('?i')])
    IsFixed = Predicate('?i', domain=[Item('?i')])
    IsClass = Predicate('?i ?c', domain=[Item('?i'), Class('?c')])
    IsArm = Predicate('?r', domain=[Part('?r')])
    IsStackable = Predicate('?i1 ?i2', domain=[Item('?i1'), Item('?i2')])

    HandHolding = Predicate('?r ?b', domain=[IsArm('?r'), Item('?b')])
    HandEmpty = Predicate('?r', domain=[IsArm('?r')])
    On = Predicate('?i1 ?i2', domain=[Item('?i1'), Item('?i2')])
    Nearby = Predicate('?s', domain=[IsFixed('?s')])

    Found = Predicate('?b', domain=[Item('?b')])
    Localized = Predicate('?b', domain=[Item('?b')])

    Holding = Predicate('?c', domain=[Class('?c')])

    rename_functions(locals())

    actions = [
        Action(name='Pick', param='?a ?i ?s',
               pre=[IsArm('?a'), IsStackable('?i', '?s'), HandEmpty('?a'),
                    Nearby('?s'), On('?i', '?s'), Localized('?i')],
               eff=[HandHolding('?a', '?i'), ~HandEmpty('?a'), ~On('?i', '?s')]),
        Action(name='Place', param='?a ?i ?s',
               pre=[IsArm('?a'), IsStackable('?i', '?s'),
                    Nearby('?s'), HandHolding('?a', '?i'), Localized('?s')],
               eff=[HandEmpty('?a'), On('?i', '?s'), ~HandHolding('?a', '?i')]),
        Action(name='ScanRoom', param='?s',
               pre=[IsFixed('?s'), ~Found('?s')],
               eff=[Found('?s')]),
        Action(name='ScanFixed', param='?s ?i',
               pre=[IsStackable('?i', '?s'), ~Found('?i')],
               eff=[Found('?i'), On('?i', '?s'), Increase(TotalCost(), 1)]),
        Action(name='Look', param='?i',
               pre=[Found('?i')],
               eff=[Localized('?i')]),
        Action(name='Move', param='?s',
               pre=[IsFixed('?s')],
               eff=[Nearby('?s')] +
                   [~Nearby(s) for s in ['table0', 'table1']] +
                   [~Localized(s) for s in ['soup0']]),
    ]
    initial_atoms = [
        initialize(TotalCost(), 0),
        HandEmpty('left'),
        HandEmpty('right'),

        Found('table0'),
        Found('table1'),
        IsClass('table0', 'table'),
        IsClass('table1', 'table'),
        IsFixed('table0'),
        IsFixed('table1'),

        IsStackable('soup0', 'table0'),
        IsStackable('soup0', 'table1'),
        IsClass('soup0', 'soup'),
        Found('soup0'),
        On('soup0', 'table0'),

        IsStackable('green0', 'table0'),
        IsStackable('green0', 'table1'),
        IsClass('green0', 'green'),
        Found('green0'),
        On('green0', 'table0'),
    ]

    axioms = [
        Axiom(param='?a ?i ?c',
              pre=[IsArm('?a'), IsClass('?i', '?c'), HandHolding('?a', '?i')],
              eff=Holding('?c')),
    ]

    goal_literals = [Holding('green'), Holding('soup')]

    problem = Problem(initial_atoms, goal_literals, actions,
                      axioms, [], objective=TotalCost())

    print problem

    plan, evaluations = incremental(problem, verbose=verbose)
    print plan

if __name__ == '__main__':
    main()

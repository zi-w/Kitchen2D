from ss.model.functions import Predicate, NonNegFunction, rename_functions, initialize, TotalCost, Increase
from ss.model.problem import Problem, dump_evaluations, get_length, get_cost
from ss.model.operators import Action, Axiom
from ss.model.streams import GenStream, TestStream
from ss.algorithms.incremental import incremental
from ss.utils import INF

import numpy as np
import math
import sys
import random

SCALE_COST = 100.


def scale_cost(cost):

    return int(math.ceil(SCALE_COST * cost))


def distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for (a, b) in zip(p1, p2)))


def create_problem(cupSize=(3, 4),
                   kettleSize=(5, 6),
                   cupInitPos=(5, -9, 0),
                   kettleInitPos=(-5, -9, 0),
                   faucetPos=(5, 1, 0),
                   kettleGoalPos=(-5, -9, 0),
                   goalPosEps=0.5,
                   maxMoveDist=10.0,
                   domain=[(-20, 20), (-9, 20), (-np.pi, np.pi)],
                   verboseFns=True):

    CUP = '?cup'
    POS1 = '?pos1'
    POS2 = '?pos2'
    POS3 = '?pos3'
    KETTLE = '?kettle'
    WATER = '?water'
    FAUCET = '?faucet'

    IsCup = Predicate([CUP])
    IsKettle = Predicate([KETTLE])
    IsWater = Predicate([WATER])
    IsFaucet = Predicate([FAUCET])
    LegalPos = Predicate([POS1])
    CanGetWater = Predicate([POS1])
    HoldsWater = Predicate([KETTLE])
    CanPour = Predicate([POS1, POS2, POS3])

    AtPos = Predicate([CUP, POS1])
    WaterInKettle = Predicate([WATER, KETTLE])
    WaterBoiled = Predicate([WATER, KETTLE])

    HasBoiledWater = Predicate([KETTLE])

    CanMove = Predicate([POS1, POS2], domain=[LegalPos(POS1), LegalPos(POS2)],
                        fn=lambda x, y: (distance(x, y) <= maxMoveDist))
    CloseTo = Predicate([POS1, POS2], domain=[LegalPos(POS1), LegalPos(POS2)],
                        fn=lambda x, y: (distance(x, y) <= goalPosEps))

    MoveCost = NonNegFunction([POS1, POS2], domain=[LegalPos(POS1), LegalPos(POS2)],
                              fn=lambda p1, p2: scale_cost(distance(p1, p2)))

    def getWaterTest(p):
        (x, y, z) = p
        result = (faucetPos[0] - 1 <= x <= faucetPos[0] + 1) and (faucetPos[1] -
                                                                  cupSize[1] - 1 <= y <= faucetPos[1] - cupSize[1]) and (-0.05 <= z <= 0.05)
        if verboseFns and not result:
            print 'cannot get water from:', p
        return result

    def fill_cost(pos):
        success_cost = 1
        fail_cost = 10
        p_success = 0.75 if getWaterTest(pos) else 0.25
        expected_cost = p_success * success_cost + (1 - p_success) * fail_cost
        return scale_cost(expected_cost)

    FillCost = NonNegFunction([POS1], domain=[LegalPos(POS1)], fn=fill_cost)

    rename_functions(locals())

    actions = [
        Action(name='Move', param=[CUP, POS1, POS2],
               pre=[IsCup(CUP), CanMove(POS1, POS2),
                    AtPos(CUP, POS1)],
               eff=[AtPos(CUP, POS2),
                    ~AtPos(CUP, POS1),
                    Increase(TotalCost(), MoveCost(POS1, POS2))]),

        Action(name='Fill', param=[CUP, FAUCET, POS1, WATER],
               pre=[HoldsWater(CUP), IsFaucet(FAUCET), CanGetWater(POS1), IsWater(WATER),
                    AtPos(CUP, POS1), AtPos(FAUCET, faucetPos)],
               eff=[WaterInKettle(WATER, CUP),
                    Increase(TotalCost(), FillCost(POS1))]),

        Action(name='Pour', param=[CUP, KETTLE, POS1, POS3, POS2, WATER],
               pre=[HoldsWater(CUP), HoldsWater(KETTLE), CanPour(POS1, POS3, POS2), IsWater(WATER),
                    AtPos(CUP, POS1), AtPos(KETTLE, POS2), WaterInKettle(WATER, CUP)],
               eff=[WaterInKettle(WATER, KETTLE), AtPos(CUP, POS3),
                    ~WaterInKettle(WATER, CUP), ~AtPos(CUP, POS1),
                    Increase(TotalCost(), scale_cost(1))]),

        Action(name='Boil', param=[KETTLE, WATER, POS1],
               pre=[IsKettle(KETTLE), IsWater(WATER), CloseTo(kettleGoalPos, POS1),
                    AtPos(KETTLE, POS1), WaterInKettle(WATER, KETTLE)],
               eff=[WaterBoiled(WATER, KETTLE),
                    Increase(TotalCost(), scale_cost(1))]),
    ]

    axioms = [
        Axiom(param=[WATER, KETTLE],
              pre=[IsWater(WATER), IsKettle(KETTLE),
                   WaterInKettle(WATER, KETTLE), WaterBoiled(WATER, KETTLE)],
              eff=HasBoiledWater(KETTLE)),
    ]

    def legalTest(rp):
        (x, y, z) = rp
        result = (domain[0][0] <= x <= domain[0][1]) and (
            domain[1][0] <= y <= domain[1][1]) and (domain[2][0] <= z <= domain[2][1])
        if verboseFns and not result:
            print 'not legal:', rp
        return result

    def randPos():
        while True:
            pos = tuple(random.uniform(a[0], a[1]) for a in domain)
            if verboseFns:
                print 'randPos:', pos
            yield (pos,)

    def genMove(p):
        while True:
            pos = tuple(random.uniform(-1, 1) * maxMoveDist + a for a in p)
            if distance(pos, p) < maxMoveDist and legalTest(pos):
                if verboseFns:
                    print 'genMove:', pos
                yield (pos,)

    def genClosePos(p):
        while True:
            pos = tuple(random.uniform(-1, 1) * goalPosEps + a for a in p)
            if (distance(pos, p) < goalPosEps) and legalTest(pos):
                if verboseFns:
                    print 'genClosePos:', pos
                yield (pos,)

    def genPourPos(kpos):

        while True:
            x = kpos[0] + random.uniform((cupSize[0] / 2.0 + kettleSize[0] / 2.0),
                                         (cupSize[0] / 2.0 + kettleSize[0] / 2.0 + cupSize[1] / 5.0), )
            y = kpos[1] + random.uniform(kettleSize[1] + 2, kettleSize[1] + 5)
            initpos = (x, y, 0)
            endpos = (x, y, np.pi / 1.4 * np.sign(x - kpos[0]))
            if legalTest(initpos) and legalTest(endpos):
                if verboseFns:
                    print 'genPourPos:', initpos, endpos
                yield (initpos, endpos)
            elif verboseFns:
                print 'genPourPos illegal:', initpos, endpos

    def genGetWaterPos():
        while True:
            pos = (random.uniform(-1, 1) * goalPosEps + faucetPos[0],
                   random.uniform(-cupSize[1] - 1, -cupSize[1]) + faucetPos[1],
                   0)
            if legalTest(pos):
                if verboseFns:
                    print 'genGetWaterPos:', pos
                yield (pos,)

    streams = [

        GenStream(inp=[], domain=[], fn=genGetWaterPos,
                  out=[POS2], graph=[LegalPos(POS2), CanGetWater(POS2)]),
        GenStream(inp=[], domain=[], fn=randPos,
                  out=[POS1], graph=[LegalPos(POS1)]),

        GenStream(inp=[POS1], domain=[LegalPos(POS1)], fn=genMove,
                  out=[POS2], graph=[LegalPos(POS2), CanMove(POS1, POS2)]),
        GenStream(inp=[POS1], domain=[LegalPos(POS1)], fn=genClosePos,
                  out=[POS2], graph=[LegalPos(POS2), CloseTo(POS1, POS2)]),
        GenStream(inp=[POS2], domain=[LegalPos(POS2)], fn=genPourPos,
                  out=[POS1, POS3], graph=[CanPour(POS1, POS3, POS2), LegalPos(POS1), LegalPos(POS3)]),


        TestStream(inp=[POS1], domain=[LegalPos(POS1)], test=getWaterTest,
                   graph=[CanGetWater(POS1)]),
    ]

    kettle = 'kettle'
    cup = 'cup'
    faucet = 'faucet'
    water = 'water'

    initial_atoms = [

        IsKettle(kettle),
        IsCup(cup),
        IsFaucet(faucet),
        IsWater(water),
        HoldsWater(cup),
        HoldsWater(kettle),
        LegalPos(cupInitPos),
        LegalPos(kettleInitPos),
        LegalPos(kettleGoalPos),
        LegalPos(faucetPos),


        AtPos(kettle, kettleInitPos),
        AtPos(faucet, faucetPos),
        AtPos(cup, cupInitPos),


        initialize(TotalCost(), 0),
    ]

    goal_literals = [
        AtPos(cup, cupInitPos),
        HasBoiledWater(kettle),
    ]

    objective = TotalCost()

    return Problem(initial_atoms, goal_literals, actions, axioms, streams, objective=objective)


TEST_ARGS = {

    0: {},
    1: {'cupSize': (4, 5),
        'kettleSize': (5, 6),
        'cupInitPos': (8, -9, 0),
        'kettleInitPos': (-5, -9, 0),
        'faucetPos': (5, 10, 0)},
    2: {'cupSize': (4, 5),
        'kettleSize': (5, 6),
        'cupInitPos': (8, -9, 0),
        'kettleInitPos': (-5, -9, 0),
        'faucetPos': (8, 5, 0)},
    3: {'cupSize': (4, 5),
        'kettleSize': (5, 6),
        'cupInitPos': (8, -9, 0),
        'kettleInitPos': (-5, -9, 0),
        'faucetPos': (9, 0, 0)},
}


def main(argv):
    testNum = int(argv[0]) if argv else 0

    print 'Test number', testNum
    print 'Args = ', TEST_ARGS[testNum]

    problem = create_problem(verboseFns=False, **TEST_ARGS[testNum])
    print problem

    plan, evaluations = incremental(problem,
                                    planner='ff-astar',
                                    max_time=5,
                                    terminate_cost=INF,
                                    verbose=False,
                                    verbose_search=False)

    if plan is None:
        print '\nFailed to find a plan'
        return
    print '\nFound a plan'
    print 'Length:', get_length(plan, evaluations)
    print 'Cost:', get_cost(plan, evaluations) / SCALE_COST
    for i, (action, args) in enumerate(plan):
        print i, action, list(args)


if __name__ == '__main__':

    main(sys.argv[1:])

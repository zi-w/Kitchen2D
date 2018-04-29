import math

from fast_downward import write,    TEMP_DIR, TRANSLATE_OUTPUT, remove_paths, run_search, ensure_dir
from ss.model.functions import Head, Literal, TotalCost, Increase
from ss.model.operators import apply, Goal
from ss.utils import INF
from collections import defaultdict, namedtuple

COST_SCALE = 1
MAX_COST = (2**31 - 1) / 100


def transform_cost(cost):
    new_cost = int(math.ceil(COST_SCALE * cost))
    assert new_cost < MAX_COST
    return new_cost


def sas_version(version=3):
    return 'begin_version\n'           '%s\n'           'end_version\n' % version


def sas_action_costs(problem):
    return 'begin_metric\n'           '%s\n'           'end_metric\n' % int(problem.costs)


def sas_variables(problem):
    s = '%s\n' % len(problem.var_order)
    for i, var in enumerate(problem.var_order):
        axiom_layer = 0 if var in problem.derived_vars else -1
        n = len(problem.index_from_var_val[var])
        assert 2 <= n
        name = 'var%s' % i

        s += 'begin_variable\n'             '%s\n'             '%s\n'             '%s\n' % (
            name, axiom_layer, n)
        for j, val in enumerate(problem.index_from_var_val[var]):
            name = 'val%s' % j

            s += '%s\n' % name
        s += 'end_variable\n'
    return s


def sas_mutexes(problem):
    s = '%s\n' % len(problem.mutexes)
    for mutex in problem.mutexes:
        s += 'begin_mutex_group\n'             '%s\n' % len(mutex)
        for fact in mutex:
            s += '%s %s\n' % problem.get_var_val(fact)
        s += 'end_mutex_group'
    return s


def sas_initial(problem):
    s = 'begin_state\n'
    for var in problem.var_order:
        s += '%s\n' % problem.get_val(var, problem.initial[var])
    s += 'end_state\n'
    return s


def sas_conditions(problem, conditions):
    s = '%s\n' % len(conditions)
    for fact in conditions:
        s += '%s %s\n' % problem.get_var_val(fact)
    return s


def sas_goal(problem):
    s = 'begin_goal\n' + sas_conditions(problem, problem.goal) + 'end_goal\n'
    return s


def sas_actions(problem):
    s = '%s\n' % len(problem.actions)
    for i, action in enumerate(problem.actions):
        s += 'begin_operator\n'             'a-%s\n' % i
        s += sas_conditions(problem, action.preconditions)
        s += '%s\n' % len(action.effects)
        for fact in action.effects:
            s += '0 %s -1 %s\n' % problem.get_var_val(fact)
        s += '%s\n'             'end_operator\n' % transform_cost(action.cost)
    return s


def sas_axioms(problem):
    s = '%s\n' % len(problem.axioms)
    for axiom in problem.axioms:
        s += 'begin_rule\n'
        s += sas_conditions(problem, axiom.preconditions)
        s += '%s -1 %s\n' % problem.get_var_val(axiom.effect)
        s += 'end_rule\n'
    return s


def to_sas(problem):
    return sas_version() + sas_action_costs(problem) + sas_variables(problem) + sas_mutexes(problem) + sas_initial(problem) + sas_goal(problem) + sas_actions(problem) + sas_axioms(problem)


Fact = namedtuple('Fact', ['var', 'val'])
Action = namedtuple('Action', ['original', 'preconditions', 'effects', 'cost'])
Axiom = namedtuple('Axiom', ['preconditions', 'effect'])


class DownwardProblem(object):
    costs = True
    values = (False, True)
    default = False

    def __init__(self, initial, goal, actions, axioms):
        self.index_from_var = {}
        self.var_order = []
        self.index_from_var_val = {}
        self.val_order_from_var = {}
        self.mutexes = []

        self.initial = apply(initial, defaultdict(lambda: self.default))
        self.fluent_heads = set()
        for op in (actions + axioms):
            for literal in op.effects:
                if isinstance(literal, Literal) and (literal.value != self.initial[literal.head]):
                    self.fluent_heads.add(literal.head)
        self.condition_heads = set()
        for op in (actions + axioms + [Goal(goal)]):
            for literal in op.preconditions:
                assert isinstance(literal, Literal)
                if literal.head not in self.fluent_heads:
                    continue
                self.condition_heads.add(literal.head)

        self.goal = self._get_conditions(goal)
        self.actions = list(self._get_actions(actions))
        self.axioms = list(self._get_axioms(axioms))
        self.derived_vars = {axiom.effect.var for axiom in self.axioms}

    def _add_var(self, var):
        if var not in self.index_from_var:
            self.index_from_var[var] = len(self.index_from_var)
            self.var_order.append(var)
            self.index_from_var_val[var] = {}
            self.val_order_from_var[var] = []
            for val in self.values:
                self._add_val(var, val)

    def _add_val(self, var, val):
        self._add_var(var)
        if val not in self.index_from_var_val[var]:
            self.index_from_var_val[var][val] = len(
                self.index_from_var_val[var])
            self.val_order_from_var[var].append(val)

    def _get_conditions(self, conditions):
        facts = []
        for literal in conditions:
            assert isinstance(literal, Literal)
            if literal.head in self.fluent_heads:
                self._add_var(literal.head)
                facts.append(Fact(literal.head, literal.value))
            elif self.initial[literal.head] != literal.value:
                return None
        return facts

    def _get_effects(self, effects):
        facts = []
        cost = 0
        for literal in effects:
            if isinstance(literal, Literal):
                if literal.head in self.condition_heads:
                    facts.append(Fact(literal.head, literal.value))
            elif isinstance(literal, Increase):
                assert literal.head == TotalCost()
                if isinstance(literal.value, Head):
                    cost += self.initial[literal.value]
                else:
                    cost += literal.value
            else:
                raise ValueError(literal)
        return facts, cost

    def _get_actions(self, actions):
        for action in actions:
            conditions = self._get_conditions(action.preconditions)
            effects, cost = self._get_effects(action.effects)
            if (conditions is None) or (not effects):
                continue
            yield Action(action, conditions, effects, cost)

    def _get_axioms(self, axioms):
        for axiom in axioms:
            conditions = self._get_conditions(axiom.preconditions)
            effects, _ = self._get_effects(axiom.effects)
            if (conditions is None) or (not effects):
                continue
            yield Axiom(conditions, effects[0])

    def get_var(self, var):
        return self.index_from_var[var]

    def get_val(self, var, val):
        return self.index_from_var_val[var][val]

    def get_var_val(self, fact):
        return self.get_var(fact.var), self.get_val(fact.var, fact.val)

    def __repr__(self):
        return '{}\n'               'Variables: {}\n'               'Actions: {}\n'               'Axioms: {}\n'               'Goals: {}'.format(self.__class__.__name__,
                                                                                                                                                  len(
                                                                                                                                                      self.var_order),
                                                                                                                                                  len(
                                                                                                                                                      self.actions),
                                                                                                                                                  len(
                                                                                                                                                      self.axioms),
                                                                                                                                                  self.goal if (self.goal is None) else len(self.goal))


def convert_solution(solution, problem):

    plan = []
    for line in solution.split('\n')[:-2]:
        index = int(line.strip('( )')[2:])
        plan.append(problem.actions[index].original)

    return plan


def solve_sas(problem, planner='max-astar', max_time=INF, max_cost=INF, verbose=False, clean=True, temp_dir=TEMP_DIR):
    if problem.goal is None:
        return None
    if not problem.goal:
        return []

    remove_paths(temp_dir)
    ensure_dir(temp_dir)
    write(temp_dir + TRANSLATE_OUTPUT, to_sas(problem))
    plan = run_search(planner, max_time, max_cost, verbose, temp_dir)
    if clean:
        remove_paths(temp_dir)
    if plan is None:
        return None
    return convert_solution(plan, problem)

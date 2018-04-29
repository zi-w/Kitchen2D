from collections import defaultdict, deque

from functions import Atom, initialize
from ss.model.functions import TotalCost, Predicate
from ss.model.operators import apply, Initial, Goal, DurativeAction
from ss.utils import INF


def state_from_evals(evals):
    return apply(evals, {})


def evals_from_state(state):
    return {initialize(*item) for item in state.items()}


def state_sequence(initial, actions, default=None):
    states = [apply(initial, defaultdict(default))]
    for action in actions:
        assert not action.parameters
        states.append(action.apply(states[-1]))
    return states


def is_solution(evals, plan, goal):

    state = defaultdict(bool)
    instances = [Initial(evals)] + [action.instantiate(args)
                                    for action, args in plan] + [Goal(goal)]
    for instance in instances:
        if not instance.applicable(state):
            return False
        state = instance.apply(state)
    return True


def instantiate_plan(plan):
    return [action.instantiate(args) for action, args in plan]


class Problem(object):

    def __init__(self, initial, goal, actions, axioms, streams, objective=None, max_cost=INF):
        self.initial = sorted(initial)

        self.goal = goal
        self.actions = actions
        self.axioms = axioms
        self.streams = streams
        self.objective = objective
        self.max_cost = max_cost

    def get_action(self, name):
        for action in self.actions:
            if action.name == name:
                return action
        return None

    def predicate_uses(self):
        predicate_to_signs = defaultdict(set)
        for op in (self.actions + [Goal(self.goal)]):
            for pre in op.preconditions:
                if isinstance(pre.head.function, Predicate):
                    predicate_to_signs[pre.head.function].add(pre.value)
        for ax in self.axioms:
            for sign in predicate_to_signs[ax.effect.head.function]:
                for pre in ax.preconditions:
                    if isinstance(pre.head.function, Predicate):
                        predicate_to_signs[pre.head.function].add(
                            sign == pre.value)

        return predicate_to_signs

    def is_temporal(self):

        return any(isinstance(action, DurativeAction) for action in self.actions)

    def fluents(self):
        return {e.head.function for a in self.actions for e in a.effects}

    def derived(self):
        return {ax.effect.head.function for ax in self.axioms}

    def graph(self):
        return {e.head.function for s in self.streams for e in s.graph}

    def functions(self):
        return {f for op in (self.actions + self.axioms + [Goal(self.goal)]) for f in op.functions()}

    def external(self):
        return {f for f in self.functions() if f.is_defined()}

    def dump(self):
        print 'Initial'
        dump_evaluations(self.initial)

    def fluent_streams(self):

        fluents = self.fluents()
        fluent_streams = []
        for stream in self.streams:
            domain_functions = {a.head.function for a in stream.domain}

            if domain_functions & fluents:
                fluent_streams.append(stream)
        return fluent_streams

    def __repr__(self):
        return '{}\n'               'Initial: {}\n'               'Goal: {}\n'               'Actions: {}\n'               'Axioms: {}\n'               'Streams: {}\n'.format(self.__class__.__name__,
                                                                                                                                                                               self.initial, self.goal,
                                                                                                                                                                               self.actions, self.axioms, self.streams)


def reset_derived(derived, state):
    for head in list(state):
        if head.function in derived:
            state[head] = False
        if state[head] is False:
            del state[head]


def axiom_achievers(axiom_instances, state):

    axioms_from_pre = defaultdict(list)
    for ax in axiom_instances:
        for p in ax.preconditions:
            assert isinstance(p, Atom)
            axioms_from_pre[p].append(ax)

    axiom_from_eff = {}
    queue = deque()
    for head, val in state.items():
        if not isinstance(head.function, Predicate) or (val != True):
            continue
        eval = initialize(head, val)
        if isinstance(eval, Atom) and (eval not in axiom_from_eff):
            axiom_from_eff[eval] = None
            queue.append(eval)
    while queue:
        pre = queue.popleft()
        for ax in axioms_from_pre[pre]:
            if (ax.effect not in axiom_from_eff) and all(p in axiom_from_eff for p in ax.preconditions):

                axiom_from_eff[ax.effect] = ax
                queue.append(ax.effect)
    return axiom_from_eff


def supporting_axioms_helper(goal, axiom_from_eff, supporters):

    axiom = axiom_from_eff[goal]
    if (axiom is None) or (axiom in supporters):
        return
    for pre in axiom.preconditions:
        supporting_axioms_helper(pre, axiom_from_eff, supporters)
    supporters.append(axiom)


def supporting_axioms(state, goals, axiom_instances):

    supporters = []
    needed_goals = filter(lambda g: not g.holds(state), goals)

    if not needed_goals:
        return supporters
    axiom_from_eff = axiom_achievers(axiom_instances, state)
    for goal in needed_goals:

        supporting_axioms_helper(goal, axiom_from_eff, supporters)
    return supporters


def plan_supporting_axioms(evaluations, plan_instances, axiom_instances, goal):
    states = state_sequence(evaluations, plan_instances, default=bool)
    for state, action in zip(states, plan_instances + [Goal(goal)]):
        yield supporting_axioms(state, action.preconditions, axiom_instances)


def apply_axioms(axiom_instances, state):
    for eval in axiom_achievers(axiom_instances, state):
        eval.assign(state)


def get_length(plan, evaluations):
    if plan is None:
        return INF
    return len(plan)


def get_cost(plan, evaluations):
    if plan is None:
        return INF
    plan_instances = [action.instantiate(args) for action, args in plan]
    return state_sequence(evaluations, plan_instances)[-1][TotalCost()]


def dump_evaluations(evaluations):
    eval_from_function = defaultdict(set)
    for eval in evaluations:
        eval_from_function[eval.head.function].add(eval)
    for fn in sorted(eval_from_function, key=lambda fn: fn.name):
        print len(eval_from_function[fn]), sorted(eval_from_function[fn], key=lambda eval: eval.head.args)

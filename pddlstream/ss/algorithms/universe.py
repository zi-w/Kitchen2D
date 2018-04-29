from collections import defaultdict, deque
from itertools import product

from ss.model.functions import Object, Function, Predicate, initialize, process_domain, Atom, Predicate, NegatedAtom
from ss.model.problem import reset_derived, apply_axioms, dump_evaluations
from ss.model.operators import applicable, apply, Goal
from ss.model.streams import Stream
from ss.to_pddl import pddl_domain, pddl_problem


def get_mapping(atoms1, atoms2, initial={}):
    assert len(atoms1) == len(atoms2)
    mapping = initial.copy()
    for t, a in zip(atoms1, atoms2):
        assert t.head.function is a.head.function
        for p, v in zip(t.head.args, a.head.args):
            if mapping.get(p, v) == v:
                mapping[p] = v
            else:
                return None
    return mapping


class Universe(object):
    _domain_name = 'stripstream'
    _problem_name = _domain_name

    def __init__(self, problem, initial, use_bounds, only_eager, evaluate=True):

        self.problem = problem
        self.use_bounds = use_bounds
        self.only_eager = only_eager
        self.evaluate = evaluate
        self.evaluations = set()
        self.value_from_head = {}
        self.atoms_from_predicate = defaultdict(set)
        self.fluents = self.problem.fluents()
        self.computed = set()

        self.name_from_object = {}
        self.object_from_name = {}
        self.action_from_name = {}
        for action in problem.actions:
            assert action.name not in self.action_from_name
            self.action_from_name[action.name] = action
        self.axioms_from_derived = defaultdict(list)
        for axiom in problem.axioms:
            self.axioms_from_derived[axiom.effect.head.function].append(axiom)

        self.functions = set()
        for action in (problem.actions + problem.axioms):
            self.functions.update(action.functions())
        for literal in problem.goal:
            self.functions.add(literal.head.function)

        self.streams_from_predicate = defaultdict(list)
        self.stream_queue = deque()
        self.stream_instances = set()
        for stream in problem.streams:
            if only_eager and not stream.eager:
                continue

            for i, atom in enumerate(stream.domain):
                self.streams_from_predicate[
                    atom.head.function].append((stream, i))

        self.defined_functions = {f for f in self.functions if f.is_defined()}
        for func in self.defined_functions:
            if (use_bounds and (not func.is_bound_defined())) or (only_eager and (not func.eager)):

                continue
            if use_bounds and isinstance(func, Predicate) and (func.bound is False):

                continue
            for i, atom in enumerate(func.domain):
                self.streams_from_predicate[
                    atom.head.function].append((func, i))

        for atom in initial:
            self.add_eval(atom)
        for action in problem.actions:
            for obj in action.constants():
                self.add_eval(Object(obj))
        for literal in problem.goal:
            for obj in literal.head.args:
                self.add_eval(Object(obj))
        for stream in problem.streams:
            if only_eager and not stream.eager:
                continue
            if not stream.domain:
                self._add_instance(stream, {})

    def _add_object(self, obj):
        if obj in self.name_from_object:
            return
        name = 'x{}'.format(len(self.name_from_object))

        self.name_from_object[obj] = name
        assert name not in self.object_from_name
        self.object_from_name[name] = obj

    def _add_instance(self, relation, mapping):
        inputs = tuple(mapping[p] for p in relation.inputs)
        if isinstance(relation, Stream):
            instance = relation.get_instance(inputs)
            if not instance.enumerated and (instance not in self.stream_instances):
                self.stream_instances.add(instance)
                self.stream_queue.append(instance)
        elif isinstance(relation, Function):

            head = relation.get_head(inputs)
            if not head.computed() and (head not in self.computed):
                if self.use_bounds:
                    self.add_eval(head.get_bound())
                else:
                    self.add_eval(head.get_eval())
        else:
            raise ValueError(relation)

    def _update_stream_instances(self, atom):
        if not self.evaluate:
            return
        for relation, i in self.streams_from_predicate[atom.head.function]:

            values = [self.atoms_from_predicate.get(a.head.function, {}) if i != j else {atom}
                      for j, a in enumerate(relation.domain)]
            for combo in product(*values):

                mapping = get_mapping(relation.domain, combo)
                if mapping is not None:
                    self._add_instance(relation, mapping)

    def is_fluent(self, e):

        return e.head.function in self.fluents

    def is_derived(self, e):
        return e.head.function in self.axioms_from_derived

    def is_static(self, e):
        return not self.is_derived(e) and not self.is_fluent(e)

    def _operator_instances(self, operator):

        static_atoms = process_domain({a for a in operator.preconditions if isinstance(a, Atom) and self.is_static(a)}
                                      | {Object(p) for p in operator.parameters})
        values = [self.atoms_from_predicate.get(
            a.head.function, {}) for a in static_atoms]
        for combo in product(*values):

            mapping = get_mapping(static_atoms, combo)
            if mapping is not None:
                yield operator, tuple(mapping[p] for p in operator.parameters)

    def action_instances(self):
        for action in self.action_from_name.values():
            for instance in self._operator_instances(action):
                yield instance

    def axiom_instances(self):
        for axioms in self.axioms_from_derived.values():
            for axiom in axioms:
                for instance in self._operator_instances(axiom):
                    yield instance

    def add_eval(self, eval):
        if eval in self.evaluations:
            return
        if self.value_from_head.get(eval.head, eval.value) != eval.value:
            raise ValueError('{}: {} != {}'.format(
                eval.head, self.value_from_head[eval.head], eval.value))
        self.value_from_head[eval.head] = eval.value
        self.functions.add(eval.head.function)
        self.evaluations.add(eval)
        for obj in eval.head.args:
            self._add_object(obj)
        if isinstance(eval, Atom) and (eval not in self.atoms_from_predicate[eval.head.function]):
            self.atoms_from_predicate[eval.head.function].add(eval)
            self._update_stream_instances(eval)
        for implied in eval.head.implied():
            self.add_eval(implied)

    def pddl(self):
        predicates = set(
            filter(lambda f: isinstance(f, Predicate), self.functions))
        functions = self.functions - predicates

        initial_str = [e.substitute(self.name_from_object)
                       for e in self.evaluations if not isinstance(e, NegatedAtom)]
        goal_str = [l.substitute(self.name_from_object)
                    for l in self.problem.goal]
        return pddl_domain(self._domain_name,
                           self.object_from_name.keys(),
                           predicates, functions,
                           [a.substitute_constants(self.name_from_object)
                            for a in self.action_from_name.values()],
                           [a.substitute_constants(self.name_from_object)
                            for axioms in self.axioms_from_derived.values() for a in axioms]),               pddl_problem(self._domain_name, self._problem_name,
                                                                                                                          [],
                                                                                                                          initial_str, goal_str,
                                                                                                                          self.problem.objective)

    def state_fluents(self, state):
        return frozenset(filter(lambda e: not self.is_static(e),
                                (initialize(h, v) for h, v in state.iteritems())))

    def print_plan(self, plan):
        plan_instances = [self.action_from_name[
            name].instantiate(args) for name, args in plan]
        print plan_instances
        axioms = [axiom.instantiate(args)
                  for axiom, args in self.axiom_instances()]
        state = apply(self.evaluations, defaultdict(bool))
        reset_derived(self.axioms_from_derived, state)
        print 0, self.state_fluents(state)
        for i, instance in enumerate(plan_instances):
            apply_axioms(axioms, state)
            assert instance.applicable(state)
            state = instance.apply(state)
            reset_derived(self.axioms_from_derived, state)
            print i + 1, self.state_fluents(state)
        apply_axioms(axioms, state)
        assert applicable(self.problem.goal, state)

    def is_solution(self, plan):
        plan_instances = [action.instantiate(
            args) for action, args in plan] + [Goal(self.problem.goal)]
        axiom_instances = [axiom.instantiate(
            args) for axiom, args in self.axiom_instances()]
        state = apply(self.evaluations, defaultdict(bool))
        for instance in plan_instances:
            reset_derived(self.axioms_from_derived, state)
            apply_axioms(axiom_instances, state)
            if not instance.applicable(state):
                return False
            state = instance.apply(state)
        return True

    def convert_plan(self, plan):
        if plan is None:
            return None
        new_plan = []
        for action_name, arg_names in plan:
            action = self.action_from_name[action_name]
            args = tuple(self.object_from_name[a] for a in arg_names)
            new_plan.append((action, args))
        return new_plan

    def dump(self):
        print 'Evaluations'
        dump_evaluations(self.evaluations)

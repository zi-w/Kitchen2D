from collections import namedtuple, defaultdict
from heapq import heappop, heappush
from itertools import product

from ss.algorithms.universe import get_mapping
from ss.model.functions import Atom, TotalCost, Function, Increase, initialize

Node = namedtuple('Node', ['effort', 'stream'])


def determine_atom_effort(evaluations, bound_streams, op=sum):

    unprocessed_from_atom = defaultdict(list)
    node_from_atom = {None: Node(0, None)}
    for bs in bound_streams:
        bs._conditions = bs.stream.domain() + [None]
        bs._remaining = len(bs._conditions)
        for atom in bs._conditions:
            unprocessed_from_atom[atom].append(bs)

    for atom in evaluations:
        if isinstance(atom, Atom):
            node_from_atom[atom] = Node(0, None)

    queue = [(node.effort, atom) for atom, node in node_from_atom.items()]
    while queue:
        _, atom = heappop(queue)
        if atom not in unprocessed_from_atom:
            continue
        for bs in unprocessed_from_atom[atom]:
            bs._remaining -= 1
            if not bs._remaining:
                cost = op(node_from_atom[
                          con].effort for con in bs._conditions) + bs.stream.get_effort()
                for eff in bs.bound_atoms:
                    if (eff not in node_from_atom) or (cost < node_from_atom[eff].effort):
                        node_from_atom[eff] = Node(cost, bs)
                        heappush(queue, (cost, eff))
        del unprocessed_from_atom[atom]
    del node_from_atom[None]
    return node_from_atom


def initialize_effort_functions(problem):

    problem.objective = TotalCost()
    for action in problem.actions:

        action._stream_atoms = filter(
            lambda a: a.head.function in problem.graph(), action.preconditions)
        stream_args = {p for a in action._stream_atoms for p in a.head.args}
        stream_params = [p for p in action.parameters if p in stream_args]
        Func = Function(['?x{}'.format(i) for i in xrange(len(stream_params))])
        action._stream_head = Func(*stream_params)
        action.effects = action.effects + \
            (Increase(TotalCost(), action._stream_head),)


def add_effort_evaluations(evaluations, universe, bound_streams, operation=sum):
    node_from_atom = determine_atom_effort(
        evaluations, bound_streams, op=operation)
    for action in universe.problem.actions:
        values = [universe.atoms_from_predicate.get(
            a.head.function, {}) for a in action._stream_atoms]
        for combo in product(*values):
            mapping = get_mapping(action._stream_atoms, combo)
            if mapping is None:
                continue
            ground_atoms = [a.substitute(mapping)
                            for a in action._stream_atoms]
            cost = 1

            cost += operation(node_from_atom[a].effort for a in ground_atoms)
            stream_head = action._stream_head.substitute(mapping)
            universe.add_eval(initialize(stream_head, cost))
    return node_from_atom

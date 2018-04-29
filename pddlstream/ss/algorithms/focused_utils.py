from ss.algorithms.universe import Universe
from ss.utils import INF
from collections import defaultdict
from heapq import heappush, heappop
from ss.model.functions import Atom
from ss.model.problem import initialize, instantiate_plan
from ss.algorithms.incremental import solve_universe


class BoundStream(object):

    def __init__(self, stream, bound_outputs, bound_atoms):
        self.stream = stream
        self.bound_outputs = bound_outputs
        self.bound_atoms = bound_atoms

    def __repr__(self):
        return '{}{}->{}'.format(self.stream.stream.name, self.stream.inputs, self.bound_outputs)


def evaluate_stream_instances(queue, evaluations, max_evals):
    num_evals = 0
    while queue and (num_evals < max_evals):
        num_evals += 1
        instance = queue.popleft()
        if not instance.enumerated:
            evaluations.update(instance.next_atoms())
        if not instance.enumerated:
            queue.append(instance)


def evaluate_eager(problem, evaluations):
    eager_universe = Universe(problem, evaluations,
                              use_bounds=False, only_eager=True)
    evaluate_stream_instances(
        eager_universe.stream_queue, eager_universe.evaluations, INF)
    return eager_universe.evaluations


def solve_eager(problem, evaluations, solve, **kwargs):
    eager_universe = Universe(problem, evaluations,
                              use_bounds=False, only_eager=True)
    evaluate_stream_instances(
        eager_universe.stream_queue, eager_universe.evaluations, INF)

    if not solve:
        return None, eager_universe.evaluations
    return solve_universe(eager_universe, **kwargs), eager_universe.evaluations


def isolated_reset_fn(disabled, evaluations):

    evaluate_stream_instances(disabled, evaluations, len(disabled))


def revisit_reset_fn(disabled, evaluations):
    while disabled:
        instance = disabled.popleft()
        instance.disabled = False


def disable_stream(disabled, instance):
    if not instance.disabled and not instance.enumerated:
        disabled.append(instance)
    instance.disabled = True


def partial_order_streams(evaluations, bound_streams):

    orders = set()
    achievers = {}
    for eval in evaluations:
        if isinstance(eval, Atom):
            achievers[eval] = None
    for bs in bound_streams:
        for atom in bs.stream.domain():
            if achievers[atom] is not None:
                orders.add((achievers[atom], bs))
        for atom in bs.bound_atoms:
            if atom not in achievers:
                achievers[atom] = bs

    return orders


def stream_achievers(state, bound_streams):
    achievers = {}
    for head, value in state.items():
        eval = initialize(head, value)
        if isinstance(eval, Atom):
            achievers[eval] = None
    for bs in bound_streams:
        for atom in bs.stream.domain():
            assert atom in achievers
        for atom in bs.bound_atoms:
            if atom not in achievers:
                achievers[atom] = bs
    return achievers


def supporting_streams_helper(goal, stream_from_eff, sequence):
    stream = stream_from_eff[goal]
    if (stream is None) or (stream in sequence):
        return
    for pre in stream.stream.domain():
        supporting_streams_helper(pre, stream_from_eff, sequence)
    sequence.append(stream)


def supporting_streams(state, goals, streams):
    stream_from_eff = stream_achievers(state, streams)
    sequence = []
    for goal in goals:
        if goal not in stream_from_eff:
            print goal
            return None
        supporting_streams_helper(goal, stream_from_eff, sequence)
    return sequence


def effort_priority(bs):
    if not isinstance(bs, BoundStream):
        return -1
    return bs.stream.get_effort()


def zero_priority(v):
    return 0


def topological_sort(vertices, edges, priority_fn=effort_priority):
    incoming_edges = defaultdict(set)
    outgoing_edges = defaultdict(set)
    for v1, v2 in edges:
        incoming_edges[v2].add(v1)
        outgoing_edges[v1].add(v2)

    ordering = []
    queue = []
    for v in vertices:
        if not incoming_edges[v]:
            heappush(queue, (priority_fn(v), v))
    while queue:
        _, v1 = heappop(queue)
        ordering.append(v1)
        for v2 in outgoing_edges[v1]:
            incoming_edges[v2].remove(v1)
            if not incoming_edges[v2]:
                heappush(queue, (priority_fn(v2), v2))
    return ordering


def partial_ordered(plan):

    instances = instantiate_plan(plan)
    orders = set()
    primary_effects = set()
    for i in reversed(xrange(len(instances))):
        for pre in instances[i].preconditions:
            for j in reversed(xrange(i)):

                if any(eff == pre for eff in instances[j].effects):
                    orders.add((j, i))
                    primary_effects.add((j, pre))
                    break
        for eff in instances[i].effects:
            for j in xrange(i):
                if any((pre.head == eff.head) and (pre.value != eff.value) for pre in instances[j].preconditions):
                    orders.add((j, i))
            if (i, eff) in primary_effects:
                for j in xrange(i):
                    if any((eff2.head == eff.head) and (eff2.value != eff.value) for eff2 in instances[j].effects):
                        orders.add((j, i))

    for i, (action, args) in enumerate(plan):
        print i, action, args
    print orders
    print primary_effects
    print topological_sort(range(len(plan)), orders, lambda v: hasattr(plan[v][0], 'stream'))

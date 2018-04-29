import time
from collections import deque

from ss.algorithms.focused_utils import evaluate_eager, BoundStream
from ss.algorithms.incremental import solve_universe
from ss.algorithms.universe import Universe
from ss.model.functions import infer_evaluations
from ss.model.problem import get_length, get_cost
from ss.utils import INF
from ss.algorithms.dual_focused import bound_stream_instances, solve_streams


class StreamSequence(object):

    def __init__(self, sequence, plan, cost):

        self.sequence = sequence
        self.plan = plan
        self.cost = cost

    def __repr__(self):
        return repr(self.sequence)


def add_sequence(evaluations, disabled, sequence):

    if not sequence.sequence:
        return
    bound = sequence.sequence[0]
    instance = bound.stream
    if instance.enumerated or not (set(instance.domain()) <= evaluations):
        return
    if not instance.disabled:

        instance.disabled = True
        instance.sequences = []
        disabled.append(instance)
    instance.sequences.append(sequence)


def evaluate_instance(evaluations, disabled, instance):
    if instance.enumerated:
        return
    for outputs in instance.next_outputs():

        evaluations.update(instance.substitute_graph(outputs))

        for sequence in instance.sequences:
            bound = sequence.sequence[0]
            assert bound.stream is instance
            bindings = dict(zip(bound.bound_outputs, outputs))
            new_sequence = []
            for bound2 in sequence.sequence[1:]:
                new_inputs = [bindings.get(inp, inp)
                              for inp in bound2.stream.inputs]
                new_instance = bound2.stream.stream.get_instance(new_inputs)
                new_sequence.append(BoundStream(
                    new_instance, bound2.bound_outputs, bound2.bound_atoms))

            add_sequence(evaluations, disabled, StreamSequence(
                new_sequence, sequence.plan, sequence.cost))


def evaluate_sequences(evaluations, disabled, max_evals):

    next_disabled = deque()
    num_evals = 0
    while disabled and (num_evals < max_evals):
        num_evals += 1
        instance = disabled.popleft()
        evaluate_instance(evaluations, disabled, instance)
        if not instance.enumerated:
            next_disabled.append(instance)
    disabled.extend(next_disabled)


def prune_sequences(disabled, max_cost):
    for _ in xrange(len(disabled)):
        instance = disabled.popleft()
        instance.sequences = filter(
            lambda seq: seq.cost < max_cost, instance.sequences)
        if instance.sequences:
            disabled.append(instance)
        else:
            instance.disabled = False


def sequence_focused(problem, max_time=INF, max_cost=INF, terminate_cost=INF,
                     planner='ff-astar', waves=1, verbose=False):

    start_time = time.time()
    num_epochs = 1
    num_iterations = 0
    evaluations = infer_evaluations(problem.initial)
    disabled = deque()
    best_plan = None
    best_cost = INF
    search_time = 0
    stream_time = 0
    while (time.time() - start_time) < max_time:
        num_iterations += 1
        print '\nEpoch: {} | Iteration: {} | Disabled: {} | Cost: {} | '              'Search time: {:.3f} | Stream time: {:.3f} | Total time: {:.3f}'.format(
            num_epochs, num_iterations, len(disabled), best_cost, search_time, stream_time, time.time() - start_time)
        evaluations = evaluate_eager(problem, evaluations)
        universe = Universe(problem, evaluations,
                            use_bounds=True, only_eager=False)
        if not all(f.eager for f in universe.defined_functions):
            raise NotImplementedError(
                'Non-eager functions are not yet supported')
        bound_streams = bound_stream_instances(universe)
        t0 = time.time()

        plan = solve_universe(universe, planner=planner,
                              max_time=(max_time - (time.time() - start_time)),
                              max_cost=min(best_cost, max_cost), verbose=verbose)
        search_time += (time.time() - t0)
        cost = get_cost(plan, universe.evaluations)
        print 'Actions | Length: {} | Cost: {} | {}'.format(
            get_length(plan, universe.evaluations), cost, plan)
        t0 = time.time()
        streams = solve_streams(universe, evaluations,
                                plan, bound_streams, start_time, max_time)
        stream_time += (time.time() - t0)
        print 'Streams | Length: {} | {}'.format(get_length(streams, None), streams)
        if streams:

            add_sequence(evaluations, disabled,
                         StreamSequence(streams, plan, cost))
        elif (streams is not None) and (cost < best_cost):
            best_plan = plan
            best_cost = cost
            prune_sequences(disabled, best_cost)
        if (best_cost < terminate_cost) or not disabled:
            break
        for _ in xrange(waves):

            evaluate_sequences(evaluations, disabled, INF)
        num_epochs += 1
    return best_plan, evaluations

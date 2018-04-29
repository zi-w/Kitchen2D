import time
from collections import deque
from itertools import product

from ss.algorithms.focused_utils import evaluate_eager, revisit_reset_fn, partial_ordered, instantiate_plan, BoundStream
from ss.algorithms.incremental import solve_universe
from ss.algorithms.universe import Universe
from ss.model.functions import Predicate, Increase, infer_evaluations, TotalCost
from ss.model.operators import Action
from ss.model.problem import Problem, get_length, get_cost
from ss.utils import INF
from ss.algorithms.focused_binding import call_streams, multi_bind_call_streams


def bound_stream_instances(universe):
    abstract_evals = set()
    while universe.stream_queue:
        instance = universe.stream_queue.popleft()
        outputs_list = instance.bound_outputs()

        for outputs in outputs_list:
            params = instance.inputs + outputs
            universe.add_eval(instance.stream.predicate(*params))
            for atom in instance.substitute_graph(outputs):
                if atom not in universe.evaluations:
                    abstract_evals.add(atom)
                    universe.add_eval(atom)
    return abstract_evals


def make_stream_operators(problem, effort_weight):
    stream_actions = []
    stream_axioms = []
    fluents = problem.fluents()
    for stream in problem.streams:

        params = stream.inputs + stream.outputs
        stream.predicate = Predicate(params)
        preconditions = list(stream.domain) + [stream.predicate(*params)]
        effects = list(stream.graph)
        if effort_weight is not None:
            effort = 1

            effects.append(Increase(TotalCost(), effort_weight * effort))

        action = Action(stream.name, params, preconditions, effects)
        action.stream = stream
        stream.fluent_domain = tuple(
            a for a in stream.domain if a.head.function in fluents)
        stream.domain = tuple(
            a for a in stream.domain if a not in stream.fluent_domain)
        stream_actions.append(action)
    return stream_actions, stream_axioms


def is_stream_action(action):
    return hasattr(action, 'stream') or hasattr(action, 'bound_stream')


def split_indices(plan):
    stream_indices = []
    action_indices = []
    for i, (action, args) in enumerate(plan):
        if is_stream_action(action):
            stream_indices.append((len(action_indices), i))
        else:
            action_indices.append(i)
    return stream_indices, action_indices


def prioritize_streams(plan):
    stream_indices, action_indices = split_indices(plan)
    initial_size = len(action_indices)
    instances = instantiate_plan(plan)
    for start_index, i in stream_indices:
        best_index = len(action_indices) - (initial_size - start_index)
        while 0 < best_index:
            j = action_indices[best_index - 1]
            if (set(instances[j].effects) & set(instances[i].preconditions)) or any((a1.head == a2.head) and (a1.value != a2.value)
                                                                                    for a1, a2 in product(instances[j].preconditions, instances[i].effects)):
                break
            best_index -= 1
        action_indices.insert(best_index, i)
    return [plan[i] for i in action_indices]


def defer_streams(plan):

    stream_indices, action_indices = split_indices(plan)
    instances = instantiate_plan(plan)
    for start_index, i in reversed(stream_indices):
        best_index = start_index
        while best_index < len(action_indices):
            j = action_indices[best_index]
            if (set(instances[i].effects) & set(instances[j].preconditions)) or any((a1.head == a2.head) and (a1.value != a2.value)
                                                                                    for a1, a2 in product(instances[i].preconditions, instances[j].effects)):
                break
            best_index += 1
        action_indices.insert(best_index, i)
    return [plan[i] for i in action_indices]


def detangle_plan(evaluations, plan):

    stream_instances = []
    action_instances = []
    for action, args in plan:
        if not action_instances and is_stream_action(action):
            stream = action.stream
            inputs = args[:len(stream.inputs)]
            outputs = args[len(stream.inputs):]
            instance = stream.get_instance(inputs)
            bs = BoundStream(instance, outputs,
                             instance.substitute_graph(outputs))
            stream_instances.append(bs)

        else:
            action_instances.append((action, args))
    return stream_instances, action_instances


def plan_focused(problem, max_time=INF, max_cost=INF, terminate_cost=INF, effort_weight=1,
                 planner='ff-astar', max_planner_time=10, reset_fn=revisit_reset_fn, bind=False,
                 verbose=False, verbose_search=False, defer=False):
    start_time = time.time()
    num_iterations = 0
    num_epochs = 1
    if effort_weight is not None:
        problem.objective = TotalCost()
        for action in problem.actions:
            action.effects = action.effects + (Increase(TotalCost(), 1),)
    evaluations = infer_evaluations(problem.initial)
    disabled = deque()

    stream_actions, stream_axioms = make_stream_operators(
        problem, effort_weight)
    stream_problem = Problem([], problem.goal, problem.actions + stream_actions,
                             problem.axioms + stream_axioms, problem.streams, objective=TotalCost())
    best_plan = None
    best_cost = INF
    while (time.time() - start_time) < max_time:
        num_iterations += 1
        print '\nEpoch: {} | Iteration: {} | Disabled: {} | Cost: {} | '              'Time: {:.3f}'.format(num_epochs, num_iterations, len(disabled), best_cost, time.time() - start_time)
        evaluations = evaluate_eager(problem, evaluations)
        universe = Universe(stream_problem, evaluations,
                            use_bounds=True, only_eager=False)
        if not all(f.eager for f in universe.defined_functions):
            raise NotImplementedError(
                'Non-eager functions are not yet supported')

        abstract_evals = bound_stream_instances(universe)
        universe.evaluations -= abstract_evals

        mt = (max_time - (time.time() - start_time))
        if disabled:
            mt = min(max_planner_time, mt)
        combined_plan = solve_universe(universe, planner=planner, max_time=mt,
                                       max_cost=min(best_cost, max_cost), verbose=verbose_search)
        if combined_plan is None:
            if not disabled:
                break
            reset_fn(disabled, evaluations)
            num_epochs += 1
            continue

        combined_plan = defer_streams(
            combined_plan) if defer else prioritize_streams(combined_plan)
        stream_plan, action_plan = detangle_plan(evaluations, combined_plan)
        print 'Length: {} | Cost: {} | Streams: {}'.format(
            get_length(combined_plan, universe.evaluations),
            get_cost(combined_plan, universe.evaluations), len(stream_plan))
        print 'Actions:', action_plan
        print 'Streams:', stream_plan
        if not stream_plan:
            cost = get_cost(combined_plan, universe.evaluations)
            if cost < best_cost:
                best_plan = combined_plan
                best_cost = cost
            if (best_cost < terminate_cost) or not disabled:
                break
            reset_fn(disabled, evaluations)
            num_epochs += 1
            continue
        negative_atoms = []
        if bind:

            reattempt = multi_bind_call_streams(
                evaluations, disabled, stream_plan, negative_atoms)
        else:
            reattempt = call_streams(
                evaluations, disabled, stream_plan, negative_atoms)

    return best_plan, evaluations

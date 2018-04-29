import time
from collections import defaultdict, deque

from ss.algorithms.effort import initialize_effort_functions, add_effort_evaluations
from ss.algorithms.incremental import solve_universe
from ss.algorithms.focused_utils import evaluate_eager, isolated_reset_fn, revisit_reset_fn
from ss.algorithms.universe import Universe
from ss.algorithms.focused_utils import BoundStream
from ss.model.functions import Literal, Increase, infer_evaluations, Head
from ss.model.problem import get_cost, get_length, supporting_axioms
from ss.model.operators import apply, Operator
from ss.model.streams import StreamInstance
from ss.utils import INF


def bound_stream_instances(universe):

    stream_from_head = {}
    bound_streams = []
    while universe.stream_queue:
        instance = universe.stream_queue.popleft()
        for outs in instance.bound_outputs():
            atoms = instance.substitute_graph(outs)
            if not atoms:
                continue
            bound_streams.append(BoundStream(instance, outs, atoms))
            for eval in atoms:
                if eval not in universe.evaluations:
                    universe.add_eval(eval)
                    stream_from_head[eval.head] = instance
    return stream_from_head, bound_streams


def required_heads(universe, plan):

    actions = [action.instantiate(args) for action, args in plan]
    axioms = [axiom.instantiate(args)
              for axiom, args in universe.axiom_instances()]
    goal = Operator([], universe.problem.goal, [])
    state = apply(universe.evaluations, defaultdict(bool))

    heads = set()
    image = {}
    for act in actions + [goal]:
        preconditions = {
            p for p in act.preconditions if not universe.is_derived(p)}
        for ax in supporting_axioms(state, act.preconditions, axioms):
            preconditions.update(
                p for p in ax.preconditions if not universe.is_derived(p))

        for p in preconditions:
            assert isinstance(p, Literal)
            assert p.head.function not in universe.axioms_from_derived
            if p.head in image:
                assert image[p.head] == p.value
            else:
                heads.update(p.heads())
        for e in act.effects:
            if isinstance(e, Literal):
                image[e.head] = e.value
            elif isinstance(e, Increase):

                heads.update(e.heads())
            else:
                raise NotImplementedError(e)
        state = act.apply(state)
    return heads


def retrace_streams(stream_from_head, evaluated, heads):

    streams = set()
    for head in heads:
        if head in evaluated:
            continue
        if head.function.fn is not None:
            stream = head
        elif head in stream_from_head:
            stream = stream_from_head[head]
        else:
            continue
        streams |= ({stream} | retrace_streams(stream_from_head, evaluated,
                                               (atom.head for atom in stream.domain())))
    return streams


def focused(problem, planner='ff-astar', max_planner_time=10, max_time=INF, max_cost=INF, effort_weight=1,
            single=False, reset_fn=revisit_reset_fn, verbose=False):

    start_time = time.time()
    num_epochs = 0
    num_iterations = 0
    if effort_weight is not None:

        initialize_effort_functions(problem)
    evaluations = infer_evaluations(problem.initial)
    disabled = deque()
    while (time.time() - start_time) < max_time:
        num_iterations += 1
        print '\nEpoch: {} | Iteration: {} | Time: {:.3f}'.format(num_epochs, num_iterations, time.time() - start_time)

        evaluations = evaluate_eager(problem, evaluations)
        universe = Universe(problem, evaluations,
                            use_bounds=True, only_eager=False)

        stream_from_head, bound_streams = bound_stream_instances(universe)
        if effort_weight is not None:
            add_effort_evaluations(evaluations, universe, bound_streams)
        mt = (max_time - (time.time() - start_time))
        if disabled:
            mt = min(max_planner_time, mt)
        plan = solve_universe(universe, planner=planner,
                              max_time=mt, max_cost=max_cost, verbose=verbose)
        print 'Length: {} | Cost: {}'.format(get_length(plan, universe.evaluations),
                                             get_cost(plan, universe.evaluations))
        print 'Plan:', plan
        if plan is None:
            if not disabled:
                break
            reset_fn(disabled, evaluations)
            num_epochs += 1
            continue
        evaluated = {e.head for e in evaluations}
        supporting_heads = required_heads(universe, plan)

        useful_streams = retrace_streams(
            stream_from_head, evaluated, supporting_heads)

        print useful_streams
        evaluable_streams = filter(lambda s: set(
            s.domain()) <= evaluations, useful_streams)

        if not evaluable_streams:
            return plan, evaluations
        if single:
            evaluable_streams = evaluable_streams[:1]

        for instance in evaluable_streams:
            if isinstance(instance, Head):
                evaluations.update(infer_evaluations([instance.get_eval()]))
            elif not instance.enumerated:
                evaluations.update(infer_evaluations(instance.next_atoms()))
                instance.disabled = True
                disabled.append(instance)

    return None, evaluations

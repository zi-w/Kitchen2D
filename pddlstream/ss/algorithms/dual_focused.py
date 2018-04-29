import time
from collections import deque, defaultdict
from itertools import product

from ss.algorithms.downward import DownwardProblem, solve_sas
from ss.algorithms.focused_binding import call_streams, multi_bind_call_streams
from ss.algorithms.focused_utils import solve_eager, isolated_reset_fn, revisit_reset_fn, BoundStream
from ss.algorithms.incremental import solve_universe
from ss.algorithms.universe import Universe, get_mapping
from ss.algorithms.plan_focused import prioritize_streams, defer_streams
from ss.algorithms.effort import initialize_effort_functions, add_effort_evaluations
from ss.model.functions import Increase, infer_evaluations, TotalCost, Atom, Literal, Head, Object
from ss.model.functions import NegatedAtom
from ss.model.functions import Predicate
from ss.model.operators import Operator, Initial, Goal, Axiom
from ss.model.problem import get_length, get_cost, state_sequence
from ss.utils import INF


def bound_stream_instances(universe):

    bound_streams = []
    while universe.stream_queue:
        instance = universe.stream_queue.popleft()
        for outs in instance.bound_outputs():
            atoms = instance.substitute_graph(outs)
            if atoms:
                bound_streams.append(BoundStream(instance, outs, atoms))
            for eval in atoms:
                universe.add_eval(eval)
    return bound_streams


def literal_sequence(initial, actions):
    state = {}
    for action in [Initial(initial)] + actions:
        assert not action.parameters
        for atom in action.effects:
            if isinstance(atom, Literal):
                atom.assign(state)
        yield state.copy()


def plan_preimage(universe, evaluations, plan):

    action_instances = [action.instantiate(args) for action, args in plan]
    lazy_states = list(literal_sequence(
        universe.evaluations, action_instances))
    real_states = list(literal_sequence(evaluations, action_instances))
    axiom_instances = [axiom.instantiate(
        args) for axiom, args in universe.axiom_instances()]
    preimage = set()
    remapped_axioms = []
    for rs, ls, action in reversed(zip(real_states, lazy_states, action_instances + [Goal(universe.problem.goal)])):
        preimage -= set(action.effects)

        derived = filter(lambda a: isinstance(a, Atom) and
                         (a.head.function in universe.axioms_from_derived), action.preconditions)
        preimage |= (set(action.preconditions) - set(derived))
        if derived:
            derived_map = {f: f.__class__(f.inputs)
                           for f in universe.axioms_from_derived}
            remap_fn = lambda a: a.__class__(
                Head(derived_map[a.head.function], a.head.args))
            preimage.update(map(remap_fn, derived))
            for ax in axiom_instances:
                preconditions = []
                for atom in ax.preconditions:
                    if atom.head.function in derived_map:
                        preconditions.append(remap_fn(atom))
                    elif ls.get(atom.head, False) != atom.value:
                        break
                    elif (atom.head not in rs) and (not atom.head.function.is_defined()):
                        preconditions.append(atom)

                else:
                    remapped_axioms.append(
                        Axiom([], preconditions, remap_fn(ax.effect)))
    return preimage, remapped_axioms


def stream_action_instances(evaluations, bound_streams):

    actions = []
    axioms = []
    for bound_stream in bound_streams:
        instance = bound_stream.stream
        effort = instance.get_effort()
        assert effort != INF

        preconditions = [a for a in instance.domain() if a not in evaluations]
        effects = [a for a in bound_stream.bound_atoms if a not in evaluations]

        if instance.stream.eager and (len(effects) == 1):

            axioms.append(Axiom([], preconditions, effects[0]))
        else:
            action = Operator([], preconditions, effects +
                              [Increase(TotalCost(), effort)])
            action.bound_stream = bound_stream
            actions.append(action)
    return actions, axioms


def solve_streams(universe, evaluations, plan, bound_streams, start_time, max_time, defer, **kwargs):

    planner = 'ff-astar'
    if plan is None:
        return None, plan
    if not plan:
        return [], plan
    preimage, axioms = plan_preimage(universe, evaluations, plan)
    preimage_goal = preimage - evaluations
    if not preimage_goal:
        return [], plan
    stream_actions, stream_axioms = stream_action_instances(
        evaluations, bound_streams)
    downward_problem = DownwardProblem(
        evaluations, preimage_goal, stream_actions, axioms + stream_axioms)
    stream_plan = solve_sas(downward_problem, planner=planner,
                            max_time=(max_time - (time.time() - start_time)),
                            verbose=False, **kwargs)
    if stream_plan is None:

        return None, plan

    return [a.bound_stream for a in stream_plan], plan


def solve_streams_new(universe, evaluations, plan, bound_streams, start_time, max_time, defer, **kwargs):
    planner = 'ff-astar'
    if plan is None:
        return None, plan
    if not plan:
        return [], plan
    OrderPreds = [Predicate([]) for _ in xrange(len(plan) + 1)]
    action_from_instance = {}
    for i, (action, args) in enumerate(plan):

        instance = action.instantiate(args)
        instance.preconditions = filter(lambda p: isinstance(p, Atom) or not universe.is_derived(p),
                                        instance.preconditions) + (OrderPreds[i](),)
        instance.effects = list(instance.effects) + \
            [~OrderPreds[i](), OrderPreds[i + 1]()]
        action_from_instance[instance] = (action, args)
    axioms = [axiom.instantiate(args)
              for axiom, args in universe.axiom_instances()]
    stream_actions, stream_axioms = stream_action_instances(
        evaluations, bound_streams)

    goal = [OrderPreds[-1]()]
    downward_problem = DownwardProblem(evaluations | {OrderPreds[0]()}, goal,
                                       action_from_instance.keys() + stream_actions,
                                       axioms + stream_axioms)
    combined_plan = solve_sas(downward_problem, planner=planner,
                              max_time=(max_time - (time.time() - start_time)),
                              verbose=False, **kwargs)
    if combined_plan is None:
        return None, plan

    combined_plan = [action_from_instance.get(
        op, (op, tuple())) for op in combined_plan]
    combined_plan = defer_streams(
        combined_plan) if defer else prioritize_streams(combined_plan)

    stream_plan, action_plan = [], []
    for i, (action, args) in enumerate(combined_plan):
        if (action, args) in plan:
            action_plan.append((action, args))
        elif action_plan:

            action_plan.append((action, args))

        else:
            stream_plan.append(action.bound_stream)
    return stream_plan, action_plan


def negative_axioms(universe, plan):
    negative_atoms = set()
    if plan is None:
        return negative_atoms

    action_instances = [action.instantiate(
        args) for action, args in plan] + [Goal(universe.problem.goal)]
    states = list(state_sequence(universe.evaluations,
                                 action_instances, default=bool))
    for state, action in zip(states, action_instances):
        atoms_from_predicate = defaultdict(list)
        for head, val in state.items():
            if (val is True) and (head.function in universe.fluents):
                atoms_from_predicate[head.function].append(Atom(head))
        for pre in action.preconditions:
            if not isinstance(pre, NegatedAtom) or not universe.is_derived(pre):
                continue
            for axiom in universe.axioms_from_derived[pre.head.function]:
                assert not any(universe.is_derived(a)
                               for a in axiom.preconditions)
                external = filter(lambda a: a.head.function.is_defined() and
                                  (a.head.function.bound is False), axiom.preconditions)
                if not external:
                    continue
                fluents = filter(universe.is_fluent, axiom.preconditions)
                values = [atoms_from_predicate.get(
                    a.head.function, []) for a in fluents]
                initial_mapping = dict(
                    zip(axiom.effect.head.args, pre.head.args))
                for combo in product(*values):

                    mapping = get_mapping(
                        fluents, combo, initial=initial_mapping)
                    if mapping is None:
                        continue
                    assert all(p in mapping for p in axiom.parameters)

                    if all(pre.substitute(mapping).holds(state) for pre in (set(axiom.preconditions) - set(external))):
                        negative_atoms.update(~pre.substitute(
                            mapping) for pre in external)
    return filter(lambda a: not a.head.computed(), negative_atoms)


def evaluate_negative_atoms(universe, evaluations, opt_plan):
    success = True
    negative_atoms = negative_axioms(universe, opt_plan)
    for atom in negative_atoms:
        if set(atom.head.domain()) <= evaluations:
            eval = atom.head.get_eval()
            evaluations.add(eval)
            success &= (atom.value == eval.value)
        else:
            success = False
    return success, negative_atoms

X = '?x'
Computed = Predicate([X], name='Computed')
Computable = Predicate([X], name='Computable')


def initialize_fluent_streams(problem):

    fluents = problem.fluents()
    new_axioms = []
    for stream in problem.streams:

        stream.fluent_domain = tuple(
            a for a in stream.domain if a.head.function in fluents)
        stream.domain = tuple(
            a for a in stream.domain if a not in stream.fluent_domain)
        stream.predicates = []
        for i, o in enumerate(stream.outputs):

            params = stream.inputs + (X,)
            predicate = Predicate(params, name='{}{}'.format(stream.name, i))
            stream.predicates.append(predicate)
            new_axioms.append(Axiom(params, pre=[predicate(*params)] +
                                    [Computable(i) for i in stream.inputs],
                                    eff=Computable(X)))

    new_axioms.append(Axiom(X, pre=[Computed(X)], eff=Computable(X)))
    for action in problem.actions:
        action.preconditions = action.preconditions + \
            tuple(map(Computable, action.parameters))
    problem.axioms += new_axioms


def add_computed_evals(evaluations):
    for eval in list(evaluations):
        if eval.head.function is Object:
            [o] = eval.head.args
            atom = Computed(o)
            evaluations.add(atom)


def add_fluent_streams(evaluations, bound_streams, universe, initial_computed=True):
    for bs in bound_streams:
        instance = bs.stream
        mapping = instance.domain_mapping()
        fluent_atoms = {atom.substitute(mapping)
                        for atom in instance.stream.fluent_domain}
        if initial_computed and (fluent_atoms <= evaluations):
            for o in bs.bound_outputs:
                atom = Computed(o)
                universe.add_eval(atom)
        else:
            for predicate, o in zip(instance.stream.predicates, bs.bound_outputs):
                params = instance.inputs + (o,)
                atom = predicate(*params)
                universe.add_eval(atom)


def dual_focused(problem, max_time=INF, max_cost=INF, terminate_cost=INF, effort_weight=None, solve=False, defer=False,
                 use_context=False, planner='ff-astar', max_planner_time=10, reset_fn=revisit_reset_fn,
                 bind=False, revisit=False, verbose=False,
                 verbose_search=False, **kwargs):
    start_time = time.time()
    num_epochs = 1
    num_iterations = 0
    if effort_weight is not None:

        initialize_effort_functions(problem)
    evaluations = infer_evaluations(problem.initial)
    disabled = deque()
    best_plan = None
    best_cost = INF
    search_time = 0
    stream_time = 0
    reattempt = True
    has_fluent_streams = len(problem.fluent_streams()) != 0

    assert not has_fluent_streams
    if has_fluent_streams:

        initialize_fluent_streams(problem)
    while (time.time() - start_time) < max_time:
        num_iterations += 1
        if verbose:
            print '\nEpoch: {} | Iteration: {} | Disabled: {} | Cost: {} | '              'Search time: {:.3f} | Stream time: {:.3f} | Total time: {:.3f}'.format(
                num_epochs, num_iterations, len(disabled), best_cost, search_time, stream_time, time.time() - start_time)

        real_plan, evaluations = solve_eager(problem, evaluations, solve=(solve and reattempt), planner=planner,
                                             max_time=(
                                                 max_time - (time.time() - start_time)),
                                             max_cost=min(best_cost, max_cost), verbose=verbose, **kwargs)

        reattempt = False
        real_cost = get_cost(real_plan, evaluations)
        if real_cost < best_cost:

            best_plan = real_plan
            best_cost = real_cost
            if best_cost < terminate_cost:
                break

        if has_fluent_streams:
            add_computed_evals(evaluations)
        universe = Universe(problem, evaluations,
                            use_bounds=True, only_eager=False)
        if not all(f.eager for f in universe.defined_functions):
            raise NotImplementedError(
                'Non-eager functions are not yet supported')
        bound_streams = bound_stream_instances(universe)
        if effort_weight is not None:
            add_effort_evaluations(evaluations, universe, bound_streams)
        if has_fluent_streams:
            add_fluent_streams(evaluations, bound_streams, universe)

        mt = (max_time - (time.time() - start_time))
        if disabled:
            mt = min(max_planner_time, mt)
        t0 = time.time()

        opt_plan = solve_universe(universe, planner=planner, max_time=mt,
                                  max_cost=min(best_cost, max_cost), verbose=verbose_search, **kwargs)
        search_time += (time.time() - t0)
        if verbose:
            print 'Actions | Length: {} | Cost: {} | {}'.format(get_length(opt_plan, universe.evaluations),
                                                                get_cost(opt_plan, universe.evaluations), opt_plan)

        if use_context:
            success, negative_atoms = evaluate_negative_atoms(
                universe, evaluations, opt_plan)
            if verbose:
                print 'External | Success: {} | {}'.format(success, negative_atoms)
        else:
            success, negative_atoms = True, set()

        t0 = time.time()
        solve_streams_fn = solve_streams_new if (
            defer or has_fluent_streams) else solve_streams
        stream_plan, action_plan = solve_streams_fn(
            universe, evaluations, opt_plan, bound_streams, start_time, max_time, defer, **kwargs)
        stream_time += (time.time() - t0)
        if verbose:
            print 'Streams | Length: {} | {}'.format(get_length(stream_plan, []), stream_plan)

        if stream_plan:
            if revisit:
                isolated_reset_fn(disabled, evaluations)

            if bind:

                reattempt = multi_bind_call_streams(
                    evaluations, disabled, stream_plan, negative_atoms, verbose=verbose)
            else:
                reattempt = call_streams(
                    evaluations, disabled, stream_plan, negative_atoms, verbose=verbose)
            if verbose:
                print 'Reattempt:', reattempt
            continue

        cost = get_cost(action_plan, universe.evaluations)
        if success and (stream_plan is not None) and (cost < best_cost):
            best_plan = action_plan
            best_cost = cost
        if (best_cost < terminate_cost) or not disabled:
            break

        reset_fn(disabled, evaluations)
        num_epochs += 1
    return best_plan, evaluations

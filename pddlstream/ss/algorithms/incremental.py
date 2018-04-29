import time

from ss.algorithms.fast_downward import fast_downward
from ss.algorithms.downward import DownwardProblem, solve_sas
from ss.algorithms.universe import Universe
from ss.utils import INF
from ss.model.problem import get_cost
from ss.algorithms.tpshe import tpshe
from ss.algorithms.tfd import tfd
from ss.algorithms.smtplan import smtplan


def evaluate_stream_instances(universe, max_evals, start_time, max_time, verbose=False):
    num_evals = 0
    while universe.stream_queue and (num_evals < max_evals) and ((time.time() - start_time) < max_time):
        num_evals += 1
        instance = universe.stream_queue.popleft()
        new_atoms = instance.next_atoms()
        if verbose:
            print instance, new_atoms
        for eval in new_atoms:
            universe.add_eval(eval)
        if not instance.enumerated:
            universe.stream_queue.append(instance)


def solve_universe(universe, **kwargs):

    if not universe.problem.goal:
        return []

    domain_pddl, problem_pddl = universe.pddl()
    if universe.problem.is_temporal():
        plan = tpshe(domain_pddl, problem_pddl, **kwargs)

    else:
        plan = fast_downward(domain_pddl, problem_pddl, **kwargs)
    return universe.convert_plan(plan)


def solve_universe_manual(universe, **kwargs):
    if not universe.problem.goal:
        return []

    action_mapping = {action.instantiate(args): (
        action, args) for action, args in universe.action_instances()}
    action_instances = action_mapping.keys()
    axiom_instances = [axiom.instantiate(
        args) for axiom, args in universe.axiom_instances()]
    print 'Actions: {} | Axioms: {}'.format(len(action_instances), len(axiom_instances))
    problem = DownwardProblem(
        universe.evaluations, universe.problem.goal, action_instances, axiom_instances)

    plan = solve_sas(problem, **kwargs)
    if plan is None:
        return None
    return [action_mapping[ai] for ai in plan]


def incremental(problem, max_time=INF, max_cost=INF, terminate_cost=INF, planner='ff-astar',
                max_planner_time=10, verbose=False, verbose_search=False):

    start_time = time.time()
    search_time = 0
    num_iterations = 0
    universe = Universe(problem, problem.initial,
                        use_bounds=False, only_eager=False)
    best_plan = None
    best_cost = INF
    while (time.time() - start_time) < max_time:
        num_iterations += 1
        elapsed_time = (time.time() - start_time)
        print 'Iteration: {} | Evaluations: {} | Cost: {} | '              'Search time: {:.3f} | Total time: {:.3f}'.format(num_iterations, len(universe.evaluations),
                                                                                                                             best_cost, search_time, elapsed_time)
        t0 = time.time()
        plan = solve_universe(universe, planner=planner,
                              max_time=min(max_planner_time,
                                           (max_time - elapsed_time)),
                              max_cost=min(best_cost, max_cost), verbose=verbose_search)
        search_time += (time.time() - t0)
        cost = get_cost(plan, universe.evaluations)
        if cost < best_cost:
            best_plan = plan
            best_cost = cost
        if (best_cost != INF) and (best_cost <= terminate_cost):
            break
        if not universe.stream_queue:
            break
        evaluate_stream_instances(universe, len(
            universe.stream_queue), start_time, max_time, verbose=verbose)
    return best_plan, universe.evaluations


def exhaustive(problem, max_time=INF, max_cost=INF, search_time=5, verbose=False, verbose_search=False):
    stream_time = max_time - search_time
    start_time = time.time()
    last_print = time.time()
    universe = Universe(problem, problem.initial,
                        use_bounds=False, only_eager=False)
    while universe.stream_queue and ((time.time() - start_time) < stream_time):
        evaluate_stream_instances(
            universe, 1, start_time, max_time, verbose=verbose)
        if 5 <= (time.time() - last_print):
            print 'Evaluations: {} | Total time: {:.3f}'.format(len(universe.evaluations), (time.time() - start_time))
            last_print = time.time()
    plan = solve_universe(universe, max_time=search_time,
                          max_cost=max_cost, verbose=verbose_search)
    return plan, universe.evaluations


def finite(problem, max_cost=INF, search_time=INF, verbose=False):
    universe = Universe(problem, problem.initial,
                        use_bounds=False, only_eager=False)
    evaluate_stream_instances(universe, INF, time.time(), INF, verbose=verbose)
    plan = solve_universe(universe, max_time=search_time,
                          max_cost=max_cost, verbose=verbose)
    return plan, universe.evaluations

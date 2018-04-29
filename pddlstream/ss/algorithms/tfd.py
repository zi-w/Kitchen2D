
import os
import subprocess

from ss.algorithms.fast_downward import TEMP_DIR, DOMAIN_INPUT, PROBLEM_INPUT, write, read, ensure_dir, safe_remove, safe_rm_dir
from ss.utils import INF

ENV_VAR = 'TFD_PATH'


COMMAND = 'plan.py y+Y+e+O+1+C+1+b {} {} {}'


PLAN_FILE = 'plan'


"""
Usage: search <option characters>  (input read from stdin)
Options are:
  a - enable anytime search (otherwise finish on first plan found)
  t <timeout secs> - total timeout in seconds for anytime search (when plan found)
  T <timeout secs> - total timeout in seconds for anytime search (when no plan found)
  m <monitor file> - monitor plan, validate a given plan
  g - perform greedy search (follow heuristic)
  l - disable lazy evaluation (Lazy = use parent's f instead of child's)
  v - disable verbose printouts
  y - cyclic cg CEA heuristic
  Y - cyclic cg CEA heuristic - preferred operators
  x - cyclic cg makespan heuristic 
  X - cyclic cg makespan heuristic - preferred operators
  G [m|c|t|w] - G value evaluation, one of m - makespan, c - pathcost, t - timestamp, w [weight] - weighted / Note: One of those has to be set!
  Q [r|p|h] - queue mode, one of r - round robin, p - priority, h - hierarchical
  K - use tss known filtering (might crop search space)!
  n - no_heuristic
  r - reschedule_plans
  O [n] - prefOpsOrderedMode, with n being the number of pref ops used
  C [n] - prefOpsCheapestMode, with n being the number of pref ops used
  E [n] - prefOpsMostExpensiveMode, with n being the number of pref ops used
  e - epsilonize internally
  f - epsilonize externally
  p <plan file> - plan filename prefix
  M v - monitoring: verify timestamps
  u - do not use cachin in heuristic
"""


def has_tfd():
    return True


def get_tfd_root():
    return '/home/caelan/Programs/Planners/tfd-src-0.4/downward'

    if not has_tfd():
        raise RuntimeError('Environment variable %s is not defined.' % ENV_VAR)
    return os.environ[ENV_VAR]


def tfd(domain_pddl, problem_pddl, max_time=INF, max_cost=INF, verbose=True):
    safe_rm_dir(TEMP_DIR)
    ensure_dir(TEMP_DIR)
    domain_path = TEMP_DIR + DOMAIN_INPUT
    problem_path = TEMP_DIR + PROBLEM_INPUT
    write(domain_path, domain_pddl)
    write(problem_path, problem_pddl)

    plan_path = os.path.join(TEMP_DIR, PLAN_FILE)

    paths = [os.path.join(os.getcwd(), p)
             for p in (domain_path, problem_path, plan_path)]
    command = os.path.join(get_tfd_root(), COMMAND.format(*paths))
    if verbose:
        print command

    stdout = None if verbose else open(os.devnull, 'w')

    stderr = None
    try:
        proc = subprocess.Popen(command.split(
            ' '), cwd=get_tfd_root(), stdout=stdout, stderr=stderr)
        proc.wait()

    except subprocess.CalledProcessError, e:
        print "Subprocess error", e.output
        raw_input("Continue?")

    temp_path = os.path.join(os.getcwd(), TEMP_DIR)
    plan_files = sorted([f for f in os.listdir(
        temp_path) if f.startswith(PLAN_FILE)])
    if verbose:
        print plan_files

    if not plan_files:
        return None
    output = None
    with open(os.path.join(temp_path, plan_files[-1]), 'r') as f:
        output = f.read()

    plan = []
    for line in output.split('\n')[:-1]:
        entries = line[line.find('(') + 1:line.find(')')].split(' ')
        action, args = entries[0], entries[1:]
        if args == ['']:
            args = []
        plan.append((action, args))
    if not verbose:
        safe_rm_dir(TEMP_DIR)
    return plan

from time import time
from fast_downward import read, write, safe_rm_file, safe_rm_dir, ensure_dir, INF, TEMP_DIR
import os
import re
import math

DOMAIN_PATH = 'domain.pddl'
PROBLEM_PATH = 'problem.pddl'
OUTPUT_PATH = 'sas_plan'
TMP_OUTPUT_PATH = 'tmp_sas_plan'

ENV_VAR = 'TPSHE_PATH'

COMMAND = 'python {}bin/plan.py she {} {} --time {} --iterated'


def get_tpshe_root():
    if ENV_VAR not in os.environ:
        raise RuntimeError('Environment variable %s is not defined.' % ENV_VAR)
    return os.environ[ENV_VAR]


def run_tpshe(max_time, verbose):
    command = COMMAND.format(get_tpshe_root(), TEMP_DIR + DOMAIN_PATH,
                             TEMP_DIR + PROBLEM_PATH, int(math.ceil(max_time)))
    t0 = time()
    p = os.popen(command)
    if verbose:
        print command
    output = p.read()
    if verbose:
        print
        print output
        print 'Runtime:', time() - t0

    plan_files = sorted(f for f in os.listdir(
        '.') if f.startswith(TMP_OUTPUT_PATH))
    if not plan_files:
        return None

    best_plan, best_makespan = None, INF
    for plan_file in plan_files:
        plan, duration = parse_tmp_solution(read(plan_file))
        print plan_file, len(plan), duration
        if duration <= best_makespan:
            best_plan, best_makespan = plan, duration

    return best_plan


def parse_solution(solution):
    lines = solution.split('\n')[:-2]
    plan = []
    for line in lines:
        entries = line.strip('( )').split(' ')
        name = entries[0][3:]
        plan.append((name, tuple(entries[1:])))
    return plan


def parse_tmp_solution(solution):
    total_duration = 0
    plan = []
    regex = r'(\d+.\d+): \(\s*(\w+(?:\s\w+)*)\s*\) \[(\d+.\d+)\]'
    for start_time, action, duration in re.findall(regex, solution):
        total_duration = max(float(start_time) +
                             float(duration), total_duration)
        entries = action.lower().split(' ')
        plan.append((entries[0], tuple(entries[1:])))
    return plan, total_duration

PATHS = [OUTPUT_PATH, TMP_OUTPUT_PATH,
         'dom.pddl', 'ins.pddl', 'output', 'output.sas',
         'plan.validation', 'tdom.pddl', 'tins.pddl']


def remove_paths():
    safe_rm_dir(TEMP_DIR)
    for f in os.listdir('.'):
        if any(f.startswith(p) for p in PATHS):
            safe_rm_file(f)


def tpshe(domain_pddl, problem_pddl, max_time=30, verbose=True, clean=False, **kwargs):
    remove_paths()
    ensure_dir(TEMP_DIR)
    write(TEMP_DIR + DOMAIN_PATH, domain_pddl)
    write(TEMP_DIR + PROBLEM_PATH, problem_pddl)
    solution = run_tpshe(max_time, verbose)
    if clean:
        remove_paths()
    return solution

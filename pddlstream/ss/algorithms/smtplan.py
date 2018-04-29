import os

from ss.algorithms.fast_downward import TEMP_DIR, DOMAIN_INPUT, PROBLEM_INPUT, write, read, ensure_dir, safe_remove, safe_rm_dir
from ss.utils import INF

ENV_VAR = 'SMTPLAN_PATH'


COMMAND = 'SMTPlan %s %s -l 1 -u %s -c 2 -s 1'

FAILURE = 'No plan found'


def has_smtplan():
    return ENV_VAR in os.environ


def get_smtplan_root():
    if not has_smtplan():
        raise RuntimeError('Environment variable %s is not defined.' % ENV_VAR)
    return os.environ[ENV_VAR]


def smtplan(domain_pddl, problem_pddl, max_length=20, verbose=False, **kwargs):
    safe_rm_dir(TEMP_DIR)
    ensure_dir(TEMP_DIR)
    domain_path = TEMP_DIR + DOMAIN_INPUT
    problem_path = TEMP_DIR + PROBLEM_INPUT
    write(domain_path, domain_pddl)
    write(problem_path, problem_pddl)

    command = os.path.join(get_smtplan_root(), COMMAND %
                           (domain_path, problem_path, max_length))
    if verbose:
        print command
    p = os.popen(command)
    output = p.read()
    if verbose:
        print output

    if FAILURE in output:
        return None
    plan = []
    for line in output.split('\n')[:-1]:

        if 'arning' in line or line == '':
            continue
        entries = line[line.find('(') + 1:line.find(')')].split(' ')
        plan.append((entries[0], entries[1:]))
    if not verbose:
        remove_dir(TEMP_DIRECTORY)
    return plan


"""
0.0:  |(pickup block_1)0_sta| [0.0]
2.0:  |(stack block_1 block_2)1_sta| [0.0]
4.0:  |(pickup block_0)2_sta| [0.0]
6.0:  |(stack block_0 block_1)3_sta| [0.0]
Goal at [6.0]
"""


"""
0.0:  |(pickup block_1)0_sta| [0.0]
0.0:  |(armempty)0_0|
0.0:  |(clear block_0)0_0|
0.0:  |(clear block_0)0_1|
0.0:  |(clear block_1)0_0|
0.0:  |(clear block_2)0_0|
0.0:  |(clear block_2)0_1|
0.0:  |(ontable block_0)0_0|
0.0:  |(ontable block_0)0_1|
0.0:  |(ontable block_1)0_0|
0.0:  |(ontable block_2)0_0|
0.0:  |(ontable block_2)0_1|
0.0:  |(holding block_1)0_1|
0.0:  |(total-cost)0_0| == |(total-cost)0_0|
0.0:  |(total-cost)0_1| == |(total-cost)0_1|
2.0:  |(stack block_1 block_2)1_sta| [0.0]
2.0:  |(armempty)1_1|
2.0:  |(clear block_0)1_0|
2.0:  |(clear block_0)1_1|
2.0:  |(clear block_1)1_1|
2.0:  |(clear block_2)1_0|
2.0:  |(ontable block_0)1_0|
2.0:  |(ontable block_0)1_1|
2.0:  |(ontable block_2)1_0|
2.0:  |(ontable block_2)1_1|
2.0:  |(holding block_1)1_0|
2.0:  |(on block_1 block_2)1_1|
2.0:  |(total-cost)1_0| == |(total-cost)1_0|
2.0:  |(total-cost)1_1| == |(total-cost)1_1|
4.0:  |(pickup block_0)2_sta| [0.0]
4.0:  |(armempty)2_0|
4.0:  |(clear block_0)2_0|
4.0:  |(clear block_1)2_0|
4.0:  |(clear block_1)2_1|
4.0:  |(ontable block_0)2_0|
4.0:  |(ontable block_2)2_0|
4.0:  |(ontable block_2)2_1|
4.0:  |(holding block_0)2_1|
4.0:  |(on block_1 block_2)2_0|
4.0:  |(on block_1 block_2)2_1|
4.0:  |(total-cost)2_0| == |(total-cost)2_0|
4.0:  |(total-cost)2_1| == |(total-cost)2_1|
6.0:  |(stack block_0 block_1)3_sta| [0.0]
6.0:  |(armempty)3_1|
6.0:  |(clear block_0)3_1|
6.0:  |(clear block_1)3_0|
6.0:  |(ontable block_2)3_0|
6.0:  |(ontable block_2)3_1|
6.0:  |(holding block_0)3_0|
6.0:  |(on block_0 block_1)3_1|
6.0:  |(on block_1 block_2)3_0|
6.0:  |(on block_1 block_2)3_1|
6.0:  |(total-cost)3_0| == |(total-cost)3_0|
6.0:  |(total-cost)3_1| == |(total-cost)3_1|
Goal at [6.0]
"""

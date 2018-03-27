import sys
sys.path.append('pddlstream/')
from ss.algorithms.dual_focused import dual_focused
import numpy as np

import kitchen2d.kitchen_stuff as ks
from kitchen_tasks.kitchen_utils import print_plan, execute_plan, step_plan, kitchen_creation_methods, initialize_objects_in_kitchen, update_sizes

from kitchen_tasks.kitchen_problem import create_problem
from kitchen_tasks.kitchen_predicates import *
##################################################


DEFAULT_TASK = {
    'kitchen_params': {
        'do_gui': False,
        'sink_w': 10.,
        'sink_h': 5.,
        'sink_d': 1.,
        'sink_pos_x': 20.,
        'left_table_width': 100.,
        'right_table_width': 100.,
        'faucet_h': 15.,
        'faucet_w': 5.,
        'faucet_d': 0.5,
        'obstacles': [],
        'liquid_name': 'coffee',
        'overclock': 100 # Set overclock higher if run visualization faster
    },
    'initial_poses': {
        'gripper': (0., 15., 0.),
        'cup': (7.5, 0., 0.),
        'sugar_cup': (-10., 0., 0.),
        'cream_cup': (15., 0, 0),
        'spoon': (0.5, 0.5, 0),
        'stirrer': (20, 0.5, 0),
        'block': (-20., 0, 0),
    },
    'object_sizes': {
        'spoon': (0.2, 3, 1.),
        'stirrer': (0.2, 5., 0.5),
        'block': (5, 1, 0),
        'holder': (1., 2., 0.5), # default size of a holder
    },
    'expid_pour': 0,
    'expid_scoop': 0,
}

'''
We use a holder for each spoon and stirrer object to
ensure that these objects can be grasped.
The size and location of each holder depends on the 
spoon or stirrer it is holding. See function make_kitchen
in kitchen_tasks/kitchen_utils.py for how these holders 
are created.
'''

##################################################

def sample_task_instance():
    '''
    Randomly sample a task instance, which has the same
    format as DEFAULT_TASK.
    '''
    task = DEFAULT_TASK.copy()
    expid_pour = task['expid_pour']
    expid_scoop = task['expid_scoop']
    sizes = task['object_sizes']
    sizes = update_sizes(expid_pour, expid_scoop, sizes)

    faucet_w = task['kitchen_params']['faucet_w']
    holder_d = task['object_sizes']['holder'][2]
    delta_dist = 2.
    grasp_dist = 3.
    left = -ks.SCREEN_WIDTH/2 + delta_dist + grasp_dist
    right = ks.SCREEN_WIDTH/2 - delta_dist - grasp_dist

    def sample_kitchen():
        while True:
            sink_pos_x = np.random.uniform(left, right)
            sink_w = np.random.uniform(faucet_w+delta_dist, faucet_w*2)
            if sink_pos_x + sink_w < right:
                return sink_pos_x, sink_w
    sink_pos_x, sink_w = sample_kitchen()
    task['kitchen_params']['sink_pos_x'] = sink_pos_x
    task['kitchen_params']['sink_w'] = sink_w
    
    x_intervals = [[left, sink_pos_x - grasp_dist], 
                   [sink_pos_x + sink_w + grasp_dist, right]]
    
    gripper_x = np.random.uniform(left, right)

    
    
    names = ['cup', 'sugar_cup', 'cream_cup', 'spoon', 'stirrer', 'block']

    def sample_from_region(region):
        weight = [np.prod(a[1]-a[0]) for a in region]
        weight = weight / np.sum(weight)
        rid = np.random.choice(len(weight), p=weight)
        return np.random.uniform(region[rid][0], region[rid][1])

    def test_feasible(x_poses, debug=False):
        '''
        We use the following constraints: 
        1, the distance between two objects is at least delta_dist. 
        The distance between two objects is measured by the Euclidean
        distance from the right side of the left object to the left side 
        of the right object.
        2, The distance from the center of any object to any side of other 
        objects is at least grasp_dist.
        3, The objects are put on the table (table y axis is 0), but not in
        the sink (sink y axis is -sink_h).
        4, block is to the left of everything.
        '''
        for i in range(len(names)):
            if names[i] == 'block':
                if x_poses[i] > sink_pos_x:
                    return False
            for j in range(len(names)):
                if names[i] == 'block':
                    if x_poses[i] > x_poses[j]:
                        return False
                if i <= j:
                    continue
                if (i == 1 or j == 1) and debug:
                    import pdb; pdb.set_trace()
                if x_poses[i] > x_poses[j]:
                    # ensure object i is to the left of j
                    i_ = j; j_ = i
                else:
                    i_ = i; j_ = j
                r_i = x_poses[i_] + max(sizes[names[i_]][0], sizes[names[i_]][2])
                l_j = x_poses[j_] - max(sizes[names[j_]][0], sizes[names[i_]][2])
                if l_j - r_i < delta_dist:
                    return False
                if x_poses[j_] - r_i < grasp_dist:
                    return False
                if l_j - x_poses[i_] < grasp_dist:
                    return False

        return True

    while True:
        x_poses = [sample_from_region(x_intervals) for i in range(len(names))]
        if test_feasible(x_poses):
            break
    test_feasible(x_poses, False)
    cup_x, sugar_x, cream_x, spoon_x, stir_x, block_x = x_poses
    initial_poses = {
        'gripper': (gripper_x, 20., 0.),
        'cup': (cup_x, 0., 0.),
        'sugar_cup': (sugar_x, 0., 0.),
        'cream_cup': (cream_x, 0, 0),
        # Notice that spoons and stirrers will be put in a holder cup
        'spoon': (spoon_x, holder_d, 0),
        'stirrer': (stir_x, holder_d, 0),
        'block': (block_x, 0, 0),
    }
    task['initial_poses'] = initial_poses

    return task


##################################################

def solve_task(kitchen_params, initial_poses, object_sizes, expid_pour, expid_scoop):
    '''
    Solve for a plan of a task and visualize the results.
    Args:
        kitchen_params: init parameters of a Kitchen2D object. See Kitchen2D in 
            kitchen2d/kitchen_stuff.py
        initial_poses: a dictionary mapping from the object names to their poses.
        object_sizes: a dictionary mapping from the object names to their sizes.
        expid_pour: experiment ID for learning to pour.
        expid_scoop: experiment ID for learning to scoop.
    '''
    make_kitchen, make_body = kitchen_creation_methods(expid_pour, expid_scoop, 
        kitchen_params, initial_poses, object_sizes)

    block_goal = (-ks.SCREEN_WIDTH/2 + 5, 0, 0)

    '''
    Initial state of the system.
    Poses of the objects will be added 
    later in create_problem.
    '''
    initial_atoms = [
        # Static atoms
        IsPose('block', block_goal),
        # Fluent atoms
        Empty('gripper'),
        CanMove('gripper'),
        HasSugar('sugar_cup'),
        HasCream('cream_cup'),
        IsPourable('cream_cup'),
        Stackable('cup', 'block'),
        Clear('block')
    ]

    # Goal literals (list of conjunctive literals)
    goal_literals = [
        AtPose('block', block_goal),
        On('cup', 'block'),
        HasCoffee('cup'),
        HasCream('cup'),
        HasSugar('cup'),
        Mixed('cup'),
        Empty('gripper'),
    ]

    # Construct the simulator
    kitchen = make_kitchen(planning=False)
    body_from_name = initialize_objects_in_kitchen(kitchen, make_body, initial_poses, initial_atoms)
    kitchen.enable_gui()

    # Create a problem for STRIPStream
    problem = create_problem(initial_poses, initial_atoms, goal_literals, make_kitchen, make_body,
                   expid_pour, expid_scoop, do_motion=False, do_collisions=True, do_grasp=True)

    # Solve the problem in STRIPStream
    raw_input('Start running STRIPStream solver?')
    plan, evaluations = dual_focused(problem, planner='ff-wastar3', max_time=600, verbose=True, 
        max_planner_time=100, bind=True, use_context=True)


    print_plan(plan, evaluations)
    if plan is None:
        return

    
    if raw_input('Show plan? [y/n]') in ['y', 'Y']:
        # Show plan simulation
        step_plan(plan, kitchen, body_from_name)
    if raw_input('Show plan execution? [y/n]') in ['y', 'Y']:
        # Show execution simulation
        kitchen = make_kitchen(planning=False)
        body_from_name = initialize_objects_in_kitchen(kitchen, make_body, initial_poses, initial_atoms)
        kitchen.enable_gui()
        execute_plan(plan, kitchen, body_from_name)

    raw_input('Done!')


if __name__ == '__main__':
    print 'Sampling task instances...'
    task = sample_task_instance()
    solve_task(**task)

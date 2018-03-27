# author: Caelan Garrett and Zi Wang
import numpy as np
import sys
sys.path.append('motion-planners')
try:
    from motion_planners.rrt_connect import birrt
    from motion_planners.rrt_connect import direct_path
except ImportError:
    raise RuntimeError('Requires https://github.com/caelan/motion-planners')
from kitchen_constants import *

def motion_planner(pos, angle, dpos, dangle, check_point_collision, 
                   check_path_collision, motion_angle=0, linear=False, step_size=0.1, 
                   iters=100, restarts=2, **kwargs):
    '''
    A motion planner wrapper. We use a bi-directional rapidly exploring 
    random tree from 
    https://github.com/caelan/motion-planners

    Alternatively, one can change the implementation of the motion planner 
    to any valid ones such as RRT* or probabilistic roadmap.
    '''
    return rrt(pos, angle, dpos, dangle, check_point_collision, check_path_collision, 
        motion_angle, linear, step_size, iters, restarts)


def linear_rotate(pos, angle, dangle, check_point_collision, step_size=np.pi/16):
    '''
    Helper function for rrt.
    '''
    path = [np.hstack([pos, angle])]
    delta = dangle - angle
    distance = np.abs(delta)
    for t in np.arange(step_size, distance, step_size):
        theta = angle + (t * delta / distance)
        if check_point_collision(pos, theta):
            return None
        path.append(np.hstack([pos, theta]))
    path.append(np.hstack([pos, dangle]))
    return np.array(path)

def rrt(pos, angle, dpos, dangle, check_point_collision, _, motion_angle=0,
        linear=False, step_size=0.1, iters=100, restarts=2, **kwargs):
    '''
    Return a RRT planned path from position pos, angle angle to position dpos, angle dangle.
    '''
    rotate_path1 = linear_rotate(pos, angle, motion_angle, check_point_collision)
    if rotate_path1 is None: return None
    rotate_path2 = linear_rotate(dpos, motion_angle, dangle, check_point_collision)
    if rotate_path2 is None: return None

    def sample_fn():
        lower = [-SCREEN_WIDTH / 2., 0]
        upper = [SCREEN_WIDTH / 2, SCREEN_HEIGHT - TABLE_HEIGHT]
        return b2Vec2(np.random.uniform(lower, upper))

    def distance_fn(q1, q2):
        return np.linalg.norm(np.array(q2) - np.array(q1))

    def extend_fn(q1, q2):
        delta = np.array(q2) - np.array(q1)
        distance = np.linalg.norm(delta)
        for t in np.arange(step_size, distance, step_size):
            yield q1 + (t*delta/distance)
        yield q2

    def collision_fn(q):
        return check_point_collision(q, motion_angle)

    if linear:
        path = direct_path(pos, dpos, extend_fn, collision_fn)
    else:
        path = birrt(pos, dpos, distance_fn, sample_fn, extend_fn, collision_fn,
                     restarts=restarts, iterations=iters, smooth=50)
    if path is None:
        return None
    translate_path = np.array([np.hstack([pos, motion_angle]) for pos in path])
    return np.vstack([rotate_path1[1:-1], translate_path, rotate_path2[1:-1]])
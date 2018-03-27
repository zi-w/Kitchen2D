# Author: Zi Wang
import kitchen2d.kitchen_stuff as ks
from kitchen2d.kitchen_stuff import Kitchen2D
from kitchen2d.gripper import Gripper
import numpy as np
import time
import active_learners.helper as helper
SETTING = {
    'do_gui': False,
    'sink_w': 10.,
    'sink_h': 5.,
    'sink_d': 1.,
    'sink_pos_x': -3.,
    'left_table_width': 50.,
    'right_table_width': 50.,
    'faucet_h': 12.,
    'faucet_w': 5.,
    'faucet_d': 0.5,
    'planning': False,
    'overclock': 50 # number of frames to skip when showing graphics.
}


def query_gui(action_type, kitchen):
    choice = raw_input('Show GUI for {}? [y/n]'.format(action_type))
    if choice == 'y':
        print('Enabling GUI...')
        kitchen.enable_gui()
    else:
        print('Disabling GUI...')
        kitchen.disable_gui()

def main():
    kitchen = Kitchen2D(**SETTING)
    expid_pour, expid_scoop = 0, 0 

    # Try setting is_adaptive to be False. It will use the adaptive sampler that samples uniformly from
    # the feasible precondition. If is_adaptive is False, it uses the diverse sampler. If flag_lk is False,
    # and is_adaptive is False, it uses the diverse sampler with a fixed kernel. If flag_lk is True,
    # and is uniform is Ture, it uses the diverse sampler with the kernel parameters learned from plan feedbacks.
    gp_pour, c_pour = helper.process_gp_sample(expid_pour, exp='pour', is_adaptive=False, flag_lk=False)
    pour_to_w, pour_to_h, pour_from_w, pour_from_h = c_pour

    gp_scoop, c_scoop = helper.process_gp_sample(expid_scoop, exp='scoop', is_adaptive=True, flag_lk=False)
    scoop_w, scoop_h = c_scoop
    holder_d = 0.5

    # Create objects 
    gripper = Gripper(kitchen, (0,8), 0)
    cup1 = ks.make_cup(kitchen, (10,0), 0, pour_from_w, pour_from_h, holder_d)
    cup2 = ks.make_cup(kitchen, (-25,0), 0, pour_to_w, pour_to_h, holder_d)
    block = ks.make_block(kitchen, (-9,0), 0, 4,4)
    large_cup = ks.make_cup(kitchen, (23, 0), 0, scoop_w, scoop_h, holder_d)
    '''
    # Move
    query_gui('MOVE', kitchen)
    gripper.find_path((-5., 10), 0, maxspeed=0.5)

    # Push
    query_gui('PUSH', kitchen)
    gripper.push(block, (1.,0.), -6, 0.5)

    # Pick
    query_gui('GRASP', kitchen)
    # Sample from the super level set of the GP learned for pour
    grasp, rel_x, rel_y, dangle, _, _, _, _ = gp_pour.sample(c_pour)
    dangle *= np.sign(rel_x)

    gripper.find_path((15, 10), 0)
    gripper.grasp(cup1, grasp)

    # Get water
    query_gui('GET-WATER', kitchen)
    gripper.get_liquid_from_faucet(5)

    # Pour
    query_gui('POUR', kitchen)
    print gripper.pour(cup2, (rel_x, rel_y), dangle)

    # Place
    query_gui('PLACE', kitchen)
    gripper.place((10, 0), 0)
    '''
    # Scoop
    kitchen.gen_liquid_in_cup(large_cup, 1000, 'sugar')
    kitchen.gen_liquid_in_cup(cup1, 200)
    spoon = ks.make_spoon(kitchen, (23, 10), 0, 0.2, 3, 1.)

    query_gui('SCOOP', kitchen)
    rel_x1, rel_y1, rel_x2, rel_y2, rel_x3, rel_y3, grasp, _, _ = gp_scoop.sample(c_scoop)
    rel_pos1 = (rel_x1, rel_y1); rel_pos2 = (rel_x2, rel_y2); rel_pos3 = (rel_x3, rel_y3)
    gripper.set_grasped(spoon, grasp, (23, 10), 0)
    print gripper.scoop(large_cup, rel_pos1, rel_pos2, rel_pos3)

    # Dump
    query_gui('DUMP', kitchen)
    gripper.dump(cup1, 0.9)
    
    # Place
    query_gui('PLACE', kitchen)
    gripper.place((26, 10.), 0)

    # Stir
    query_gui('STIR', kitchen)
    stirrer = ks.make_stirrer(kitchen, (0, 3.5), 0., 0.2, 5., 0.5)
    gripper.set_grasped(stirrer, 0.8, (10, 10), 0)
    gripper.stir(cup1, (0, 0.0), (1, 0.0))
    gripper.find_path(gripper.position + [0, 5], 0)


if __name__ == '__main__':
    main()
# Author: Zi Wang
from Box2D import *
from Box2D.b2 import *
from kitchen_constants import *
import scipy.interpolate
import numpy as np
from motion_planner import motion_planner
from scipy.spatial import ConvexHull
import kitchen_stuff as ks

def create_gripper(world, lgripper_pos, rgripper_pos, 
                   init_angle, pj_motorSpeed, 
                   gripper_width=GRIPPER_WIDTH, gripper_height=GRIPPER_HEIGHT):
    '''
    Create a gripper with two sides, connected by a prismatic joint.
    Args: 
        lgripper_pos: position of the left side gripper.
        rgripper_pos: position of the right side gripper.
        init_angle: initial angle of the gripper.
        pj_motorSpeed: the motor speed of the prismatic joint. If pj_motorSpeed > 0, 
        the gripper will open; otherwise it will close. The recommended values are MOTOR_SPEED
        and -MOTOR_SPEED.
        gripper_width: the width of the box shape of either side of gripper.
        gripper_height: the height of the box shape of either side of gripper.

    '''
    lgripper = world.CreateDynamicBody(
            position=lgripper_pos, angle=init_angle, gravityScale=1., allowSleep=True)
    rgripper = world.CreateDynamicBody(
        position=rgripper_pos, angle=init_angle, gravityScale=1., allowSleep=True)
    init_width = np.linalg.norm(lgripper_pos - rgripper_pos) - gripper_width
    lgripper.CreateFixture(
        shape=b2PolygonShape(box=(gripper_width/2, gripper_height/2)),
        density=100.,
        friction=10.
    )
    rgripper.CreateFixture(
        shape=b2PolygonShape(box=(gripper_width/2, gripper_height/2)),
        density=100.,
        friction=10.
    )
    pj = world.CreatePrismaticJoint(
        bodyA=lgripper,
        bodyB=rgripper,
        anchor=lgripper.worldCenter,
        axis=(np.cos(init_angle), np.sin(init_angle)),  # (1, 0),
        lowerTranslation=-init_width,
        upperTranslation=OPEN_WIDTH - init_width,
        enableLimit=True,
        maxMotorForce=10000.0,
        motorSpeed=pj_motorSpeed,
        enableMotor=True,
    )
    lgripper.userData = 'gripper'
    rgripper.userData = 'gripper'
    return lgripper, rgripper, pj

def get_gripper_lrpos(init_pos, init_angle, width):
    '''
    Returns the left side gripper position and the right side 
    gripper position if the gripper is at position init_pos 
    with angle init_angle, and the distance between the two sides
    of gripper is width.
    '''
    init_pos = b2Vec2(init_pos)
    lgripper_pos = init_pos - width/2. * \
        b2Vec2((np.cos(init_angle), np.sin(init_angle)))
    rgripper_pos = init_pos + width/2. * \
        b2Vec2((np.cos(init_angle), np.sin(init_angle)))
    return lgripper_pos, rgripper_pos
def posa_metric(pos, angle, pos2, angle2):
    '''
    A distance metric for position and angle.
    '''
    d1 = np.linalg.norm(np.array(pos) - np.array(pos2))
    def angle2sincos(angle):
        return np.array([np.sin(angle), np.cos(angle)])
    d2 = np.linalg.norm(angle2sincos(angle) - angle2sincos(angle2))
    return d1 + d2

def incup(cup, particles, p_range=SCREEN_WIDTH/2, debug=0):
    '''
    Returns particles that are inside the cup, outside the cup, and whether the particles 
    stopped moving.
    '''
    incupparticles = []
    outcupparticles = []
    stopped = 1
    for p in particles:
        ppos = p.position - cup.position
        tp = cup.angle
        trans = np.array([[np.cos(tp), np.sin(tp)], [-np.sin(tp), np.cos(tp)]])
        ppos = np.dot(trans, ppos)+cup.shift

        if ppos[0] <= cup.usr_w and ppos[0] >= 0.\
                and ppos[1] >= 0. and ppos[1] <= cup.usr_h:
            incupparticles += [p]
        else:
            outcupparticles += [p]
        if (np.abs(p.linearVelocity[1]) > 0.01) and np.abs(p.position[0]) < p_range \
                and p.position[1]>=0.1:
            stopped = 0
    return incupparticles, outcupparticles, stopped

def inspoon(spoon, particles):
    '''
    Returns particles that are inside the spoon, outside the spoon, and whether the particles 
    stopped moving.
    '''
    incupparticles = []
    outcupparticles = []
    stopped = 1
    angle = spoon.angle
    center = spoon.position + spoon.usr_d*np.array([-np.sin(angle), np.cos(angle)])
    for p in particles:
        ppos = p.position 
        if posa_metric(center, 0, ppos, 0) < spoon.usr_d-spoon.usr_w:
            incupparticles += [p]
            if (np.linalg.norm(p.linearVelocity) > 0.1):
                stopped = 0
        else:
            outcupparticles += [p]
    return incupparticles, outcupparticles, stopped

def buffer_shape(shape, radius, sides=4):
    '''
    Return a buffered version of a shape.
    The buffer radius is radius.
    '''
    if isinstance(shape, b2CircleShape):
        return b2CircleShape(pos=shape.pos, radius=shape.radius+radius)
    thetas = [(i+.5)*2*np.pi/sides for i in xrange(sides)]
    buffer_vertices = [radius*np.array([np.cos(theta), np.sin(theta)]) for theta in thetas]
    new_vertices = []
    for v in shape.vertices:
        for b in buffer_vertices:
            new_vertices.append(np.array(v) + b)
    hull = ConvexHull(new_vertices)
    hull_vertices = [tuple(new_vertices[i]) for i in hull.vertices]
    return b2PolygonShape(vertices=hull_vertices)

def copy_kitchen(kitchen, attached_obj=None, buffer_radius=0):
    '''
    Make a copy of kitchen with objects as obstacles for collision checking.
    '''
    if buffer_radius is None:
        buffer_radius = 0
    b2w = ks.b2WorldInterface(do_gui=False, caption='Buffer')
    b2w.planning = True
    b2w.liquid = ks.Liquid(b2w.world) #kitchen.liquid
    for b in kitchen.world.bodies:
        if (b.userData in COPY_IGNORE) or (b == attached_obj):
            continue
        obstacle = b2w.world.CreateStaticBody(
            position=b.position, angle=b.angle, userData='obstacle')
        for fixture in b.fixtures:
            if buffer_radius == 0:
                obstacle.CreateFixture(shape=fixture.shape, isSensor=True)
            else:
                obstacle.CreateFixture(shape=buffer_shape(fixture.shape, buffer_radius), 
                                       isSensor=True)
    return b2w


class Gripper(object):
    def __init__(self, 
                 b2world_interface, 
                 init_pos=None, 
                 init_angle=None, 
                 lgripper_pos=None, 
                 rgripper_pos=None,
                 is_open=True,
                 gripper_width=GRIPPER_WIDTH, 
                 gripper_height=GRIPPER_HEIGHT, 
                 is_copy=False):
        '''
        A two-side gripper with a prismatic joint.
        Args:
            b2world_interface: a Kitchen2D object or b2WorldInterface object.
            init_pos: initial position of the gripper.
            init_angle: initial angle of the gripper.
            lgripper_pos: position of the left gripper.
            rgripper_pos: position of the right gripper.
            If lgripper_pos is None or rgripper_pos is None, these two parameters
            are computed from init_pos and init_angle; otherwise, init_pos is 
            computed from lgripper_pos and rgripper_pos.
            is_open: True if the gripper is open initially; otherwise False.
            gripper_width: the width of the box shape of either side of gripper.
            gripper_height: the height of the box shape of either side of gripper.
            is_copy: indicator of whether the object is a copy of a gripper in the
            kitchen. This is used for collision checking. Collision checking is done
            by creating a copied simulator that simulates all the objects (except 
            particles) as static sensory obstacles. See simulate_itself.
        '''
        assert(init_angle is not None)
        world = b2world_interface.world
        self.b2w = b2world_interface
        self.gripper_width = gripper_width
        self.gripper_height = gripper_height
        
        if is_open:
            pj_motorSpeed = MOTOR_SPEED
            init_width = OPEN_WIDTH 
        else:
            pj_motorSpeed = -MOTOR_SPEED
            init_width = 0
        if lgripper_pos is None or rgripper_pos is None:
            assert(init_pos is not None)
            init_pos = b2Vec2(init_pos)
            lgripper_pos, rgripper_pos = get_gripper_lrpos(init_pos,
                init_angle, init_width+self.gripper_width)

        self.lgripper, self.rgripper, self.pj = create_gripper(world,
            lgripper_pos, rgripper_pos, init_angle, pj_motorSpeed, 
            gripper_width=self.gripper_width, gripper_height=self.gripper_height)
        self.lerror_integral = b2Vec2((0, 0))
        self.laerror_integral = 0.
        self.rerror_integral = b2Vec2((0, 0))
        self.raerror_integral = 0.
        self.attached = False
        self.attached_obj = None
        self.reset_mass()
        self.planning = self.b2w.planning
        self.is_copy = is_copy
    def reset_mass(self):
        '''
        Reset the mass of the gripper.
        '''
        self.mass = self.lgripper.mass + self.rgripper.mass
    def open(self, timeout=1.):
        '''
        Open the gripper. The simulation will be run for timeout seconds.
        '''
        if self.pj.motorSpeed == MOTOR_SPEED:
            return
        self.pj.motorSpeed = MOTOR_SPEED
        t = 0
        pos = self.position.copy()
        angle = self.angle.copy()
        while t < timeout / TIME_STEP:
            t += self.apply_lowlevel_control(pos, angle)
        self.reset_mass()
        self.attached_obj = None
        self.attached = False

    def close(self, timeout=1.):
        '''
        Close the gripper. The simulation will be run for timeout seconds.
        '''
        if self.pj.motorSpeed == -MOTOR_SPEED:
            return
        self.pj.motorSpeed = -MOTOR_SPEED
        t = 0
        pos = self.position.copy()
        angle = self.angle.copy()
        while t < timeout / TIME_STEP:
            t += self.apply_lowlevel_control(pos, angle)


    def pour(self, to_obj, rel_pos, dangle, exact_control=False,  
             stop_ratio=1.0, topour_particles=1, p_range=SCREEN_WIDTH):
        '''
        Use the gripper to pour from the grapsed cup object to 
        to_obj, another cup object.
        Args:
            to_obj: a Box2D body that resembles a cup.
            rel_pos: relative position to to_obj.
            dangle: a control parameter that controls the rotation angle
            of the gripper when performing pouring.
            exact_control: indicator of whether to stop before all the particles 
            are poured.
            stop_ratio: a ratio on the particles to be poured. 
            The gripper starts to rotate back to verticle pose 
            if the particles poured reaches this ratio. This is used 
            only if exact_control is True.
            topour_particles: number of the particles to pour from the grapsed cup.
            p_range: horizontal distance to the to_obj, within which 
            the particles' speed are used to see if pour is completed (all particles
            stops moving).
        Returns: 
            an idicator of whether the pouring action is successful.
            a score of the ratio of successfully poured particles.
        '''
        assert(self.attached)
        dpos = to_obj.position + rel_pos
        found = self.find_path(dpos, 0.)
        if not found:
            return False, 0
        if self.planning:
            return True, 1.
        incupparticles, stopped = self.compute_post_grasp_mass()
        n_particles = len(incupparticles)
        assert n_particles > 0, 'No liquid in cup.'
        stopped = False
        t = 0
        while (not stopped and t * TIME_STEP < 30.):
            t += self.apply_lowlevel_control(dpos, dangle, maxspeed=0.1)
            post_incupparticles, stopped = self.compute_post_grasp_mass(p_range)
            if exact_control and n_particles - len(post_incupparticles) >= topour_particles*stop_ratio:
                self.apply_lowlevel_control(dpos, 0, maxspeed=0.1)
                break
        in_to_cup, _, _ = incup(to_obj, incupparticles)
        self.apply_lowlevel_control(dpos, 0., maxspeed=0.5)
        return True, len(in_to_cup) * 1.0 / n_particles

    def get_cup_feasible(self, from_obj):
        '''
        Returns the feaible range inside a cup Box2D object, from_obj.
        cup_left_offset: the position of the bottom left corner inside the cup.
        cup_feasible: a tuple delta position such that cup_left_offset + cup_feasible
        is the position of the top right corner inside the cup.
        EPS was used to make sure the positions within feasible is at least EPS away 
        from touching the cup.
        '''
        if self.attached_obj.userData == 'spoon':
            offset_l = GRIPPER_WIDTH + self.attached_obj.usr_w/2.
            offset_r = np.max((self.attached_obj.usr_d, offset_l))
        else:
            offset_l = GRIPPER_WIDTH + self.attached_obj.usr_w/2.
            offset_r = offset_l
        # position of the left lower corner of the cup inside
        cup_left_offset = from_obj.position + \
                        np.array([-from_obj.usr_w/2.+from_obj.usr_d+offset_l+EPS, from_obj.usr_d+EPS])

        # feasible range of spoon tip positions relative to cup_left_offset
        cup_feasible = np.array([from_obj.usr_w-from_obj.usr_d*2-offset_r-offset_l-EPS*2., 
            from_obj.usr_h-from_obj.usr_d-EPS])
        return cup_left_offset, cup_feasible

    def get_dump_init_end_pose(self, to_obj, rel_pos_x):
        '''
        Returns the initial and ending pose (including position and angle)
        of the dump action.
        rel_pos_x: relative ratio of the ending horizontal position
        of the spoon. For example, if rel_pos_x is 0, the ending position 
        of the spoon is at the top left corner of to_obj; if rel_pos_x is 1, 
        the ending position of the spoon is at the top right corner of to_obj.
        '''
        assert(self.attached)
        assert(self.attached_obj.userData == 'spoon')
        assert to_obj.userData == 'cup'
        spoon_d = np.max((self.attached_obj.usr_d, GRIPPER_WIDTH))

        delta_pos = np.linalg.norm(self.position - self.attached_obj.position)

        pos_x = delta_pos - to_obj.usr_w / 2. + rel_pos_x * to_obj.usr_w
        dpos1 = to_obj.position + (pos_x, to_obj.usr_h + spoon_d + 1.)
        
        dpos2 = dpos1.copy()
        dpos2[0] -= delta_pos
        dpos2[1] += delta_pos

        return np.hstack((dpos1, -np.pi/2)), np.hstack((dpos2, 0.))
    def dump(self, to_obj, rel_pos_x):
        '''
        Dump a scoop of particles to a Box2D cup, to_obj.
        rel_pos_x: relative ratio of the ending horizontal position
        of the spoon. For example, if rel_pos_x is 0, the ending position 
        of the spoon is at the top left corner of to_obj; if rel_pos_x is 1, 
        the ending position of the spoon is at the top right corner of to_obj.
        '''
        dposa1, dposa2 = self.get_dump_init_end_pose(to_obj, rel_pos_x)
        if not self.find_path(dposa1[:2], dposa1[2], motion_angle=dposa1[2]):
            print 'Failed to find a path to dump'
            return False
        self.apply_control([dposa2])
        return True

    def get_scoop_init_end_pose(self, from_obj, rel_pos1, rel_pos3):
        '''
        Returns the initial and ending poses of the scoop action.
        Args:
            from_obj: the Box2D cup object that the gripper is scooping
            from.
            rel_pos1: control parameters. range: (0,0)->(1,1)
            rel_pos3: control parameters. range: (0,0)->(1,1)
        '''
        delta_pos = np.linalg.norm(self.position - self.attached_obj.position)
        
        cup_left_offset, cup_feasible = self.get_cup_feasible(from_obj)
        if cup_feasible[0] <=0 or cup_feasible[1] <= 0:
            print 'impossible to scoop with the cup size'
            return False, 0

        verticle_offset = np.array([0, delta_pos])
        horizontal_offset = np.array([delta_pos, 0.])

        # initial position of scoop
        dpos1 =  cup_left_offset \
            + np.array(rel_pos1) * cup_feasible \
            + verticle_offset
        dpos3 = cup_left_offset + horizontal_offset\
            + np.array(rel_pos3) * cup_feasible

        dpos = dpos3.copy()
        dpos[1] = from_obj.position[1] + from_obj.usr_h + self.attached_obj.usr_d + 1.

        return np.hstack((dpos1, 0.)), np.hstack((dpos, -np.pi/2))

    def scoop(self, from_obj, rel_pos1, rel_pos2, rel_pos3, maxspeed= 1.0):
        '''
        Scoop with a spoon grasped by the gripper. 
        Args:
            from_obj: the Box2D cup object that the gripper is scooping
            from.
            rel_pos1: control parameters. range: (0,0)->(1,1)
            rel_pos2: control parameters. range: (0,0)->(1,1)
            rel_pos3: control parameters. range: (0,0)->(1,1)
            maxspeed: maximum speed used by the DMP controller.
        Returns:
            an indicator of whether the scooping action is successful.
            a score of scooping scaled to the range [0,1];
            the score is propotional to the number of particles scooped in
            the spoon substracting the number of particles outside the spoon
            and the cup it is scooping from.
            If any particle is both in the spoon and the cup it scooped from,
            we think the scoop action failed. This typically happens if the 
            spoon got stuck in the cup.
        '''
        assert(self.attached)
        assert(self.attached_obj.userData == 'spoon')

        incupparticles, _, _ = incup(from_obj, self.b2w.liquid.particles)


        delta_pos = np.linalg.norm(self.position - self.attached_obj.position)
        
        cup_left_offset, cup_feasible = self.get_cup_feasible(from_obj)
        if cup_feasible[0] <=0 or cup_feasible[1] <= 0:
            print 'Impossible to scoop with the cup size.'
            return False, 0

        verticle_offset = np.array([0, delta_pos])
        horizontal_offset = np.array([delta_pos, 0.])

        # Initial position of scoop
        dpos1 =  cup_left_offset \
            + np.array(rel_pos1) * cup_feasible \
            + verticle_offset
        
        found = self.find_path(dpos1, 0.)
        if not found:
            return False, 0
        dpos2 = cup_left_offset \
            + np.array(rel_pos2) * cup_feasible \
            + verticle_offset

        dpos3 = cup_left_offset + horizontal_offset\
            + np.array(rel_pos3) * cup_feasible

        traj = np.vstack((np.hstack((dpos2, 0.)), np.hstack((dpos3, -np.pi/2))))

        dpos = dpos3.copy()
        dpos[1] = from_obj.position[1] + from_obj.usr_h + self.attached_obj.usr_d + 1.

        traj = np.vstack((traj, np.hstack((dpos, -np.pi/2))))

        self.apply_dmp_control(traj, maxspeed=maxspeed)

        
        stopped = False

        cnt = 0
        max_cnt = 10
        while not stopped:
            print 'Balancing the spoon...'
            self.apply_dmp_control(np.hstack((dpos, -np.pi/2)), maxspeed=0.1)
            in_spoon, stopped = self.compute_post_grasp_mass()
            cnt += 1
            if cnt > max_cnt:
                break
        incup_spoon, _, _ = incup(from_obj, in_spoon)
        if len(incup_spoon) > 0:
            return False, 0.
        incupparticles2, _, _ = incup(from_obj, incupparticles)
        outside = (len(in_spoon) + len(incupparticles2) - len(incupparticles))

        return True,  (len(in_spoon) + outside) / (0.5*self.attached_obj.usr_d**2*np.pi/(self.b2w.liquid.radius**2*4))

    def get_liquid_from_faucet_pos(self):
        '''
        Returns the gripper position to get liquid from the faucet.
        '''
        obj_pos = np.array(self.b2w.faucet_location) - (0, 0.1)
        if self.position[1] + self.gripper_height/2 - self.attached_obj.position[1] - self.attached_obj.usr_h > 0:
            gripper_pos = obj_pos - (0, self.gripper_height/2)
        else:
            obj_pos[1] -= self.attached_obj.usr_h
            gripper_pos = obj_pos + self.position - self.attached_obj.position
        return gripper_pos

    def get_liquid_from_faucet(self, runtime):
        '''
        Get liquid from the faucet in the kitchen.
        Args:
            runtime: the duration of time (in seconds) that the 
            gripper stays under the faucet.
        '''
        assert(self.attached)
        gripper_pos = self.get_liquid_from_faucet_pos()
        found = self.find_path(gripper_pos, 0)
        if not found:
            return False
        if self.planning:
            return True
        t = 0
        while t < runtime/TIME_STEP:
            t += self.apply_lowlevel_control(gripper_pos, 0, maxspeed=0.1)
            self.compute_post_grasp_mass()
        return True

    def get_stir_init_end_pose(self, from_obj, rel_pos1, rel_pos2):
        '''
        Returns the initial and ending poses of the stir action.
        Args:
            from_obj: the Box2D cup object that the gripper is stirring.
            rel_pos1: control parameters. range: (0,0)->(1,1)
            rel_pos3: control parameters. range: (0,0)->(1,1)
        '''
        delta_pos = np.linalg.norm(self.position - self.attached_obj.position)
        
        cup_left_offset, cup_feasible = self.get_cup_feasible(from_obj)
        if (cup_feasible[0] <= 0) or (cup_feasible[1] <= 0):
            print 'impossible to scoop with the cup size'
            return False, 0

        vertical_offset = np.array([0, delta_pos])
        # initial position of stir
        dpos1 = cup_left_offset \
            + np.array(rel_pos1) * cup_feasible \
            + vertical_offset
        # ending position of stir
        dpos2 = cup_left_offset \
            + np.array(rel_pos2) * cup_feasible \
            + vertical_offset
        return np.hstack((dpos1, 0.)), np.hstack((dpos2, 0))

    def stir(self, obj, rel_pos1, rel_pos2, num_stirs=5):
        '''
        Scoop with a spoon grasped by the gripper. 
        Args:
            obj: the Box2D cup object that the gripper is strring.
            rel_pos1: control parameters. range: (0,0)->(1,1)
            rel_pos2: control parameters. range: (0,0)->(1,1)
            num_stirs: number of stirs. One stir is a movement from left 
            to right to left.
        Returns:
            an indicator of whether the stirring action is successful.
        '''
        dposa1, dposa2 = self.get_stir_init_end_pose(obj, rel_pos1, rel_pos2)
        dposa1[2] = 0.
        dposa2[2] = 0.
        if not self.find_path(dposa1[:2], dposa1[2]):
            print('Failed to stir because no path is found.')
            return False
        for i in range(num_stirs):
            self.apply_lowlevel_control(dposa1[:2], dposa1[2], maxspeed=1.)
            self.apply_lowlevel_control(dposa2[:2], dposa2[2], maxspeed=1.)
        return True

    def bounding_box(self, body):
        '''
        Returns the min and max bounding box vertices of a Box2D body.
        '''
        vertices = np.array(ks.get_body_vertices(body))
        return vertices.min(axis=0), vertices.max(axis=0)

    def grasp_translation(self, body, pos_ratio):
        """
        Returns the gripper position of a grasp relative to the body frame
        """
        if body.userData == 'spoon' or body.userData == 'stir':
            grasp_w = body.usr_d*2 + GRIPPER_HEIGHT/2 + EPS + (body.usr_h*0.9-2*EPS) * pos_ratio
            gripper_pos = body.position + grasp_w*np.array([-np.sin(body.angle), np.cos(body.angle)])
        else:
            assert(np.isclose(body.angle, 0, atol=1e-3))
            lower, upper = self.bounding_box(body)
            body_h = (upper[1] - lower[1]) - EPS
            gripper_pos = np.array([body.position[0],
                lower[1] + (pos_ratio * body_h) + (self.gripper_height / 2) + (EPS / 2.)])
        return gripper_pos - body.position

    def get_grasp_poses(self, body, pos_ratio):
        '''
        Args:
            body: a Box2D object to be grasped.
            pos_ratio: a control parameter; range: 0->1
        Returns:
            dpos: the target position of the gripper for grasping
            dpos_up: the target position on top of the object, before reaching
            dpos.
        '''
        dpos = self.grasp_translation(body, pos_ratio) + body.position
        _, upper = self.bounding_box(body)
        dpos_up = np.array([dpos[0],
            upper[1] + 0.5 + (self.gripper_height / 2)])
        return dpos, dpos_up

    def place(self, obj_dpos, obj_dangle):
        '''
        Place the grasped object down.
        Args:
            obj_dpos: target position of the attached object.
            obj_dangle: target angle of the attached object.
        Returns:
            indicator of whether placing is successful.
        '''
        assert(self.attached)
        g2 = self.simulate_itself()
        g2.set_position(obj_dpos, 0)
        rel_xy = g2.position - g2.attached_obj.position
        dpos_gripper = rel_xy + obj_dpos
        dpos_gripper[1] += EPS
        found = self.find_path(dpos_gripper, obj_dangle)

        if not found:
            print 'Failed to place because no path was found.'
            return False
        _, dpos_up = self.get_grasp_poses(self.attached_obj, 0)
        self.open()

        self.apply_lowlevel_control(dpos_up, 0)
        return True
    def grasp_updown(self, dpos, dpos_up, body):
        '''
        Grasp a Box2D body.
        Args:
            dpos: target position to grasp the body.
            dpos_up: the target position on top of the object, 
            before reaching dpos.
            body: a Box2D body to be grasped.
        Returns:
            indicator of whether the grasp is successful.
        '''
        dpos = b2Vec2(dpos)
        dpos_up = b2Vec2(dpos_up)
        self.open()
        found = self.find_path(dpos_up, 0)
        if not found:
            return False
        found = self.find_path(dpos, 0)
        if not found:
            return False
        self.close()

        print('mass before grasp check={}'.format(self.mass))
        self.check_grasp(body)
        print('mass after grasp check={}'.format(self.mass))
        dpos[1] += EPS
        self.apply_lowlevel_control(dpos, 0)
        return self.attached

    def grasp(self, body, pos_ratio):
        '''
        Grasp a Box2D body.
        Args:
            body: a Box2D body to be grasped.
            pos_ratio: a control parameter; range: 0->1
        Returns:
            indicator of whether the grasp is successful.
        '''
        dpos, dpos_up = self.get_grasp_poses(body, pos_ratio)
        self.open()
        found = self.find_path(dpos_up, 0)
        if not found:
            return False
        found = self.find_path(dpos, 0)
        if not found:
            return False
        self.close()

        print('mass before grasp check={}'.format(self.mass))
        self.check_grasp(body)
        print('mass after grasp check={}'.format(self.mass))
        # avoid collision checker to think "on countertop" is a collision
        dpos[1] += EPS
        self.apply_lowlevel_control(dpos, 0)
        return self.attached

    def is_graspable(self, body):
        '''
        Returns True if the Box2D body can be grasped.
        '''
        return body.usr_w <= OPEN_WIDTH

    def set_closed(self):
        '''
        Set the gripper to be closed.
        '''
        width = self.gripper_width
        self.lgripper.position, self.rgripper.position = get_gripper_lrpos(self.position, self.angle, width)
        self.pj.motorSpeed = -MOTOR_SPEED
    def set_open(self):
        '''
        Set the gripper to be open.
        '''
        width = OPEN_WIDTH + self.gripper_width
        self.lgripper.position, self.rgripper.position = get_gripper_lrpos(self.position, self.angle, width)
        self.pj.motorSpeed = MOTOR_SPEED

    def get_push_pos(self, body, goal_x):
        '''
        Computes the push positions.
        Args:
            body: a Box2D body to be pushed.
            goal_x: goal_x is the x axis of the goal position.
        Returns:
            start_gripper_pos: starting position to push body to goal_x.
            end_gripper_pos: ending position to push body to goal_x
        '''
        lower, upper = self.bounding_box(body)

        start_body_pos = np.array(body.position)
        end_body_pos = np.array([goal_x, start_body_pos[1]])

        block_offset = 0.1
        if end_body_pos[0] < start_body_pos[0]:
            x = upper[0] + (self.gripper_width + block_offset)
        else:
            x = lower[0] - (self.gripper_width + block_offset)
        ground_offset = 0.5 # 0.1

        y = lower[1] + self.gripper_height / 2. + ground_offset
        start_gripper_pos = (x, y)
        end_gripper_pos = tuple(end_body_pos + (np.array(start_gripper_pos) - start_body_pos))
        return start_gripper_pos, end_gripper_pos

    def push(self, body, rel_pos, goal_x, maxspeed):
        '''
        Push an object.
        Args:
            body: a Box2D body to be pushed.
            rel_pos: relative horizontal distance to the object before push starts.
            goal_x: goal_x is the x axis of the goal position.
            maxspeed: maximum speed of pushing.
        Returns:
            indicator of whether the push is successful.
        '''
        if goal_x == body.position[0]:
            return True
        assert(rel_pos[0] > 0)

        lower, upper = self.bounding_box(body)
        dpos = lower + rel_pos + self.gripper_height/2. + 0.1
        if goal_x < body.position[0]:
            dpos[0] = upper[0] + rel_pos[0] + self.gripper_width
        else:
            dpos[0] = lower[0] - rel_pos[0] - self.gripper_width
        self.close()
        found = self.find_path(dpos, 0)
        if not found:
            print found
            return False
        if self.planning:
            return True
        if goal_x < body.position[0]:
            dpos[0] = goal_x + body.usr_w/2. + self.gripper_width
        else:
            dpos[0] = goal_x - body.usr_w/2. - self.gripper_width
        dangle = 0
        adjust_cnt = 0
        while posa_metric(self.position, self.angle, dpos, dangle) > 0.01:
            self.apply_lowlevel_control(dpos, dangle, maxspeed=maxspeed)
            adjust_cnt += 1
            if adjust_cnt > 10:
                break

    def attach(self, body):
        '''
        Attach body to gripper.
        '''
        self.pj.motorSpeed = -MOTOR_SPEED # Need this for force closure
        self.attached = True
        self.attached_obj = body

    def release(self):
        '''
        Release the attached object from gripper.
        '''
        self.pj.motorSpeed = MOTOR_SPEED # Need this for force closure
        self.attached = False
        self.attached_obj = None
        self.reset_mass()

    def set_grasped(self, body, pos_ratio, pos, angle):
        '''
        Configures a Box2D body as if it is grasped by the gripper. 
        This is used mainly for planning.
        Args:
            body: a Box2D body to be grasped.
            pos_ratio: the control paramter for grasp. See grasp.
            pos: position of gripper after grasping body.
            angle: angle of gripper after grasping body.
        Returns:
            indicator of whether body can be grasped.
        '''
        self.pj.motorSpeed = -MOTOR_SPEED
        self.attached = True
        self.attached_obj = body
        self.compute_post_grasp_mass()
        gripper_pos, _ = self.get_grasp_poses(body, pos_ratio)
        if not self.is_graspable(body):
            return False
        width = body.usr_w + self.gripper_width
        lgripper_pos, rgripper_pos = get_gripper_lrpos(gripper_pos, body.angle, width)
        self.lgripper.position = lgripper_pos
        self.rgripper.position = rgripper_pos
        self.set_position(pos, angle)
        return True

    def check_grasp(self, obj, update_mass=True):
        '''
        Checks if an object is grasped by the gripper. Notice that 
        the step function needs to be called from the world associated 
        with the gripper before checking after set_grasped.
        Args:
            obj: a Box2D body object.
            update_mass: indicator of whether or not to update the mass
            of gripper (to include the grasped object).
        Returns:
            flag: indicator of successful grasp.
        '''
        flag = False
        for cs in self.lgripper.contacts:
            c = cs.contact
            cond = obj in [c.fixtureA.body, c.fixtureB.body]
            if c.touching and cond: #and c.manifold.localNormal[0]==1:
                flag = True
        if not flag:
            return False
        flag = False
        for cs in self.rgripper.contacts:
            c = cs.contact
            cond = obj in [c.fixtureA.body, c.fixtureB.body]
            if c.touching and cond: #and c.manifold.localNormal[0]==-1:
                flag = True
        if flag:
            self.attached = flag
            self.attached_obj = obj
            self.compute_post_grasp_mass()

        return flag

    def compute_post_grasp_mass(self, p_range=SCREEN_WIDTH):
        '''
        Computes the mass of gripper, which includes whatever objects it is holding.
        '''
        obj = self.attached_obj
        self.reset_mass()
        self.mass += obj.mass
        incupparticles = None
        stopped = True
        if obj.userData == 'cup':
            incupparticles, outcupparticles, stopped = incup(obj, self.b2w.liquid.particles, p_range=p_range)
            self.mass += np.sum([p.mass for p in incupparticles])
        elif obj.userData == 'spoon':
            incupparticles, outcupparticles, stopped = inspoon(obj, self.b2w.liquid.particles)
            self.mass += np.sum([p.mass for p in incupparticles])
        return incupparticles, stopped

    def get_traj_weight(self, traj, a, b):
        '''
        Helper function for apply_dmp_control.
        Adapted from https://github.com/studywolf/pydmps
        '''
        timesteps = traj.shape[1]
        traj = np.hstack((traj, traj[:,-1:]))
        dtraj = np.diff(traj) / TIME_STEP
        dtraj = np.hstack((self.velocity[:,None], dtraj))
        ddtraj = np.diff(dtraj) / TIME_STEP

        f_target = ddtraj - a * (b * (traj[:, -1:] - traj[:, :-1]) - dtraj[:, :-1])
        x_track = np.exp(-np.arange(timesteps)*TIME_STEP)
        nbfs = 200
        c = np.exp(-np.linspace(0, timesteps*TIME_STEP, nbfs))
        self.c = c
        h = np.ones(nbfs) * nbfs**1.5 / c
        self.h = h
        psi_track = np.exp(- h * (x_track[:, None] - c)**2)
        w = np.zeros((3, nbfs))
        for d in range(3):
            k = (traj[d, -1] - traj[d, 0])
            if k == 0:
                k = 0.0000001
            for b in range(nbfs):
                numer = np.sum(x_track * psi_track[:, b] * f_target[d,:])
                denom = np.sum(x_track**2 * psi_track[:, b])

                w[d, b] = numer / (k * denom)
        return np.nan_to_num(w)

    def apply_dmp_control(self, traj, maxspeed=0.5, a=1., b=.5):
        '''
        Apply Dynamic Movement Primitives (DMP) controller to the gripper.
        Adapted from https://github.com/studywolf/pydmps
        Args:
            traj: a t x d numpy array where t is the number of poses along the 
            trajectory and d is 3 (the dimension of pose).
            maxspeed: the maximum speed of moving.
            a: control parameter of DMP
            b: control parameter of DMP
        '''
        if traj.ndim == 1:
            traj = traj[None,:]
        cur_pos = np.hstack((self.position, self.angle))

        if np.linalg.norm(cur_pos-traj[0]) != 0:
            traj = np.vstack((cur_pos, traj))
        des_pos = traj[-1]
        assert maxspeed > 0
        segment_lengths = np.array([0.] + [np.linalg.norm(traj[i] - traj[i-1]) for i in range(1,len(traj))])
        tot_length = np.sum(segment_lengths)
        assert(tot_length > 0)
        max_timesteps = int(tot_length / maxspeed / TIME_STEP + 1) + 1
        timesteps = max_timesteps
        runtime = timesteps * TIME_STEP
        path_gen = scipy.interpolate.interp1d(np.cumsum(segment_lengths)*runtime/tot_length, traj.T, fill_value='extrapolate')
        traj = path_gen(np.arange(timesteps) * runtime / (timesteps-1))

        w = self.get_traj_weight(traj, a, b)
        for step in range(timesteps+2):
            x = np.exp(-step*TIME_STEP)
            psi = np.exp(- self.h * (x - self.c)**2)
            f = x * (des_pos - cur_pos) * np.dot(psi, w.T) / np.sum(psi)
            error = (des_pos - np.hstack((self.position, self.angle)))
            #import pdb;pdb.set_trace()
            ddy = a * (b * (error) - self.velocity) + f
            lforce = (ddy[:2] - GRAVITY) * self.mass/2
            rforce = (ddy[:2] - GRAVITY) * self.mass/2
            aforce = (ddy[2] * self.inertia) / self.radius
            if self.attached:
                aforce += (-GRAVITY[1]) * self.attached_obj.mass * (self.attached_obj.worldCenter[0] - self.position[0])/ self.radius
            g_aforce = aforce * np.array([np.sin(self.angle), -np.cos(self.angle)]) / 2

            lforce += g_aforce
            rforce -= g_aforce
            self.lgripper.ApplyForce(lforce, self.lgripper.position, wake=True)
            self.rgripper.ApplyForce(rforce, self.rgripper.position, wake=True)
            self.b2w.step()
        return True

    def apply_lowlevel_control(self, dpos, dangle, maxspeed=1., collision_check=False):
        '''
        Apply a lowlevel controller to the gripper, which moves to the target pose 
        by following the straight line interpolation from the current pose.
        Args:
            dpos: target position.
            dangle: target angle.
            maxspeed: the maximum speed of moving.
            collision_check: indicator of whether to check collision along the way.
        Returns: 
            if collision_check is True, it returns the trajectory until collision happened;
            otherwise, return the length of the trajectory.
        '''
        dpos = b2Vec2(dpos)

        dposa = np.hstack((dpos, dangle))
        cur_pos = np.hstack((self.position, self.angle))

        t0 = 1 # timesteps for acceleration and deceleration
        assert maxspeed > 0
        max_timesteps = int(np.max(np.abs(dposa - cur_pos)/maxspeed) / TIME_STEP + 1) + t0
        t = max_timesteps

        cur_v = self.velocity
        v = (dposa - cur_pos - cur_v*t0*TIME_STEP/2) / (t-t0) / TIME_STEP
        ddy = np.array([(v-cur_v)/t0/TIME_STEP]*t0 + [[0]*3]*(t-2*t0) + [-v/t0/TIME_STEP]*t0)
        traj = [(self.position, self.angle)]
        for step in range(len(ddy)):
            lforce = (ddy[step,:2] - GRAVITY) * self.mass/2
            rforce = (ddy[step,:2] - GRAVITY) * self.mass/2
            aforce = (ddy[step, 2] * self.inertia) / self.radius
            if self.attached:
                aforce += (-GRAVITY[1]) * self.attached_obj.mass * (self.attached_obj.worldCenter[0] - self.position[0])/ self.radius
            g_aforce = aforce * np.array([np.sin(self.angle), -np.cos(self.angle)]) / 2
            lforce += g_aforce
            rforce -= g_aforce
            self.lgripper.ApplyForce(lforce, self.lgripper.position, wake=True)
            self.rgripper.ApplyForce(rforce, self.rgripper.position, wake=True)
            self.b2w.step()
            traj.append((self.position, self.angle))
            if collision_check and self.check_collision():
                return traj, True

        if collision_check:
            return traj, False
        else:
            return len(ddy)

    def check_point_collision(self, pos, angle=0.):
        '''
        Check if the gripper is colliding with objects in the scene if
        the gripper is at position pos and angle angle.
        '''
        self.set_position(pos, angle)
        force = -self.b2w.gravity * self.mass/2
        self.lgripper.ApplyForce(force, self.lgripper.position, wake=True)
        self.rgripper.ApplyForce(force, self.rgripper.position, wake=True)
        self.b2w.step()
        return self.check_collision()
    def check_path_collision(self, pos, angle, dpos, dangle):
        '''
        Checks if the gripper is colliding with objects in the scene if 
        it moves from position pos and angle angle to position dpos and angle
        dangle following a straight line.
        '''
        self.set_position(pos, angle)
        traj, collision = self.apply_lowlevel_control(dpos, dangle, maxspeed=1.0, collision_check=True)
        if not collision:
            return traj, collision
        res_id = np.max([-10, -len(traj)])
        while(self.check_point_collision(*traj[res_id])):
            if res_id == -len(traj):
                return [], True
            res_id -= 1
            res_id = np.max([res_id, -len(traj)])
        return traj[:res_id+1], collision

    def check_collision(self):
        assert self.is_copy, 'only copied gripper can check collisions'
        contacts = self.lgripper.contacts + self.rgripper.contacts
        if self.attached:
            contacts += self.attached_obj.contacts
        for c in contacts:
            if c.contact.touching and (c.contact.fixtureA.body.userData is 'obstacle' \
                                       or c.contact.fixtureB.body.userData is 'obstacle'):
                return True
        return False

    @property
    def position(self):
        return (self.lgripper.position+self.rgripper.position)/2

    @property
    def angle(self):
        dpos = self.rgripper.position - self.lgripper.position
        angle = np.arctan2(dpos[1], dpos[0])
        return angle

    @property
    def position_angle(self):
        return np.hstack((self.position, self.angle))
    pose = position_angle

    @property
    def radius(self):
        return np.linalg.norm((self.lgripper.position - self.position), ord=2)

    @property
    def inertia(self):
        inertia = self.lgripper.inertia + self.lgripper.mass * (self.radius**2)
        inertia += self.rgripper.inertia + self.rgripper.mass * (self.radius**2)
        if self.attached:
            radius_cup = np.linalg.norm(self.attached_obj.worldCenter - self.position)
            inertia += self.attached_obj.mass * (radius_cup**2)
        return inertia

    @property
    def velocity(self):
        linear_v = (self.lgripper.linearVelocity + self.rgripper.linearVelocity)/2
        trans = np.array([-np.sin(self.angle), np.cos(self.angle)])
        r_av = np.sum(trans * self.rgripper.linearVelocity)
        l_av = -np.sum(trans * self.lgripper.linearVelocity)
        angular_v = (r_av + l_av)/2/self.radius
        return np.hstack((linear_v, angular_v))

    def set_position(self, pos, angle):
        '''
        Set the pose of gripper to be at position pos and angle angle.
        The pose of the attached object is adjusted to ensure what is 
        attached is still attached after this function is called.
        '''
        self.b2w.world.ClearForces()
        if self.attached:
            rel_a = angle - self.angle
            p = self.attached_obj.position - self.position
            p = np.array([[np.cos(rel_a), -np.sin(rel_a)],
                [np.sin(rel_a), np.cos(rel_a)]]).dot(p)
            self.attached_obj.position = p + pos
            self.attached_obj.angle += rel_a
            self.attached_obj.linearVelocity = b2Vec2((0,0))
            self.attached_obj.angularVelocity = 0
        lpos, rpos = get_gripper_lrpos(pos, angle, self.radius*2)
        self.lgripper.position = lpos
        self.rgripper.position = rpos
        self.lgripper.angle = angle
        self.rgripper.angle = angle
        self.set_angular_velocity()
        self.set_linear_velocity()

    def set_pose(self, pose):
        '''
        Set the pose of gripper to be pose.
        The pose of the attached object is adjusted to ensure what is 
        attached is still attached after this function is called.
        '''
        self.set_position(pose[:2], pose[2])

    def set_angular_velocity(self, ravel=0):
        '''
        Set the angular velocity of the gripper to be ravel.
        '''
        self.lgripper.angularVelocity = ravel
        self.rgripper.angularVelocity = ravel

    def set_linear_velocity(self, rlvel=(0, 0)):
        '''
        Set the linear velocity of the gripper to be rlvel.
        '''
        self.lgripper.linearVelocity = b2Vec2(rlvel)
        self.rgripper.linearVelocity = b2Vec2(rlvel)

    @property
    def info(self):
        return (self.lgripper.position, self.lgripper.angle, self.rgripper.position,
                self.rgripper.angle, self.pj.motorSpeed)

    def simulate_itself(self, collision_buffer=None):
        '''
        Returns a copy of the gripper, where the objects in its world are sensory obstacles,
        except the gripper itself and what it is grasping. The sensory obstacles uses a buffer
        collision_buffer so that the obstacles are enlarged in the copy.
        '''
        b2w2 = copy_kitchen(self.b2w, self.attached_obj, collision_buffer)
        g2 = Gripper(b2w2,
                     lgripper_pos=self.lgripper.position.copy(),
                     rgripper_pos=self.rgripper.position.copy(),
                     init_angle=self.angle,
                     is_open=True if self.pj.motorSpeed > 0 else False,
                     is_copy=True)
        if self.attached:
            g2.attached = True
            name = self.attached_obj.userData
            attached_obj_pose = np.hstack((self.attached_obj.position, self.attached_obj.angle))
            attached_obj_size = {name:(self.attached_obj.usr_w, self.attached_obj.usr_h, self.attached_obj.usr_d)}
            g2.attached_obj = ks.make_body(kitchen=b2w2, 
                                           name=name, 
                                           pose=attached_obj_pose, 
                                           args=attached_obj_size)
            
            g2.mass += g2.attached_obj.mass
        return g2

    def apply_control(self, path, maxspeed=0.5):
        '''
        Apply the lowlevel controller to follow a path.
        '''
        for q in path:
            self.apply_lowlevel_control(q[:2], q[2], maxspeed=maxspeed)

    def plan_path(self, dpos, dangle, collision_buffer=None, motion_angle=0):
        '''
        Plan a path from the current pose to a target pose, avoiding obstacles.
        Args:
            dpos: target position.
            dangle: target angle.
            collision_buffer: the buffer of objects when checking collisions.
            motion_angle: the angle of the gripper when the motion is applied.
        Returns:
            traj: a trajectory planned by rapidly exploring random tree.
        '''
        dpos = b2Vec2(dpos)
        g2 = self.simulate_itself(collision_buffer=collision_buffer)
        traj = motion_planner(self.position, self.angle, dpos, dangle,
                   g2.check_point_collision, g2.check_path_collision, motion_angle=motion_angle)
        return traj

    def find_path(self, dpos, dangle, collision_buffer=None, motion_angle=0, maxspeed=1.0):
        '''
        Finds and follows a path from the current pose to a target pose, avoiding obstacles.
        Args:
            dpos: target position.
            dangle: target angle.
            collision_buffer: the buffer of objects when checking collisions.
            motion_angle: the angle of the gripper when the motion is applied.
            maxspeed: maximum speed of movement.
        Returns:
            indicator of if a path is found.
        '''
        if posa_metric(self.position, self.angle, dpos, dangle) < ACC_THRES:
            return True

        traj = self.plan_path(dpos, dangle, collision_buffer=collision_buffer, motion_angle=motion_angle)
        if traj is None:
            return False

        if self.planning:
            g2 = self.simulate_itself(collision_buffer=None)
            col = g2.check_point_collision(self.position, self.angle)
            if col:
                return False
            col = g2.check_point_collision(dpos, dangle)
            if col:
                return False
            self.set_position(dpos, dangle)
            return True
        else:
            self.apply_control(traj, maxspeed=maxspeed)
        adjust_cnt = 0
        while posa_metric(self.position, self.angle, dpos, dangle) > ACC_THRES:
            self.apply_lowlevel_control(dpos, dangle, maxspeed=maxspeed)
            adjust_cnt += 1
            if adjust_cnt > 5:
                break
        return True
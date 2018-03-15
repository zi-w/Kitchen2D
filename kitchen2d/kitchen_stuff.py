# Author: Zi Wang
from Box2D import *
from Box2D.b2 import *
import scipy.interpolate
import pygame
import numpy as np
from numpy import linalg as LA
import json
import sys
import copy
from scipy.spatial import ConvexHull
import utils.helper as helper
import sys


# constants
GRAVITY = b2Vec2(0.0, -10.0)
PPM = 10.0  # pixels per meter
TARGET_FPS = 100
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX = 640, 480
#SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX = 250, 180

SCREEN_WIDTH = SCREEN_WIDTH_PX / PPM
SCREEN_HEIGHT = SCREEN_HEIGHT_PX / PPM
GRIPPER_WIDTH = 0.6
GRIPPER_HEIGHT = 2.0
OPEN_WIDTH = 5.0
TABLE_HEIGHT = 20
#TABLE_HEIGHT = 1

TABLE_THICK = 2
ACC_THRES = 0.1
VEL_ITERS = 10
POS_ITERS = 10
PID_STABLE_TIMESTEPS = 100
SAFE_LEVEL = 25.
SAFE_MOVE_THRES = 1
MOTOR_SPEED = 5.0

EPS = 0.2
OVERCLOCK = 50 # None | 50 | 100

class guiWorld:
    def __init__(self, caption='PyBox2D Simulator'):
        self.screen = pygame.display.set_mode(
            (SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX), 0, 32)
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        # in body frame
        self.screen_origin = b2Vec2(SCREEN_WIDTH/2., TABLE_HEIGHT)
        self.colors = {
            'countertop': (50, 50, 50, 255),
            'gripper': (244, 170, 66, 255),
            'cup': (15, 0, 100, 0),
            'stirrer': (163, 209, 224, 255),
            'obstacle': (123, 128, 120, 255),
            'sensor': (255, 255, 255, 20),
            #'sensor': (155, 155, 155, 255), # To visualize the sensor
            'default': (81, 81, 81, 255),
            'faucet': (175, 175, 175, 255),
            'water': (26, 130, 252),
            'block': (0, 99, 0),
            'coffee': (165, 42, 42),
            'cream': (225, 225, 225),
            'sugar': (255, 192, 203),
        }

    def draw(self, bodies, bg_color=(255, 255, 255, 1)):
        # def draw(self, bodies, bg_color=(0,0,0,0)):
        def my_draw_polygon(polygon, body, fixture):
            vertices = [(self.screen_origin + body.transform*v)
                        * PPM for v in polygon.vertices]
            vertices = [(v[0], SCREEN_HEIGHT_PX-v[1]) for v in vertices]
            color = self.colors.get(body.userData, self.colors['default'])

            pygame.draw.polygon(self.screen, color, vertices)

        def my_draw_circle(circle, body, fixture):
            position = (self.screen_origin + body.transform*circle.pos)*PPM
            position = (position[0], SCREEN_HEIGHT_PX-position[1])
            color = self.colors.get(body.userData, self.colors['default'])
            pygame.draw.circle(self.screen, color, [int(x) for x in position],
                               int(circle.radius*PPM))

        b2PolygonShape.draw = my_draw_polygon
        b2CircleShape.draw = my_draw_circle
        # draw the world
        self.screen.fill(bg_color)
        if OVERCLOCK is None:
            self.clock.tick(TARGET_FPS)
        pygame.event.get() # TODO(caelan): this is needed to update on OS X for some reason...
        for body in bodies:
            for fixture in body.fixtures:
                fixture.shape.draw(body, fixture)
        pygame.display.flip()

# this is the interface to pybox2d


class b2WorldInterface(object):
    def __init__(self, do_gui=True, save_fig=False, caption='PyBox2D Simulator'):
        self.world = b2World(gravity=GRAVITY, doSleep=True)
        #self.liquid = Liquid(self.world)
        self.do_gui = do_gui
        if do_gui:
            self.gui_world = guiWorld(caption=caption)
        else:
            self.gui_world = None
        self.save_fig = save_fig
        self.planning = False
        self.image_idx = 0
        self.image_name = 'test'
        self.gravity = GRAVITY
        self.num_steps = 0
    def enable_gui(self, caption='PyBox2D Simulator'):
        self.do_gui = True
        self.gui_world = guiWorld(caption=caption)
    def disable_gui(self):
        self.do_gui = False
    def draw(self):
        if not self.do_gui:
            return
        self.gui_world.draw(self.world.bodies)
        if self.save_fig and self.image_idx % 100 == 0:
            pygame.image.save(self.gui_world.screen,
                              'tmp_images/{num:05d}_{nm}'.format(num=self.image_idx/100, nm= self.image_name)+'.png')
        self.image_idx += 1
    def step(self, idx=0):
        self.world.Step(TIME_STEP, VEL_ITERS, POS_ITERS)
        self.world.ClearForces()
        self.num_steps += 1
        if (OVERCLOCK is None) or (self.num_steps % OVERCLOCK == 0):
            self.draw()

class Kitchen2D(b2WorldInterface):
    def __init__(self, do_gui, sink_w, sink_h, sink_d, sink_pos_x, 
                 left_table_width, right_table_width, faucet_h, faucet_w, faucet_d, planning=True,
                 obstacles=None, save_fig=False, liquid_name='water', frequency=0.2):
        # sink_w: sink length (horizontal)
        # sink_h: sink height
        # sink_d: table/sink thickness
        # faucet_h: height of faucet
        # faucet_w: horizontal length of the faucet
        # height and length all include the thickness of the material
        super(Kitchen2D, self).__init__(do_gui, save_fig=save_fig)
        self.planning = planning
        world = self.world
        self.liquid = Liquid(world, liquid_name=liquid_name, frequency=frequency)
        self.disable_liquid = False
        self.base = world.CreateStaticBody(
            position=(sink_pos_x, -sink_h),
            shapes=[b2PolygonShape(vertices=[(0, 0),
                                             (sink_w, 0),
                                             (sink_w, sink_d),
                                             (0, sink_d)]),
                    b2PolygonShape(vertices=[(0, sink_d),
                                             (sink_d, sink_d),
                                             (sink_d, sink_h),
                                             (0, sink_h)]),
                    b2PolygonShape(vertices=[(sink_w, sink_d),
                                             (sink_w, sink_h),
                                             (sink_w-sink_d, sink_h),
                                             (sink_w-sink_d, sink_d)]),
                    b2PolygonShape(vertices=[(-left_table_width, sink_h),
                                             (0, sink_h),
                                             (0, sink_h-sink_d),
                                             (-left_table_width, sink_h-sink_d)]),
                    b2PolygonShape(vertices=[(sink_w, sink_h),
                                             (sink_w+right_table_width, sink_h),
                                             (sink_w+right_table_width, sink_h-sink_d),
                                             (sink_w, sink_h-sink_d)])],
            userData='countertop'
        )
        self.faucet = world.CreateStaticBody(
            position=(sink_pos_x-sink_d, -sink_h),
            shapes=[b2PolygonShape(vertices=[(sink_w, sink_h),
                                             (sink_w+faucet_d, sink_h),
                                             (sink_w+faucet_d, sink_h+faucet_h),
                                             (sink_w, sink_h+faucet_h)]),
                    b2PolygonShape(vertices=[(sink_w-faucet_w, sink_h+faucet_h-faucet_d),
                                             (sink_w, sink_h+faucet_h-faucet_d),
                                             (sink_w, sink_h+faucet_h),
                                             (sink_w-faucet_w, sink_h+faucet_h)]),
                    b2PolygonShape(vertices=[(sink_w-faucet_w, sink_h+faucet_h-2*faucet_d),
                                             (sink_w-faucet_w+faucet_d,
                                              sink_h+faucet_h-2*faucet_d),
                                             (sink_w-faucet_w+faucet_d,
                                              sink_h+faucet_h-faucet_d),
                                             (sink_w-faucet_w, sink_h+faucet_h-faucet_d)])],
            userData='faucet')
        self.faucet_location = (sink_pos_x - sink_d + sink_w -
                                faucet_w + faucet_d/2, faucet_h-3*faucet_d)
        self.sensor = world.CreateStaticBody(
            position=(sink_pos_x - sink_d + sink_w - faucet_w +
                      faucet_d/2, (faucet_h-2*faucet_d + sink_h - sink_d)/2 - sink_h + sink_d),
            userData='sensor')
        self.sensor.CreateFixture(
            shape=b2PolygonShape(box=(0.2, (faucet_h-3*faucet_d + sink_h - sink_d)/2)),
            isSensor=True)
        # TODO(caelan): is this sensor the correct size?
        self.cups = []
        self.add_obstacles(obstacles)
    def get_body(self, name):
        for body in self.world.bodies:
            if body.userData == name:
                return body
        return None
    def add_obstacles(self, obstacles=None):
        if obstacles is None:
            return
        for obs in obstacles:
            pos, shape = obs
            base = self.world.CreateStaticBody(
                position=pos,
                shapes=[b2PolygonShape(box=shape)],
                userData='countertop')

    def step(self):
        if (not self.planning) and (not self.disable_liquid) and sensor_touching_test(self.sensor):
            pos = np.random.normal(0, 0.1, size=(2)) + \
                np.array(self.faucet_location)
            self.liquid.make_particles(pos)
        super(Kitchen2D, self).step()
    def gen_liquid_in_cup(self, cup, N, userData='water'):
        assert np.abs(cup.angle) < 0.1, 'cup is not verticle'
        grid_x = (cup.usr_w - cup.usr_d*2)/ self.liquid.radius / 2
        grid_y = (cup.usr_w - cup.usr_d) / self.liquid.radius / 2
        assert grid_x*grid_y > N, 'cup cannot hold {} particles'.format(N)
        for i in range(N):
            x, y = i % grid_x, i / grid_x
            pos = (self.liquid.radius*(2*x+1)+cup.usr_d, self.liquid.radius*(2*y+1)+cup.usr_d)
            self.liquid.make_one_particle(cup.position - cup.shift + pos, userData)

    def make_cup(self, pos, angle, w, h, d):
        self.cups.append(make_cup(self, pos, angle, w, h, d))
        return self.cups[-1]

    def simulate(self, simulate_time):
        total_time = 0
        while total_time < simulate_time:
            self.step()
            total_time += TIME_STEP

LIQUID_NAMES = ('water', 'coffee', 'sugar', 'cream')

def sensor_touching_test(sensor):
    if len(sensor.contacts) == 0:
        return False
    for c in sensor.contacts:
        if c.contact.touching and (c.contact.fixtureA.body.userData not in LIQUID_NAMES) \
           and (c.contact.fixtureB.body.userData not in LIQUID_NAMES):
            return True
    return False

def get_body_vertices(body):
    vertices = []
    for fixture in body.fixtures:
        if isinstance(fixture.shape, b2PolygonShape):
            vertices.extend([body.transform * v for v in fixture.shape.vertices])
        elif isinstance(fixture.shape, b2CircleShape):
            center = body.transform * fixture.shape.pos
            vertices.extend([center + [fixture.shape.radius]*2, center - [fixture.shape.radius]*2])
    return vertices

class Liquid(object):
    def __init__(self, world, frequency=0.2, density=0.01, friction=0.0, radius=0.05, shape_type='circle', liquid_name='water'):
        self.particles = []
        self.density = density
        self.friction = friction
        self.radius = radius
        shapes = {'triangle': b2PolygonShape(vertices=[(0,0), (radius,0), (radius*np.cos(np.pi/3), radius*np.sin(np.pi/3))]), 
                  'square': b2PolygonShape(box=(radius,radius)),
                  'circle': b2CircleShape(radius=radius)}
        self.shape = shapes[shape_type]
        self.frequency = frequency
        self.particle_calls = 0
        self.world = world
        self.liquid_name = liquid_name
    def make_particles(self, pos):
        if self.frequency < 1:
            if self.particle_calls % int(1./self.frequency) == 0:
                self.make_one_particle(pos, self.liquid_name)
                self.particle_calls = 0
            self.particle_calls += 1
        else:
            for i in range(int(self.frequency)):
                self.make_one_particle(pos, self.liquid_name)
    def make_one_particle(self, pos, userData):
        p = self.world.CreateDynamicBody(position=pos, userData=userData)
        p.CreateFixture(
            shape=self.shape,
            friction=self.friction,
            density=self.density
        )
        self.particles.append(p)

def create_gripper(world, lgripper_pos, rgripper_pos, init_angle, pj_motorSpeed, gripper_width=GRIPPER_WIDTH, gripper_height=GRIPPER_HEIGHT):
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
        lowerTranslation=-init_width,#-OPEN_WIDTH,
        upperTranslation=OPEN_WIDTH - init_width, #0.0,
        enableLimit=True,
        maxMotorForce=10000.0,
        motorSpeed=pj_motorSpeed,
        enableMotor=True,
    )
    lgripper.userData = 'gripper'
    rgripper.userData = 'gripper'
    return lgripper, rgripper, pj

def get_gripper_lrpos(init_pos, init_angle, width):
    init_pos = b2Vec2(init_pos)
    lgripper_pos = init_pos - width/2. * \
        b2Vec2((np.cos(init_angle), np.sin(init_angle)))
    rgripper_pos = init_pos + width/2. * \
        b2Vec2((np.cos(init_angle), np.sin(init_angle)))
    return lgripper_pos, rgripper_pos

class Gripper(object):
    def __init__(self, b2world_interface, init_pos=None, init_angle=None,
                 lgripper_pos=None, rgripper_pos=None, pj_motorSpeed=MOTOR_SPEED, init_width=OPEN_WIDTH,
                 gripper_width=GRIPPER_WIDTH, gripper_height=GRIPPER_HEIGHT):
        assert(init_angle is not None)
        world = b2world_interface.world
        self.b2w = b2world_interface
        self.gripper_width = gripper_width
        self.gripper_height = gripper_height
        if lgripper_pos is None or rgripper_pos is None:
            assert(init_pos is not None)
            init_pos = b2Vec2(init_pos)
            lgripper_pos, rgripper_pos = get_gripper_lrpos(init_pos,
                init_angle, init_width+self.gripper_width)
        self.lgripper, self.rgripper, self.pj = create_gripper(world,
            lgripper_pos, rgripper_pos, init_angle, pj_motorSpeed, gripper_width=self.gripper_width, gripper_height=self.gripper_height)
        self.lerror_integral = b2Vec2((0, 0))
        self.laerror_integral = 0.
        self.rerror_integral = b2Vec2((0, 0))
        self.raerror_integral = 0.
        self.grasped = False
        self.grasped_obj = None
        self.reset_mass()
        self.planning = self.b2w.planning
    def reset_mass(self):
        self.mass = self.lgripper.mass + self.rgripper.mass
    def open(self, timeout=1.):
        if self.pj.motorSpeed == MOTOR_SPEED:
            return
        self.pj.motorSpeed = MOTOR_SPEED
        t = 0
        pos = self.position.copy()
        angle = self.angle.copy()
        while t < timeout / TIME_STEP:
            t += self.apply_lowlevel_control(pos, angle)
        self.reset_mass()
        self.grasped_obj = None
        self.grasped = False

    def close(self, timeout=1.):
        if self.pj.motorSpeed == -MOTOR_SPEED:
            return
        self.pj.motorSpeed = -MOTOR_SPEED
        t = 0
        pos = self.position.copy()
        angle = self.angle.copy()
        while t < timeout / TIME_STEP:
            t += self.apply_lowlevel_control(pos, angle)


    def pour(self, to_obj, rel_pos, dangle, stop_ratio=1.0, topour_particles=1, exact_control=False, p_range=SCREEN_WIDTH):
        # find_path needs to be moved outside pour
        dpos = to_obj.position + rel_pos
        found = self.find_path(dpos, 0)
        if not found:
            return False, 0
        if self.planning:
            return True, 1.
        incupparticles, stopped = self.compute_post_grasp_mass()
        n_particles = len(incupparticles)
        stopped = False
        #self.b2w.enable_gui()
        t = 0
        while (not stopped and t * TIME_STEP < 30.):
            t += self.apply_lowlevel_control(dpos, dangle, maxspeed=0.1)
            post_incupparticles, stopped = self.compute_post_grasp_mass(p_range)
            if exact_control and n_particles - len(post_incupparticles) >= topour_particles*stop_ratio:
                self.apply_lowlevel_control(dpos, 0, maxspeed=0.1)
                break
        in_to_cup, _, _ = incup(to_obj, incupparticles)
        self.apply_lowlevel_control(dpos, 0)
        # print n_particles, len(in_to_cup)
        return True, len(in_to_cup) * 1.0 / n_particles

    def cup_scoop(self, from_obj, rel_pos1, dangle):
        assert(self.grasped)
        assert(self.grasped_obj.userData == 'cup')

    def get_cup_feasible(self, from_obj):
        # position of the left lower corner of the cup inside
        cup_left_offset = from_obj.position + \
                        np.array([-from_obj.usr_w/2.+from_obj.usr_d+GRIPPER_WIDTH+EPS, from_obj.usr_d+EPS])
        spoon_w = np.max((self.grasped_obj.usr_w, GRIPPER_WIDTH))
        # feasible range of spoon tip positions relative to cup_left_offset
        cup_feasible = np.array([from_obj.usr_w-from_obj.usr_d*2-spoon_w-GRIPPER_WIDTH-EPS*2., 
            from_obj.usr_h-from_obj.usr_d-EPS])
        return cup_left_offset, cup_feasible
    def get_dump_init_end_pose(self, to_obj, rel_pos_x):
        assert(self.grasped)
        assert(self.grasped_obj.userData == 'spoon')
        assert to_obj.userData == 'cup'
        spoon_w = np.max((self.grasped_obj.usr_w, GRIPPER_WIDTH))

        delta_pos = np.linalg.norm(self.position - self.grasped_obj.position)

        pos_x = delta_pos - to_obj.usr_w / 2. + rel_pos_x * to_obj.usr_w
        dpos1 = to_obj.position + (pos_x, to_obj.usr_h + spoon_w + 1.)
        
        dpos2 = dpos1.copy()
        dpos2[0] -= delta_pos
        dpos2[1] += delta_pos

        return np.hstack((dpos1, -np.pi/2)), np.hstack((dpos2, 0))
    def dump(self, to_obj, rel_pos_x):
        dposa1, dposa2 = self.get_dump_init_end_pose(to_obj, rel_pos_x)
        self.find_path(dposa1[:2], dposa1[2])
        #self.apply_dmp_control(dposa2)
        self.apply_control([dposa2])

    def get_scoop_init_end_pose(self, from_obj, rel_pos1, rel_pos3):
        delta_pos = np.linalg.norm(self.position - self.grasped_obj.position)
        
        cup_left_offset, cup_feasible = self.get_cup_feasible(from_obj)
        if cup_feasible[0] <=0 or cup_feasible[1] <= 0:
            print 'impossible to scoop with the cup size'
            return False, 0

        verticle_offset = np.array([0, delta_pos])
        horizontal_offset = np.array([delta_pos, 0])

        # initial position of scoop
        dpos1 =  cup_left_offset \
            + np.array(rel_pos1) * cup_feasible \
            + verticle_offset
        dpos3 = cup_left_offset + horizontal_offset\
            + np.array(rel_pos3) * cup_feasible

        #print 'dpos3', dpos3
        dpos = dpos3.copy()#self.position.copy()
        dpos[1] = from_obj.position[1] + from_obj.usr_h + self.grasped_obj.usr_w + 1.

        return np.hstack((dpos1, 0.)), np.hstack((dpos, -np.pi/2))

    def scoop(self, from_obj, rel_pos1, rel_pos2, rel_pos3, maxspeed= 1.0):
        assert(self.grasped)
        assert(self.grasped_obj.userData == 'spoon')
        #assert(np.abs(self.angle) < 0.1)
        
        incupparticles, _, _ = incup(from_obj, self.b2w.liquid.particles)


        delta_pos = np.linalg.norm(self.position - self.grasped_obj.position)
        
        cup_left_offset, cup_feasible = self.get_cup_feasible(from_obj)
        if cup_feasible[0] <=0 or cup_feasible[1] <= 0:
            print 'impossible to scoop with the cup size'
            return False, 0

        verticle_offset = np.array([0, delta_pos])
        horizontal_offset = np.array([delta_pos, 0])

        # initial position of scoop
        dpos1 =  cup_left_offset \
            + np.array(rel_pos1) * cup_feasible \
            + verticle_offset
        
        #print 'dpos1', dpos1
        #import pdb; pdb.set_trace()
        found = self.find_path(dpos1, 0)
        if not found:
            return False, 0
        dpos2 = cup_left_offset \
            + np.array(rel_pos2) * cup_feasible \
            + verticle_offset

        #print 'dpos2', dpos2

        dpos3 = cup_left_offset + horizontal_offset\
            + np.array(rel_pos3) * cup_feasible

        traj = np.vstack((np.hstack((dpos2, 0)), np.hstack((dpos3, -np.pi/2))))

        #print 'dpos3', dpos3
        dpos = dpos3.copy()#self.position.copy()
        dpos[1] = from_obj.position[1] + from_obj.usr_h + self.grasped_obj.usr_w + 1.

        #print 'dpos', dpos
        traj = np.vstack((traj, np.hstack((dpos, -np.pi/2))))

        self.apply_dmp_control(traj, maxspeed=maxspeed)

        in_spoon, _, stopped = inspoon(self.grasped_obj, incupparticles)

        cnt = 0
        max_cnt = 100 # 10
        while not stopped:
            print 'scoop particles still moving '
            self.apply_dmp_control(np.hstack((dpos, -np.pi/2)), maxspeed=0.1)
            in_spoon, stopped = self.compute_post_grasp_mass()
            #in_spoon, _, stopped = inspoon(self.grasped_obj, incupparticles)
            cnt += 1
            if cnt > max_cnt:
                break
        incup_spoon, _, _ = incup(from_obj, in_spoon)
        if len(incup_spoon) > 0:
            return False, 0.
        incupparticles2, _, _ = incup(from_obj, incupparticles)
        outside = (len(in_spoon) + len(incupparticles2) - len(incupparticles))
        #import pdb; pdb.set_trace()
        return True,  (len(in_spoon) + outside) / (0.5*self.grasped_obj.usr_w**2*np.pi/(self.b2w.liquid.radius**2*4))

    def get_water_pos(self):
        obj_pos = np.array(self.b2w.faucet_location) - (0, 0.1)
        if self.position[1] + self.gripper_height/2 - self.grasped_obj.position[1] - self.grasped_obj.usr_h > 0:
            gripper_pos = obj_pos - (0, self.gripper_height/2)
        else:
            obj_pos[1] -= self.grasped_obj.usr_h
            gripper_pos = obj_pos + self.position - self.grasped_obj.position
        return gripper_pos

    def get_water(self, runtime):
        assert(self.grasped)
        gripper_pos = self.get_water_pos()
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
        delta_pos = np.linalg.norm(self.position - self.grasped_obj.position)
        
        cup_left_offset, cup_feasible = self.get_cup_feasible(from_obj)
        if (cup_feasible[0] <= 0) or (cup_feasible[1] <= 0):
            print 'impossible to scoop with the cup size'
            return False, 0

        vertical_offset = np.array([0, delta_pos])
        #horizontal_offset = np.array([delta_pos, 0])

        # initial position of scoop
        dpos1 = cup_left_offset \
            + np.array(rel_pos1) * cup_feasible \
            + vertical_offset
        dpos2 = cup_left_offset \
            + np.array(rel_pos2) * cup_feasible \
            + vertical_offset
        return np.hstack((dpos1, 0.)), np.hstack((dpos2, 0))
    def stir(self, obj, rel_pos1, rel_pos2, num_stirs=2):
        dposa1, dposa2 = self.get_stir_init_end_pose(obj, rel_pos1, rel_pos2)
        dposa1[2] = 0.
        dposa2[2] = 0.
        for i in range(num_stirs):
            self.apply_dmp_control(dposa1)
            self.apply_dmp_control(dposa2)

    def bounding_box(self, body):
        vertices = np.array(get_body_vertices(body))
        return vertices.min(axis=0), vertices.max(axis=0)

    def grasp_translation(self, body, pos_ratio):
        """
        Returns the gripper position relative to the body frame
        """
        # TODO(caelan): incorporate orientation
        assert(np.isclose(body.angle, 0, atol=1e-3))
        lower, upper = self.bounding_box(body)
        delta = 0.1  # TODO(caelan): is this just a buffer?
        body_h = (upper[1] - lower[1]) - delta # TODO(caelan): usr_h?
        gripper_pos = np.array([body.position[0],
            lower[1] + (pos_ratio * body_h) + (self.gripper_height / 2) + (delta / 2)])
        return gripper_pos - body.position

    def gripper_from_body(self, body, pos_ratio):
        return self.grasp_translation(body, pos_ratio) + body.position

    def body_from_gripper(self, body, pos_ratio):
        return self.position - self.grasp_translation(body, pos_ratio)

    def get_grasp_poses(self, body, pos_ratio):
        dpos = self.gripper_from_body(body, pos_ratio)
        _, upper = self.bounding_box(body)
        dpos_up = np.array([dpos[0],
            upper[1] + 0.5 + (self.gripper_height / 2)])
        return dpos, dpos_up

    def place(self, obj_dpos, obj_dangle):
        assert(self.grasped)
        g2 = self.simulate_itself()
        g2.set_position(obj_dpos, 0)
        rel_xy = g2.position - g2.grasped_obj.position
        dpos_gripper = rel_xy + obj_dpos
        dpos_gripper[1] += EPS
        found = self.find_path(dpos_gripper, obj_dangle)

        if not found:
            return False
        _, dpos_up = self.get_grasp_poses(self.grasped_obj, 0)
        self.open()

        self.apply_lowlevel_control(dpos_up, 0)
        return True
    def grasp2(self, dpos, dpos_up, body):
        dpos = b2Vec2(dpos)
        dpos_up = b2Vec2(dpos_up)
        self.open()
        found = self.find_path(dpos_up, 0)
        if not found:
            return False
        found = self.find_path(dpos, 0)
        if not found:
            return False
        #self.safe_move(dpos, KP=KP, KD=KD, KI=KI)
        self.close()

        print('mass before grasp check={}'.format(self.mass))
        self.check_grasp(body)
        print('mass after grasp check={}'.format(self.mass))
        # avoid collision checker to think "on countertop" is a collision
        dpos[1] += EPS
        #import pdb; pdb.set_trace()
        self.apply_lowlevel_control(dpos, 0)
        return self.grasped

    def grasp(self, body, pos_ratio):
        dpos, dpos_up = self.get_grasp_poses(body, pos_ratio)
        self.open()
        found = self.find_path(dpos_up, 0)
        if not found:
            return False
        found = self.find_path(dpos, 0)
        if not found:
            return False
        #self.safe_move(dpos, KP=KP, KD=KD, KI=KI)
        self.close()

        print('mass before grasp check={}'.format(self.mass))
        self.check_grasp(body)
        print('mass after grasp check={}'.format(self.mass))
        # avoid collision checker to think "on countertop" is a collision
        dpos[1] += EPS
        #import pdb; pdb.set_trace()
        self.apply_lowlevel_control(dpos, 0)
        return self.grasped

    def grasp_spoon(self, body, pos_ratio):
        assert body.userData == 'spoon'
        dangle = body.angle
        assert dangle > -np.pi/2 and dangle < np.pi/2
        dpos, dpos_up = self.get_grasp_poses(body, pos_ratio)
        self.open()
        found = self.find_path(dpos_up, 0)
        if not found:
            return False
        found = self.find_path(dpos, 0)
        if not found:
            return False
        #self.safe_move(dpos, KP=KP, KD=KD, KI=KI)
        self.close()

        print('mass before grasp check={}'.format(self.mass))
        self.check_grasp(body)
        print('mass after grasp check={}'.format(self.mass))
        # avoid collision checker to think "on countertop" is a collision
        dpos[1] += EPS
        #import pdb; pdb.set_trace()
        self.apply_lowlevel_control(dpos, 0)
        return self.grasped
    def is_graspable(self, body):
        return body.usr_w <= OPEN_WIDTH

    def set_closed(self, body=None):
        width = self.gripper_width if body is None else (body.usr_w + self.gripper_width)
        #width = body.usr_w/2 + self.gripper_width # TODO(caelan): make this tight
        self.lgripper.position, self.rgripper.position = get_gripper_lrpos(self.position, self.angle, width)

    def set_open(self):
        width = OPEN_WIDTH + self.gripper_width
        self.lgripper.position, self.rgripper.position = get_gripper_lrpos(self.position, self.angle, width)

    def get_push_pos(self, body, goal_x):
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
        # TODO(caelan): incorporate this below

    def get_stack_pos(self, body):
        lower, upper = self.bounding_box(body)
        return (body.position[0], upper[1])

    def push(self, body, rel_pos, goal_x, maxspeed):
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
        while helper.posa_metric(self.position, self.angle, dpos, dangle) > 0.01:
            self.apply_lowlevel_control(dpos, dangle, runtime=100., maxspeed=maxspeed)
            adjust_cnt += 1
            if adjust_cnt > 10:
                break

    def attach(self, body):
        assert self.is_graspable(body)
        self.pj.motorSpeed = -MOTOR_SPEED # Need this for force closure
        self.grasped = True
        self.grasped_obj = body
        self.compute_post_grasp_mass()
        #self.set_closed(body)
        # TODO: this probably should compute object position from robot not vice versa

    def release(self):
        self.pj.motorSpeed = MOTOR_SPEED # Need this for force closure
        self.grasped = False
        self.grasped_obj = None
        self.reset_mass()
        #self.set_open()

    def set_grasped(self, body, pos_ratio, pos, angle):
        # only body is involved, no liquid
        self.pj.motorSpeed = -MOTOR_SPEED
        self.grasped = True
        self.grasped_obj = body
        self.compute_post_grasp_mass()
        #import pdb; pdb.set_trace()
        #assert(np.isclose(self.grasped_obj.angle,0, atol=0.01))
        gripper_pos, _ = self.get_grasp_poses(body, pos_ratio)
        if not self.is_graspable(body):
            return False
        width = body.usr_w + self.gripper_width
        lgripper_pos, rgripper_pos = get_gripper_lrpos(gripper_pos, 0, width)
        self.lgripper.position = lgripper_pos
        self.rgripper.position = rgripper_pos
        self.set_position(pos, angle)
        return True

    def set_spoon_grasped(self, body, pos_ratio, pos, angle):
        assert body.userData == 'spoon'
        self.pj.motorSpeed = -MOTOR_SPEED
        self.grasped = True
        self.grasped_obj = body
        self.compute_post_grasp_mass()

        grasp_w = body.usr_w*2 + GRIPPER_HEIGHT/2 + EPS + (body.usr_h*0.9-2*EPS) * pos_ratio
        gripper_pos = body.position + grasp_w*np.array([-np.sin(body.angle), np.cos(body.angle)])
        assert body.usr_d < OPEN_WIDTH
        width = body.usr_d + GRIPPER_WIDTH
        lgripper_pos, rgripper_pos = get_gripper_lrpos(gripper_pos, body.angle, width)
        self.lgripper.position = lgripper_pos
        self.rgripper.position = rgripper_pos
        self.set_position(pos, angle)
        return True

    def check_grasp(self, obj, update_mass=True):
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
            self.grasped = flag
            self.grasped_obj = obj
            self.compute_post_grasp_mass()

        return flag

    def compute_post_grasp_mass(self, p_range=SCREEN_WIDTH):
        obj = self.grasped_obj
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
        timesteps = traj.shape[1]
        traj = np.hstack((traj, traj[:,-1:]))
        dtraj = np.diff(traj) / TIME_STEP
        #import pdb; pdb.set_trace()
        dtraj = np.hstack((self.velocity[:,None], dtraj))
        ddtraj = np.diff(dtraj) / TIME_STEP


        #f_target = ddtraj - a * (b * (traj[:, -1:] - traj[:, :-2]) - dtraj[:, :-1])
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
            # spatial scaling term
            k = (traj[d, -1] - traj[d, 0])
            if k == 0:
                k = 0.0000001
            for b in range(nbfs):
                numer = np.sum(x_track * psi_track[:, b] * f_target[d,:])
                #import pdb;pdb.set_trace()
                denom = np.sum(x_track**2 * psi_track[:, b])

                w[d, b] = numer / (k * denom)
                if np.isnan(w[d, b]):
                    import pdb; pdb.set_trace()
        return np.nan_to_num(w)

    def apply_dmp_control(self, traj, runtime=100., maxspeed=0.5, a=1., b=.5):
        # traj is t x d
        # traj can be dposa
        if traj.ndim == 1:
            traj = traj[None,:]
        cur_pos = np.hstack((self.position, self.angle))

        if np.linalg.norm(cur_pos-traj[0]) != 0:
            traj = np.vstack((cur_pos, traj))
        des_pos = traj[-1]
        timesteps = int(runtime / TIME_STEP)
        #If maxspeed = 0, the speed is decided by runtime instead
        if maxspeed > 0:
            segment_lengths = np.array([0.] + [np.linalg.norm(traj[i] - traj[i-1]) for i in range(1,len(traj))])
            tot_length = np.sum(segment_lengths)
            assert(tot_length > 0)
            max_timesteps = int(tot_length / maxspeed / TIME_STEP + 1) + 1
            timesteps = max_timesteps#np.min([timesteps, max_timesteps])
        runtime = timesteps * TIME_STEP
        #path_gen = scipy.interpolate.interp1d(np.arange(len(traj))*runtime/(len(traj)-1), traj.T)
        path_gen = scipy.interpolate.interp1d(np.cumsum(segment_lengths)*runtime/tot_length, traj.T, fill_value='extrapolate')
        #import pdb; pdb.set_trace()
        try:
            traj = path_gen(np.arange(timesteps) * runtime / (timesteps-1))
        except:
            import pdb; pdb.set_trace()
        w = self.get_traj_weight(traj, a, b)
        print('timesteps={}'.format(timesteps))
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
            if self.grasped:
                aforce += (-GRAVITY[1]) * self.grasped_obj.mass * (self.grasped_obj.worldCenter[0] - self.position[0])/ self.radius
            g_aforce = aforce * np.array([np.sin(self.angle), -np.cos(self.angle)]) / 2

            lforce += g_aforce
            rforce -= g_aforce
            self.lgripper.ApplyForce(lforce, self.lgripper.position, wake=True)
            self.rgripper.ApplyForce(rforce, self.rgripper.position, wake=True)
            self.b2w.step()
        print self.position
        print('position diff={}'.format(self.position_angle - des_pos))
        #import pdb; pdb.set_trace()
        return True


    def get_traj(self, cpos, cangle, cvel, dpos, dangle, runtime, maxspeed):
        dpos = b2Vec2(dpos)
        dposa = np.hstack((dpos, dangle))
        cur_pos = np.hstack((cpos, cangle))

        t = int(runtime / TIME_STEP)
        #import pdb;pdb.set_trace()
        t0 = 10 # timesteps for acceleration and deceleration
        if maxspeed > 0:
            max_timesteps = int(np.max(np.abs(dposa - cur_pos)/maxspeed) / TIME_STEP + 1) + t0
            t = np.min([t, max_timesteps])
        print('expected timesteps to goal = {}'.format(t))

        cur_v = cvel
        v = (dposa - cur_pos - cur_v*t0*TIME_STEP/2) / (t-t0) / TIME_STEP
        ddy = np.array([(v-cur_v)/t0/TIME_STEP]*t0 + [[0]*3]*(t-2*t0) + [-v/t0/TIME_STEP]*t0)
        dy = np.vstack((cur_v, cur_v+np.cumsum(ddy, axis=0)*TIME_STEP))
        y = np.vstack((cur_pos, cur_pos+np.cumsum(dy, axis=0)*TIME_STEP))
        return y.T, dy.T, ddy.T

    def apply_lowlevel_control(self, dpos, dangle, runtime=100., maxspeed=1., collision_check=False):
        # at least 2 timesteps
        dpos = b2Vec2(dpos)

        dposa = np.hstack((dpos, dangle))
        cur_pos = np.hstack((self.position, self.angle))

        t = int(runtime / TIME_STEP)
        #import pdb;pdb.set_trace()
        t0 = 1 # timesteps for acceleration and deceleration
        if maxspeed > 0:
            max_timesteps = int(np.max(np.abs(dposa - cur_pos)/maxspeed) / TIME_STEP + 1) + t0
            t = np.min([t, max_timesteps])
        #print('expected timesteps to goal = {}'.format(t))

        cur_v = self.velocity
        v = (dposa - cur_pos - cur_v*t0*TIME_STEP/2) / (t-t0) / TIME_STEP
        ddy = np.array([(v-cur_v)/t0/TIME_STEP]*t0 + [[0]*3]*(t-2*t0) + [-v/t0/TIME_STEP]*t0)
        traj = [(self.position, self.angle)]
        for step in range(len(ddy)):
            lforce = (ddy[step,:2] - GRAVITY) * self.mass/2
            rforce = (ddy[step,:2] - GRAVITY) * self.mass/2
            aforce = (ddy[step, 2] * self.inertia) / self.radius
            if self.grasped:
                aforce += (-GRAVITY[1]) * self.grasped_obj.mass * (self.grasped_obj.worldCenter[0] - self.position[0])/ self.radius
            g_aforce = aforce * np.array([np.sin(self.angle), -np.cos(self.angle)]) / 2
            lforce += g_aforce
            rforce -= g_aforce
            self.lgripper.ApplyForce(lforce, self.lgripper.position, wake=True)
            self.rgripper.ApplyForce(rforce, self.rgripper.position, wake=True)
            self.b2w.step()
            traj.append((self.position, self.angle))
            if collision_check and self.check_collision():
                return traj, True

        #print('position diff={}, angle diff={}, trajlen={}'.format(self.position - dpos, self.angle - dangle, len(ddy)))
        if collision_check:
            return traj, False
        else:
            return len(ddy)
    def check_point_collision(self, pos, angle=0.):
        self.set_position(pos, angle)
        force = -self.b2w.gravity * self.mass/2
        self.lgripper.ApplyForce(force, self.lgripper.position, wake=True)
        self.rgripper.ApplyForce(force, self.rgripper.position, wake=True)
        self.b2w.step()
        return self.check_collision()
    def check_path_collision(self, pos, angle, dpos, dangle):
        # start position has to be not in collision
        #assert(not self.check_point_collision(pos, angle))
        self.set_position(pos, angle)
        traj, collision = self.apply_lowlevel_control(dpos, dangle, 100., maxspeed=1.0, collision_check=True)
        if not collision:
            return traj, collision
        res_id = np.max([-10, -len(traj)])
        while(self.check_point_collision(*traj[res_id])):
            if res_id == -len(traj):
                return [], True
            res_id -= 1
            res_id = np.max([res_id, -len(traj)])
        return traj[:res_id+1], collision#traj[res_id::-10], collision
    def check_collision(self):
        contacts = self.lgripper.contacts + self.rgripper.contacts
        if self.grasped:
            contacts += self.grasped_obj.contacts
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
        angle = np.arctan2(dpos[1], dpos[0])#self.lgripper.angle
        #if not np.isclose(angle, self.lgripper.angle, atol=0.1):
        #    import pdb; pdb.set_trace()
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
        if self.grasped:
            radius_cup = np.linalg.norm(self.grasped_obj.worldCenter - self.position)
            inertia += self.grasped_obj.mass * (radius_cup**2)
        return inertia #self.mass * (self.radius**2)
    @property
    def velocity(self):
        linear_v = (self.lgripper.linearVelocity + self.rgripper.linearVelocity)/2
        trans = np.array([-np.sin(self.angle), np.cos(self.angle)])
        r_av = np.sum(trans * self.rgripper.linearVelocity)
        l_av = -np.sum(trans * self.lgripper.linearVelocity)
        angular_v = (r_av + l_av)/2/self.radius
        return np.hstack((linear_v, angular_v))

    def set_position(self, pos, angle):
        self.b2w.world.ClearForces()
        if self.grasped:
            rel_a = angle - self.angle
            p = self.grasped_obj.position - self.position
            p = np.array([[np.cos(rel_a), -np.sin(rel_a)],
                [np.sin(rel_a), np.cos(rel_a)]]).dot(p)
            self.grasped_obj.position = p + pos
            self.grasped_obj.angle += rel_a
            self.grasped_obj.linearVelocity = b2Vec2((0,0))
            self.grasped_obj.angularVelocity = 0
        lpos, rpos = get_gripper_lrpos(pos, angle, self.radius*2)
        self.lgripper.position = lpos
        self.rgripper.position = rpos
        self.lgripper.angle = angle
        self.rgripper.angle = angle
        self.set_angular_velocity()
        self.set_linear_velocity()

    def set_pose(self, pose):
        self.set_position(pose[:2], pose[2])

    def set_angular_velocity(self, ravel=0):
        self.lgripper.angularVelocity = ravel
        self.rgripper.angularVelocity = ravel

    def set_linear_velocity(self, rlvel=(0, 0)):
        self.lgripper.linearVelocity = b2Vec2(rlvel)
        self.rgripper.linearVelocity = b2Vec2(rlvel)

    @property
    def info(self):
        return (self.lgripper.position, self.lgripper.angle, self.rgripper.position,
                self.rgripper.angle, self.pj.motorSpeed)

    def simulate_itself(self, buffer=None):
        if buffer is None:
            b2w2 = copy_kitchen(self.b2w, self.grasped_obj)
        else:
            b2w2 = buffer_kitchen(self.b2w, self.grasped_obj, buffer)
        g2 = Gripper(b2w2,
                     lgripper_pos=self.lgripper.position.copy(),
                     rgripper_pos=self.rgripper.position.copy(),
                     init_angle=self.angle,
                     pj_motorSpeed=self.pj.motorSpeed)
        if self.grasped:
            #b2w2.enable_gui()
            g2.grasped = True
            if self.grasped_obj.userData == 'cup':
                g2.grasped_obj = make_cup(b2w2, self.grasped_obj.position, self.grasped_obj.angle,
                    self.grasped_obj.usr_w, self.grasped_obj.usr_h, self.grasped_obj.usr_d)
            elif self.grasped_obj.userData == 'spoon':
                g2.grasped_obj = make_good_spoon(b2w2, self.grasped_obj.position, self.grasped_obj.angle,
                    self.grasped_obj.usr_w, self.grasped_obj.usr_h, self.grasped_obj.usr_d)
            elif self.grasped_obj.userData == 'block':
                g2.grasped_obj = make_block(b2w2, self.grasped_obj.position, self.grasped_obj.angle,
                    self.grasped_obj.usr_w, self.grasped_obj.usr_h)
            elif self.grasped_obj.userData == 'stirrer':
                g2.grasped_obj = make_stirrer(b2w2, self.grasped_obj.position, self.grasped_obj.angle,
                    self.grasped_obj.usr_w, self.grasped_obj.usr_h, self.grasped_obj.usr_d)
            else:
                raise ValueError('grasped object not identified.')
            
            g2.mass += g2.grasped_obj.mass
        return g2
    """
    def find_path(self, dpos, dangle):
        dpos = b2Vec2(dpos)
        # gripper must be straight when this is called
        # assert(np.isclose(self.angle, 0, atol=0.1))

        g2 = self.simulate_itself()
        # Try interpolation first
        if g2.check_point_collision(self.position, self.angle):
            return False
        if g2.check_point_collision(dpos, dangle):
            return False
        #assert(not g2.check_point_collision(self.position, self.angle))
        if self.planning:
            self.set_position(dpos, dangle)
            return True
        traj, collision = g2.check_path_collision(self.position, self.angle, dpos, dangle)
        if collision and len(traj) == 0:
            return False
        if collision:
            # RRT
            traj = None
            for i in range(2):
                print 'rrt attempt ', i
                traj = rrt(self.position, self.angle, dpos, dangle, g2.check_point_collision, g2.check_path_collision)
                if traj is not None:
                    break
            if traj is not None:
                #traj = np.hstack((traj, np.zeros((len(traj),1))))
                if not self.planning:
                    self.apply_dmp_control(traj)
            else:
                return False # this means no path found
        if self.planning:
            self.set_position(dpos, dangle)
            return True
        adjust_cnt = 0
        while helper.posa_metric(self.position, self.angle, dpos, dangle) > 0.01:
            self.apply_lowlevel_control(dpos, dangle)
            adjust_cnt += 1
            if adjust_cnt > 5:
                break
            #    return False # this means it failed online test
        return True
    """

    def apply_control(self, path, maxspeed=0.5):
        for q in path:
            self.apply_lowlevel_control(q[:2], q[2], maxspeed=maxspeed)
        """
        self.apply_dmp_control(np.array(path), maxspeed=maxspeed)
        adjust_cnt = 0
        dpos, dangle = path[-1][:2], path[-1][2]
        while (helper.posa_metric(self.position, self.angle, dpos, dangle) > 0.01) and (adjust_cnt <= 5):
            self.apply_lowlevel_control(dpos, dangle)
            adjust_cnt += 1
        """

    def plan_linear(self, dpos, dangle):
        dpos = b2Vec2(dpos)
        g2 = self.simulate_itself()
        return rrt(self.position, self.angle, dpos, dangle, g2.check_point_collision, g2.check_path_collision, linear=True)
        """
        if g2.check_point_collision(self.position, self.angle) or g2.check_point_collision(dpos, dangle):
             return None
         traj, collision = g2.check_path_collision(self.position, self.angle, dpos, dangle)
        if collision:
             return None
        return np.array([(vec.x, vec.y, theta) for vec, theta in traj])
        """

    def plan_path(self, dpos, dangle, buffer=None, motion_angle=0):
        dpos = b2Vec2(dpos)
        # gripper must be straight when this is called
        # assert(np.isclose(self.angle, 0, atol=0.1))
        g2 = self.simulate_itself(buffer=buffer)

        # if buffer is not None:
        #     g2.b2w.enable_gui()
        #     g2.b2w.draw()
        #     print buffer
        #     raw_input('Plan?')

        """
        # Try interpolation first
        if g2.check_point_collision(self.position, self.angle) or g2.check_point_collision(dpos, dangle):
            return None
        # assert(not g2.check_point_collision(self.position, self.angle))
        traj, collision = g2.check_path_collision(self.position, self.angle, dpos, dangle)
        if not collision:
            return np.array([(vec.x, vec.y, theta) for vec, theta in traj])
        if len(traj) == 0:
            return None
        """
        traj = rrt(self.position, self.angle, dpos, dangle,
                   g2.check_point_collision, g2.check_path_collision, motion_angle=motion_angle)

        # if (traj is not None) and (buffer is not None):
        #     for conf in traj:
        #         g2.set_pose(conf)
        #         g2.b2w.draw()
        #         raw_input('Step')

        if (traj is not None) and (not self.planning):
            #self.apply_dmp_control(traj)
            self.apply_control(traj, maxspeed=1)
            #for t in traj:
            #    self.apply_lowlevel_control(t[:2], t[2])
        return traj

    def find_path(self, dpos, dangle, buffer=None, motion_angle=0):
        if helper.posa_metric(self.position, self.angle, dpos, dangle) < ACC_THRES:
            return True
            
        traj = self.plan_path(dpos, dangle, buffer=buffer, motion_angle=motion_angle)
        if traj is None:
            return False  # this means no path found
        if self.planning:
            g2 = self.simulate_itself(buffer=None)
            col = g2.check_point_collision(self.position, self.angle)
            if col:
                return False
            col = g2.check_point_collision(dpos, dangle)
            if col:
                return False
            self.set_position(dpos, dangle)
            return True
        #else:
        #    self.apply_dmp_control(traj)
        adjust_cnt = 0
        while helper.posa_metric(self.position, self.angle, dpos, dangle) > ACC_THRES:
            self.apply_lowlevel_control(dpos, dangle)
            adjust_cnt += 1
            if adjust_cnt > 5:
                break
            #    return False # this means it failed online test
        return True

"""
class Node(object):
    def __init__(self, pos, angle, parent=None):
        self.pos = pos
        self.angle = angle
        self.parent = parent
    def retrace(self):
        traj = []
        node = self
        while node is not None:
            traj.append(np.hstack((node.pos, node.angle)))
            node = node.parent
        return traj[::-1]

def rrt(pos, angle, dpos, dangle, check_point_collision, check_path_collision, goal_prob=0.05, iters=1000):
    nodes = [Node(pos, angle)]
    for i in xrange(iters):
        if np.random.rand() < goal_prob:
            s_pos = dpos
            s_a = dangle
            #import pdb;pdb.set_trace()
        else:
            s_pos = np.random.uniform([-SCREEN_WIDTH/2.,0],[SCREEN_WIDTH/2,SCREEN_HEIGHT-TABLE_HEIGHT])
            if np.random.rand() < 0.5:
                s_a = angle
            else:
                s_a = dangle
            #if np.random.rand() < 0.1:
            #    s_pos = pos - np.array((5., 0.))
            #elif np.random.rand() < 0.2:
            #    s_pos = pos + np.array((0. ,5.))
            #    s_pos[1] = np.min([SCREEN_HEIGHT-TABLE_HEIGHT-10.])
        closest_id = np.argmin([helper.posa_metric(s_pos, s_a, 
                                                   node.pos, node.angle) for node in nodes])
        last = nodes[closest_id]
        if last.angle == dangle:
            s_a = dangle
        traj, collision = check_path_collision(last.pos, last.angle, s_pos, s_a)
        print i, len(traj), len(nodes) # TODO(caelan): the size of the tree grows quickly...
        for p in traj:
            last = Node(p[0], p[1], last)
            nodes.append(last)
            if helper.posa_metric(p[0], p[1], dpos, dangle) < 0.05:
                return np.array(last.retrace())
    return None
"""

def linear_rotate(pos, angle, dangle, check_point_collision, step_size=np.pi/16):
    path = [np.hstack([pos, angle])]
    delta = dangle - angle
    distance = np.abs(delta)
    # TODO(caelan): wrap angles and choose which direction to move in based on which is closer
    for t in np.arange(step_size, distance, step_size):
        theta = angle + (t * delta / distance)
        if check_point_collision(pos, theta):
            return None
        path.append(np.hstack([pos, theta]))
    path.append(np.hstack([pos, dangle]))
    return np.array(path)
    # TODO: sparsify the path for execution

def rrt(pos, angle, dpos, dangle, check_point_collision, _, motion_angle=0,
        linear=False, step_size=0.1, iters=100, restarts=2, **kwargs):
    #assert np.abs(angle - dangle) < 1e-3
    rotate_path1 = linear_rotate(pos, angle, motion_angle, check_point_collision)
    if rotate_path1 is None: return None
    rotate_path2 = linear_rotate(dpos, motion_angle, dangle, check_point_collision)
    if rotate_path2 is None: return None

    def sample_fn():
        #lower = [-SCREEN_WIDTH / 2., 0, motion_angle]
        #upper = [SCREEN_WIDTH / 2, SCREEN_HEIGHT - TABLE_HEIGHT, motion_angle]
        lower = [-SCREEN_WIDTH / 2., 0]
        upper = [SCREEN_WIDTH / 2, SCREEN_HEIGHT - TABLE_HEIGHT]
        return b2Vec2(np.random.uniform(lower, upper))

    def distance_fn(q1, q2):
        return np.linalg.norm(np.array(q2) - np.array(q1))
        #return helper.posa_metric(q1, motion_angle, q2, motion_angle)

    def extend_fn(q1, q2):
        # TODO: can also space these farther apart
        delta = np.array(q2) - np.array(q1)
        distance = np.linalg.norm(delta)
        for t in np.arange(step_size, distance, step_size):
            yield q1 + (t*delta/distance)
        yield q2

    def collision_fn(q):
        #return False
        return check_point_collision(q, motion_angle)

    try:
        from motion_planners.rrt_connect import birrt
        from motion_planners.rrt_connect import direct_path
    except ImportError:
        raise RuntimeError('Requires https://github.com/caelan/motion-planners')

    if linear:
        path = direct_path(pos, dpos, extend_fn, collision_fn)
    else:
        path = birrt(pos, dpos, distance_fn, sample_fn, extend_fn, collision_fn,
                     restarts=restarts, iterations=iters, smooth=50)
    if path is None:
        return None
    translate_path = np.array([np.hstack([pos, motion_angle]) for pos in path])
    return np.vstack([rotate_path1[1:-1], translate_path, rotate_path2[1:-1]])

COPY_IGNORE = ('gripper', 'water', 'sensor', 'coffee', 'cream', 'sugar')

def copy_kitchen(kitchen, grasped_obj):
    b2w = b2WorldInterface(do_gui=False, caption='Copy')
    for b in kitchen.world.bodies:
        if (b.userData in COPY_IGNORE) or (b == grasped_obj):
            continue
        #countertop = b.userData is 'countertop'
        #if countertop:
        #    userData = 'countertop'
        #else:
        #    userData = 'obstacle'
        obstacle = b2w.world.CreateStaticBody(
            position=b.position, angle=b.angle, userData='obstacle')
        for fixture in b.fixtures:
            obstacle.CreateFixture( shape=fixture.shape, isSensor=True)#not countertop)
    return b2w

def buffer_shape(shape, radius, sides=4):
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

def buffer_kitchen(kitchen, grasped_obj=None, radius=0.1):
    b2w = b2WorldInterface(do_gui=False, caption='Buffer')
    b2w.planning = True
    b2w.liquid = Liquid(b2w.world) #kitchen.liquid
    for b in kitchen.world.bodies:
        if (b.userData in COPY_IGNORE) or (b == grasped_obj):
            continue
        obstacle = b2w.world.CreateStaticBody(
            position=b.position, angle=b.angle, userData='obstacle')
        for fixture in b.fixtures:
            obstacle.CreateFixture(shape=buffer_shape(fixture.shape, radius), isSensor=True)
    return b2w

def make_bad_spoon(b2world_interface, pos, angle, w, h, l, d, shifth=1.0):
    # w: width of the spoon bottom
    # h: height of the spoon side
    # l: spoon's handle length
    # d: thickness of the spoon
    shift = np.array([w*3, shifth*h])
    #print shift
    world = b2world_interface.world
    body = world.CreateDynamicBody(
        position=pos,
        angle=angle
    )
    # spoon bottom
    polygon_shape = [(0, 0), (w, 0), (w, d), (0, d)]
    polygon_shape = [(v[0] - shift[0], v[1]-shift[1]) for v in polygon_shape]
    body.CreateFixture(
        shape=b2PolygonShape(vertices=polygon_shape),
        friction=1,
        density=1
    )
    # spoon left side
    polygon_shape = [(0, d), (d, d), (-0.5*h+d, h), (-0.5*h, h)]
    polygon_shape = [(v[0] - shift[0], v[1]-shift[1]) for v in polygon_shape]
    body.CreateFixture(
        shape=b2PolygonShape(vertices=polygon_shape),
        friction=1,
        density=1
    )
    # spoon right side
    polygon_shape = [(w, d), (w+0.5*h, h), (w+0.5*h-d, h), (w-d, d)]
    polygon_shape = [(v[0] - shift[0], v[1]-shift[1]) for v in polygon_shape]
    body.CreateFixture(
        shape=b2PolygonShape(vertices=polygon_shape),
        friction=1,
        density=1
    )
    polygon_shape = [(w+0.5*h-d, h), (w+0.5*h+l-d, h),
                     (w+0.5*h+l-d, h+d), (w+0.5*h-d, h+d)]
    polygon_shape = [(v[0] - shift[0], v[1]-shift[1]) for v in polygon_shape]
    body.CreateFixture(
        shape=b2PolygonShape(vertices=polygon_shape),
        friction=1,
        density=1
    )
    body.usr_w = d
    body.usr_h = l+h+w
    body.usr_d = w
    body.shift = shift
    body.userData = "spoon"
    return body

def make_good_spoon(b2world_interface, pos, angle, w, h, d, shifth=0.0):
    shift = np.array([d/2., shifth*(h+2*w)])
    #print shift
    world = b2world_interface.world
    body = world.CreateDynamicBody(
        position=pos,
        angle=angle
    )
    polygon_shape = [(0, w*2.-d), (d, w*2.-d),
                     (d, h+w*2.), (0, h+w*2.)]
    polygon_shape = [(v[0] - shift[0], v[1]-shift[1]) for v in polygon_shape]
    body.CreateFixture(
        shape=b2PolygonShape(vertices=polygon_shape),
        friction=1,
        density=0.5
    )

    n_points = 10
    outer_shape = []
    inner_shape = []
    center = b2Vec2((0, w))
    for a in np.linspace(0, np.pi, n_points):
        outer_shape.append(center + w*np.array([np.sin(a), np.cos(a)]))
    for a in np.linspace(0, np.pi, n_points):
        inner_shape.append(center + (w-d)*np.array([np.sin(a), np.cos(a)]))
    #import matplotlib.pyplot as plt
    #polygon_shape = np.array(polygon_shape)
    #plt.plot(polygon_shape[:,0], polygon_shape[:,1], 'x')
    #plt.show()
    for i in range(n_points-1):
        polygon_shape = [outer_shape[i], outer_shape[i+1], inner_shape[i+1], inner_shape[i]]
        body.CreateFixture(
            shape=b2PolygonShape(vertices=polygon_shape),
            friction=1,
            density=0.5
        )
    body.usr_w = w*1.
    body.usr_h = h*1.
    body.usr_d = d*1.
    body.userData = "spoon"
    return body

def make_block(b2world_interface, pos, angle, w, h, shifth=0.0):
    shift = np.array([w/2.0, shifth*h])
    #print shift
    world = b2world_interface.world
    body = world.CreateDynamicBody(
        position=pos,
        angle=angle
    )
    polygon_shape = [(0, 0), (w, 0), (w, h), (0, h)]
    polygon_shape = [(v[0] - shift[0], v[1]-shift[1]) for v in polygon_shape]
    body.CreateFixture(
        shape=b2PolygonShape(vertices=polygon_shape),
        friction=1,
        density=0.05
    )
    body.usr_w = w
    body.usr_h = h
    body.usr_d = None
    body.shift = shift
    body.userData = "block"
    return body


def make_cup(b2world_interface, pos, angle, w, h, d, shifth=0.0):
    shift = np.array([w/2.0, shifth*h])
    #print shift
    world = b2world_interface.world
    body = world.CreateDynamicBody(
        position=pos,
        angle=angle
    )
    polygon_shape = [(d/2, 0), (w-d/2, 0), (w-d/2, d), (d/2, d)]
    polygon_shape = [(v[0] - shift[0], v[1]-shift[1]) for v in polygon_shape]
    body.CreateFixture(
        shape=b2PolygonShape(vertices=polygon_shape),
        friction=1,
        density=0.5
    )
    polygon_shape = [(0, 0), (d, 0), (d, h), (0, h)]
    polygon_shape = [(v[0] - shift[0], v[1]-shift[1]) for v in polygon_shape]
    body.CreateFixture(
        shape=b2PolygonShape(vertices=polygon_shape),
        friction=1,
        density=0.5
    )
    polygon_shape = [(w, 0), (w, h), (w-d, h), (w-d, 0)]
    polygon_shape = [(v[0] - shift[0], v[1]-shift[1]) for v in polygon_shape]
    body.CreateFixture(
        shape=b2PolygonShape(vertices=polygon_shape),
        friction=1,
        density=0.5
    )
    body.usr_w = w*1.0
    body.usr_h = h*1.0
    body.usr_d = d*1.0
    body.shift = shift
    body.userData = "cup"
    return body


def make_stirrer(b2world_interface, pos, angle, w, h, d, shifth=0.0):
    world = b2world_interface.world
    body = world.CreateDynamicBody(
        position=pos,
        angle=angle
    )
    body.CreateFixture(
        shape=b2CircleShape(pos=(0, -h/2), radius=d),
        friction=0.1,
        density=1
    )
    body.CreateFixture(
        shape=b2PolygonShape(box=(w/2, h/2)),
        friction=0.1,
        density=1
    )
    body.usr_w = w*1.0
    body.usr_h = h*1.0
    body.usr_d = d*1.0
    body.userData = "stirrer"
    return body


def make_static_cup(b2world_interface, pos, angle, w, h, d):
    world = b2world_interface.world
    body = world.CreateStaticBody(
        position=pos,
        angle=angle,
        shapes=[b2PolygonShape(vertices=[(0, 0), (w, 0), (w, d), (0, d)]),
                b2PolygonShape(vertices=[(0, d), (d, d), (d, h), (0, h)]),
                b2PolygonShape(vertices=[(w, d), (w, h), (w-d, h), (w-d, d)])]
    )
    body.userData = 'static_cup'
    return body


def inspoon(spoon, particles):
    incupparticles = []
    outcupparticles = []
    stopped = 1
    angle = spoon.angle
    center = spoon.position + spoon.usr_w*np.array([-np.sin(angle), np.cos(angle)])
    for p in particles:
        ppos = p.position 
        if helper.posa_metric(center, 0, ppos, 0) < spoon.usr_w-spoon.usr_d:
            incupparticles += [p]
            if (np.linalg.norm(p.linearVelocity) > 0.1):
                stopped = 0
        else:
            outcupparticles += [p]
    return incupparticles, outcupparticles, stopped

def incup(cup, particles, p_range=SCREEN_WIDTH/2, debug=0):
    incupparticles = []
    outcupparticles = []
    stopped = 1
    for p in particles:
        ppos = p.position - cup.position
        tp = cup.angle
        trans = np.array([[np.cos(tp), np.sin(tp)], [-np.sin(tp), np.cos(tp)]])
        ppos = np.dot(trans, ppos)+cup.shift

        #if ppos[0] <= cup.usr_w+cup.usr_d/2.0 and ppos[0] >= -cup.usr_d/2.0\
        #        and ppos[1] >= -cup.usr_d/2.0 and ppos[1] <= cup.usr_h+cup.usr_d/2.0:
        if ppos[0] <= cup.usr_w and ppos[0] >= 0.\
                and ppos[1] >= 0. and ppos[1] <= cup.usr_h:
            incupparticles += [p]
        else:
            outcupparticles += [p]
        # else:
            # if debug:
            #  import pdb; pdb.set_trace()
        if (np.abs(p.linearVelocity[1]) > 0.01) and np.abs(p.position[0]) < p_range \
                and p.position[1]>=0.1:
            stopped = 0
    #if stopped:
    #    import pdb; pdb.set_trace()
    return incupparticles, outcupparticles, stopped

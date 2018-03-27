# Author: Zi Wang
from Box2D import *
from Box2D.b2 import *
from kitchen_constants import *

import pygame
import numpy as np


class guiWorld:
    def __init__(self, caption='PyBox2D Simulator', overclock=None):
        '''
        Graphics wrapper for visualization of Pybox2D with pygame. 
        Adapted from kitchen2d/push_world.py
        Args:
            caption: caption on the window.
            overclock: number of frames to skip when showing graphics.
        '''
        self.screen = pygame.display.set_mode(
            (SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX), 0, 32)
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self.overclock = overclock
        self.screen_origin = b2Vec2(SCREEN_WIDTH/2., TABLE_HEIGHT)
        self.colors = {
            'countertop': (50, 50, 50, 255),
            'gripper': (244, 170, 66, 255),
            'cup': (15, 0, 100, 0),
            'stirrer': (163, 209, 224, 255),
            'spoon': (73, 11, 61, 255),
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
        '''
        Draw bodies in the world with pygame.
        Adapted from examples/simple/simple_02.py in pybox2d.
        Args:
            bodies: a list of box2d bodies
            bg_color: background color 
        '''
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
        self.screen.fill(bg_color)
        if self.overclock is None:
            self.clock.tick(TARGET_FPS)
        pygame.event.get()
        for body in bodies:
            for fixture in body.fixtures:
                fixture.shape.draw(body, fixture)
        pygame.display.flip()


class b2WorldInterface(object):
    def __init__(self, do_gui=True, save_fig=False, caption='PyBox2D Simulator', overclock=None):
        '''
        Interface between Pybox2D and the graphics wrapper guiWorld.
        Args:
            do_gui: True if showing graphics; otherwise False.
            save_fig: True if saving figures of simulations; otherwise False. The figures are
            saved in folder tmp_images/
            caption: caption on the simulator graphics window.
            overclock: number of frames to skip when showing graphics. If overclock is None, 
            this feature is not used.
        '''
        self.world = b2World(gravity=GRAVITY, doSleep=True)
        self.do_gui = do_gui
        if do_gui:
            self.gui_world = guiWorld(caption=caption, overclock=overclock)
        else:
            self.gui_world = None
        self.save_fig = save_fig
        self.planning = False
        self.image_idx = 0
        self.image_name = 'test'
        self.gravity = GRAVITY
        self.num_steps = 0
        self.overclock = overclock
        self.liquid = Liquid(self.world)
    def enable_gui(self, caption='PyBox2D Simulator'):
        '''
        Enable visualization.
        '''
        self.do_gui = True
        self.gui_world = guiWorld(caption=caption, overclock=self.overclock)
    def disable_gui(self):
        '''
        Disable visualization.
        '''
        self.do_gui = False
    def draw(self):
        '''
        Visualize the current scene if do_gui is Trues.
        '''
        if not self.do_gui:
            return
        self.gui_world.draw(self.world.bodies)
        if self.save_fig and self.image_idx % 100 == 0:
            pygame.image.save(self.gui_world.screen,
                              'tmp_images/{num:05d}_{nm}'.format(num=self.image_idx/100, nm= self.image_name)+'.png')
        self.image_idx += 1
    def step(self):
        '''
        Wrapper of the step function of b2World.
        '''
        self.world.Step(TIME_STEP, VEL_ITERS, POS_ITERS)
        self.world.ClearForces()
        self.num_steps += 1
        if (self.overclock is None) or (self.num_steps % self.overclock == 0):
            self.draw()

class Kitchen2D(b2WorldInterface):
    def __init__(self, do_gui, sink_w, sink_h, sink_d, sink_pos_x, 
                 left_table_width, right_table_width, faucet_h, faucet_w, faucet_d, planning=True,
                 obstacles=None, save_fig=False, liquid_name='water', liquid_frequency=0.2, overclock=None):
        '''
        Args:
            sink_w: sink length (horizontal).
            sink_h: sink height.
            sink_d: table/sink thickness.
            sink_pos_x: position of the top left corner of the sink.
            left_table_width: width of the table on the left side of the sink.
            right_table_width: width of the table on the right side of the sink.
            faucet_h: height of faucet.
            faucet_w: horizontal length of the faucet.
            faucet_d: thickness of the faucet.
            height and length all include the thickness of the material.
            planning: indicator of whether the simulator is used for planning. If True, 
            the simulation for liquid will be skipped.
            obstacles: a list of box shapes that are static obstacles. See add_obstacles.
            save_fig: True if saving figures of simulations; otherwise False. The figures are
            saved in folder tmp_images/.
            liquid_name: name of the default liquid from the faucet.
            liquid_frequency: number of particles generated by the faucet per step.
            overclock: number of frames to skip when showing graphics. If overclock is None, 
            this feature is not used.
        '''
        super(Kitchen2D, self).__init__(do_gui, save_fig=save_fig)
        self.planning = planning
        world = self.world
        self.liquid = Liquid(world, liquid_name=liquid_name, liquid_frequency=liquid_frequency)
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
        self.add_obstacles(obstacles)
        self.overclock = overclock

    def add_obstacles(self, obstacles=None):
        '''
        Add statics obstacles to the scene.
        obstacles: a list of box shapes, e.g. [(w,d)], where w is the width 
        and d is the height of the box.
        '''
        if obstacles is None:
            return
        for obs in obstacles:
            pos, shape = obs
            base = self.world.CreateStaticBody(
                position=pos,
                shapes=[b2PolygonShape(box=shape)],
                userData='countertop')

    def step(self):
        '''
        Wrapper of step function of b2WorldInterface. It includes simulations 
        for liquid generated from the faucet.
        '''
        if (not self.planning) and (not self.disable_liquid) and sensor_touching_test(self.sensor):
            pos = np.random.normal(0, 0.1, size=(2)) + \
                np.array(self.faucet_location)
            self.liquid.make_particles(pos)
        super(Kitchen2D, self).step()

    def gen_liquid_in_cup(self, cup, N, userData='water'):
        '''
        Generate N liquid particles named userData uniformly in an empty cup.
        '''
        assert np.abs(cup.angle) < 0.1, 'cup is not verticle'
        grid_x = (cup.usr_w - cup.usr_d*2)/ self.liquid.radius / 2
        grid_y = (cup.usr_w - cup.usr_d) / self.liquid.radius / 2
        assert grid_x*grid_y > N, 'cup cannot hold {} particles'.format(N)
        for i in range(N):
            x, y = i % grid_x, i / grid_x
            pos = (self.liquid.radius*(2*x+1)+cup.usr_d, self.liquid.radius*(2*y+1)+cup.usr_d)
            self.liquid.make_one_particle(cup.position - cup.shift + pos, userData)




def sensor_touching_test(sensor):
    '''
    Indicator of whether any Box2D body (other than the ones 
    in LIQUID_NAMES) is touching the sensor.
    '''
    if len(sensor.contacts) == 0:
        return False
    for c in sensor.contacts:
        if c.contact.touching and (c.contact.fixtureA.body.userData not in LIQUID_NAMES) \
           and (c.contact.fixtureB.body.userData not in LIQUID_NAMES):
            return True
    return False

def get_body_vertices(body):
    '''
    Returns the list of vertices of the shapes that constructs a Box2D body.
    '''
    vertices = []
    for fixture in body.fixtures:
        if isinstance(fixture.shape, b2PolygonShape):
            vertices.extend([body.transform * v for v in fixture.shape.vertices])
        elif isinstance(fixture.shape, b2CircleShape):
            center = body.transform * fixture.shape.pos
            vertices.extend([center + [fixture.shape.radius]*2, center - [fixture.shape.radius]*2])
    return vertices

class Liquid(object):
    def __init__(self, world, liquid_frequency=0.2, density=0.01, friction=0.0, radius=0.05, shape_type='circle', liquid_name='water'):
        '''
        This class manages the liquid particles in the kitchen.
        Args:
            world: a Kitchen2D object.
            liquid_frequency: number of particles generated by the faucet per step.
            density: density of particles.
            friction: friction of particles.
            radius: radius of the shape of particles (see the dictionary shapes).
            shape_type: type of the shape of particles ('circle' | 'square' | 'triangle')
            liquid_name: name of particles (for visualization purposes).
        '''
        self.particles = []
        self.density = density
        self.friction = friction
        self.radius = radius
        shapes = {'triangle': b2PolygonShape(vertices=[(0,0), (radius,0), (radius*np.cos(np.pi/3), radius*np.sin(np.pi/3))]), 
                  'square': b2PolygonShape(box=(radius,radius)),
                  'circle': b2CircleShape(radius=radius)}
        self.shape = shapes[shape_type]
        self.liquid_frequency = liquid_frequency
        self.particle_calls = 0
        self.world = world
        self.liquid_name = liquid_name
    def make_particles(self, pos):
        '''
        Make new particles at position pos, following liquid_frequency specified in __init__.
        '''
        if self.liquid_frequency < 1:
            if self.particle_calls % int(1./self.liquid_frequency) == 0:
                self.make_one_particle(pos, self.liquid_name)
                self.particle_calls = 0
            self.particle_calls += 1
        else:
            for i in range(int(self.liquid_frequency)):
                self.make_one_particle(pos, self.liquid_name)
    def make_one_particle(self, pos, userData):
        '''
        Make one particle of name userData at position pos.
        '''
        p = self.world.CreateDynamicBody(position=pos, userData=userData)
        p.CreateFixture(
            shape=self.shape,
            friction=self.friction,
            density=self.density
        )
        self.particles.append(p)

def make_body(kitchen, name, pose, args):
    '''
    A wrapper to create Box2d bodies based on their name, pose and size.
    '''
    if 'gripper' in name:
        from gripper import Gripper
        body = Gripper(kitchen, init_pos=pose[:2], init_angle=pose[2])
    elif 'cup' in name:
        body = make_cup(kitchen, pose[:2], pose[2], *args[name])
    elif 'spoon' in name:
        body = make_spoon(kitchen, pose[:2], pose[2], *args[name])
    elif 'stir' in name:
        body = make_stirrer(kitchen, pose[:2], pose[2], *args[name])
    elif 'block' in name:
        body = make_block(kitchen, pose[:2], pose[2], *args[name])
    else:
        raise NotImplementedError(name)
    body.name = name
    return body

def make_spoon(b2world_interface, pos, angle, w, h, d):
    '''
    Return a Box2D body that resembles a spoon.
    The center of the spoon is at the bottom of the tip of half sphere.
    Args:
        b2world_interface: an instance of b2WorldInterface
        pos: position of spoon
        angle: angle of spoon
        w: thickness of the spoon
        h: spoon's handle length
        d: width of the spoon bottom
    '''
    world = b2world_interface.world
    body = world.CreateDynamicBody(
        position=pos,
        angle=angle
    )
    polygon_shape = [(0, d*2.-w), (w, d*2.-w),
                     (w, h+d*2.), (0, h+d*2.)]
    shift = np.array([w/2., 0.])
    polygon_shape = [(v[0] - shift[0], v[1]-shift[1]) for v in polygon_shape]
    body.CreateFixture(
        shape=b2PolygonShape(vertices=polygon_shape),
        friction=1,
        density=0.5
    )

    n_points = 10
    outer_shape = []
    inner_shape = []
    center = b2Vec2((0, d))
    for a in np.linspace(0, np.pi, n_points):
        outer_shape.append(center + d*np.array([np.sin(a), np.cos(a)]))
    for a in np.linspace(0, np.pi, n_points):
        inner_shape.append(center + (d-w)*np.array([np.sin(a), np.cos(a)]))
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

def make_block(b2world_interface, pos, angle, w, h, d=None):
    '''
    Return a Box2D body that resembles a block.
    The center of the block is at its bottom center.
    Args:
        b2world_interface: an instance of b2WorldInterface
        pos: position
        angle: angle
        w: width
        h: height
    '''
    shift = np.array([w/2.0, 0])
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
    '''
    Return a Box2D body that resembles a cup.
    The center of the cup is at its bottom center.
    Args:
        b2world_interface: an instance of b2WorldInterface
        pos: position
        angle: angle
        w: width
        h: height
        d: thickness of the cup material
    '''
    shift = np.array([w/2.0, shifth*h])
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


def make_stirrer(b2world_interface, pos, angle, w, h, d):
    '''
    Return a Box2D body that resembles a stirrer.
    The center is at its bottom center.
    Args:
        b2world_interface: an instance of b2WorldInterface
        pos: position
        angle: angle
        w: thickness (or width) of handle
        h: height of handle
        d: radius of the bottom circle
    '''
    assert d < GRIPPER_WIDTH, 'Stirrer bottom is too large.'
    shift = np.array([w/2., 0.])
    world = b2world_interface.world
    body = world.CreateDynamicBody(
        position=pos,
        angle=angle
    )
    body.CreateFixture(
        shape=b2CircleShape(pos=(0, d), radius=d),
        friction=0.1,
        density=1
    )
    polygon_shape = [(0, d), (w, d),
                     (w, h+d*2), (0, h+d*2)]
    polygon_shape = [(v[0] - shift[0], v[1]-shift[1]) for v in polygon_shape]
    body.CreateFixture(
        shape=b2PolygonShape(vertices=polygon_shape),
        friction=1,
        density=0.5
    )
    body.usr_d = d*1.0
    body.usr_h = h*1.0
    body.usr_w = w*1.0
    body.userData = "stirrer"
    return body


def make_static_cup(b2world_interface, pos, angle, w, h, d):
    '''
    Returns a static cup of width w, height h and thickness d,
    at position pos with orientation angle.
    '''
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




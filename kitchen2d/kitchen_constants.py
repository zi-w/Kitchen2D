
from Box2D import b2Vec2
# Constants of world
GRAVITY = b2Vec2(0.0, -10.0) # gravity of the world
PPM = 10.0  # pixels per meter
TARGET_FPS = 100 # frame per second
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX = 800, 500 # screen width and height in px

SCREEN_WIDTH = SCREEN_WIDTH_PX / PPM # screen width in meter
SCREEN_HEIGHT = SCREEN_HEIGHT_PX / PPM # screen height in meter

# Constants of kitchen
GRIPPER_WIDTH = 0.6
GRIPPER_HEIGHT = 2.0
OPEN_WIDTH = 5.0
TABLE_HEIGHT = 20

TABLE_THICK = 2
ACC_THRES = 0.1 # threshold for accuracy of positions
VEL_ITERS = 10
POS_ITERS = 10
SAFE_MOVE_THRES = 1
MOTOR_SPEED = 5.0

EPS = 0.1 # epsilon distance used for grasping, placing etc


COPY_IGNORE = ('gripper', 'water', 'sensor', 'coffee', 'cream', 'sugar')


LIQUID_NAMES = ('water', 'coffee', 'sugar', 'cream')
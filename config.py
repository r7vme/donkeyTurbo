""" 
CAR CONFIG 

This file is read by your car application's manage.py script to change the car
performance. 

EXMAPLE
-----------
import dk
cfg = dk.load_config(config_path='~/d2/config.py')
print(cfg.CAMERA_RESOLUTION)

"""


import os

#PATHS
CAR_PATH = PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(CAR_PATH, 'data')
MODELS_PATH = os.path.join(CAR_PATH, 'models')

#VEHICLE
DRIVE_LOOP_HZ = 20
MAX_LOOPS = 100000

#CAMERA
CAMERA_RESOLUTION = (78, 160)
CAMERA_FRAMERATE = DRIVE_LOOP_HZ
CAMERA_ZOOM = (0.0, 0.35, 1.0, 1.0)

#STEERING
STEERING_CHANNEL = 1
STEERING_LEFT_PWM = 460 #490
# middle 400
STEERING_RIGHT_PWM = 340 #290

#THROTTLE
THROTTLE_CHANNEL = 0
THROTTLE_FORWARD_PWM = 420
THROTTLE_STOPPED_PWM = 410
THROTTLE_REVERSE_PWM = 300

#TRAINING
BATCH_SIZE = 128
TRAIN_TEST_SPLIT = 0.8


#JOYSTICK
USE_JOYSTICK_AS_DEFAULT = True
JOYSTICK_MAX_THROTTLE = 1.0
JOYSTICK_STEERING_SCALE = 1.0
AUTO_RECORD_ON_THROTTLE = True

#DONKEYTURBO
DT_PILOT_CONFIG = {
               'angle_sma_n': 10,
               'throttle_max': 1.0,
               'throttle_min': 0.5,
                }

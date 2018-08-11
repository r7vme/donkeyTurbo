'''
Module contains pilot class and Keras model.
'''

from collections import deque
from donkeycar.parts.keras import KerasPilot
from donkeycar.util import data as dkutil
from donkeyturbo.model import dt_categorical
from tensorflow.python.keras.models import load_model

# Main features:
# - linear throttle computation based on angle.
# - angle post computed with SMA.
class DTKerasPilot(KerasPilot):
    def __init__(self, config=None):
        super(DTKerasPilot, self).__init__()

        if config is None:
            # Set default values.
            config = {
                    'angle_sma_n': 3,
                    'throttle_max': 1.0,
                    'throttle_min': 0.5,
                    'obstacle_model': '',
                    }

        # TODO: do validation
        self.config = config

        self.model = dt_categorical()

        # Initialize fixed length FIFO queue for angles historical data.
        self.angles = deque([], self.config['angle_sma_n'])

        # Load obstacle model.
        self.obstacle = None
        if self.config['obstacle_model']:
            self.obstacle = load_model(self.config['obstacle_model'])

    def run(self, img_arr):
        # Prepare image array.
        img_arr = img_arr.reshape((1,) + img_arr.shape)

        # Compute angle.
        angle_binned, _ = self.model.predict(img_arr)
        angle_unbinned = dkutil.linear_unbin(angle_binned[0])

        # If enabled, do obstacle detection.
        if self.obstacle:
            angle_obstacle = self.compute_angle_obstacle(img_arr)
            # In case we detected obstacle use fallback angle.
            if angle_obstacle:
                angle_unbinned = angle_obstacle
                print('obstacle detected, using fallback mode.')

        angle = self.compute_angle(angle_unbinned)
        throttle = self.compute_throttle(angle)

        print('throttle:', throttle, 'angle:', angle)
        return angle, throttle

    # _compute_sma computes SMA (simple moving average).
    # https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average
    @staticmethod
    def _compute_sma(data):
        return sum(data) / len(data)

    def compute_angle(self, angle):
        # Append new angle to historial data.
        self.angles.append(angle)
        # Return SMA value.
        return self._compute_sma(self.angles)

    def compute_throttle(self, angle):
        b = self.config['throttle_min']
        s = self.config['throttle_max'] - self.config['throttle_min']
        x = 1 - abs(angle)

        # b - base, minimally allowed speed
        # s - scale, basically delta between max and min limits
        # x - linear dependency from angle
        th = b + s * x

        return th

    def compute_angle_obstacle(self, img_arr):
        # no obstacle
        # obstacle on the left -> turn right
        # obstacle on the right -> turn left
        obstacle_to_angle = {
                0: 0,
                1: 1,
                2: -1,
                }

        # Do NN prediction.
        binned = self.obstacle.predict(img_arr)[0].tolist()
        obstacle = binned.index(max(binned))
        return obstacle_to_angle[obstacle]

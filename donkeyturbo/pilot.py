'''
Module contains pilot class and Keras model.
'''

import donkeycar as dk
from collections import deque
from donkeycar.parts.keras import KerasPilot

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
                    }

        # TODO: do validation
        self.config = config

        # Initialize fixed length FIFO queue for angles historical data.
        self.angles = deque([], self.config['angle_sma_n'])

    def run(self, img_arr):
        # Do NN prediction.
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        angle_binned, _ = self.model.predict(img_arr)
        angle_unbinned = dk.utils.linear_unbin(angle_binned)

        # Do post processing.
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

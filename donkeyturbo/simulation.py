'''
Module that runs websocket simulation server.
'''
import base64
import numpy as np
import socketio
import donkeycar.management.base as dkbase
from PIL import Image
from io import BytesIO
from donkeycar.management.base import BaseCommand
from donkeycar.parts.simulation import SteeringServer
from donkeyturbo.keras import DTKerasPilot

class Sim(object):
    '''
    Start a websocket SocketIO server to talk to a donkey simulator.
    '''
    def run(self, args):
        cfg = dkbase.load_config("./config.py")

        if cfg is None:
            return

        # NOTE(r7vme): Use hardcoded pilot.
        kl = DTKerasPilot()

        # Load keras model.
        kl.load(args["--model"])

        #start socket server framework
        sio = socketio.Server()

        #start sim server handler
        ss = DTSteeringServer(sio, kpart=kl, top_speed=3.0)

        #register events and pass to server handlers

        @sio.on('telemetry')
        def telemetry(sid, data):
            ss.telemetry(sid, data)

        @sio.on('connect')
        def connect(sid, environ):
            ss.connect(sid, environ)

        ss.go(('0.0.0.0', 9090))

# Copy of SteeringServer with only one change to crop image.
class DTSteeringServer(SteeringServer):
    def __init__(self, *args, **kwargs):
        super(DTSteeringServer, self).__init__(*args, **kwargs)

    def telemetry(self, sid, data):
        '''
        Callback when we get new data from Unity simulator.
        We use it to process the image, do a forward inference,
        then send controls back to client.
        Takes sid (?) and data, a dictionary of json elements.
        '''
        if data:
            # The current steering angle of the car
            last_steering = float(data["steering_angle"])

            # The current throttle of the car
            last_throttle = float(data["throttle"])

            # The current speed of the car
            speed = float(data["speed"])

            # The current image from the center camera of the car
            imgString = data["image"]

            # decode string based data into bytes, then to Image
            image = Image.open(BytesIO(base64.b64decode(imgString)))

            # then as numpy array
            image_array = np.asarray(image)

            # optional change to pre-preocess image before NN sees it
            if self.image_part is not None:
                image_array = self.image_part.run(image_array)

            # NOTE(r7vme): i have no idea why image_part does not pass.
            image_array = np.delete(image_array, np.s_[0:42:], axis=0)

            # forward pass - inference
            steering, throttle = self.kpart.run(image_array)

            # filter throttle here, as our NN doesn't always do a greate job
            throttle = self.throttle_control(last_steering, last_throttle, speed, throttle)

            # simulator will scale our steering based on it's angle based input.
            # but we have an opportunity for more adjustment here.
            steering *= self.steering_scale

            # send command back to Unity simulator
            self.send_control(steering, throttle)

        else:
            # NOTE: DON'T EDIT THIS.
            self.sio.emit('manual', data={}, skip_sid=True)

        self.timer.on_frame()

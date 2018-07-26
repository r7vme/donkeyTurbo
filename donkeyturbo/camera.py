'''
Module with camera class.
'''

import time
from donkeycar.parts.camera import PiCamera as DonkeyCamera


# NOTE(r7vme): The only difference is hardcoded resolution and added zoom.
class DTPiCamera(DonkeyCamera):
    def __init__(self, resolution=(120, 160), framerate=20):
        from picamera.array import PiRGBArray
        from picamera import PiCamera
        # initialize the camera and stream
        self.camera = PiCamera() #PiCamera gets resolution (height, width)

        # Crop image from the top.
        self.camera.resolution = (160, 78)
        self.camera.zoom = (0.0, 0.35, 1.0, 1.0)

        self.camera.framerate = framerate
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,
            format="rgb", use_video_port=True)

        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.on = True

        print('PiCamera loaded.. .warming camera')
        time.sleep(2)

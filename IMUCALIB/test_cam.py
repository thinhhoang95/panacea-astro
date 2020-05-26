from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time
# Initialization of camera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 2
# camera.shutter_speed = camera.exposure_speed
# camera.iso = 800
# camera.exposure_mode = 'off'
time.sleep(5)
# g = camera.awb_gains
# camera.awb_mode = 'off'
# camera.awb_gains = g
#rawCapture = PiRGBArray(camera, size=(640,480))
time.sleep(0.5)
print('(CAM) Camera initialized')

with camera:
        camera.capture_sequence(['test.jpg'], use_video_port=True)
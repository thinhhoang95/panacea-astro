import picamera
import time

with picamera.PiCamera() as camera:
    camera.resolution = (640, 480)
    camera.framerate = 12
    camera.start_recording('my_video.h264')
    time.sleep(30)
    camera.stop_recording()
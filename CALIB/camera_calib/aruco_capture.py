# This script is used to obtain several chessboard images to calibrate the camera

# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
rawCapture = PiRGBArray(camera, size=(640,480))
# allow the camera to warmup
time.sleep(0.1)
# grab an image from the camera
for i in range(20):
    print('Now capture image #', i,'/20')
    time.sleep(3)
    camera.capture(rawCapture, format='bgr')
    image = rawCapture.array
    cv2.imwrite('img_'+str(i)+'.jpg', image)
    print('Image #', i, ' is captured!')
    rawCapture.truncate(0)
    time.sleep(1)
print('All images have been captured. Please run the calibration script to obtain the camera matrix')
print('Goodbye')
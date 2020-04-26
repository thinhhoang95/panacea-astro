# DAQ for Panacea Hardware (MPU9250 + CAMERA)
# The purpose is to output accelerometer, Euler angles and a movie for optical flow analysis
#
# Author: Hoang Dinh Thinh (thinh@neuralmetrics.net)
# Department of Aerospace Engineering, University of Corruption, Ho Chi Minh City, Vietnam

from threading import Lock
import RTIMU
import sys 
import math
import keyboard
import os
import io

from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time
import threading
import glob

import numpy as np

# Lock declarations
imu_terminate_lock = Lock()
smile_lock = Lock()
smile_flag = False
imu_terminate_flag = False
imu_start_flag = False

# Initialization of global variables
# Initialization of RTIMULib
SETTINGS_FILE_NAME = "RTIMULib"

rtimu_settings = RTIMU.Settings(SETTINGS_FILE_NAME)
imu = RTIMU.RTIMU(rtimu_settings)

# Initialization of camera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 8
time.sleep(3)
camera.shutter_speed = camera.exposure_speed
camera.exposure_mode = 'off'
g = camera.awb_gains
camera.awb_mode = 'off'
camera.awb_gains = g
rawCapture = PiRGBArray(camera, size=(640,480))

time.sleep(0.5)
print('Camera initialized')

# Initialization of video codec and writer
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# frameout = cv2.VideoWriter('output.avi', fourcc, 30.0, (640, 480))
# print('Codec and Video Writer initialized')

# Gyro offsets
yawoff = 0.0
pitchoff = 0.0
rolloff = 0.0

print('Gyro offsets set')

# Timers
t_print = time.time()
t_damp = time.time()
t_fail = time.time()
t_fail_timer = 0.0
t_shutdown = 0.0

print('Timers set')

if (not imu.IMUInit()):
    hack = time.time()
    if (hack - t_print) > 1.0:
        print('Still waiting IMU to get ready...')
        t_print = hack
        t_shutdown += 1
        if t_shutdown > 9:
            sys.exit(1)

imu.setSlerpPower(0.02)
imu.setGyroEnable(True)
imu.setAccelEnable(True)
imu.setCompassEnable(True)
poll_interval = imu.IMUGetPollInterval()

print('IMU enabled')

# Attitude variables
roll = 0.0
pitch = 0.0
yaw = 0.0
heading = 0.0
rollrate = 0.0
pitchrate = 0.0
yawrate = 0.0

print('Attitude set')

# Attitude history variables
tme = []
accX = []
accY = []
accZ = []
phi = []
the = []
psi = []

print('Logging enabled')

# Magnetic Deviation constant
magnetic_deviation = -13.7

# Initialization of clocks
clock = time.time()
t_begin = time.time()

print('Variable initialization completed')

# DAQ for MPU (acceleration and Euler angles)
def mpu_daq():
    
    global hack, t_damp, imu_terminate_flag, imu_start_flag, clock, imu, t_print, smile_lock, smile_flag
    global tme, psi, the, phi
    while not imu_terminate_flag:
        if not imu_start_flag:
            # print('(MPU) Heartbeat Skip')
            continue

        hack = time.time()
        clock = hack

        # print('(MPU) Heartbeat')

        if imu.IMURead():
            data = imu.getIMUData()
            fusionPose = data["fusionPose"]
            Gyro = data["gyro"]
            Accel = data["accel"]
            t_fail_timer = 0.0
            clock = time.time()

            if (hack - t_damp) > .01:
                roll = round(math.degrees(fusionPose[0]) - rolloff, 4)
                pitch = round(math.degrees(fusionPose[1]) - pitchoff, 4)
                yaw = round(math.degrees(fusionPose[2]) - yawoff, 4)
                rollrate = round(math.degrees(Gyro[0]), 4)
                pitchrate = round(math.degrees(Gyro[1]), 4)
                yawrate = round(math.degrees(Gyro[2]), 4)
                if yaw < 0.1:
                    yaw = yaw + 360
                if yaw > 360:
                    yaw = yaw - 360
                if yaw < 0.1:
                    yaw = yaw + 360.0

                # yaw is magnetic heading, convert to true heading
                heading = yaw - magnetic_deviation
                if heading < 0.1:
                    heading = heading + 360
                if heading > 360:
                    heading = heading - 360

                t_damp = hack

                if (hack - t_print) > .01:

                    # print('(MPU) Roll: ' + str(roll) + ', Pitch: ' + str(pitch) + ', Yaw: ' + str(yaw) + 'degrees \r'),

                    if (Accel[0]**2 + Accel[1]**2 + Accel[2]**2 > 25):  # upto 5g acceleration
                        raise Exception('Accelerometer failed: Total acceleration exceeds 5g')
                    
                    print('(MPU) Roll: ', roll, ' Pitch: ', pitch, ' Yaw: ', yaw)

                    # with smile_lock:
                    #     if smile_flag: 
                    #         tme.append(clock)
                    #         psi.append(math.radians(yaw))
                    #         the.append(math.radians(pitch))
                    #         phi.append(math.radians(roll))
                    #         smile_flag = False
                    #         print('> (MPU) Roll: ', roll, ' Pitch: ', pitch, ' Yaw: ', yaw)

                    t_print = hack
            time.sleep(poll_interval*1.0/1000.0)

# DAQ for Pi camera
def cam_daq():
    global imu_terminate_flag, imu_start_flag, camera, rawCapture
    while not imu_terminate_flag:
        if not imu_start_flag:
            continue
        for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
            image = frame.array
            frameout.write(image)
            rawCapture.truncate(0)
            if imu_terminate_flag:
                frameout.release()
                print('(CAM) Video file has been written successfully')
                return

# Use PiCamera VideoWriter instead of OpenCV's
def cam_daq_2():
    global imu_terminate_flag, imu_start_flag, camera
    while not imu_terminate_flag:
        if not imu_start_flag:
            continue
        with camera:
            camera.start_recording(str(time.time())+'.h264')
            print('(CAM) Recording started for 20 seconds')
            camera.wait_recording(20)
            camera.stop_recording()
            print('(CAM) Recording completed')
            return

def filenames():
    global imu_terminate_flag, smile_flag, smile_lock
    frame = 0
    while not imu_terminate_flag:
        yield 'image%04d_%s.jpg' % (frame,str(time.time()))
        frame += 1

def cam_daq_3():
    global imu_terminate_flag, imu_start_flag, camera, smile_lock, smile_flag
    while not imu_terminate_flag:
        if not imu_start_flag:
            continue
        with camera:
            captured = 0
            while True:
                if imu_terminate_flag:
                    return
                print('(CAM) Capturing image #', captured)
                camera.capture(str(time.time())+'.jpg')
                with smile_lock:
                    smile_flag = True # notify the MPU thread to record data
                time.sleep(1)
                captured = captured + 1

# Read keyboard key to end the experiment
def read_keyboard_key():
    global imu_terminate_lock, imu_terminate_flag, tme, psi, the, phi
    global imu_thread, cam_thread
    print('(KEY) MPU recording will be effective in 20 seconds')
    while True:
        if time.time() - t_begin > 5:
            print('(KEY) Experiment terminated')
            with imu_terminate_lock:
                imu_terminate_flag = True
            # Wait for IMU and camera thread to finish
            imu_thread.join()
            cam_thread.join()

            # Write the values to files
            if os.path.exists('accel.txt'):
                os.remove('accel.txt')
                print('(KEY) File accel.txt has been deleted')
            # accel_file = open('accel.txt', 'a')
            # print('Time vector: ', len(tme), ', AccX vector: ', len(accX), ', AccY vector: ', len(accY), ', AccZ vector: ', len(accZ))
            # for i in range(len(tme)):
            #     accel_row_to_write = '{:.8f},{:.8f},{:.8f},{:.8f}\r\n'.format(tme[i], accX[i], accY[i], accZ[i])
            #     accel_file.write(accel_row_to_write)
            # print('(KEY) File accel.txt has been written successfully')
            if os.path.exists('eulang.txt'):
                os.remove('eulang.txt')
                print('(KEY) File eulang.txt has been deleted')
            eulang_file = open('eulang.txt', 'a')
            for i in range(len(tme)):
                eulang_row_to_write = '{:.8f},{:.8f},{:.8f},{:.8f}\r\n'.format(tme[i], psi[i], the[i], phi[i])
                eulang_file.write(eulang_row_to_write)
            print('(KEY) File eulang.txt has been written successfully')
            return

# Thread declarations
key_thread = threading.Thread(target=read_keyboard_key)
imu_thread = threading.Thread(target=mpu_daq)
cam_thread = threading.Thread(target=cam_daq_3)

def main():
    print('=========================================================')
    print('Panacea DAQ - Data Acquisition for Panacea Landing System')
    print('Author: Hoang Dinh Thinh - University of Corruption')
    print('=========================================================')
    # Initialize the IMU
    # Perform IMU check
    global imu
    if (not imu.IMUInit()):
        global hack, t_print, t_shutdown, t_damp
        hack = time.time()
        if (hack - t_print) > 1.0:
            t_print = hack
            t_shutdown += 1
            if t_shutdown > 9:
                print('Cannot connect to the IMU. Please check connection!')
                sys.exit()
    global imu_terminate_lock, imu_start_flag
    with imu_terminate_lock:
        imu_start_flag = True
        print('Algorithm is set to run in 20 seconds')
    image_list = glob.glob('*.jpg')
    print('Found ', len(image_list), ' image files to delete')
    for image in image_list:
        try:
            os.remove(image)
        except:
            print('Cannot delete image: ', image)
    print('Starting threads...')
    imu_thread.start()
    print('MPU thread started 1/3')
    cam_thread.start()
    print('CAM thread started 2/3')
    key_thread.start()
    print('KEY thread started 3/3')

main()
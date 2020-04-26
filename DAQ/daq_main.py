# DAQ for Panacea Hardware (MPU9250 + CAMERA)
# The purpose is to output accelerometer, Euler angles and a movie for optical flow analysis
#
# Author: Hoang Dinh Thinh (thinh@neuralmetrics.net)
# Department of Aerospace Engineering, University of Corruption, Ho Chi Minh City, Vietnam

# This is the edition that allows multiprocessing
from multiprocessing import Lock, Process, Queue, current_process

# DAQ for MPU (acceleration and Euler angles)
def mpu_daq(smile_queue):
    import RTIMU
    import sys 
    import math
    import keyboard
    import os
    import io
    import time
    import glob
    import queue

    # Initialization of RTIMULib
    SETTINGS_FILE_NAME = "RTIMULib"

    rtimu_settings = RTIMU.Settings(SETTINGS_FILE_NAME)
    imu = RTIMU.RTIMU(rtimu_settings)
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
    while True:

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
                    
                    # print('(MPU) Roll: ', roll, ' Pitch: ', pitch, ' Yaw: ', yaw)

                    try:
                        smile = smile_queue.get_nowait()
                    except queue.Empty:
                        pass
                    else:
                        if smile == True:
                            print('(MPU) Keyframe pose data save')
                            tme.append(clock)
                            psi.append(math.radians(yaw))
                            the.append(math.radians(pitch))
                            phi.append(math.radians(roll))
                            accX.append(Accel[0])
                            accY.append(Accel[1])
                            accZ.append(Accel[2])
                        else:
                            if os.path.exists('eulang.txt'):
                                os.remove('eulang.txt')
                                print('(KEY) File eulang.txt has been deleted')
                            eulang_file = open('eulang.txt', 'a')
                            for i in range(len(tme)):
                                eulang_row_to_write = '{:.8f},{:.8f},{:.8f},{:.8f}\r\n'.format(tme[i], psi[i], the[i], phi[i])
                                eulang_file.write(eulang_row_to_write)
                            print('(KEY) File eulang.txt has been written successfully')
                            if os.path.exists('accel.txt'):
                                os.remove('accel.txt')
                                print('(KEY) File accel.txt has been deleted')
                            accel_file = open('accel.txt', 'a')
                            for i in range(len(tme)):
                                accel_row_to_write = '{:.8f},{:.8f},{:.8f},{:.8f}\r\n'.format(tme[i], accX[i], accY[i], accZ[i])
                                accel_file.write(accel_row_to_write)
                            print('(KEY) File accel.txt has been written successfully')
                            return
                    finally: 
                        print('(MPU) Roll: ', roll, ' Pitch: ', pitch, ' Yaw: ', yaw)
                        t_print = hack
                        time.sleep(poll_interval*1.0/1000.0)

def cam_daq_3(smile_queue):
    import queue
    from picamera.array import PiRGBArray
    from picamera import PiCamera
    import cv2
    import time
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
    print('(CAM) Camera initialized')

    clock = time.time()
    run_until = clock + 25 

    while True:
        with camera:
            captured = 0
            while True:
                if time.time() < run_until:
                    print('(CAM) Capturing image #', captured)
                    smile_queue.put(True)
                    camera.capture('img_'+str(captured)+'.jpg')
                    time.sleep(1)
                    captured = captured + 1
                else:
                    smile_queue.put(False)
                    return

def main():
    import glob
    import time
    import os 
    print('=========================================================')
    print('Panacea DAQ - Data Acquisition for Panacea Landing System')
    print('Author: Hoang Dinh Thinh - University of Corruption')
    print('=========================================================')
    image_list = glob.glob('*.jpg')
    print('Found ', len(image_list), ' image files to delete')
    for image in image_list:
        try:
            os.remove(image)
        except:
            print('Cannot delete image: ', image)
    smile_queue = Queue()
    
    mpu_daq_p = Process(target=mpu_daq, args=(smile_queue,))
    cam_daq_p = Process(target=cam_daq_3, args=(smile_queue,))
    
    print('Processes started at ' + str(time.time()))
    mpu_daq_p.start()
    cam_daq_p.start()

    mpu_daq_p.join()
    cam_daq_p.join()
    print('Processes finished at ' + str(time.time()))


if __name__ == '__main__':
    main()
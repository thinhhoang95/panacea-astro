# DAQ for Panacea Hardware (MPU9250 + CAMERA)
# The purpose is to output accelerometer, Euler angles and a movie for optical flow analysis
#
# Author: Hoang Dinh Thinh (thinh@neuralmetrics.net)
# Department of Aerospace Engineering, University of Corruption, Ho Chi Minh City, Vietnam

# This is the edition that allows multiprocessing
from multiprocessing import Lock, Process, Queue, current_process

# Maximum runs control
max_runs = 10
runs = ['run_{:d}'.format(i) for i in range(max_runs)]
current_dir_lock = Lock()
current_dir = 0 # first entry in runs

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

    global current_dir, runs

    # Initialization of RTIMULib
    SETTINGS_FILE_NAME = "RTIMULib"

    rtimu_settings = RTIMU.Settings(SETTINGS_FILE_NAME)
    imu = RTIMU.RTIMU(rtimu_settings)
    # Gyro offsets
    yawoff = 0.0
    pitchoff = 5.17
    rolloff = -4.99

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
    entry = 0 # only output 50 initial MPU readings

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

                    # print('(MPU) Keyframe pose data save')
                    tme.append(clock)
                    psi.append(math.radians(yaw))
                    the.append(math.radians(pitch))
                    phi.append(math.radians(roll))
                    accX.append(Accel[0])
                    accY.append(Accel[1])
                    accZ.append(Accel[2])

                    entry = entry + 1 # count data obtained

                    try:
                        smile = smile_queue.get_nowait()
                    except queue.Empty:
                        pass
                    else:
                        if not smile: # terminate signall obtained from the camera process
                            suffix = runs[current_dir]
                            if os.path.exists(suffix+'/inertial/eulang.txt'):
                                os.remove(suffix+'/inertial/eulang.txt')
                                print('(KEY) File eulang.txt has been deleted')
                            eulang_file = open(suffix+'/inertial/eulang.txt', 'a')
                            for i in range(len(tme)):
                                eulang_row_to_write = '{:.8f},{:.8f},{:.8f},{:.8f}\r\n'.format(tme[i], psi[i], the[i], phi[i])
                                eulang_file.write(eulang_row_to_write)
                            print('(KEY) File eulang.txt has been written successfully')
                            if os.path.exists(suffix+'/inertial/accelerometer.txt'):
                                os.remove(suffix+'/inertial/accelerometer.txt')
                                print('(KEY) File accelerometer.txt has been deleted')
                            accel_file = open(suffix+'/inertial/accelerometer.txt', 'a')
                            for i in range(len(tme)):
                                accel_row_to_write = '{:.8f},{:.8f},{:.8f},{:.8f}\r\n'.format(tme[i], accX[i], accY[i], accZ[i])
                                accel_file.write(accel_row_to_write)
                            print('(KEY) File accel.txt has been written successfully')
                            return
                    finally: 
                        # print('(MPU) Roll: ', roll, ' Pitch: ', pitch, ' Yaw: ', yaw)
                        if entry<50:
                            print('(MPU) Roll: ', roll, ' Pitch: ', pitch, ' Yaw: ', yaw)
                        elif entry==50:
                            print('(MPU) MPU data will now be hidden')
                        t_print = hack
                        time.sleep(poll_interval*1.0/1000.0)

def filenames():
    global current_dir, runs
    import time
    frame = 0
    while frame<15: # maximum camera samples goes here
        yield runs[current_dir] + '/images/image%04d_%.6f.jpg' % (frame,time.time())
        frame += 1

def cam_daq_3(smile_queue):
    global current_dir, runs, max_runs
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
    time.sleep(15)
    print('(CAM) You may now pick up the device, capture will begin in 7 seconds')
    time.sleep(7)
    #capture_duration = 10
    #print('(CAM) Now begin capturing in {:d} seconds'.format(capture_duration))
    #clock = time.time()
    #run_until = clock + capture_duration

    print('Now begin capturing for 15 camera samples...')

    while True:
        with camera:
            camera.capture_sequence(filenames(), use_video_port=True)
            smile_queue.put(False) # send terminate signal to the IMU thread
            # switch to the next run (next directory in line)
            with current_dir_lock:
                if current_dir < max_runs:
                    current_dir = current_dir + 1
                else:
                    print('Run has completed')
            return

def main():
    import glob
    import time
    import os 
    print('=========================================================')
    print('Panacea DAQ - Data Acquisition for Panacea Landing System')
    print('Variant: IMU Calibration with Finite Runs')
    print('Author: Hoang Dinh Thinh - University of Corruption')
    print('Email: hoangdinhthinh@ieee.org')
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
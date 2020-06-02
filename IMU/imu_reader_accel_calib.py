import os
import socket
import operator
import math
import time
import os.path
import RTIMU
import sys
import getopt

sys.path.append('.')

SETTINGS_FILE = "RTIMULib"

s = RTIMU.Settings(SETTINGS_FILE)
imu = RTIMU.RTIMU(s)

# offsets
yawoff = 0.0
pitchoff = 0.0
rolloff = 0.0

# timers
t_print = time.time()
t_damp = time.time()
t_fail = time.time()
t_fail_timer = 0.0
t_shutdown = 0

program_duration = input('How long to read accelerometer\'s data? ')

if (not imu.IMUInit()):
    hack = time.time()
    imu_sentence = "$IIXDR,IMU_FAILED_TO_INITIALIZE*7C"
    if (hack - t_print) > 1.0:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # sock.sendto(imu_sentence, (IMU_IP, IMU_PORT))
        t_print = hack
        t_shutdown += 1
        if t_shutdown > 9:
            sys.exit(1)

imu.setSlerpPower(0.02)
imu.setGyroEnable(True)
imu.setAccelEnable(True)
imu.setCompassEnable(True)

poll_interval = imu.IMUGetPollInterval()

# data variables
roll = 0.0
pitch = 0.0
yaw = 0.0
heading = 0.0
rollrate = 0.0
pitchrate = 0.0
yawrate = 0.0

# raw data of acceleration
tme = []
accX = []
accY = []
accZ = []
gyrX = []
gyrY = []
gyrZ = []

# magnetic deviation

magnetic_deviation = -13.7

clock = time.time()
t_begin = time.time()
t_end = time.time() + float(program_duration)
    
print('Program begins at ' + str(t_begin))
print('Program will run until ' + str(t_end))

while clock < t_end:

    hack = time.time()
    clock = hack

    # if it's been longer than 5 seconds since last print
    if (hack - t_damp) > 5.0:

        if (hack - t_fail) > 1.0:
            t_fail = hack
            t_shutdown += 1

    if imu.IMURead():
        data = imu.getIMUData()
        fusionPose = data["fusionPose"]
        Gyro = data["gyro"]
        Accel = data["accel"]
        t_fail_timer = 0.0
        clock = time.time()

        if (hack - t_damp) > .01:
            # print('Clock: ' + str(hack))
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
                # Print log data
                print('Accelerometer X: ', Accel[0], ', Y: ', Accel[1], ' Z: ', Accel[2], ' g')
                print('Roll: ' + str(roll) + ', Pitch: ' + str(pitch) + ', Yaw: ' + str(yaw) + 'degrees \n')
                
                # Append to vectors
                tme.append(hack - t_begin)
                accX.append(Accel[0])
                accY.append(Accel[1])
                accZ.append(Accel[2])
                gyrZ.append(yawrate)
                gyrY.append(pitchrate)
                gyrX.append(rollrate)

                if (Accel[0]**2 + Accel[1]**2 + Accel[2]**2 > 25): # upto 5g acceleration
                    print('IMU FAILED - INSTABILITY DETECTED')
                    raise Exception('IMU FAILED - UNSTABLE READINGS')
                
                t_print = hack
        time.sleep(poll_interval*1.0/1000.0)

print('Time samples collected: ', len(tme))
print('Writing accelerometer values to file... ')
accel_file = open('accelerometer.txt', 'a')
for record in range(len(tme)):
    accel_file.write(str(tme[record])+','+str(accX[record]) +
                     ','+str(accY[record])+','+str(accZ[record])+'\n')
accel_file.close()
print('Writing gyroscope values to file...')
gyro_file = open('gyroscope.txt', 'a')
for record in range(len(tme)):
    gyro_file.write(str(tme[record])+','+str(gyrZ[record]) +
                     ','+str(gyrY[record])+','+str(gyrX[record])+'\n')
accel_file.close()

print('Data has been successfully written to accelerometer.txt and gyroscope.txt file')

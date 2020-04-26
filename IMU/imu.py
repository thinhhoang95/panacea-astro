import sys, getopt

sys.path.append('.')
import RTIMU
import os.path
import time
import math
import operator
import socket
import os

IMU_IP = "127.0.0.2"
IMU_PORT = 5005

MON_IP = "127.0.0.5"
MON_PORT = 5005

SETTINGS_FILE = "RTIMULib"

s = RTIMU.Settings(SETTINGS_FILE)
imu = RTIMU.RTIMU(s)

# offsets
yawoff =152.0
pitchoff = 0.0
rolloff = 0.0

# timers
t_print = time.time()
t_damp = time.time()
t_fail = time.time()
t_fail_timer = 0.0
t_shutdown = 0

if (not imu.IMUInit()):
    hack = time.time()
    imu_sentence = "$IIXDR,IMU_FAILED_TO_INITIALIZE*7C"
    if (hack - t_print) > 1.0:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(imu_sentence, (IMU_IP, IMU_PORT))
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

# magnetic deviation

magnetic_deviation = -13.7

# dampening variables
t_one = 0
t_three = 0
roll_total = 0.0
roll_run = [0] * 10
heading_cos_total = 0.0
heading_sin_total = 0.0
heading_cos_run = [0] * 30
heading_sin_run = [0] * 30

while True:

  hack = time.time()

  # if it's been longer than 5 seconds since last print
  if (hack - t_damp) > 5.0:
      
      if (hack - t_fail) > 1.0:
          t_one = 0
          t_three = 0
          roll_total = 0.0
          roll_run = [0] * 10
          heading_cos_total = 0.0
          heading_sin_total = 0.0
          heading_cos_run = [0] * 30
          heading_sin_run = [0] * 30
          t_fail_timer += 1
          imu_sentence = "IIXDR,IMU_FAIL," + str(round(t_fail_timer / 60, 1))
          cs = format(reduce(operator.xor,map(ord,imu_sentence),0),'X')
          if len(cs) == 1:
                cs = "0" + cs
          imu_sentence = "$" + imu_sentence + "*" + cs
          sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
          sock.sendto(imu_sentence, (IMU_IP, IMU_PORT))
          t_fail = hack
          t_shutdown += 1

  if imu.IMURead():
    data = imu.getIMUData()
    fusionPose = data["fusionPose"]
    Gyro = data["gyro"]
    Accel = data["accel"]
    t_fail_timer = 0.0

    if (hack - t_damp) > .01:
        # print('Clock: ' + str(hack))
        roll = round(math.degrees(fusionPose[0]) - rolloff, 4)
        pitch = round(math.degrees(fusionPose[1]) - pitchoff, 4)
        yaw = round(math.degrees(fusionPose[2])- yawoff, 4)
        rollrate = round(math.degrees(Gyro[0]), 4)
        pitchrate = round(math.degrees(Gyro[1]), 4)
        yawrate = round(math.degrees(Gyro[2]), 4)
	if yaw < 0.1:
                    yaw = yaw + 360
	if yaw > 360:
	    yaw = yaw - 360
    
        # Dampening functions
        roll_total = roll_total - roll_run[t_one]
        roll_run[t_one] = roll
        roll_total = roll_total + roll_run[t_one]
        roll = round(roll_total / 10, 4)
        heading_cos_total = heading_cos_total - heading_cos_run[t_three]
        heading_sin_total = heading_sin_total - heading_sin_run[t_three]
        heading_cos_run[t_three] = math.cos(math.radians(yaw))
        heading_sin_run[t_three] = math.sin(math.radians(yaw))
        heading_cos_total = heading_cos_total + heading_cos_run[t_three]
        heading_sin_total = heading_sin_total + heading_sin_run[t_three]
        yaw = round(math.degrees(math.atan2(heading_sin_total/30,heading_cos_total/30)),4)
        if yaw < 0.1:
            yaw = yaw + 360.0

        # yaw is magnetic heading, convert to true heading
        heading = yaw - magnetic_deviation
        if heading < 0.1:
            heading = heading + 360
	if heading > 360:
	    heading = heading - 360

        t_damp = hack
        t_one += 1
        if t_one == 10:
            t_one = 0
        t_three += 1
        if t_three == 30:
            t_three = 0
  
        if (hack - t_print) > .01:
            
            # health monitor
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(str(hack), (MON_IP, MON_PORT))

            # iihdm magnetic heading
            hdm = "IIHDM," + str(round(yaw))[:-2] + ",M"
            hdmcs = format(reduce(operator.xor,map(ord,hdm),0),'X')
            if len(hdmcs) == 1:
                hdmcs = "0" + hdmcs
            iihdm = "$" + hdm + "*" + hdmcs

            # iihdt true heading
            hdt = "IIHDT," + str(round(heading))[:-2] + ",T"
            hdtcs = format(reduce(operator.xor,map(ord,hdt),0),'X')
            if len(hdtcs) == 1:
                hdtcs = "0" + hdtcs
            iihdt = "$" + hdt + "*" + hdtcs

	    # iixdr ahrs data
            xdr = "IIXDR,A," + str(int(round(roll))) + ",D,ROLL,A,"  + str(int(round(pitch))) + ",D,PTCH,A"
            xdrcs = format(reduce(operator.xor,map(ord,xdr),0),'X')
            if len(xdrcs) == 1:
                xdrcs = "0" + xdrcs
            iixdr = "$" + xdr + "*" + xdrcs

	    # tirot rate of turn
	    rot = "TIROT," + str(yawrate*60) + ",A"
	    rotcs = format(reduce(operator.xor,map(ord,rot),0),'X')
            if len(rotcs) == 1:
                rotcs = "0" + rotcs
            tirot = "$" + rot + "*" + rotcs

            # assemble the sentence
            imu_sentence = iihdm + '\r\n' + iihdt + '\r\n' + iixdr + '\r\n' + tirot + '\r\n'

            # to imu bus
            f = open('imu_bus', 'w')
            f.write(str(t_print) + ',' + str(heading) + ',' + str(roll)  + ',' + str(pitch))
            f.close()
            
            print('at '+ str(t_print-1500000000) + '  HDG: ' + str(math.radians(heading)) + ' PTC: ' + str(math.radians(pitch)) + ' ROL: ' + str(math.radians(roll)) + '\r\n')
            #print('            AccelX: ' + str(Accel[0]) +' AccelY: ' + str(Accel[1]) + ' AccelZ: ' + str(Accel[2]))
            #to eulang
            eulang = open('eulang.txt', 'a')
            eulang.write(str(t_print-1500000000)+','+str(math.radians(heading)) + ',' + str(math.radians(pitch))  + ',' + str(math.radians(roll)) + '\r\n')
            eulang.close
            #to accel
            accel = open('accel.txt', 'a')
            accel.write(str(t_print-1500000000)+','+str(Accel[0]) + ',' + str(Accel[1])  + ',' + str(Accel[2]) + '\r\n')
            accel.close
            # To kplex
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(imu_sentence, (IMU_IP, IMU_PORT))

            t_print = hack
    time.sleep(poll_interval*1.0/1000.0)
    

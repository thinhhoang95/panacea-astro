'''
Prepare the residual elements for lsqnonlin in MATLAB
Calibration of IMU's accelerometer precisely through camera
'''

import numpy as np
from scipy.spatial.transform import Rotation
import glob
import cv2 as cv
import os
import scipy.io as io

def get_tag_pos(tag_id):
    if tag_id==0:
        return np.array([0,2,4])
    elif tag_id==1:
        return np.array([0,3,6])
    elif tag_id==2:
        return np.array([0,4,8])
    else:
        return None

run_directories = [x[0] for x in os.walk('.')]

for dir in run_directories:

    print('Now processing directory ', dir)

    accel_file = np.genfromtxt(dir + '/inertial/accelerometer.txt', delimiter=',')
    eulang_file = np.genfromtxt(dir + '/inertial/eulang.txt', delimiter=',')
    image_path = glob.glob(dir + '/images/*.jpg')

    print('This directory has {:d} IMU entries and {:d} camera samples'.format(np.shape(accel_file)[0], len(image_path)))

    image_info = []
    # Convert image name timestamp to a numpy vector
    for img_path in image_path:
        image_info.append((img_path[17:len(img_path)-4],img_path))
    image_info.sort(key=lambda tup: tup[0])
    image_path_cursor = 0
    last_time = 0 # to calculate delta_t between timesteps
    # IMU integration matrices
    B_overline = np.eye(6)
    # Initial position of the device: processed from the first frame observation of the tags
    x_initial = np.empty((0,3))
    # Variables to hold components of measurements
    Rbi_stack = np.empty((0,3,3))
    A_overline_stack = np.empty((0,6,6))
    B_overline_stack = np.empty((0,6,6))
    rhs_stack = np.empty((0,3))
    x_init_stack = np.empty((0,3))
    weight_stack = np.empty(0)
    clock_started = 0 # record the first image capture time, so that variance of measurement with time (~T^2) is estimated

    # Processing of IMU measurements
    for k in range(np.shape(accel_file)[0]):
        time = accel_file[k,0]
        if last_time == 0 or time < image_info[0][0]:
            last_time = time # do nothing until first image is processed
        else: # from second record onward
            T = time-last_time
            last_time = time
            A_mat = np.array([[1,0,0,T,0,0],[0,1,0,0,T,0],[0,0,1,0,0,T],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
            B_overline = B_overline @ A_mat
            if time > float(image_info[image_path_cursor][0]):
                # Detect ARUCOs in the imag
                # Load the image
                image_file = image_info[image_path_cursor][1]
                frame = cv.imread(image_file)
                #Load the dictionary that was used to generate the markers.
                dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
                # Initialize the detector parameters using default values
                parameters =  cv.aruco.DetectorParameters_create()
                # Load the camera matrix and distortion from file
                cam_mat = np.load('cam_mat.pca.npy')
                dist = np.load('dist.pca.npy')
                # Detect the markers in the image
                markerCorners, markerIds, rejectedCandidates = cv.aruco.detectMarkers(frame, dictionary, parameters=parameters)
                rvecs, tvecs, *other = cv.aruco.estimatePoseSingleMarkers(markerCorners, 0.05, cam_mat, dist)
                if image_path_cursor == 0:
                    # TODO: First frame is used for determination of the initial condition
                    x0_stack = np.empty((0,6))
                    clock_started = time
                    for tag_id, rvec, tvec in zip(markerIds, rvecs, tvecs):
                        xt = get_tag_pos(tag_id)
                        Rtc = Rotation.from_rotvec(rvec).as_dcm()
                        x0 = np.concatenate((x0, Rtc @ xt - tvec))
                        print('Initial position obtained: ', x0)
                    x_initial = np.concatenate((np.average(x0, axis=0),np.zeros(3)))
                    continue # done with the first image
                image_sample = k
                # Calculation of A_overline
                A_overline = np.eye(6)
                ypr = eulang_file[k,1:]
                Rbi = Rotation.from_euler('ZYX', ypr).as_dcm()
                for kp in range(image_sample): # sum over kp
                    Acc = np.eye(6)
                    if kp+1 > image_sample:
                        pass
                    else:
                        for i in range(kp+1, image_sample):
                            Acc = Acc @ A_mat
                    B_mat_1 = np.array([[0.5*(T**2),0,0],[0,0.0*(T**2),0],[0,0,0.5*(T**2)],[T,0,0],[0,T,0],[0,0,T]]) # acceleration integration
                    # B_mat_2 = ...
                    # B_mat_3 = np.array([[self.theta[0],0,0,-self.theta[3],0,0],[0,self.theta[1],0,0,self.theta[4],0],[0,0,self.theta[2],0,0,self.theta[5]]])
                    # B_mat_3 = np.array([[self.X[self.now_pointer,6],0,0,-self.X[self.now_pointer,9]],[0,self.X[self.now_pointer,7],0,-self.X[self.now_pointer,10]],[0,0,self.X[self.now_pointer,8],-self.X[self.now_pointer,11]]])
                    a = -accel_file[k,1:]
                    ai = (Rbi @ a + np.array([0,0,1])) * 9.78206 # acceleration in the inertial frame
                    a_x = ai[0]
                    a_y = ai[1]
                    a_z = ai[2]
                    a_circle_k = np.concatenate(np.array(np.array([[a_x,a_y,a_z,0,0,0],[0,0,0,a_y,a_z,0],[0,0,0,0,0,a_x]]), np.eye(3)))
                    A_overline = A_overline + Acc @ B_mat_1 @ a_circle_k
                # A_overline and B_overline is available, we proceed onto establishment of the measurement model for each ARUCO observation
                for tag_id, rvec, tvec in zip(markerIds, rvecs, tvecs):
                    Rbi_stack = np.concatenate((Rbi_stack, Rbi[None,...].copy()), axis=0)
                    A_overline_stack = np.concatenate((A_overline_stack, A_overline[None,...].copy()), axis=0)
                    B_overline_stack = np.concatenate((B_overline_stack, B_overline[None,...].copy()), axis=0)
                    xt = get_tag_pos(tag_id)
                    Rtc = Rotation.from_rotvec(rvec).as_dcm()
                    rhs = -Rtc @ xt + tvec
                    rhs_stack = np.concatenate((rhs_stack, rhs.copy()), axis=0)
                    x_init_stack = np.concatenate((x_init_stack, x_initial), axis=0)
                    weight_stack = np.concatenate((weight_stack,(time - clock_started)**2))
                image_path_cursor = image_path_cursor + 1

# Export the matrices to MATLAB filetype
io.savemat('top.mat', {'Aoverline': A_overline_stack, 'Boverline': B_overline_stack, 'rhs': rhs_stack, 'xinit': x_init_stack, 'weight': weight_stack})
print('File top.mat has been successfully saved. Now run the MATLAB script to optimize!')
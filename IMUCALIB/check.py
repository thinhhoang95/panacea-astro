'''
Prepare the residual elements for lsqnonlin in MATLAB
Calibration of IMU's accelerometer precisely through camera
'''

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.linalg import block_diag
import glob
import cv2 as cv
import os
import scipy.io as io

def get_tag_pos(tag_id):
    if tag_id==0:
        return np.array([0,0,0])
    elif tag_id==1:
        return np.array([0.09,0,0])
    elif tag_id==2:
        return np.array([0.18,0,0])
    elif tag_id==3:
        return np.array([0,-0.092,0])
    elif tag_id==4:
        return np.array([0.09,-0.092,0])
    elif tag_id==5:
        return np.array([0.18,-0.092,0])
    else:
        return None

# flags
each_frame_as_initial_flag = True # if False, the first frame of the run will serve as initial position, otherwise, the last frame

run_directories = ['run_'+str(i) for i in range(10)]

# Variables to hold components of measurements
Rbi_stack = np.empty((0,3,3))
A_overline_stack = np.empty((0,6,9))
a_overline_stack = np.empty((0,6,3))
B_overline_stack = np.empty((0,6,6))
Rtc_stack = np.empty((0,3,3))
rhs_stack = np.empty((0,3))
x_init_stack = np.empty((0,6)) # initial position from beginning of run
x_init_fstack = np.empty((0,6)) # initial position from last frame obtained
xt_stack = np.empty((0,3))
weight_stack = np.empty(0)
T_stack = np.empty(0)
# debug stacks
ai_stack_dbg = np.empty((0,3))
x_stack_dbg = np.empty((0,6))
x_dbg = np.zeros(6)

# Useful constants
Ritip = np.array([[1,0,0],[0,-1,0],[0,0,-1]]) # flip y and z axis

for dir in run_directories:

    print('Now processing directory ', dir)

    accel_file = np.genfromtxt(dir + '/inertial/accelerometer.txt', delimiter=',')
    eulang_file = np.genfromtxt(dir + '/inertial/eulang.txt', delimiter=',')
    image_path = glob.glob(dir + '/images/*.jpg')

    print('This directory has {:d} IMU entries and {:d} camera samples'.format(np.shape(accel_file)[0], len(image_path)))

    image_info = []
    # Convert image name timestamp to a numpy vector
    for img_path in image_path:
        image_info.append((img_path[23:len(img_path)-4],img_path))
    image_info.sort(key=lambda tup: tup[0])
    image_path_cursor = 0
    last_time = 0 # to calculate delta_t between timesteps
    # IMU integration matrices
    B_overline = np.eye(6)
    # Initial position of the device: processed from the first frame observation of the tags
    x_initial = np.empty((0,3))
    clock_started = 0 # record the first image capture time, so that variance of measurement with time (~T^2) is estimated
    # Mark the k which first image is available (true start of the process)
    k_start = 0
    x_hat = np.zeros(6)

    # Processing of IMU measurements
    for k in range(np.shape(accel_file)[0]):
        time = accel_file[k,0]
        if last_time == 0 or time < float(image_info[0][0]):
            T_stack = np.concatenate((T_stack, [0.01]))
            last_time = time # do nothing until first image is processed
            k_start = k
        else: # from second record onward
            T = time-last_time
            T_stack = np.concatenate((T_stack, [T]))
            last_time = time
            A_mat = np.array([[1,0,0,T,0,0],[0,1,0,0,T,0],[0,0,1,0,0,T],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
            B_mat_1 = np.array([[0.5*(T**2),0,0],[0,0.5*(T**2),0],[0,0,0.5*(T**2)],[T,0,0],[0,T,0],[0,0,T]]) # acceleration integration
            B_overline = B_overline @ A_mat
            print('IMU sample ', k)
            
            
            if image_path_cursor < len(image_info) and time > float(image_info[image_path_cursor][0]): # upon hitting an image
                print('Now processing image', image_info[image_path_cursor][1])
                # Detect ARUCOs in the imag
                # Load the image
                image_file = image_info[image_path_cursor][1]
                frame = cv.imread(image_file)
                image_draw = frame.copy()
                #Load the dictionary that was used to generate the markers.
                dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
                # Initialize the detector parameters using default values
                parameters =  cv.aruco.DetectorParameters_create()
                # Load the camera matrix and distortion from file
                cam_mat = np.load('cam_mat.pca.npy')
                dist = np.load('dist.pca.npy')
                # Detect the markers in the image
                markerCorners, markerIds, rejectedCandidates = cv.aruco.detectMarkers(frame, dictionary, parameters=parameters)
                rvecs, tvecs, *other = cv.aruco.estimatePoseSingleMarkers(markerCorners, 0.086, cam_mat, dist)
                if image_path_cursor == 0:
                    # TODO: First frame is used for determination of the initial condition
                    x0_stack = np.empty((0,6))
                    clock_started = time
                    for tag_id, rvec, tvec in zip(markerIds, rvecs, tvecs):
                        xt = get_tag_pos(tag_id[0])
                        Rtc = Rotation.from_rotvec(rvec[0]).as_matrix()
                        x0 = np.concatenate((Ritip @ (xt + Rtc.T @ -tvec[0]), np.array([0,0,0])))
                        x0_stack = np.concatenate((x0_stack, [x0]), axis=0)
                        print('Initial position obtained: ', x0[0:3])
                        # cv.aruco.drawAxis(image_draw, cam_mat, dist, rvec[0], tvec[0], 0.15)
                    # cv.imshow('Image '+image_info[image_path_cursor][1], image_draw)
                    # cv.waitKey(30000)
                    # cv.destroyAllWindows()
                    Ripi = Rotation.from_euler('ZYX', np.array([-0.0678,0,0])).as_matrix() # I' (tag's z down) -> I (north aligned)
                    x_initial = block_diag(Ripi, Ripi) @ np.average(x0_stack, axis=0) # TODO: delete or comment out this 
                    x_dbg = x_initial.copy()
                    B_overline = np.eye(6) # reset acceleration integrators upon reaching the first image
                    image_path_cursor = image_path_cursor + 1
                    continue # done with the first image
                image_sample = k
                # Calculation of A_overline => 
                A_overline = np.zeros((6,9))
                print(k)
                a_overline = np.zeros((6,3))
                ypr = eulang_file[k,1:]
                
                for kp in range(k_start, image_sample): # kp = 0 to N-1
                    # print('kp: ', kp)
                    Acc = np.eye(6)
                    if kp+1 > image_sample-1:
                        pass
                    else:
                        for i in range(kp+1, image_sample): # i = kp+1 to N-1
                            # print('i: ', i)
                            Am = np.array([[1,0,0,T_stack[i],0,0],[0,1,0,0,T_stack[i],0],[0,0,1,0,0,T_stack[i]],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
                            Acc = Acc @ Am # actually A_mat changes with T
                    Bm = np.array([[0.5*(T_stack[kp]**2),0,0],[0,0.5*(T_stack[kp]**2),0],[0,0,0.5*(T_stack[kp]**2)],[T_stack[kp],0,0],[0,T_stack[kp],0],[0,0,T_stack[kp]]]) # acceleration integration
                    # B_mat_2 = ...
                    # B_mat_3 = np.array([[self.theta[0],0,0,-self.theta[3],0,0],[0,self.theta[1],0,0,self.theta[4],0],[0,0,self.theta[2],0,0,self.theta[5]]])
                    # B_mat_3 = np.array([[self.X[self.now_pointer,6],0,0,-self.X[self.now_pointer,9]],[0,self.X[self.now_pointer,7],0,-self.X[self.now_pointer,10]],[0,0,self.X[self.now_pointer,8],-self.X[self.now_pointer,11]]])
                    a = -accel_file[kp,1:]
                    ypr = eulang_file[kp,1:]
                    Rbi = Rotation.from_euler('ZYX', ypr).as_matrix()
                    ai = (Rbi @ a + np.array([0,0,1])) * 9.78206 # acceleration in the inertial frame
                    a_x = a[0]*9.78206
                    a_y = a[1]*9.78206
                    a_z = a[2]*9.78206
                    a_circle_k = np.concatenate((np.array([[a_x,a_y,a_z,0,0,0],[0,0,0,a_y,a_z,0],[0,0,0,0,0,a_z]]), np.eye(3)), axis=1)
                    A_overline = A_overline + Acc @ Bm @ Rbi @ a_circle_k
                    a_overline = a_overline + Acc @ Bm
                
                
                # A_overline and B_overline is available, we proceed onto establishment of the measurement model for each ARUCO observation
                if markerIds is None or rvecs is None or tvecs is None:
                    print('This image does contain a ARUCO tag')
                    image_path_cursor = image_path_cursor + 1
                    continue
                print('Found {:d} ARUCO tags'.format(len(markerIds)))
                cam_pos_estimate = np.empty((0,6))
                for tag_id, rvec, tvec in zip(markerIds, rvecs, tvecs):
                    Rbi_stack = np.concatenate((Rbi_stack, Rbi[None,...].copy()), axis=0)
                    A_overline_stack = np.concatenate((A_overline_stack, A_overline[None,...].copy()), axis=0)
                    B_overline_stack = np.concatenate((B_overline_stack, B_overline[None,...].copy()), axis=0)
                    a_overline_stack = np.concatenate((a_overline_stack, a_overline[None,...].copy()), axis=0)
                    xt = get_tag_pos(tag_id[0])
                    Rtc = Rotation.from_rotvec(rvec[0]).as_matrix()
                    Rtc_stack = np.concatenate((Rtc_stack, Rtc[None,...].copy()), axis=0)
                    # rhs = -Rtc @ xt + tvec
                    rhs = -tvec[0].copy()
                    rhs_stack = np.concatenate((rhs_stack, rhs[None,...].copy()), axis=0)
                    x_init_stack = np.concatenate((x_init_stack, [x_initial.copy()]), axis=0)
                    weight_stack = np.concatenate((weight_stack,[(time - clock_started)]))
                    xt_stack = np.concatenate((xt_stack, [xt.copy()]))
                    # Draw the markers
                    cv.aruco.drawAxis(image_draw, cam_mat, dist, rvec[0], tvec[0], 0.15)
                    cam_pos = np.concatenate((Ritip @ (xt + Rtc.T @ -tvec[0]), np.array([0,0,0])))
                    cam_pos_estimate = np.concatenate((cam_pos_estimate, [cam_pos]), axis=0)
                    # cv.putText(image_draw, str(cam_pos[0:3]), (10,10*np.shape(cam_pos_estimate)[0]), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255,255), 1)
                cam_pos_average = np.average(cam_pos_estimate, axis=0)
                if each_frame_as_initial_flag: 
                    # HERE!!! ===>
                    Ripi = Rotation.from_euler('ZYX', np.array([0,0,0])).as_matrix() # I' (tag's z down) -> I (north aligned)
                    # HERE!! <====
                    x_initial = block_diag(Ripi, Ripi) @ cam_pos_average.copy() # TODO: delete or comment out this 
                    B_overline = np.eye(6) # reset B_overline
                    print('Initial has been reset to ', x_initial)
                    k_start = k # reset A_overline
                ai_stack_dbg = np.concatenate((ai_stack_dbg, [ai]), axis=0)
                # x_hat = A_overline @ np.array([0.8894,0.0297,-0.1052,0.9500,-0.0847,0.9028,-1.0043,-0.7695,-0.7234]).T + a_overline @ np.array([0,0,9.78206]).T + B_overline @ x_initial.T
                # REMEMBER TO CHANGE THE VALUE IN Ripi ABOVE!
                x_hat = A_overline @ np.array([1,0,0,1,0,1,0,0,0]).T + a_overline @ np.array([0,0,9.78206]).T + B_overline @ x_initial.T
                #print('x_hat: ', x_hat)
                #print('x_dbg: ', x_dbg)
                cv.putText(image_draw, str('Euclidean Distance: ' + str(np.linalg.norm(x_hat[0:3] - x_initial[0:3]))), (10,10), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255,255), 1)
                # cv.putText(image_draw, str('x_dbg: ' + str(x_dbg)), (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255,255), 1)
                cv.putText(image_draw, str('x_init (reset): ' + str(x_initial)), (10,20), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255,255), 1)
                cv.putText(image_draw, str('x_init (reset): ' + str(x_initial)), (10,20), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255,255), 1)
                
                cv.imshow('Image '+image_info[image_path_cursor][1], image_draw)
                cv.waitKey(1000)
                cv.destroyAllWindows()
                image_path_cursor = image_path_cursor + 1
            
            # Calculation of "naive" double integration
            ypr = eulang_file[k,1:]
            a = -accel_file[k,1:]
            Rbi = Rotation.from_euler('ZYX', ypr).as_matrix()
            ai = (Rbi @ a + np.array([0,0,1])) * 9.78206 # acceleration in the inertial frame
            x_dbg = A_mat @ x_dbg + B_mat_1 @ ai
            #print('HAT: ', x_hat)
            #print('DBG: ', x_dbg)
            x_stack_dbg = np.concatenate((x_stack_dbg, [x_dbg]), axis=0)
            

# Export the matrices to MATLAB filetype
#io.savemat('top.mat', {'Aoverline': A_overline_stack, 'Boverline': B_overline_stack, 'rhs': rhs_stack, 'xinit': x_init_stack, 'weight': weight_stack, 'Rbi': Rbi_stack, 'xt': xt_stack, 'aoverline': a_overline_stack, 'Rtc': Rtc_stack, 'x_dbg': x_stack_dbg})
#print('File top.mat has been successfully saved. Now run the MATLAB script to optimize!')
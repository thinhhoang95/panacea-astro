import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation
import time
import glob


for run in range(10):

    image_path = glob.glob('run_' + str(run) + '/*.jpg')
    image_info = []
    image_pointer = 0

    for img_path in image_path:
        image_info.append((img_path[17:len(img_path)-4],img_path))
    image_info.sort(key=lambda tup: tup[0])

    optimization_file = np.empty((0,6))
    i=0
    eulang_file = np.genfromtxt('eulang.txt', delimiter=',')
    accel_file = np.genfromtxt('accelerometer.txt', delimiter=',')

    t_old = 0
    A_overline = np.eye(6)
    B_overline = np.eye(6)
    
    for image_file in image_list:
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


        output_img = np.copy(frame)
        rvec_stack = np.empty((0,3))
        if rvecs is None:
            i = i+1
            continue
        for rvec, tvec in zip(rvecs, tvecs):
            # cv.aruco.drawAxis(output_img, cam_mat, dist, rvec, tvec, 0.05)
            print(rvec)
            if rvec[0,0]<0:
                rvec = -rvec
            print('Individual rvec: ', rvec)
            rvec_stack = np.append(rvec_stack, rvec, axis=0)
            optimization_file_row = np.hstack((rvec[0],eulang_file[i,1:4]))
            print('Optimization row: ', optimization_file_row)
            optimization_file = np.vstack((optimization_file, np.hstack((rvec[0],eulang_file[i,1:4]))))
        #print('rvec_stack: ', rvec_stack)
        #rvec_mean = np.mean(rvec_stack, axis=0)
        #print('rvec mean: ', rvec_mean)
        #cv.aruco.drawAxis(output_img, cam_mat, dist, rvec_mean, tvec[0], 0.15)
        #optimization_file[i,:] = np.hstack((rvec_mean,eulang_file[i,1:4]))
        i = i+1
        if i<10:
            cv.imshow(image_file, output_img)
            cv.waitKey(0)

    np.savetxt('optimize.txt', optimization_file, delimiter=',')
    print('Optimization file has been successfully saved')
        # cv.imshow('Output '+image_file, output_img)
        # cv.waitKey()
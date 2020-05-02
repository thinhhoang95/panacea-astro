import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation
import time
import math

image_list = ['img_'+str(i)+'.jpg' for i in range(38)]

optimization_file = np.zeros((38,6))
i=0
eulang_file = np.genfromtxt('eulang.txt', delimiter=',')

cam_mat = np.load('cam_mat.pca.npy')
dist = np.load('dist.pca.npy')

theta = [2.9339,    0.4031,   -0.0441,    2.2710]
# theta = [6.5034,0.7350]

R1 = Rotation.from_rotvec(theta[0:3]).as_dcm()

RITPT = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
RITPTZ = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
RITO = np.array([[math.cos(theta[0]), -math.sin(theta[0]), 0],[math.sin(theta[0]), math.cos(theta[0]), 0],[0,0,1]])
RIBC = np.array([[math.cos(theta[3]), -math.sin(theta[3]), 0],[math.sin(theta[3]), math.cos(theta[3]), 0],[0,0,1]])

i=0
for image_file in image_list:
    frame = cv.imread(image_file)
    dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
    parameters =  cv.aruco.DetectorParameters_create()
    markerCorners, markerIds, rejectedCandidates = cv.aruco.detectMarkers(frame, dictionary, parameters=parameters)
    rvecs, tvecs, *other = cv.aruco.estimatePoseSingleMarkers(markerCorners, 0.05, cam_mat, dist)
    if rvecs is None:
        i = i + 1
        continue
    output_img = np.copy(frame)
    RIOB = Rotation.from_euler('ZYX',eulang_file[i,1:4]).as_dcm()
    R = R1 @ RIOB @ RIBC
    print('R1 ', R1)
    print('RIOB ', RIOB)
    print('RIBC ', RIBC)
    rvec_mean_reconstructed,_ = cv.Rodrigues(R)
    rvec_stack = np.empty((0,3))
    for rvec, tvec in zip(rvecs, tvecs):
        # cv.aruco.drawAxis(output_img, cam_mat, dist, rvec, tvec, 0.05)
        # print(rvec)
        if rvec[0,0]<0:
            rvec = -rvec
        print('rvec: ', rvec, ', angle: ', np.linalg.norm(rvec)/math.pi*180)
        rvec_stack = np.append(rvec_stack, rvec, axis=0)
    print('rvec_stack: ', rvec_stack)
    rvec_mean = np.mean(rvec_stack, axis=0)

    R_prime, _ = cv.Rodrigues(rvec_mean)
    print('R_prime ', R_prime)
    # raise Exception('quit!')
    diff_angle_mat = R_prime.T @ R
    diff_angle_rotvec, _ = cv.Rodrigues(diff_angle_mat)
    print('axang ', diff_angle_rotvec)
    print('File ', image_file, ' angle difference is ', math.degrees(np.linalg.norm(diff_angle_rotvec)))
    cv.aruco.drawAxis(output_img, cam_mat, dist, rvec_mean_reconstructed, tvecs[0], 0.15)
    cv.aruco.drawAxis(output_img, cam_mat, dist, rvec_mean, tvecs[0], 0.15)
    i = i+1
    # cv.imshow(image_file, output_img)
    # cv.waitKey(0)
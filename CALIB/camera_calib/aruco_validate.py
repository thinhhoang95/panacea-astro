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

theta = [30*math.pi/180, 1.570796]
# theta = [6.5034,0.7350]

RITPT = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
RITPTZ = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
RITO = np.array([[math.cos(theta[0]), -math.sin(theta[0]), 0],[math.sin(theta[0]), math.cos(theta[0]), 0],[0,0,1]])
RIBC = np.array([[math.cos(theta[1]), -math.sin(theta[1]), 0],[math.sin(theta[1]), math.cos(theta[1]), 0],[0,0,1]])

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
    # R = RITPT @ RITO @ RIOB @ RIBC
    R = RITPT @ RITO @ RIOB @ RIBC
    # print(RIOB)
    # raise Exception('quit!')
    rvec_mean,_ = cv.Rodrigues(R)
    print('File ', image_file, ' rvec_mean is ', rvec_mean)
    print('Meanwhile, rvec1 is ', rvecs[0])
    R_prime, _ = cv.Rodrigues(rvecs[0])
    diff_angle_mat = R_prime.T @ R
    diff_angle_rotvec, _ = cv.Rodrigues(diff_angle_mat)
    print('axang ', diff_angle_rotvec)
    print('File ', image_file, ' angle difference is ', math.degrees(np.linalg.norm(diff_angle_rotvec)))
    
    cv.aruco.drawAxis(output_img, cam_mat, dist, rvec_mean, tvecs[0], 0.15)
    cv.aruco.drawAxis(output_img, cam_mat, dist, rvecs[0], tvecs[0], 0.05)
    i = i+1
    #cv.imshow(image_file, output_img)
    #cv.waitKey(0)
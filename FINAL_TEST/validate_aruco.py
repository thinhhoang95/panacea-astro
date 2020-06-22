import cv2 as cv
import numpy as np
import time
import glob
import math
from scipy.spatial.transform import Rotation

image_path = glob.glob('images/*.jpg')
image_info = []
# Convert image name timestamp to a numpy vector
for img_path in image_path:
    image_info.append((img_path[17:len(img_path)-4],img_path))
image_info.sort(key=lambda tup: tup[0])
image_path_cursor = 0

total_image_files = len(image_info)

camera_pos_stack = np.empty((0,4))

i=0

psi1_angle = math.radians(26) # 26 degrees between tag's X and true north
tag_position_offset = np.array([0,0,0]) # position of tag

for image_file_no in range(total_image_files):
    image_file = image_path[image_file_no]

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
    rvecs, tvecs, *other = cv.aruco.estimatePoseSingleMarkers(markerCorners, 0.189, cam_mat, dist)

    if rvecs is None:
        continue

    for rvec, tvec, markerId in zip(rvecs, tvecs, markerIds):
        if markerId == 0:
            psi_mat = Rotation.from_euler('ZYX', [psi1_angle, 0, 0], degrees=False).as_dcm()
            rvec_mat = Rotation.from_rotvec(rvec).as_dcm().T
            flip_mat = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
            camera_pos = psi_mat @ flip_mat @ (tag_position_offset.T + rvec_mat @ (-tvec.T))
            tme = float(image_info[image_file_no])
            camera_pos_stack_row = np.concatenate((tme,camera_pos))
            camera_pos_stack = np.concatenate((camera_pos_stack, camera_pos_stack_row), axis=0)
            print('Time {:.f} camera position is '.format(tme), camera_pos)
    i = i+1

np.savetxt('truth.txt', camera_pos_stack, delimiter=',')
print('Ground truth values have been successfully saved to file truth.txt')
print('Program will now exit')
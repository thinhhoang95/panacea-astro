import cv2 as cv
import numpy as np

# Load the predefined dictionary
dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)

aruco_id = [21,22,23,24,25]

for id in aruco_id:
    print('Generating aruco tag for ID ', id)
    markerImage = np.zeros((400, 400), dtype=np.uint8)
    markerImage = cv.aruco.drawMarker(dictionary, id, 400, markerImage, 1);
    cv.imwrite('marker{:02d}.png'.format(id), markerImage)
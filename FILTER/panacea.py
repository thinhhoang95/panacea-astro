'''
PANACEA Data Broker (PDM)
Author: Hoang Dinh Thinh, Dept. Aerospace Engineering, Univ. Corruption
Email: hdinhthinh@gmail.com
'''

from FILTER.panacea_inertial import PanaceaInertial
from FILTER.panacea_lk_flow import PanaceaLKFlow
import numpy as np
import glob

def main():
    filter = PanaceaInertial()
    accel_data = np.genfromtxt('inertial/accel.txt', delimiter=',')
    ypr_data = np.genfromtxt('inertial/eulang.txt', delimiter=',')
    image_path = glob.glob('images/*.jpg')
    lk_flow = PanaceaLKFlow(None, None)
    # Convert image name timestamp to a numpy vector
    for i, img_path in enumerate(image_path):
        image_path[i] = img_path[25:len(img_path)-4]
    image_path.sort()
    image_path_cursor = 0
    last_time = 0 # to calculate delta_t between timesteps
    img0_path = ''
    img1_path = ''
    for accel_row, ypr_row in zip(accel_data, ypr_data):
        time = accel_row[0]
        if last_time!=0:
            accel = accel_row[1:4]
            ypr = ypr_row[1:4]
            filter.imu_propagate(accel, ypr, time - last_time)
            # upon reaching the moment with an image captured
            if time > float(image_path[image_path_cursor]):
                if not img0_path:
                    img0_path = 'FILTER/images/' + image_path[image_path_cursor] + '.jpg'
                else:
                    img1_path = 'FILTER/images/' + image_path[image_path_cursor] + '.jpg'
                    # Both images are available, perform filtering!
                    lk_flow.set_frame(img0_path, img1_path)
                    flow_tracks = lk_flow.calculate() # calculate the optical flow between two images
                    filter.cam_correction(flow_tracks) # perform all Panacea filtering and window shifting
                    img0_path = img1_path
                    img1_path = ''
                image_path_cursor = image_path_cursor + 1
            last_time = time
        else:
            last_time = time # skip the first IMU measurement to calculate delta_t

if __name__=='main':
    main()
'''
PANACEA Data Broker (PDM)
Author: Hoang Dinh Thinh, Dept. Aerospace Engineering, Univ. Corruption
Email: hdinhthinh@gmail.com
'''

from panacea_inertial_3ds import PanaceaInertial3DS
from panacea_lk_flow_mss import PanaceaLKFlowMSS
import numpy as np
import glob
from matplotlib import pyplot as plt

def main():
    filter = PanaceaInertial3DS()
    accel_data = np.genfromtxt('inertial/accel.txt', delimiter=',')
    ypr_data = np.genfromtxt('inertial/eulang.txt', delimiter=',')
    # image_path = glob.glob('images/*0004*.jpg') + glob.glob('images/*0008*.jpg')
    image_path = glob.glob('images/*.jpg')
    image_info = []
    lk_flow = PanaceaLKFlowMSS(None, None)
    # Convert image name timestamp to a numpy vector
    for img_path in image_path:
        image_info.append((img_path[17:len(img_path)-4],img_path))
    image_info.sort(key=lambda tup: tup[0])
    image_path_cursor = 0
    last_time = 0 # to calculate delta_t between timesteps
    img0_path = ''
    img1_path = ''
    # log and plot
    t_log = np.empty((0))
    x_log = np.empty((0,3))
    t_first = 0
    frame_skip_count = 1
    frame_skip = 0 # skip 1 image frame
    for accel_row, ypr_row in zip(accel_data, ypr_data):
        time = accel_row[0]
        if (t_first == 0):
            t_first = time
        if last_time!=0:
            accel = accel_row[1:4]
            ypr = ypr_row[1:4]
            # print('Clock hit {:.3f}'.format(time))
            # print('Position: ', filter.X[filter.now_pointer, 0:3])
            x_log = np.vstack((x_log, filter.X[filter.now_pointer, 0:3]))
            t_log = np.append(t_log, time - t_first)
            # print('IMU Calibration: ', filter.theta)
            filter.imu_propagate(accel, ypr, time - last_time)
            # upon reaching the moment with an image captured
            if image_path_cursor >= len(image_path):
                break # no more images to process, stop!
            if time > float(image_info[image_path_cursor][0]):
                if frame_skip_count < frame_skip:
                    frame_skip_count = frame_skip_count + 1
                    image_path_cursor = image_path_cursor + 1
                    continue
                frame_skip_count = 0
                if not img0_path:
                    # First image frame detection
                    img0_path = image_info[image_path_cursor][1]
                    filter.set_img_pointer(filter.now_pointer, 0)
                    lk_flow.first_frame_track_generate(img0_path)
                else:
                    # print('Img1 set at clock ', time)
                    img1_path = image_info[image_path_cursor][1]
                    # Both images are available, perform filtering!
                    lk_flow.set_frame(img0_path, img1_path)
                    filter.set_img_pointer(filter.img0_pointer, filter.now_pointer)
                    ftracks, tracks_marginalize, oldest_state_index, longest_flow_components = lk_flow.detect_and_match(filter.roll_pointer + filter.img0_pointer, filter.roll_pointer + filter.img1_pointer)
                    print('Track size: {:d}; Track to marginalize: {:d}, Oldest State Index: {:d}/{:d}, Longest components: {:d}'.format(len(ftracks), len(tracks_marginalize), oldest_state_index, filter.roll_pointer + filter.img1_pointer, longest_flow_components))
                    flow_tracks = lk_flow.calculate() # calculate the optical flow between two images
                    if flow_tracks is not None:
                        # pass
                        print('OF Analysis: ', img0_path, ' > ', img1_path)
                        x_k, x_kp, Ric_k, Ric_kp, camera_ray_length_k, camera_ray_length_kp, tracks = filter.cam_correction(flow_tracks) # perform all Panacea filtering and window shifting
                        # show the result
                        lk_flow.revisualize_of(tracks, x_k, x_kp, camera_ray_length_k, camera_ray_length_kp, Ric_k, Ric_kp)
                    img0_path = img1_path
                    img1_path = ''
                image_path_cursor = image_path_cursor + 1
            last_time = time
        else:
            last_time = time # skip the first IMU measurement to calculate delta_t
    position_fig = plt.figure(1)
    plt.plot(t_log, x_log)
    residue_fig = plt.figure(2)
    plt.plot(filter.res_norm_log)
    plt.plot(filter.res_norm_corrected_log)
    accel_inertial_fig = plt.figure(3)
    plt.plot(filter.res_norm_corrected_ols_log)
    plt.show()
    

if __name__=='__main__':
    main()
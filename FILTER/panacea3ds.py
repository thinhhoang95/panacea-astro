'''
PANACEA Data Broker (PDM)
Author: Hoang Dinh Thinh, Dept. Aerospace Engineering, Univ. Corruption
Email: hdinhthinh@gmail.com
'''

from panacea_inertial_3dsei import PanaceaInertial3DS
from panacea_lk_flow_msse import PanaceaLKFlowMSS
import numpy as np
import glob
from matplotlib import pyplot as plt

def main():
    filter = PanaceaInertial3DS()
    accel_data = np.genfromtxt('inertial_ll/accel.txt', delimiter=',')
    ypr_data = np.genfromtxt('inertial_ll/eulang.txt', delimiter=',')
    # image_path = glob.glob('images/*0125*.jpg') + glob.glob('images/*0126*.jpg') + glob.glob('images/*0127*.jpg') + glob.glob('images/*0128*.jpg') + glob.glob('images/*0129*.jpg') + glob.glob('images/*0130*.jpg') + glob.glob('images/*0131*.jpg') + glob.glob('images/*0132*.jpg')
    image_path = glob.glob('images_ll/*.jpg')
    image_info = []
    lk_flow = PanaceaLKFlowMSS(None, None)
    # Convert image name timestamp to a numpy vector
    for img_path in image_path:
        image_info.append((img_path[20:len(img_path)-4],img_path))
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
    oldest_state_index = 0
    frame_skip = 0 # skip 1 image frame
    accel_row_index = 0
    for accel_row, ypr_row in zip(accel_data, ypr_data):
        accel_row_index = accel_row_index + 1
        time = accel_row[0]
        if (t_first == 0):
            t_first = time
        if last_time!=0:
            accel = -accel_row[1:4]
            ypr = ypr_row[1:4]
            # print('Clock hit {:.3f}'.format(time))
            # print('Position: ', filter.X[filter.now_pointer, 0:3])
            x_log = np.vstack((x_log, filter.X[filter.now_pointer, 0:3]))
            t_log = np.append(t_log, time - t_first)
            # print('IMU Calibration: ', filter.theta)
            reset_vel = False
            if accel_row_index % 5 == 0:
                reset_vel = True
            filter.imu_propagate(accel, ypr, time - last_time, reset_vel)
            # upon reaching the moment with an image captured
            if image_path_cursor >= len(image_path):
                break # no more images to process, stop!
            if time > float(image_info[image_path_cursor][0]):
                if frame_skip_count < frame_skip:
                    frame_skip_count = frame_skip_count + 1
                    image_path_cursor = image_path_cursor + 1
                frame_skip_count = 0
                if not img0_path:
                    # First image frame detection
                    img0_path = image_info[image_path_cursor][1]
                    filter.set_img_pointer(filter.now_pointer, 0)
                    img0_mss_path = img0_path
                    lk_flow.first_frame_track_generate(img0_path)
                    filter = PanaceaInertial3DS() # reset the filter at the beginning of first frame captured
                else:
                    # print('Img1 set at clock ', time)
                    img1_path = image_info[image_path_cursor][1]
                    # Both images are available, perform filtering!
                    lk_flow.set_frame(img0_path, img1_path) # set img1_path for MSS too!
                    lk_flow.set_mss_frame(img0_mss_path) # mss_path might not coincide with img0 path as they might not be swapped with img1 
                    filter.set_img_pointer(filter.img0_pointer, filter.now_pointer)
                    flow_tracks = lk_flow.calculate() # calculate the optical flow between two images
                    if len(flow_tracks)>200:
                        flow_tracks = flow_tracks[0:200]
                    valid_tracks_in_flow_tracks = len(flow_tracks)
                    for track in flow_tracks:
                        if len(track)==1:
                            valid_tracks_in_flow_tracks = valid_tracks_in_flow_tracks - 1
                    if(valid_tracks_in_flow_tracks < 7): # at least 7 flow tracks are required
                        # too little information for odometry via OF
                        # move the image path to the next frame in line
                        print('<!> Frame ', img1_path, ': OF was skipped due to insufficient odometry information available')
                        # If optical flow is not available, high chance is that img1 contains little keypoints!
                        # Let's perform detect_and_match without moving the img0 pointer forward (to the little keypoints frame)
                        #ftracks, tracks_marginalize, oldest_state_index_ftracks, oldest_state_index_marginalize, longest_flow_components, longest_frame_count = lk_flow.detect_and_match(filter.roll_pointer + filter.img0_mss_pointer, filter.roll_pointer + filter.img1_pointer)
                        # filter.mss_cam_correction(tracks_marginalize) # no altering img0_mss path or pointer
                        #oldest_state_index = min(oldest_state_index_ftracks, oldest_state_index_marginalize)
                        img0_path = img1_path # Skip to the next consecutive pair
                        img1_path = ''
                        # adjusting the image pointer
                        filter.slide_window(oldest_state_index - filter.roll_pointer) # set pointer 0 to 1 included!
                    else: # when optical flow resumes, img0_path (in particular MSS's img0_path) will be set to img1_path 
                        if flow_tracks is not None:
                            # perform MSS correction
                            ftracks, tracks_marginalize, oldest_state_index_ftracks, oldest_state_index_marginalize, longest_flow_components, longest_frame_count = lk_flow.detect_and_match(filter.roll_pointer + filter.img0_mss_pointer, filter.roll_pointer + filter.img1_pointer)
                            if len(tracks_marginalize) > 0:
                                first_track_first_imu_sample = tracks_marginalize[0][0][-1]
                                if first_track_first_imu_sample != 0:
                                    filter.mss_cam_correction(tracks_marginalize)
                            oldest_state_index = min(oldest_state_index_ftracks, oldest_state_index_marginalize)
                            # perform OF correction
                            print('OF Analysis: ', img0_path, ' > ', img1_path)
                            x_k, x_kp, Ric_k, Ric_kp, camera_ray_length_k, camera_ray_length_kp, tracks = filter.cam_correction(flow_tracks, oldest_state_index - filter.roll_pointer) # perform all Panacea filtering and window shifting
                            # Slide the MSS pointer and image source to match optical flow's
                            lk_flow.revisualize_of(tracks, x_k, x_kp, camera_ray_length_k, camera_ray_length_kp, Ric_k, Ric_kp)
                            filter.img0_mss_pointer = filter.img0_pointer
                            img0_mss_path = img1_path
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
    plt.plot(filter.accel_log)
    plt.show()

    # Saving x_log to file
    np.savetxt('x_log.csv', x_log, delimiter=',')
    np.savetxt('t_log.csv', t_log, delimiter=',')

if __name__=='__main__':
    main()
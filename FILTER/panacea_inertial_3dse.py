import numpy as np
from scipy.ndimage.interpolation import shift
from scipy.spatial.transform import Rotation
from numpy.linalg import inv
from scipy.linalg import block_diag
from scipy.ndimage.interpolation import shift
import math
import time

class PanaceaInertial3DS:

    def __init__(self):
        # DEFAULT VALUES
        self.window_size = 9999 # maximum number of states, depending on how fast algorithm operates
        # state includes 3 position, 3 velocity, 3 scale factor, 3 bias terms
        self.X = np.zeros((self.window_size, 6)) # state vector
        self.X[0,:] = np.array([0,0,-1.67,0,0,0]) # initial condition
        self.YPR = np.zeros((self.window_size, 3)) # yaw pitch roll of Rbi 
        self.P = np.zeros((self.window_size, 6, 6)) # covariance matrix of state
        self.theta = np.array([1,1,1,0,0,0]) # 3 scale factors and 3 bias factors
        self.thetaP = 0.05 * np.identity(6) # covariance matrix of scale and bias factors
        self.accel = np.zeros((self.window_size, 3)) # logging the acceleration
        self.deltat = np.zeros(self.window_size) # dt matrix
        self.now_pointer = 0 # pointer of current IMU integration state in the window
        self.img0_pointer = 0 # pointer of the acquired img0 in the window, for optical flow analysis
        self.img0_mss_pointer = 0 # pointer of the acquired img0 in the window, for MSS analysis
        self.img1_pointer = 0 # pointer of the acquired img1 in the window
        self.roll_pointer = 0
        self.Q_mat = np.diag(1E-4 * np.array([5,5,5])) # covariance of acceleration
        self.R_mat = 1E-8 * np.identity(2) # covariance of optical flow measurement
        # TODO: kwargs for dynamic settings

        # IMPORTING CAMERA PARAMETERS
        cam_mat = np.load('cam_mat.pca.npy')
        focal_length_x = cam_mat[0,0]
        focal_length_y = cam_mat[1,1]
        cx = cam_mat[0,2]
        cy = cam_mat[1,2]
        self.fx = 3.04E-3
        self.fy = 3.04E-3
        self.px_scale_x = 5.875E-6
        self.px_scale_y = 5.875E-6
        # self.fx = focal_length_x * self.px_scale_x
        # self.fy = focal_length_y * self.px_scale_y
        self.cx = 320 * self.px_scale_x
        self.cy = 240 * self.px_scale_y

        # IMPORTING RIG CALIBRATION PARAMETERS
        #self.Rito = np.load('rito.pca.npy')
        #self.Rbc = np.load('rbc.pca.npy')
        #self.Ritpt = np.array([[1,0,0],[0,-1,0],[0,0,-1]]) # z upward to z downward

        # theta = [26.0982*math.pi/180, 84.3910*math.pi/180] # 5.025 first residue
        # theta = [-26.0982*math.pi/180, 84.3910*math.pi/180] # 6.894 first residue
        theta = [0*math.pi/180, -84.3910*math.pi/180] # 3.318 first residue


        self.Ritpt = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
        self.Ritptz = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
        self.Rito = np.array([[math.cos(theta[0]), -math.sin(theta[0]), 0],[math.sin(theta[0]), math.cos(theta[0]), 0],[0,0,1]])
        self.Rbc = np.array([[math.cos(theta[1]), -math.sin(theta[1]), 0],[math.sin(theta[1]), math.cos(theta[1]), 0],[0,0,1]])

        # Logging variables
        self.res_norm_log = np.empty(0)
        self.res_norm_corrected_log = np.empty(0)
        self.res_norm_corrected_ols_log = np.empty(0)
        self.accel_log = np.empty((0,3))


    '''
    Set the cursor 
    '''
    def set_img_pointer(self, cursor0, cursor1):
        self.img0_pointer = cursor0
        self.img1_pointer = cursor1

    def set_img_pointer_mss(self, cursor0):
        self.img0_mss_pointer = cursor0

    '''
    Propagate IMU state upon receive of new acceleration
    '''
    def imu_propagate(self, ab, ypr, T):
        if (self.now_pointer + 1 >= self.window_size):
            # Pointer is now outside of the window
            raise Exception('The window size is too small! Try to increase it')
        #A_mat = np.array([[1,0,T,0],[0,1,0,T],[0,0,1,0],[0,0,0,1]]) # fixed z
        #B_mat_1 = np.array([[0.5*(T**2),0],[0,0.5*(T**2)],[T,0],[0,T]]) # acceleration integration
        A_mat = np.array([[1,0,0,T,0,0],[0,1,0,0,T,0],[0,0,1,0,0,T],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
        B_mat_1 = np.array([[0.5*(T**2),0,0],[0,0.5*(T**2),0],[0,0,0.5*(T**2)],[T,0,0],[0,T,0],[0,0,T]]) # acceleration integration
        # B_mat_2 = ...
        # B_mat_3 = np.array([[self.theta[0],0,0,-self.theta[3]],[0,self.theta[1],0,-self.theta[4]],[0,0,self.theta[2],-self.theta[5]]])
        # _mat_3 = np.array([[self.X[self.now_pointer,6],0,0,-self.X[self.now_pointer,9]],[0,self.X[self.now_pointer,7],0,-self.X[self.now_pointer,10]],[0,0,self.X[self.now_pointer,8],-self.X[self.now_pointer,11]]])
        Rbi = Rotation.from_euler('ZYX', ypr).as_dcm()
        # u = np.concatenate(((Rbi @ ab.T - np.array([0,0,1]))*9.78206, np.array([1])))
        # u = np.concatenate(((Rbi @ self.Ritpt @ ab.T + np.array([0,0,1]))*9.78206, np.array([1])))
        u = np.array([0,0,0]) # disable IMU updates, uses camera only
        # self.accel_log = np.vstack((self.accel_log, (Rbi @ self.Ritpt @ ab.T + np.array([0,0,1]))*9.78206))
        B = B_mat_1
        next_state = A_mat @ self.X[self.now_pointer,:].T + B @ u.T
        next_state[3:6] = np.zeros(3) # zeroing out the velocity so that there won't be any drift
        # next_state[2] = -1.67 # fixing the height of the camera to 1.67m
        next_P = A_mat @ self.P[self.now_pointer,:] @ A_mat.T + B @ self.Q_mat @ B.T
        # next_P = np.diag([5, 5])
        # Writing history
        self.accel[self.now_pointer, :] = ab
        self.now_pointer = self.now_pointer + 1
        self.YPR[self.now_pointer, :] = ypr
        self.P[self.now_pointer,:] = next_P 
        self.X[self.now_pointer,:] = next_state.T
        self.deltat[self.now_pointer] = T
        

    '''
    Propagate with acceleration from pointer 1 to pointer 2
    '''
    def imu_repropagate(self, pointer1, pointer2):
        for k in range(pointer1, pointer2):
            T = self.deltat[k]
            A_mat = np.array([[1,0,0,T,0,0],[0,1,0,0,T,0],[0,0,1,0,0,T],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
            B_mat_1 = np.array([[0.5*(T**2),0,0],[0,0.0*(T**2),0],[0,0,0.5*(T**2)],[T,0,0],[0,T,0],[0,0,T]]) # acceleration integration
            # B_mat_2 = ...
            B_mat_3 = np.array([[self.theta[0],0,0,-self.theta[3],0,0],[0,self.theta[1],0,0,self.theta[4],0],[0,0,self.theta[2],0,0,self.theta[5]]])
            # B_mat_3 = np.array([[self.X[self.now_pointer,6],0,0,-self.X[self.now_pointer,9]],[0,self.X[self.now_pointer,7],0,-self.X[self.now_pointer,10]],[0,0,self.X[self.now_pointer,8],-self.X[self.now_pointer,11]]])
            ypr = self.YPR[k, :]
            a = self.accel[k,:]
            Rbi = Rotation.from_euler('ZYX', ypr).as_dcm()
            u = np.concatenate(((Rbi @ self.Ritpt @ a.T + np.array([0,0,1]))*9.78206, np.array([1])))
            B = B_mat_1 @ B_mat_3
            next_state = A_mat @ self.X[self.now_pointer,:].T + B @ u.T
            next_P = A_mat @ self.P[self.now_pointer,:] @ A_mat.T + B @ self.Q_mat @ B.T
            self.X[k+1,:] = next_state
            self.P[k+1,:] = next_P 

    '''
    MAP estimation of the accelerometer parameters
    '''
    def map_theta(self, X, Pp):
        B_overline = np.identity(6)
        A_overline = np.identity(6)
        for k in range(self.img0_pointer, self.img1_pointer):
            T = self.deltat[k]
            A_mat = np.array([[1,0,0,T,0,0],[0,1,0,0,T,0],[0,0,1,0,0,T],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
            B_overline = B_overline @ A_mat
            Acc = np.identity(6)
            if k+1 > self.img1_pointer-1:
                pass
            else:
                for q in range(k+1, self.img1_pointer):
                    Acc = Acc @ A_mat
            B_mat_1 = np.array([[0.5*(T**2),0,0],[0,0.0*(T**2),0],[0,0,0.5*(T**2)],[T,0,0],[0,T,0],[0,0,T]]) # acceleration integration
            # B_mat_2 = ...
            # B_mat_3 = np.array([[self.theta[0],0,0,-self.theta[3],0,0],[0,self.theta[1],0,0,self.theta[4],0],[0,0,self.theta[2],0,0,self.theta[5]]])
            # B_mat_3 = np.array([[self.X[self.now_pointer,6],0,0,-self.X[self.now_pointer,9]],[0,self.X[self.now_pointer,7],0,-self.X[self.now_pointer,10]],[0,0,self.X[self.now_pointer,8],-self.X[self.now_pointer,11]]])
            a = self.accel[k,:]
            ypr = self.YPR[k,:]
            Rbi = Rotation.from_euler('ZYX', ypr).as_dcm()
            A_cs = np.hstack((np.diag(Rbi @ a),-np.identity(3)))
            A_overline = A_overline + Acc @ B_mat_1 @ A_cs 
        
        # MAP estimation
        P_mat = A_overline @ self.thetaP @ A_overline.T # state (theta - accel's parameters) covariance propagation
        z = X.T - B_overline @ self.X[self.img0_pointer, :].T
        R_mat = Pp + B_overline @ self.P[self.img0_pointer,:] @ B_overline.T # covariance of z
        H = A_overline

        # Observation scale factor
        z_scale = 1E-4
        z = z * z_scale
        H = H * z_scale

        # Posteriori values
        W = inv(inv(P_mat) + H.T @ inv(R_mat) @ H)
        theta_pos = W @ (H.T @ inv(R_mat) @ z + inv(P_mat) @ self.theta.T)
        cov_pos = W

        # Assign to the state
        # self.theta = theta_pos
        # self.thetaP = cov_pos

        #print('Theta pos: ', self.theta)
        pass


    '''
    Measurement model setup
    '''
    def measurement_model(self, Ric_k, Ric_kp, x_kp, x_k, z_k, z_kp, tracks):
        H_stack = np.empty((0,3)) # 2 for fixed z
        H_stack_pos = np.empty((0,3)) # 2 for fixed z
        lhs_stack = np.empty(0)
        rhs_stack = np.empty(0)
        rpj_stack = np.empty(0)

        delta_lm_k = Ric_k.T @ np.array([0, 0, 1])
        # camera_ray_length_k = 1.67/delta_lm_k[2]
        # camera_ray_length_k = x_k[2]/delta_lm_k[2]
        camera_ray_length_k = -x_k[2]/delta_lm_k[2]
        delta_lm_kp = Ric_kp.T @ np.array([0, 0, 1])
        # camera_ray_length_kp = 1.67/delta_lm_kp[2]
        camera_ray_length_kp = -x_kp[2]/delta_lm_kp[2]
        # Construction of measurement model
        for track in tracks:
            if len(track)==1:
                continue
            track0 = (track[0][0] * self.px_scale_x, track[0][1] * self.px_scale_y)
            track1 = (track[1][0] * self.px_scale_x, track[1][1] * self.px_scale_y)
            # S = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0], [0,0,0,0,0,0,1]]) # select x,y,z in the state vector xyzvxvyvz
            # S = np.array([[1,0,0,0],[0,1,0,0]])
            Hr = np.array([[-self.fx,0,track1[0]-self.cx],[0,-self.fy,track1[1]-self.cy]]) @ Ric_kp
            # H = Hr[0:2,0:2] # fixed Z
            H = Hr # xyz
            H_stack_pos = np.concatenate((H_stack_pos, H), axis=0) # for OLS estimator
            H_stack = np.concatenate((H_stack, H), axis=0) # for Kalman filter
            # x_kp_augmented = np.concatenate((x_kp, -1.67))
            lhs = H @ x_kp[0:3].T
            lhs_stack = np.append(lhs_stack, lhs)
            # rhs_lm_pos = z_k * Ric_k.T @ np.array([(track0[0]-self.cx)/self.fx, (track0[1]-self.cy)/self.fy, 1])
            rhs_lm_pos = camera_ray_length_k * Ric_k.T @ np.array([(track0[0]-self.cx)/self.fx, (track0[1]-self.cy)/self.fy, 1])
            rhs_lm_pos = camera_ray_length_k * Ric_k.T @ np.array([(track0[0]-self.cx)/self.fx, (track0[1]-self.cy)/self.fy, 1])
            # rhs = Hr @ (rhs_lm_pos + np.concatenate((x_k[0:2], -1.67), axis=None)) # fixed z
            rhs = Hr @ (rhs_lm_pos + np.concatenate((x_k[0:2], x_k[2]), axis=None))
            # rhs = rhs + Hr[:,2] * 1.67
            #print('-- Pixel @k: ', track[0])
            #print('-- Point projection @k: ', rhs_lm_pos + x_k[0:3])
            rhs_stack = np.append(rhs_stack, rhs)
            # debug variables
            point_reprojection_0 = camera_ray_length_k * Ric_k.T @ np.array([(track0[0]-self.cx)/self.fx, (track0[1]-self.cy)/self.fy, 1]) + np.concatenate((x_k[0:2], -1.67), axis=None)
            point_reprojection_1 = camera_ray_length_kp * Ric_kp.T @ np.array([(track1[0]-self.cx)/self.fx, (track1[1]-self.cy)/self.fy, 1]) + np.concatenate((x_kp[0:2], -1.67), axis=None)
            point_reprojection_diff = np.linalg.norm(point_reprojection_1 - point_reprojection_0)
            rpj_stack = np.append(rpj_stack, point_reprojection_diff)
            # point_reprojection_0 = 1.67 * Ric_k.T @ np.array([(track0[0])/self.fx, (track0[1])/self.fy, 1]).T + x_k[0:3]
            # point_reprojection_1 = 1.67 * Ric_kp.T @ np.array([(track1[0])/self.fx, (track1[1])/self.fy, 1]).T + x_kp[0:3]
        
        #print('-- Point 0: ', point_reprojection_0)
        #print('-- Point 1: ', point_reprojection_1)
        
        residue = rhs_stack - lhs_stack
        residue_norm = np.sum(residue**2)*10E5/np.shape(residue)[0]
        #print('- Residue is: ', residue_norm)
        rpj_stack_mean = np.mean(rpj_stack)
        #print('- Reprojection residue is: ', np.mean(rpj_stack))
        return rhs_stack, lhs_stack, rpj_stack, H_stack, H_stack_pos, residue, residue_norm, rpj_stack_mean, camera_ray_length_k, camera_ray_length_kp

    def slide_window(self, pointer_destination):
        # Slide the window forward to make img1_pointer the first element
        self.X = np.roll(self.X, -pointer_destination, axis=0)
        self.roll_pointer = self.roll_pointer + pointer_destination
        self.P = np.roll(self.P, -pointer_destination, axis=0)
        self.YPR = np.roll(self.YPR, -pointer_destination, axis=0)
        self.accel = np.roll(self.accel, -pointer_destination, axis=0)
        self.deltat = np.roll(self.deltat, -pointer_destination, axis=0)
        self.now_pointer = self.now_pointer - pointer_destination
        
        self.img0_pointer = self.img1_pointer - pointer_destination
        self.img1_pointer = self.img1_pointer - pointer_destination # to be updated in subsequent run

    '''
    Perform filter correction based on measurement from the optical flow.
    The standard optical flow tracks vector has many tracks, each with 2 points
    '''
    def cam_correction(self, tracks, pointer_destination):
        img0_pointer = self.img0_pointer
        # Ric_kp = self.Ritpt @ self.Rito @ Rotation.from_euler('ZYX', self.YPR[self.img1_pointer,:], degrees=False).as_dcm() @ self.Rbc
        # Ric_k = self.Ritpt @ self.Rito @ Rotation.from_euler('ZYX', self.YPR[self.img0_pointer,:], degrees=False).as_dcm() @ self.Rbc
        Ric_kp = self.Rito @ Rotation.from_euler('ZYX', self.YPR[self.img1_pointer,:], degrees=False).as_dcm().T @ self.Rbc
        Ric_k = self.Rito @ Rotation.from_euler('ZYX', self.YPR[img0_pointer,:], degrees=False).as_dcm().T @ self.Rbc
        
        x_kp = self.X[self.img1_pointer,:]
        x_k = self.X[img0_pointer,:]
        
        #x_kp[2] = -np.abs(x_kp[2])
        #x_k[2] = -np.abs(x_k[2])

        z_k = np.abs(x_k[2])
        z_kp = np.abs(x_kp[2])
        P_kp = self.P[self.img1_pointer,0:3,0:3]

        rhs_stack, lhs_stack, rpj_stack, H_stack, H_stack_pos, residue, residue_norm, rpj_stack_mean, camera_ray_length_k, camera_ray_length_kp = self.measurement_model(Ric_k, Ric_kp, x_kp, x_k, z_k, z_kp, tracks)

        # Kalman filters
        self.res_norm_log = np.append(self.res_norm_log, np.sum(residue**2)*10E5/np.shape(residue)[0])
        R_list = [self.R_mat for i in range(int(np.shape(residue)[0]/2))]
        R_augmented = block_diag(*R_list)
        k_gain = P_kp @ H_stack.T @ inv(R_augmented + H_stack @ P_kp @ H_stack.T)
        delta_x_kp = k_gain @ residue
        # x_kp_new = x_kp + np.concatenate((delta_x_kp,np.array([0,0])), axis=None)
        # OLS Estimator
        x_kp_new_ols = inv(H_stack_pos.T @ H_stack_pos) @ H_stack_pos.T @ rhs_stack
        x_kp_new = np.concatenate((x_kp_new_ols,np.array([0,0,0])), axis=None)
        P_kp_new = (np.identity(3) - k_gain @ H_stack) @ P_kp

        print('Cov (KAL): {:e},{:e},{:e} => {:e},{:e},{:e}'.format(P_kp[0,0], P_kp[1,1], P_kp[2,2], P_kp_new[0,0], P_kp_new[1,1], P_kp_new[2,2]))

        #print('Current pos: ', x_kp[0:3])

        print('New pos (OLS/OTF): ', x_kp_new_ols)
        
        #print('Delta x (KAL): ', delta_x_kp)
        print('New pos (KAL/OTF): ', x_kp_new)
        #print('New pos (OLS): ', x_kp_new_ols)
        #print('** After correction **')
        rhs_stack, lhs_stack, rpj_stack, H_stack, H_stack_pos, residue, residue_norm, rpj_stack_mean, camera_ray_length_k, camera_ray_length_kp = self.measurement_model(Ric_k, Ric_kp, x_kp_new, x_k, z_k, z_kp, tracks)
        self.res_norm_corrected_log = np.append(self.res_norm_corrected_log, residue_norm)
        rhs_stack, lhs_stack, rpj_stack, H_stack, H_stack_pos, residue, residue_norm, rpj_stack_mean, camera_ray_length_k, camera_ray_length_kp = self.measurement_model(Ric_k, Ric_kp, np.concatenate((x_kp_new_ols, -1.67), axis=None), x_k, z_k, z_kp, tracks)
        self.res_norm_corrected_ols_log = np.append(self.res_norm_corrected_ols_log, residue_norm)
        # MAP estimation of accelerometer settings
        # self.map_theta(x_kp_new, P_kp_new)

        # Update state and repropagation from img1_pt to now
        # self.X[self.img1_pointer, :] = np.hstack((x_kp_new_ols,np.array([0,0,0]))) # <- OLS not KAL
        self.X[self.img1_pointer, :] = x_kp_new
        self.P[self.img1_pointer, 0:3, 0:3] = P_kp_new
        # self.imu_repropagate(self.img1_pointer, self.now_pointer)

        self.slide_window(pointer_destination)

        print('(OF) Roll Pointer: {:d}, Image 0 Pointer: {:d}, Image 1 Pointer: {:d}'.format(self.roll_pointer, self.img0_pointer, self.img1_pointer))

        return x_k, x_kp, Ric_k, Ric_kp, camera_ray_length_k, camera_ray_length_kp, tracks

    # Returns a list of IMU state: e.g. [130, 141, 151, 160, 171] which is noted at the point of camera image registration
    def get_track_association(self, tracks):
        kf_index_list = []
        for track in tracks:
            for keypoint in track:
                kf_index = keypoint[2]
                if not kf_index in kf_index_list:
                    kf_index_list.append(kf_index)
        kf_index_list.sort()
        kf_index_min = kf_index_list[0]
        kf_index_max = kf_index_list[-1]
        pose_count = len(kf_index_list)
        lm_count = len(tracks)
        return kf_index_list, kf_index_max, kf_index_min, pose_count, lm_count

    # def get_min_max_of_tracks(self, tracks):
    #     min_index = 9999
    #     max_index = 0
    #     lm_count = 0
    #     pose_count = 0
    #     for track in tracks:
    #         for keypoint in track:
    #             track_kf_index_min = track[0][2]
    #             track_kf_index_max = track[-1][2]
    #             if min_index != track_kf_index_min or max_index!=track_kf_index_max:
    #                 pose_count = pose_count + 1
    #             min_index = min(min_index, track_kf_index_min)
    #             max_index = max(max_index, track_kf_index_max)
    #             lm_count = lm_count + 1
    #     return min_index, max_index, lm_count, pose_count

    # Note that all indices are zero-based
    # Get the max-min of the index of the track
    def get_state_vec_index(self, kf_index_list, kf_index_max, kf_index_min, pose_count, lm_count, type, index):
        if type=='pose':
            state_index = kf_index_list.index(index) # transforms the IMU state index to the measurement state index
            a = state_index * 3
            b = (state_index + 1) * 3 - 1
        elif type=='landmark':
            pose_state_length = pose_count * 3
            a = pose_state_length + index * 2
            b = pose_state_length + (index + 1) * 2 - 1
        else:
            raise Exception('Invalid type for state vector access')
        return a, b

    # Yields the H matrix for each observation, right hand side should be zero
    def measurement_model_f(self, Ric, xe, pos_index, lm_index, state_length):
        xe1 = np.array(xe)
        xe1 = xe1 * self.px_scale_x
        H = np.array([[-self.fx,0,xe1[0]-self.cx],[0,-self.fy,xe1[1]-self.cy]]) @ Ric
        S = np.zeros((3,state_length))
        S[0,pos_index] = -1
        S[0,lm_index] = 1
        S[1,pos_index+1] = -1
        S[1,lm_index+1] = 1
        S[2,pos_index+2] = -1
        return H @ S
    
    def mss_cam_correction(self, ftracks):
        if len(ftracks) == 0:
            return
        landmark_var = 0.05
        measurement_var = 1E-8
        H_stack = None # nothing at all!
        x_vec = np.empty((0,1))
        track_association = self.get_track_association(ftracks)
        P_matrix = np.empty((0,0))
        Q_matrix = np.empty((0,0))
        # P_matrix of the state vector poses (for Kalman filter)
        for imu_true_index in track_association[0]:
            window_index = imu_true_index - self.roll_pointer
            P_pose = self.P[window_index,0:3,0:3]
            P_matrix = block_diag(P_matrix, P_pose)
            pos = self.X[window_index, 0:3]
            x_vec = np.concatenate((x_vec, pos), axis=None)
        # P_matrix of the landmarks (for Kalman filter)
        for lm_index in range(track_association[4]):
            P_matrix = block_diag(P_matrix, landmark_var*np.eye(2))
        # Processing each keyframe of each track:
        state_length = track_association[3] * 3 + track_association[4] * 2
        for track_no, track in enumerate(ftracks):
            # Preliminary prediction of the landmark's position
            first_observation_of_lm = track[0]
            imu_true_index_of_obs = first_observation_of_lm[-1]
            window_index_of_lm = imu_true_index_of_obs - self.roll_pointer
            x_at_obs = self.X[window_index_of_lm,0:3]
            Ric_at_obs = self.Rito @ Rotation.from_euler('ZYX', self.YPR[window_index_of_lm,:], degrees=False).as_dcm().T @ self.Rbc
            delta_lm_k = Ric_at_obs.T @ np.array([0, 0, 1])
            camera_ray_length_k = -x_at_obs[2]/delta_lm_k[2]
            lm_prediction = Ric_at_obs.T @ np.array([camera_ray_length_k * first_observation_of_lm[0] * self.px_scale_x / self.fx, camera_ray_length_k * first_observation_of_lm[1] * self.px_scale_y / self.fy, camera_ray_length_k])
            x_vec = np.concatenate((x_vec, lm_prediction[0:2]), axis=None)
            # Processing measurements
            for keyframe in track:
                # obtain the window states
                imu_true_index = keyframe[-1]
                window_index = imu_true_index - self.roll_pointer
                Ric = self.Rito @ Rotation.from_euler('ZYX', self.YPR[window_index,:], degrees=False).as_dcm().T @ self.Rbc
                #x = self.X[window_index,0:3] # only inertial positions are kept
                #z_alt = np.abs(x[2])
                #P = self.P[window_index,:]
                xe = keyframe[0:2]
                pos_index = self.get_state_vec_index(*track_association, 'pose', imu_true_index)[0] # beginning of the state index in the vector
                lm_index = self.get_state_vec_index(*track_association, 'landmark', track_no)[0] # beginning of the state index in the vector
                # get the measurements
                H = self.measurement_model_f(Ric, xe, pos_index, lm_index, state_length)
                if H_stack is None:
                    H_stack = np.empty((0,np.shape(H)[1]))
                H_stack = np.concatenate((H_stack, H), axis=0)
                Q_matrix = block_diag(Q_matrix, measurement_var * np.eye(2))
        # Calculate the residual before and after correction:
        residual_before = np.linalg.norm(H_stack @ x_vec)
        # Performing Kalman Filter / MAP
        z = np.zeros((np.shape(H_stack)[0]))
        rez = z - H_stack @ x_vec
        k_gain = P_matrix @ H_stack.T @ inv(Q_matrix + H_stack @ P_matrix @ H_stack.T)
        delta_x_vec = k_gain @ rez
        x_vec_new = x_vec + delta_x_vec
        k_gain_H_stack = k_gain @ H_stack
        P_matrix_new = (np.eye(np.shape(k_gain_H_stack)[0]) - k_gain_H_stack) @ P_matrix
        # New residual
        residual_after = np.linalg.norm(H_stack @ x_vec_new)
        print('Residual before/after MSS correction: {:.4f}/{:.4f}'.format(residual_before, residual_after))
        # Marginalization of states for each poses
        for no, imu_true_index in enumerate(track_association[0]):
            window_index = imu_true_index - self.roll_pointer #e.g. 131
            a, b = self.get_state_vec_index(*track_association, 'pose', imu_true_index)
            self.X[window_index,0:3] = x_vec_new[a:b+1]
            self.P[window_index,0:3,0:3] = P_matrix_new[a:b+1,a:b+1]
            if no==len(track_association[0])-1: # print upon completion of processing the last track
                print('New pos (KAL/MSS): ', x_vec_new[a:b+1])
                # time.sleep(1)
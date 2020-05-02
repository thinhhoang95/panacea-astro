import numpy as np
from scipy.ndimage.interpolation import shift
from scipy.spatial.transform import Rotation
from numpy.linalg import inv
from scipy.linalg import block_diag
from scipy.ndimage.interpolation import shift
import math

class PanaceaInertial:

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
        self.img0_pointer = 0 # pointer of the acquired img0 in the window
        self.img1_pointer = 0 # pointer of the acquired img1 in the window
        self.Q_mat = block_diag(0.3 * np.identity(3), 0) # covariance of acceleration
        self.R_mat = 5E-15 * np.identity(2) # covariance of optical flow measurement
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
        #self.fx = focal_length_x * 5.875E-6
        #self.fy = focal_length_y * 5.708E-6
        self.cx = cx * self.px_scale_x
        self.cy = cy * self.px_scale_y

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


    '''
    Set the cursor 
    '''
    def set_img_pointer(self, cursor0, cursor1):
        self.img0_pointer = cursor0
        self.img1_pointer = cursor1

    '''
    Propagate IMU state upon receive of new acceleration
    '''
    def imu_propagate(self, ab, ypr, T):
        if (self.now_pointer + 1 >= self.window_size):
            # Pointer is now outside of the window
            raise Exception('The window size is too small! Try to increase it')
        A_mat = np.array([[1,0,0,T,0,0],[0,1,0,0,T,0],[0,0,1,0,0,T],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
        B_mat_1 = np.array([[0.5*(T**2),0,0],[0,0.5*(T**2),0],[0,0,0.5*(T**2)],[T,0,0],[0,T,0],[0,0,T]]) # acceleration integration
        # B_mat_2 = ...
        B_mat_3 = np.array([[self.theta[0],0,0,-self.theta[3]],[0,self.theta[1],0,-self.theta[4]],[0,0,self.theta[2],-self.theta[5]]])
        # _mat_3 = np.array([[self.X[self.now_pointer,6],0,0,-self.X[self.now_pointer,9]],[0,self.X[self.now_pointer,7],0,-self.X[self.now_pointer,10]],[0,0,self.X[self.now_pointer,8],-self.X[self.now_pointer,11]]])
        Rbi = Rotation.from_euler('ZYX', ypr).as_dcm()
        self.YPR[self.now_pointer, :] = ypr
        u = np.concatenate(((Rbi @ ab.T - np.array([0,0,1]))*9.78206, np.array([1])))
        B = B_mat_1 @ B_mat_3
        next_state = A_mat @ self.X[self.now_pointer,:].T + B @ u.T
        next_P = A_mat @ self.P[self.now_pointer,:] @ A_mat.T + B @ self.Q_mat @ B.T
        self.accel[self.now_pointer, :] = ab
        self.now_pointer = self.now_pointer + 1
        self.P[self.now_pointer,:] = next_P 
        self.X[self.now_pointer,:] = next_state.T
        

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
            u = np.concatenate(((Rbi @ a.T - np.array([0,0,1]))*9.78206, np.array([1])))
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
            A_cs = np.hstack((np.diag(a),-np.identity(3)))
            A_overline = A_overline + Acc @ B_mat_1 @ Rbi @ A_cs 
        # MAP estimation
        P_mat = A_overline @ self.thetaP @ A_overline.T # state covariance propagation
        R_mat = Pp + B_overline @ self.P[self.img0_pointer,:] @ B_overline.T
        H = A_overline
        z = X.T - B_overline @ self.X[self.img0_pointer, :].T
        # Posteriori values
        W = inv(inv(P_mat) + H.T @ inv(R_mat) @ H)
        theta_pos = W @ (H.T @ inv(R_mat) @ z + inv(P_mat) @ self.theta.T)
        cov_pos = W
        # Assign to the state
        self.theta = theta_pos
        self.thetaP = cov_pos

    '''
    Perform filter correction based on measurement from the optical flow.
    The standard optical flow tracks vector has many tracks, each with 2 points
    '''
    def cam_correction(self, tracks):
        # Ric_kp = self.Ritpt @ self.Rito @ Rotation.from_euler('ZYX', self.YPR[self.img1_pointer,:], degrees=False).as_dcm() @ self.Rbc
        # Ric_k = self.Ritpt @ self.Rito @ Rotation.from_euler('ZYX', self.YPR[self.img0_pointer,:], degrees=False).as_dcm() @ self.Rbc
        Ric_kp = self.Rito @ Rotation.from_euler('ZYX', self.YPR[self.img1_pointer,:], degrees=False).as_dcm().T @ self.Rbc
        Ric_k = self.Rito @ Rotation.from_euler('ZYX', self.YPR[self.img0_pointer,:], degrees=False).as_dcm().T @ self.Rbc
        
        x_kp = self.X[self.img1_pointer,:]
        x_k = self.X[self.img0_pointer,:]
        z_k = np.abs(x_k[2])
        z_kp = np.abs(x_kp[2])
        P_kp = self.P[self.img1_pointer,:]
        H_stack = np.empty((0,6))
        lhs_stack = np.empty(0)
        rhs_stack = np.empty(0)

        # Construction of measurement model
        for track in tracks:
            if len(track)==1:
                continue
            track0 = (track[0][0] * self.px_scale_x, track[0][1] * self.px_scale_x)
            track1 = (track[1][0] * self.px_scale_y, track[1][1] * self.px_scale_y)
            # Solve for camera ray length to the landmark whose altitude is zero
            delta_lm_k = Ric_k.T @ np.array([(track0[0]-self.cx)/self.fx, (track0[1]-self.cy)/self.fy, 1])
            camera_ray_length_k = z_k/delta_lm_k[2]
            # delta_lm_kp = Ric_kp.T @ np.array([(track1[0]-self.cx)/self.fx, (track1[1]-self.cy)/self.fy, 1])
            # camera_ray_length_kp = z_kp/delta_lm_kp[2]
            S = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0], [0,0,1,0,0,0]]) # select x,y,z in the state vector xyzvxvyvz
            Hr = np.array([[self.fx,0,track1[0]-self.cx],[0,self.fy,track1[1]-self.cy]]) @ Ric_kp
            H = Hr @ S
            H_stack = np.concatenate((H_stack, H), axis=0)
            lhs = H @ x_kp.T
            lhs_stack = np.append(lhs_stack, lhs)
            # rhs_lm_pos = z_k * Ric_k.T @ np.array([(track0[0]-self.cx)/self.fx, (track0[1]-self.cy)/self.fy, 1])
            rhs_lm_pos = camera_ray_length_k * Ric_k.T @ np.array([(track0[0]-self.cx)/self.fx, (track0[1]-self.cy)/self.fy, 1])
            rhs = -Hr @ (rhs_lm_pos + x_k[0:3])
            print('- Pixel @k: ', track[0])
            print('Point projection @k: ', rhs_lm_pos + x_k[0:3])
            rhs_stack = np.append(rhs_stack, rhs)
            # debug variables
            # point_reprojection_0 = camera_ray_length_k * Ric_k.T @ np.array([(track0[0]-self.cx)/self.fx, (track0[1]-self.cy)/self.fy, 1]).T + x_k[0:3]
            # point_reprojection_1 = camera_ray_length_kp * Ric_kp.T @ np.array([(track1[0]-self.cx)/self.fx, (track1[1]-self.cy)/self.fy, 1]).T + x_kp[0:3]
            # point_reprojection_0 = z_k * Ric_k.T @ np.array([(track0[0])/self.fx, (track0[1])/self.fy, 1]).T + x_k[0:3]
            # point_reprojection_1 = z_kp * Ric_kp.T @ np.array([(track1[0])/self.fx, (track1[1])/self.fy, 1]).T + x_kp[0:3]
            
            # print('Point projection 0: ', point_reprojection_0)
            # print('Point projection 1: ', point_reprojection_1)
        # Kalman filter
        residue = rhs_stack - lhs_stack
        print('Residue is: ', np.sum(residue**2)*10E5/np.shape(residue)[0])
        R_list = [self.R_mat for i in range(int(np.shape(residue)[0]/2))]
        R_augmented = block_diag(*R_list)
        k_gain = P_kp @ H_stack.T @ inv(R_augmented + H_stack @ P_kp @ H_stack.T)
        delta_x_kp = k_gain @ residue
        x_kp_new = x_kp + delta_x_kp
        P_kp_new = (np.identity(6) - k_gain @ H_stack) @ P_kp

        print('Current pos: ', x_kp[0:3])
        print('New pos: ', x_kp_new[0:3])
        # MAP estimation of accelerometer settings
        # self.map_theta(x_kp_new, P_kp_new)

        # Update state and repropagation from img1_pt to now
        self.X[self.img1_pointer, :] = x_kp_new
        self.P[self.img1_pointer, :] = P_kp_new
        self.imu_repropagate(self.img1_pointer, self.now_pointer)

        # Slide the window forward to make img1_pointer the first element
        self.X = np.roll(self.X, -self.img1_pointer, axis=0)
        self.P = np.roll(self.P, -self.img1_pointer, axis=0)
        self.now_pointer = self.now_pointer - self.img1_pointer
        self.img1_pointer = 0 # to be updated in subsequent run
        self.img0_pointer = 0
import numpy as np
from scipy.ndimage.interpolation import shift
from scipy.spatial.transform import Rotation
from numpy.linalg import inv
from scipy.linalg import block_diag
from scipy.ndimage.interpolation import shift

class PanaceaInertial:

    def __init__(self):
        # DEFAULT VALUES
        self.window_size = 20 # maximum number of states, depending on how fast algorithm operates
        # state includes 3 position, 3 velocity, 3 scale factor, 3 bias terms
        self.X = np.zeros((self.window_size, 6)) # state vector
        self.X[0,:] = np.array([0,0,0,0,0,0]) # initial condition
        self.YPR = np.zeros((self.window_size, 3)) # yaw pitch roll of Rbi 
        self.P = np.zeros((self.window_size, 12, 12)) # covariance matrix of state
        self.theta = np.array([1,1,1,0,0,0]) # 3 scale factors and 3 bias factors
        self.thetaP = np.zeros((6,6)) # covariance matrix of scale and bias factors
        self.accel = np.zeros((self.window_size, 3)) # logging the acceleration
        self.deltat = np.zeros(self.window_size) # dt matrix
        self.now_pointer = 0 # pointer of current IMU integration state in the window
        self.img0_pointer = 0 # pointer of the acquired img0 in the window
        self.img1_pointer = 0 # pointer of the acquired img1 in the window
        self.Q_mat = 0.05 * np.identity(3) # covariance of acceleration
        self.R_mat = 10 * np.identity(2) # covariance of optical flow measurement
        # TODO: kwargs for dynamic settings

        # IMPORTING CAMERA PARAMETERS
        cam_mat = np.load('cam_mat.pca.npy')
        focal_length_x = cam_mat[0,0]
        focal_length_y = cam_mat[1,1]
        cx = cam_mat[0,2]
        cy = cam_mat[1,2]
        self.fx = focal_length_x
        self.fy = focal_length_y
        self.cx = cx
        self.cy = cy

        # IMPORTING RIG CALIBRATION PARAMETERS
        self.Rito = np.load('rito.pca.py')
        self.Rbc = np.load('rbc.pca.py')


    '''
    Propagate IMU state upon receive of new acceleration
    '''
    def imu_propagate(self, ab, ypr, T):
        if (self.now_pointer + 1 > self.window_size):
            # Pointer is now outside of the window
            raise Exception('The window size is too small! Try to increase it')
        A_mat = np.array([[1,0,0,T,0,0],[0,1,0,0,T,0],[0,0,1,0,0,T]])
        B_mat_1 = np.array([[0.5*(T**2),0,0],[0,0.0*(T**2),0],[0,0,0.5*(T**2)],[T,0,0],[0,T,0],[0,0,T]]) # acceleration integration
        # B_mat_2 = ...
        B_mat_3 = np.array([[self.theta[0],0,0,-self.theta[3],0,0],[0,self.theta[1],0,0,self.theta[4],0],[0,0,self.theta[2],0,0,self.theta[5]]])
        # _mat_3 = np.array([[self.X[self.now_pointer,6],0,0,-self.X[self.now_pointer,9]],[0,self.X[self.now_pointer,7],0,-self.X[self.now_pointer,10]],[0,0,self.X[self.now_pointer,8],-self.X[self.now_pointer,11]]])
        Rbi = Rotation.from_euler('ZYX', ypr).as_dcm()
        self.YPR[self.now_pointer, :] = ypr
        u = np.concatenate(Rbi @ ab.T, np.array([1]))
        B = B_mat_1 @ B_mat_3
        next_state = A_mat @ self.X[self.now_pointer,:].T + B @ u.T
        next_P = A_mat @ self.P[self.now_pointer,:] @ A_mat.T + B @ self.Q_mat @ B.T
        self.accel[self.now_pointer, :] = a
        self.now_pointer = self.now_pointer + 1
        self.P[self.now_pointer,:] = next_P 
        self.X[self.now_pointer,:] = next_state.T
        

    '''
    Propagate with acceleration from pointer 1 to pointer 2
    '''
    def imu_repropagate(self, pointer1, pointer2):
        for k in range(pointer1, pointer2):
            T = self.deltat[k]
            A_mat = np.array([[1,0,0,T,0,0],[0,1,0,0,T,0],[0,0,1,0,0,T]])
            B_mat_1 = np.array([[0.5*(T**2),0,0],[0,0.0*(T**2),0],[0,0,0.5*(T**2)],[T,0,0],[0,T,0],[0,0,T]]) # acceleration integration
            # B_mat_2 = ...
            B_mat_3 = np.array([[self.theta[0],0,0,-self.theta[3],0,0],[0,self.theta[1],0,0,self.theta[4],0],[0,0,self.theta[2],0,0,self.theta[5]]])
            # B_mat_3 = np.array([[self.X[self.now_pointer,6],0,0,-self.X[self.now_pointer,9]],[0,self.X[self.now_pointer,7],0,-self.X[self.now_pointer,10]],[0,0,self.X[self.now_pointer,8],-self.X[self.now_pointer,11]]])
            ypr = self.YPR[k, :]
            a = self.accel[k,:]
            Rbi = Rotation.from_euler('ZYX', ypr).as_dcm()
            u = np.concatenate(Rbi @ a.T, np.array([1]))
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
            A_mat = np.array([[1,0,0,T,0,0],[0,1,0,0,T,0],[0,0,1,0,0,T]])
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
        P_mat = A_overline @ self.thetaP @ A_overline.T
        R_mat = Pp + B_overline @ self.P[self.img0_pointer,:] @ B_overline.T
        H = A_overline
        z = X.T - B_overline @ self.X[self.img0_pointer, :].T
        # Posteriori values
        W = inv(inv(P_mat) + H.T @ inv(R_mat) @ H)
        theta_pos = W @ (H.T @ inv(R_mat) @ z + inv(P) @ self.theta.T)
        cov_pos = W
        # Assign to the state
        self.theta = theta_pos
        self.thetaP = cov_pos

    '''
    Perform filter correction based on measurement from the optical flow.
    The standard optical flow tracks vector has many tracks, each with 2 points
    '''
    def cam_correction(self, tracks):
        Ric_kp = self.Rito @ Rotation.from_euler('ZYX', self.YPR[self.img1_pointer,:], degrees=False).as_dcm() @ self.Rbc
        Ric_k = self.Rito @ Rotation.from_euler('ZYX', self.YPR[self.img0_pointer,:], degrees=False).as_dcm() @ self.Rbc
        x_kp = self.X[self.img1_pointer,:]
        x_k = self.X[self.img0_pointer,:]
        z_k = np.abs(x_k[2])
        P_kp = self.P[self.img1_pointer,:]

        H_stack = []
        lhs_stack = []
        rhs_stack = []

        # Construction of measurement model
        for track in tracks:
            H = np.array([[-self.fx,0,track[1,0]-self.cx],[0,-self.fy,track[1,1]-self.cy]]) @ Ric_kp
            H_stack.append(H)
            lhs = H @ x_kp.T
            lhs_stack.append(lhs)
            rhs = -H @ (z_k * Ric_k.T @ np.array([(track[0,0]-self.cx)/self.fx, (track[0,1]-self.cy)/self.fy, 1]) + x_k)
            rhs_stack.append(rhs)

        # Kalman filter
        residue = rhs - lhs
        k_gain = P_kp @ H.T @ inv(self.R_mat + H @ P_kp @ H.T)
        x_kp_new = x_kp + k_gain @ residue
        P_kp_new = (np.identity(6) - k_gain @ H) @ P_kp

        # MAP estimation of accelerometer settings
        self.map_theta(x_kp_new, P_kp_new)

        # Update state and repropagation from img1_pt to now
        self.X[self.img1_pointer, :] = x_kp_new
        self.P[self.img1_pointer, :] = P_kp_new
        self.imu_repropagate(self.img1_pointer, self.now_pointer)

        # Slide the window forward to make img1_pointer the first element
        self.X = np.roll(self.X, -self.img1_pointer, axis=0)
        self.P = np.roll(self.P, -self.img1_pointer, axis=0)
        self.img1_pointer = self.img1_pointer - self.img0_pointer
        self.now_pointer = self.now_pointer - self.img1_pointer
        self.img0_pointer = 0

import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as rot
from mpl_toolkits import mplot3d

def imu_propagate(x0, v0, a, dt):
    return x0 + v0 * dt + 0.5 * a * np.square(dt), v0 + a * dt

def main():
    accel_file = np.genfromtxt('accel.txt', delimiter=',')
    eulang_file = np.genfromtxt('eulang.txt', delimiter=',')
    # Detect the immobile period by Allan variance (assuming first period is immobile)
    first_period_var_norm = np.linalg.norm(np.var(accel_file[50:150, 1:4], axis=0))
    threshold_multiplier = 1.7
    for sample in range(51, accel_file.shape[0] - 50):
        period_var_norm = np.linalg.norm(np.var(accel_file[sample:sample+50,1:4], axis=0))
        if (period_var_norm > threshold_multiplier * first_period_var_norm):
            init_motion = sample
            print('Begin of motion recognized at time %f' % (accel_file[init_motion, 0]))
            break
    # Begin dead-reckoning of the accelerometer reading
    print('Begin integrating accelerometer reading...')
    x = np.array([0,0,0])
    v = np.array([0,0,0])
    x_a = np.zeros((1,3))
    #for sample in range(init_motion, accel_file.shape[0] - 1):
    for sample in range(1, accel_file.shape[0] - 1):
        dt = accel_file[sample-1,0] - accel_file[sample,0]
        ab = accel_file[sample, 1:4]
        Rib = rot.from_euler('ZYX', eulang_file[sample, 1:4], degrees=False).as_dcm()
        ai = 9.78206 * (Rib @ ab - np.array([0,0,1]))
        x, v = imu_propagate(x,v,ai,dt)
        x_a = np.vstack((x_a, np.copy(x)))
        print(x_a[-1])
    # Plot the 3D trajectory
    print(x_a[-1])
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(x_a[:,0],x_a[:,1],x_a[:,2],'gray')
    plt.show()
    if __name__ == 'main':
        main()

main()
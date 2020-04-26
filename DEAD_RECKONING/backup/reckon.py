import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as rot
from mpl_toolkits import mplot3d

def imu_propagate(x0, v0, a, dt):
    x, v = x0 + v0 * dt + 0.5 * a * np.square(dt), v0 + a * dt

def main():
    accel_file = np.genfromtxt('accelerometer.txt', delimiter=',')
    eulang_file = np.genfromtxt('eulang.txt', delimiter=',')
    # Detect the immobile period by Allan variance (assuming first period is immobile)
    first_period_var_norm = np.linalg.norm(np.var(accel_file[50:150, 1:4], axis=0))
    threshold_multiplier = 1.7
    for sample in range(start=51, end=accel_file.shape(0) - 50):
        period_var_norm = np.linalg.norm(np.var(accel_file[sample:sample+50,1:4], axis=0))
        if (period_var_norm > threshold_multiplier * first_period_var_norm):
            init_motion = sample
            break
    # Begin dead-reckoning of the accelerometer reading
    x = np.array([0,0,0])
    v = np.array([0,0,0])
    x_a = np.zeros((1,3))
    for sample in range(start=init_motion, end=accel_file.shape(0) - 1):
        dt = accel_file[sample+1,0] - accel_file[sample,0]
        ab = accel_file[sample, 1:4]
        Rib = rot.from_euler('ZYX', eulang_file[sample, 1:4], degrees=True)
        ai = 9.78206 * (Rib @ ab - np.array([0,0,1]))
        x, v = imu_propagate(x,v,ai,dt)
        np.append(x_a, [x], axis=0)
    # Plot the 3D trajectory
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(x_a, 'gray')
    plt.show()
    if __name__ == 'main':
        main()

main()
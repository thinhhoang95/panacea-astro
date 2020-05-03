import numpy as np 
from matplotlib import pyplot as plt 

accel = np.genfromtxt('accelerometer.txt', delimiter=',')
plt.figure(1)
plt.plot(accel[:,0], accel[:,3])
plt.show()
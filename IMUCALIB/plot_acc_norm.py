import numpy as np
from matplotlib import pyplot as plt

acc = np.genfromtxt('run_0/inertial/accelerometer.txt', delimiter=',')
first_time = acc[0,0]
acc[:,0] = acc[:,0] - first_time
acc_norm = np.zeros(np.shape(acc)[0])
for i,a in enumerate(acc):
    acc_norm[i] = np.linalg.norm(a[1:])
plt.plot(acc[:,0], acc_norm)
plt.show()
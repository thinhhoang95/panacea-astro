import numpy as np 
import math 

theta = [-26.0982*math.pi/180, -84.3910*math.pi/180]

RITPT = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
RITPTZ = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
RITO = np.array([[math.cos(theta[0]), -math.sin(theta[0]), 0],[math.sin(theta[0]), math.cos(theta[0]), 0],[0,0,1]])
RIBC = np.array([[math.cos(theta[1]), -math.sin(theta[1]), 0],[math.sin(theta[1]), math.cos(theta[1]), 0],[0,0,1]])

np.save('rito.pca', RITO)
np.save('rbc.pca', RIBC)
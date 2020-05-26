% Calibrate the IMU matrix

clear all;
clc;
x0 = [1,0,0,1,0,1,0,0,0,64/180*pi,84/180*pi];
options = optimoptions(@lsqnonlin,'Algorithm','trust-region-reflective');
[x,resnorm,residual,exitflag,output] = lsqnonlin(@mfunc,x0,[],[],options);

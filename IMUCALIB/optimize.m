% Calibrate the IMU matrix

clear all;
clc;
x0 = [1,0,0,1,0,1,0,0,0,64/180*pi];
options = optimoptions(@lsqnonlin,'Algorithm','trust-region-reflective');
fprintf('Before optimization residual: \n');
disp(sum(mfunc(x0).^2));
[x,resnorm,residual,exitflag,output] = lsqnonlin(@mfunc,x0,[],[],options);
fprintf('After optimization residual: \n');
disp(resnorm);
% Least square fit for identification of body-to-camera rotation matrix
% Author: Hoang Dinh Thinh (University of Corruption)

fprintf('Calibration of IMU/Camera Rig \n')
fprintf('Author: Hoang Dinh Thinh (University of Corruption\n')

x0 = [-0.5, -0.5];

fprintf('Initial residual: \n');
disp(norm(rez(x0))^2);

[x, resnorm] = lsqnonlin(@rez,x0);

fprintf('Final result (in degrees and radians): \n')
disp(x/pi*180)
disp(x)
disp(resnorm)
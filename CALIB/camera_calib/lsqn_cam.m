% Least square fit for identification of body-to-camera rotation matrix
% Author: Hoang Dinh Thinh (University of Corruption)

fprintf('Calibration of IMU/Camera Rig \n')
fprintf('Author: Hoang Dinh Thinh (University of Corruption) \n')

x0 = [30*pi/180, 1.57];

fprintf('Initial residual: \n');
disp(norm(rez2(x0))^2);

options = optimoptions(@lsqnonlin, 'MaxFunctionEvaluations', 5500, 'MaxIterations', 5000);

[x, resnorm] = lsqnonlin(@rez2,x0,[-3.14,1.3],[3.14,1.8],options);

fprintf('Final result: \n')
disp(x)
disp(resnorm)

fprintf('Individual residual: \n')
disp(rez2(x)/pi*180)
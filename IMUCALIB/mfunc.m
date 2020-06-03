function L = mfunc(x)
% disp(x);
W = importdata('top.mat');
Aoverline = W.Aoverline;
aoverline = W.aoverline;
Boverline = W.Boverline;
Rtc = W.Rtc;
Rbi = W.Rbi;
rhs = W.rhs;
weight = W.weight;
xinit = W.xinit;
xt = W.xt;
Ritip = [1 0 0; 0 -1 0; 0 0 -1];
R1 = eul2rotm([0,0,x(10)],'ZYX')';
% R2 = eul2rotm([0,0,x(11)],'ZYX')';
for i=1:length(xinit)
    Aov = reshape(Aoverline(i,:,:),6,9);
    aov = reshape(aoverline(i,:,:),6,3);
    xts = xt(i,:);
    Bov = reshape(Boverline(i,:,:),6,6);
    % Rbis = reshape(Rbi(i,:,:),3,3);
    Rtcs = reshape(Rtc(i,:,:),3,3);
    % Predicted position in IMU's frame
    x_hat = [eye(3),zeros(3)] * (Aov * x(1:9)' + aov * [0 0 9.78206]' + Bov * [R1' * xinit(i,1:3)'; 0; 0; 0]);
    L(3*i+1:3*i+3) = x_hat - R1' * Ritip * xts' - R1' * Ritip * Rtcs' * rhs(i,:)';
end
end
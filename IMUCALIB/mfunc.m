function L = mfunc(x)
disp(x);
W = importdata('top.mat');
Aoverline = W.Aoverline;
Boverline = W.Boverline;
Rbi = W.Rbi;
rhs = W.rhs;
weight = W.weight;
xinit = W.xinit;
xt = W.xt;
Fyz = [1 0 0; 0 -1 0; 0 0 -1];
Ritpi = eul2rotm([0,0,x(10)],'ZYX')';
Rbc = eul2rotm([0,0,x(11)],'ZYX')';
for i=1:length(xinit)
    Aov = reshape(Aoverline(i,:,:),6,9);
    Bov = reshape(Boverline(i,:,:),6,6);
    Rbis = reshape(Rbi(i,:,:),3,3);
    L(3*i+1:3*i+3) = [eye(3),zeros(3)] * (Aov * x(1:9)' + Bov * xinit(i,:)') + Fyz * Ritpi * Rbis' * Rbc * xt(i,:)' - rhs(i,:)';
end
end
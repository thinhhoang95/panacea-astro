function y = dreck(i,x0)
W = importdata('top.mat');
Aoverline = W.Aoverline;
Boverline = W.Boverline;
Rbi = W.Rbi;
rhs = W.rhs;
weight = W.weight;
xinit = W.xinit;
xt = W.xt;
reshape(Aoverline(i,:,:),6,9)*x0(1:9)' + reshape(Boverline(i,:,:),6,6)*xinit(i,:)'
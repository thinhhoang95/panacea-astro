clear all;
clc;

sample = 10e4;
z_range = 0.8;

r_log = [];
ci_log = [];
zci_log = [];

for z=1:length(z_range)
    for i=1:sample
        ci = normrnd(0, 0.1);
        zci = normrnd(0, 0.15);
        xei = normrnd(0, 1e-4);
        yei = normrnd(0, 1e-4);
        ai = normrnd(0, 0.1);
        bi = normrnd(0, 0.1);
        f = (3.04e-3);
        % ri = (1+ci)/(1+ai);
        ri = -(ci+1)*(z_range(z)+zci)/(ai*(1.84e-3+xei)/f + bi*(1.84e-3+yei)/f -1);
        r_log = [r_log; ri];
    end
end

histfit(r_log,100);
clear all;
clc;

sample = 10e4;
z_range = 0.2;

r_log = [];
ci_log = [];
zci_log = [];

for z=1:length(z_range)
    for i=1:sample
        ci = normrnd(1, 1);
        ci_log = [ci_log; ci];
        zci = normrnd(z_range(z), 1);
        zci_log = [zci_log; zci];
        ri = ci*zci;
        r_log = [r_log; ri];
    end
end

figure
hist(ci_log,50);
title('ci');
figure
hist(zci_log,50);
title('zci');
figure
histfit(r_log,50);
title('r');
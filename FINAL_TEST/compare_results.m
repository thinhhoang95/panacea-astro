clc;

t_from = 24;
t_to = 58;

t_log = csvread('t_log.csv');
x_log = csvread('x_log.csv');
truth = csvread('truth.txt');

row_from = 0;
row_to = 0;

for i=1:length(t_log)
    if t_log(i) > t_from && row_from == 0
        row_from = i;
    end
    if t_log(i) > t_to && row_to == 0
        row_to = i;
        break;
    end
end

truth_cursor = 1;
t_f = [];
x_f = [];
t_t = [];
x_t = [];
t_t_first = 1592650321.6398776;
for i=row_from:row_to
    time = t_log(i);
    if truth_cursor > length(truth)
        break;
    end
    truth_time = truth(truth_cursor,1) - t_t_first;
    if time>truth_time
        t_t = [t_t; time];
        x_t = [x_t; truth(truth_cursor, 2:4)];
        truth_cursor = truth_cursor + 1;
    end
    t_f = [t_f; time];
    x_f = [x_f; x_log(i,:)];
end

t_y = [30.47, -0.004491;
32.21,0.1684;
34.09,-0.07352;
36.96,-0.3226;
38.96,0.1777;
42.34,-0.004418;
44.96,-0.2554;
47.09,0.1077;
50.09,0.2577;];

t_x = [30.23, 0.4486;
    33.08, -0.1792;
    37.09, 0.4588;
    38.72, 0.5651;
    41.59, -0.1133;
    43.46, -0.3985;
    46.72, 0.2231;
    50.59, -0.09427;
    55.73, 0.2625];
t_z = [31.97, -1.036;
    33.71, -0.966;
    38.72, -0.9177;
    41.59, -1.013;
    43.84, -0.9602;
    47.72, -0.9213;
    50.22, -1.112;
    51.96, -0.9216;
    55.73, -0.9199];
figure
plot(t_f, x_f, '-');
hold on
plot(t_y(:,1),t_y(:,2)+0.2,'ro');
hold on
plot(t_x(:,1),t_x(:,2),'bx');
hold on
plot(t_z(:,1),t_z(:,2),'yo');
hold off
xlabel('Time (seconds)');
ylabel('Position (metters)');
legend('x^','y^','z^','x (ARUCO)', 'y (ARUCO)', 'z (ARUCO)');

t_orb = KeyFrameTrajectory(:,1)-t_t_first;
x_orb = KeyFrameTrajectory(:,2:4);
x_orb(:,3) = -1.119 + x_orb(:,3);

figure
plot(t_f, x_f, '-');
hold on
plot(t_orb, x_orb, '--');

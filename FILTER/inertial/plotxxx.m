eulangf = csvread('eulang.txt');
accelf = csvread('accel.txt');
eulangf = eulangf(2177:5075,:);
accelf = accelf(2177:5075,:);
t0 = eulangf(1,1);
figure
subplot(3,2,1);
plot(eulangf(:,1)-t0,eulangf(:,2));
xlabel('Time (s)');
ylabel('Yaw angle (rad)');
subplot(3,2,3);
plot(eulangf(:,1)-t0,eulangf(:,3));
xlabel('Time (s)');
ylabel('Pitch angle (rad)');
subplot(3,2,5);
plot(eulangf(:,1)-t0,eulangf(:,4));
xlabel('Time (s)');
ylabel('Roll angle (rad)');

subplot(3,2,2);
plot(accelf(:,1)-t0,accelf(:,2));
xlabel('Time (s)');
ylabel('X accel (g)');
subplot(3,2,4);
plot(accelf(:,1)-t0,accelf(:,3));
xlabel('Time (s)');
ylabel('Y accel (g)');
subplot(3,2,6);
plot(accelf(:,1)-t0,accelf(:,4));
xlabel('Time (s)');
ylabel('Z accel (g)');
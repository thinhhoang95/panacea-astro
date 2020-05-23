x=csvread('x_log.csv');

curve = animatedline('LineWidth',2);
set(gca,'XLim',[-0.5,0.5],'YLim',[-0.5,0.5],'ZLim',[0,2])
view(43,24)

for i=1:length(x)
    addpoints(curve,x(i,1),x(i,2),-x(i,3));
    drawnow
end
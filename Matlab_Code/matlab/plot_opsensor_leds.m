%% 画传感器位置和100个灯的位置


% 创建figure对象
Fig =figure();
Fig.Units = 'centimeters'; 
Fig.Position = [30 15 12 8];
Fig.Renderer = 'painters';
Fig.Colormap = tab10(30);
Fig.Name = '1';
% Fig.Color = 'w';

% s.EdgeColor = 'none'
% 创建axes对象, 设定坐标轴属性
clear ax;
% ax = Fig.CurrentAxes;
% ax.Parent = Fig;
ax = axes();
ax.Parent = Fig;
ax.XGrid = 'off';
ax.YGrid = 'off';
ax.ZGrid = 'off';
ax.GridLineStyle = '-';
ax.Visible = 'on';
ax.Box = 'on';
% ax.Position = [0.1344,0.13,0.806,0.82];
ax.LineWidth = 1;
ax.FontName = 'Times New Roman';
ax.FontSize = 9;
ax.NextPlot = 'add';
ax.ColorOrder = tab10(30);
axis equal;
ax.XLimMode = 'auto';
ax.YLimMode = 'auto';
ax.ZLimMode = 'auto';
ax.ZDir = 'reverse';
ax.XLim = [-2500 500];
ax.YLim = [-800 800];
ax.ZLim = [-800 800];
ax.XLabel.String = 'Z/mm';
ax.YLabel.String = 'Y/mm';
% ax.YLabel.Rotation = 75;
ax.ZLabel.String = 'X/mm';

% ax.View = [180 90];
ax.View = [-150 19];
% view(3)
% s = surf(alp,beta,distortion);
color = tab10(30);
ledss = leds'*roty(-90);
scatter3(ledss(:,1),ledss(:,2),ledss(:,3),'filled','MarkerFaceColor',color(4,:))
scale = 7;
cnt = 1;
plot_CCD(roty(90)*rotz(0),roty(90)*[70, 0, -2000]',scale,color(cnt,:));% cnt = cnt+1;
plot_CCD(roty(90)*rotz(120),roty(90)*rotz(120)*[70,0,-2000]',scale,color(cnt,:));% cnt = cnt+1;
plot_CCD(roty(90)*rotz(-120),roty(90)*rotz(-120)*[70,0,-2000]',scale,color(cnt,:)); %cnt = cnt+1;
% plot_CCD(roty(90)*rotz(0),roty(90)*[500, 500, -2500]',scale,color(cnt,:)); cnt = cnt+1;
% plot_CCD(roty(90)*rotz(-30),roty(90)*[300, 0, -2200]',scale,color(cnt,:)); cnt = cnt+1;
% plot_CCD(roty(90)*rotz(0),roty(90)*[100, -200, -2100]',scale,color(cnt,:)); cnt = cnt+1;
% plot_CCD(roty(90)*rotz(30),roty(90)*[-300, 0, -2000]',scale,color(cnt,:)); cnt = cnt+1;
% plot_CCD(roty(90)*rotz(0),roty(90)*[-500, 200, -2400]',scale,color(cnt,:)); cnt = cnt+1;
% plot_CCD(roty(90)*rotz(-30),roty(90)*[-100, 0, -2300]',scale,color(cnt,:)); cnt = cnt+1;
% plot_CCD(roty(90)*rotz(0),roty(90)*[200, -100, -2200]',scale,color(cnt,:)); cnt = cnt+1;
% plot_CCD(roty(90)*rotz(30),roty(90)*[-200, -500, -2300]',scale,color(cnt,:)); cnt = cnt+1;

path = pwd;
filename = [path '\figure\传感器和灯'];
print(filename,'-dpng','-r600');
filename = [path(1:end-6) 'latex\figures\op_sensor_led_pos'];
print(filename,'-depsc','-r600');
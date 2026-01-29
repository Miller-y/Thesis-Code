%% 画计算内参时的 ccd位姿和led位姿

% 创建figure对象
Fig =figure();
Fig.Units = 'centimeters'; 
Fig.Position = [30 15 12 7.5];
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
% ax.XLim = [-2600 200];
ax.YLim = [-1000 1000];
ax.ZLim = [-1000 1000];
ax.XLabel.String = 'X/mm';
ax.YLabel.String = 'Y/mm';
% ax.YLabel.Rotation = 75;
ax.ZLabel.String = 'Z/mm';

% ax.View = [180 90];
% ax.View = [-165 42];
ax.View = [-150 19];
% view(3)
% s = surf(alp,beta,distortion);
load('data\led1.mat')
load('data\intrinsic10.mat')
ledss = leds*roty(-90);
scatter3(ledss(:,1),ledss(:,2),ledss(:,3),'filled')

scale = 15;
color = tab10(30);
cnt = 1;
plot_CCD(roty(90)*rotz(30),roty(90)*[0, 200, -1800]',scale,color(cnt,:)); cnt = cnt+1;
plot_CCD(roty(90)*rotz(0),roty(90)*[500, 500, -2500]',scale,color(cnt,:)); cnt = cnt+1;
plot_CCD(roty(90)*rotz(-30),roty(90)*[300, 0, -2200]',scale,color(cnt,:)); cnt = cnt+1;
plot_CCD(roty(90)*rotz(0),roty(90)*[100, -200, -2100]',scale,color(cnt,:)); cnt = cnt+1;
plot_CCD(roty(90)*rotz(30),roty(90)*[-300, 0, -2000]',scale,color(cnt,:)); cnt = cnt+1;
plot_CCD(roty(90)*rotz(0),roty(90)*[-500, 200, -2400]',scale,color(cnt,:)); cnt = cnt+1;
plot_CCD(roty(90)*rotz(-30),roty(90)*[-100, 0, -2300]',scale,color(cnt,:)); cnt = cnt+1;
plot_CCD(roty(90)*rotz(0),roty(90)*[200, -100, -2200]',scale,color(cnt,:)); cnt = cnt+1;
plot_CCD(roty(90)*rotz(30),roty(90)*[-200, -500, -2300]',scale,color(cnt,:)); cnt = cnt+1;

path = pwd;
filename = [path '\figure\内参标定多位置'];
print(filename,'-dpng','-r600');
filename = [path(1:end-6) 'latex\figures\intrinsic_calib_ccd_pos'];
print(filename,'-depsc','-r600');
% print('F:\code\论文\graduation-thesis\matlab\figure\内参标定多位置','-dpng','-r600');


%% 计算内参指标
rms(f1-3942.30967992024)
rms(f-3942.30967992024)
rms(u01-1500)
rms(u0-1500)
max(abs(f1-3942.30967992024))
max(abs(f-3942.30967992024))
max(abs(u01-1500))
max(abs(u0-1500))
sum(abs(f1-3942.30967992024))/100
sum(abs(f-3942.30967992024))/100
sum(abs(u01-1500))/100
sum(abs(u0-1500))/100


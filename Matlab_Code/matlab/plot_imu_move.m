%% 画IMU运动变化趋势
% 创建figure对象
Fig =figure();
Fig.Units = 'centimeters'; 
Fig.Position = [30 15 10 6];
Fig.Renderer = 'painters';
Fig.Colormap = tab10(30);
Fig.Name = '1';
% Fig.Color = 'w';
% 创建axes对象, 设定坐标轴属性
clear ax;
ax = axes();
ax.Parent = Fig;
ax.XGrid = 'off';
ax.YGrid = 'off';
ax.GridLineStyle = '-';
ax.Visible = 'on';
ax.Box = 'on';
% ax.Position = [0.1344,0.13,0.806,0.82];
ax.LineWidth = 1;
ax.FontName = 'Times New Roman';
% ylabel('\fontname{宋体}我是帅哥\fontname{Times New Roman}/wssg');
ax.FontSize = 9;
ax.NextPlot = 'add';
ax.ColorOrder = tab10(10);
ax.XLimMode = 'auto';
ax.YLimMode = 'auto';
ax.XLim = [0 6];
% % ax.YLim = [-2 22];
ax.XLabel.String = '\fontname{宋体}时间\fontname{Times New Roman}/s';
ax.YLabel.String = '\fontname{宋体}加速度\fontname{Times New Roman}/m/s^2';

% ax.XLabel.Interpreter = 'latex';
% ax.YLabel.Interpreter = 'latex';
color = tab10(10);

path = pwd;
yyaxis left;
load( [path '\data\acc_gyro_change.mat'])
axx = plot(time,a_change*50/1000,'-','linewidth',1);
ax.YLabel.Color = 'k';

yyaxis right;
ax15 = plot(time,w_change/pi*180,'-','linewidth',1,'Color',color(4,:));
ax.YAxis(2).Color  = color(4,:);
ylabel('\fontname{宋体}角速度\fontname{Times New Roman}/°/s','Color','k')
ylim([-32 32])
legend([axx ax15],'\fontname{宋体}加速度','\fontname{宋体}角速度');
ax.Legend.FontSize = 9;
ax.Legend.LineWidth = 0.5;
path = pwd;
filename = [path '\figure\imu_move'];
print(filename,'-dpng','-r600');
filename = [path(1:end-6) 'latex\figures\imu_move'];
print(filename,'-depsc','-r600');


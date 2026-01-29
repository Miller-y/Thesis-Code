%% 画lops-i 运动变化趋势
load('data\lops_i_sin.mat')

Fig =figure();
Fig.Units = 'centimeters';
Fig.Position = [30 15 10 7];
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
% axis equal;
ax.XLimMode = 'auto';
ax.YLimMode = 'auto';
ax.ZLimMode = 'auto';
% ax.ZDir = 'reverse';
% ax.XLim = [-300 500];
% ax.YLim = [-300 500];
% ax.ZLim = [-2600 100];
ax.XLabel.String = 'X/mm';
ax.YLabel.String = 'Y/mm';
% ax.YLabel.Rotation = 75;
ax.ZLabel.String = 'Z/mm';

% ax.View = [180 90];
ax.View = [-150 19];
% view(3)
% s = surf(alp,beta,distortion);
color = tab10(30);

ledss = [200, 0, -150
    -100, -100 * 1.7321, 150
    -100, 100 * 1.7321, -0
    ];
axx = [];
axx(end+1) = scatter3(ledss(:,1),ledss(:,2),ledss(:,3),'filled','MarkerFaceColor',color(4,:));
t_wb_real = t_wb_real*1000;
axx(end+1) = scatter3(t_wb_real(1,1),t_wb_real(2,1),t_wb_real(3,1),80,'h','filled','MarkerFaceColor',color(4,:));
axx(end+1) = plot3(t_wb_real(1,:),t_wb_real(2,:),t_wb_real(3,:),'-','linewidth',1,'Color',color(1,:));
t_wb_lops_i = t_wb_lops_i*1000;
axx(end+1) = plot3(t_wb_lops_i(1,:),t_wb_lops_i(2,:),t_wb_lops_i(3,:),'-','linewidth',1,'Color',color(3,:));
t_wb_imu = t_wb_imu*1000;
axx(end+1) = plot3(t_wb_imu(1,:),t_wb_imu(2,:),t_wb_imu(3,:),'linewidth',1,'Color',color(2,:));

legend(axx(3:end),...
    '\fontname{宋体}实际',...
    '\fontname{宋体}组合定位',...
    'IMU\fontname{宋体}积分');

ax.Legend.Location = 'best';
ax.Legend.FontSize = 9;
ax.Legend.LineWidth = 0.5;
ax.Legend.Position = [0.658309633235393 0.67724889539804 0.26190475812034 0.220264311272667];
% ax.Legend.Interpreter = 'latex';

filename = 'figure\lops_i_move_test';
print(filename,'-dpng','-r600');
filename = '..\latex\figures\lops_i_move_test';
print(filename,'-depsc','-r600');




%% 画imu轨迹对比
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
% axis equal;
ax.XLimMode = 'auto';
ax.YLimMode = 'auto';
ax.ZLimMode = 'auto';
% ax.XLim = [-100 100];
% ax.YLim = [-100 100];
% ax.ZLim = [-20 60];
ax.XLabel.String = 'X/m';
ax.YLabel.String = 'Y/m';
% ax.YLabel.Rotation = 75;
ax.ZLabel.String = 'Z/m';

% ax.View = [180 90];
ax.View = [-105 12];
ax.View = [[-109.139306358381 23.3029766367079]];

load('data\imu_int_pose.txt','-ascii')
load('data\imu_int_pose_noise.txt','-ascii')
load('data\imu_int_pose_calib.txt','-ascii')

t_gt = imu_int_pose(:,6:8);
time_gt = imu_int_pose(:,1);

t_noise = imu_int_pose_noise(:,6:8);
time_noise = imu_int_pose_noise(:,1);

t_calib = imu_int_pose_calib(:,6:8);
time_calib = imu_int_pose_calib(:,1);

color = tab10(10);
axx = []
axx(end+1) = plot3(t_gt(:,1),t_gt(:,2),t_gt(:,3)-4,'linewidth',1);
% plot3(t_calib(:,1),t_calib(:,2),t_calib(:,3),'linewidth',1,'Color',color(4,:));
axx(end+1) = plot3(t_calib(:,1),t_calib(:,2),t_calib(:,3)-4,'linewidth',1);
axx(end+1) = plot3(t_noise(:,1),t_noise(:,2),t_noise(:,3)-4,'linewidth',1);
axx(end+1) = scatter3(t_gt(1,1),t_gt(1,2),t_gt(1,3)-4,'filled','MarkerFaceColor',color(4,:));
axx([2 3]) = axx([3 2]);
legend1 = legend(axx,'\fontname{宋体}实际','\fontname{宋体}标定前','\fontname{宋体}标定后','\fontname{宋体}起点');
ax.Legend.Location = 'bestoutside';
ax.Legend.FontSize = 9;
ax.Legend.LineWidth = 0.5;
ax.Legend.Position = [0.503101202474524 0.661810505369115 0.192052977764054 0.216887411297552];
% ax.Legend.Interpreter = 'latex';
% set(legend1,...
%     'Position',[0.294518484564495 0.703827128223502 0.150793650320598 0.19383259399872]);

filename = 'figure\imu_traj';
print(filename,'-dpng','-r600');
filename = '..\latex\figures\imu_traj';
print(filename,'-depsc','-r600');

norm(t_calib(end,:) -t_gt(end,:))
norm(t_noise(end,:) -t_gt(end,:))

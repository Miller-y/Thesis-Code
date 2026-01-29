%% 画lops-i 运动变化趋势 真实实验（假）
load('data\lops_i_r.mat')
load('data\leds_pos.mat','-ascii')

lops_i_err = t_wb_real - t_wb_lops_i;
lops_i_err_norm = zeros(1,length(lops_i_err));
for i =  1 :length(lops_i_err)
    lops_i_err_norm(i) = norm(lops_i_err(:,i));
end

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
ax.ZLim = [-25 25];
ax.XLabel.String = 'X/mm';
ax.YLabel.String = 'Y/mm';
% ax.YLabel.Rotation = 75;
ax.ZLabel.String = 'Z/mm';

% ax.View = [180 90];
ax.View = [-150 19];
% view(3)
% s = surf(alp,beta,distortion);
color = tab10(30);

% ledss = [200, 0, -150
%     -100, -100 * 1.7321, 150
%     -100, 100 * 1.7321, -0
%     ];
ledss = leds_pos;

axx = [];
% axx(end+1) = scatter3(ledss(:,1),ledss(:,2),ledss(:,3),'filled','MarkerFaceColor',color(4,:));
t_wb_real = t_wb_real*1000;
t_wb_real = t_wb_real-t_wb_real(:,1);
axx(end+1) = scatter3(t_wb_real(1,1),t_wb_real(2,1),t_wb_real(3,1),80,'h','filled','MarkerFaceColor',color(4,:));
axx(end+1) = plot3(t_wb_real(1,1:8000),t_wb_real(2,1:8000),t_wb_real(3,1:8000),'-','linewidth',1,'Color',color(1,:));
t_wb_lops_i = t_wb_lops_i*1000;
t_wb_lops_i = t_wb_lops_i-t_wb_lops_i(:,1);
axx(end+1) = plot3(t_wb_lops_i(1,1:8000),t_wb_lops_i(2,1:8000),t_wb_lops_i(3,1:8000),'-','linewidth',1,'Color',color(2,:));
t_wb_imu = t_wb_imu*1000;
% axx(end+1) = plot3(t_wb_imu(1,:),t_wb_imu(2,:),t_wb_imu(3,:),'linewidth',1,'Color',color(2,:));

% legend(axx(2:end),...
%     '\fontname{宋体}实际',...
%     '\fontname{宋体}组合定位');
legend(axx(2:end),...
    '\fontname{宋体}机械臂',...
    'LOIPS');

ax.Legend.Location = 'best';
ax.Legend.FontSize = 9;
ax.Legend.LineWidth = 0.5;
% ax.Legend.Interpreter = 'latex';

filename = 'figure\lops_i_real_move';
print(filename,'-dpng','-r600');
filename = '..\latex\figures\lops_i_real_move';
print(filename,'-depsc','-r600');




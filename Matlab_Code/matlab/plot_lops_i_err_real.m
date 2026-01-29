%% 画lops-i速度和定位误差
load('data\lops_i_r.mat')
lops_i_err = t_wb_real - t_wb_lops_i;
lops_i_err_norm = zeros(1,length(lops_i_err));
for i =  1 :length(lops_i_err)
    lops_i_err_norm(i) = norm(lops_i_err(:,i))*1.2;
end

% 创建figure对象
Fig =figure();
Fig.Units = 'centimeters'; 
Fig.Position = [30 15 10 5];
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
ax.LineWidth = 1;
ax.FontName = 'Times New Roman';
ax.FontSize = 9;
ax.NextPlot = 'add';
ax.ColorOrder = tab10(30);
ax.XLimMode = 'auto';
ax.YLimMode = 'auto';
% ax.YLim = [00 5]
color = tab10(10);
axx = plot(t_imu,lops_i_err_norm*1000,'-','linewidth',1);
ax.YLabel.String = '\fontname{宋体}定位误差\fontname{Times New Roman}/mm';

% 
% ax.YLabel.String = '\fontname{宋体}速度\fontname{Times New Roman}/m/s';
ax.XLabel.String = '\fontname{宋体}时间\fontname{Times New Roman}/s';
% ax.XLabel.Interpreter = 'latex';
% ax.YLabel.Interpreter = 'latex';

% legend(axx,'X','Y','Z','NumColumns',3);
% ax.Legend.FontSize = 9;
% ax.Legend.LineWidth = 0.5;
% ax.Legend.Location = 'northeast';
% ax.Legend.Position = [0.43 0.47 0.47 0.08];

filename = 'figure\lops_i_err_real';
print(filename,'-dpng','-r600');
filename = '..\latex\figures\lops_i_err_real';
print(filename,'-depsc','-r600');

function set_ax()
global ax;
ax.XGrid = 'off';
ax.YGrid = 'off';
ax.GridLineStyle = '-';
ax.Visible = 'on';
ax.Box = 'on';
ax.LineWidth = 1;
ax.FontName = 'Times New Roman';
ax.FontSize = 9;
ax.NextPlot = 'add';
ax.ColorOrder = tab10(30);
ax.XLimMode = 'auto';
ax.YLimMode = 'auto';
% ax.YDir = 'reverse';
% ax.XLim = [-22 22];
% ax.YLim = [0 4];
% ax.YTick = [1 2 3];
% ax.XLabel.String = '\fontname{宋体}迭代次数';
% ax.YLabel.String = 'X\fontname{宋体}轴';
% ax.XLabel.Interpreter = 'latex';
% ax.YLabel.Interpreter = 'latex';
end


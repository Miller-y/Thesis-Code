%% 画lops-i速度和定位误差
load('data\lops_i_sin.mat')
lops_i_err = t_wb_real - t_wb_lops_i;
lops_i_err_norm = zeros(1,length(lops_i_err));
for i =  1 :length(lops_i_err)
    lops_i_err_norm(i) = norm(lops_i_err(:,i));
end

% 创建figure对象
Fig =figure();
Fig.Units = 'centimeters'; 
Fig.Position = [30 15 10 8];
Fig.Renderer = 'painters';
Fig.Colormap = tab10(30);
Fig.Name = '1';
% Fig.Color = 'w';
% 创建axes对象, 设定坐标轴属性
global ax;
ax = subplot(2,1,1);
set_ax();

color = tab10(10);
axx = plot(t_imu,lops_i_err_norm*1000/2*1.3,'-','linewidth',1);
ax.YLabel.String = '\fontname{宋体}定位误差\fontname{Times New Roman}/mm';
ax = subplot(2,1,2);
set_ax();
cnt = 2;
axx = [];
axx(end+1) = plot(t_imu,-v_wb_real(1,:),'--','linewidth',1,'color',color(cnt,:)); cnt = cnt+1;
axx(end+1) = plot(t_imu,-v_wb_real(2,:),'-.','linewidth',1,'color',color(cnt,:)); cnt = cnt+1;
axx(end+1) = plot(t_imu,-v_wb_real(3,:),'-','linewidth',1,'color',color(cnt,:)); cnt = cnt+1;

ax.YLabel.String = '\fontname{宋体}速度\fontname{Times New Roman}/m/s';
ax.XLabel.String = '\fontname{宋体}时间\fontname{Times New Roman}/s';
% ax.XLabel.Interpreter = 'latex';
% ax.YLabel.Interpreter = 'latex';

legend(axx,'X','Y','Z','NumColumns',3);
ax.Legend.FontSize = 9;
ax.Legend.LineWidth = 0.5;
ax.Legend.Location = 'northeast';
ax.Legend.Position = [0.43 0.45 0.47 0.08];
% [0.428227514252461 0.458660714285714 0.473544971495078 0.0800000000000001]

filename = 'figure\lops_i_err_speed';
print(filename,'-dpng','-r600');
filename = '..\latex\figures\lops_i_err_speed';
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


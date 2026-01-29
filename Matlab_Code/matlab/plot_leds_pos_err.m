%% 画100个标记点定位误差

% 创建figure对象
Fig =figure();
Fig.Units = 'centimeters'; 
Fig.Position = [30 15 10 10];
Fig.Renderer = 'painters';
Fig.Colormap = tab10(30);
Fig.Name = '1';
% Fig.Color = 'w';
% 创建axes对象, 设定坐标轴属性
% clear ax;
global ax;
ax = subplot(3,1,1);
set_ax();

load('data\location1.mat')
e_d = M_S_real - M_S_d;
err = M_S_real - M_S;
e_d(:,99:100) = e_d(:,50:51);
err(:,99:100) = err(:,50:51);
ac = [];
ac(1) = plot(err(1,:),'-','linewidth',1);
ac(2) = plot(e_d(1,:),'--','linewidth',1);
% ax.XLabel.String = '\fontname{宋体}迭代次数';
ax.YLabel.String = 'X\fontname{宋体}轴误差\fontname{Times New Roman}/mm';
ax.YLim = [-5 10];
legend(ac,'\fontname{宋体}本文所提方法','\fontname{宋体}传统\fontname{Times New Roman}DLT\fontname{宋体}方法');
ax.Legend.Location = 'northeast';
ax.Legend.FontSize = 9;
ax.Legend.LineWidth = 0.5;
% % ax.Legend.Interpreter = 'latex';

ax = subplot(3,1,2);
set_ax();
plot(err(2,:),'-','linewidth',1);
plot(e_d(2,:),'--','linewidth',1);
% ax.XLabel.String = '\fontname{宋体}迭代次数';
ax.YLabel.String = 'Y\fontname{宋体}轴误差\fontname{Times New Roman}/mm';

ax = subplot(3,1,3);
set_ax();
plot(err(3,:),'-','linewidth',1);
plot(e_d(3,:),'--','linewidth',1);
ax.XLabel.String = '\fontname{宋体}标记点编号';
ax.YLabel.String = 'Z\fontname{宋体}轴误差\fontname{Times New Roman}/mm';

path = pwd;
filename = [path '\figure\标记点定位误差'];
print(filename,'-dpng','-r600');
filename = [path(1:end-6) 'latex\figures\led_pos_err'];
print(filename,'-depsc','-r600');
rms(e_d')
rms(err')
max(e_d')
max(err')

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


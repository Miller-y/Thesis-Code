%% 画外参仿真实验的像点分布
load('data\ops_calib.mat')
% 创建figure对象
Fig =figure();
Fig.Units = 'centimeters'; 
Fig.Position = [30 15 10 4];
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
ax.ColorOrder = tab10(30);
ax.XLimMode = 'auto';
ax.YLimMode = 'auto';
ax.YDir = 'reverse';
% ax.XLim = [-22 22];
ax.YLim = [0 4];
ax.YTick = [1 2 3];
ax.XLabel.String = '\fontname{宋体}像素';
ax.YLabel.String = '\fontname{宋体}相机编号';
% ax.XLabel.Interpreter = 'latex';
% ax.YLabel.Interpreter = 'latex';

ac = [];
ac(end+1) = scatter(u_buf(1,:),ones(1,length(u_buf))*1);
ac(end+1) = scatter(u_buf(2,:),ones(1,length(u_buf))*2);
ac(end+1) = scatter(u_buf(3,:),ones(1,length(u_buf))*3);
   
% [lgd,icons,plots,txt]  = legend(ac,'1\fontname{宋体}号线阵相机',...
%     '1\fontname{宋体}号线阵相机',...
%     '1\fontname{宋体}号线阵相机');
% 
% ax.Legend.Location = 'bestoutside';
% ax.Legend.FontSize = 9;
% ax.Legend.LineWidth = 0.5;
% % ax.Legend.Interpreter = 'latex';

path = pwd;
filename = [path '\figure\像点分布图'];
print(filename,'-dpng','-r600');
filename = [path(1:end-6) 'latex\figures\u_distribution'];
print(filename,'-depsc','-r600');



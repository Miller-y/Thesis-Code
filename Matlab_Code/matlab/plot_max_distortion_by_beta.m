%% 画 beta 引起的最大偏移量
load('data\Distortion.mat')
plot_Distortion;
% 创建figure对象
Fig =figure();
Fig.Units = 'centimeters'; 
Fig.Position = [30 15 10 7];
Fig.Renderer = 'painters';
Fig.Colormap = lines(4);
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
ax.ColorOrder = lines(4);
ax.XLimMode = 'auto';
ax.YLimMode = 'auto';
ax.XLim = [-22 22];
ax.YLim = [-1 22];
ax.XLabel.String = '\alpha/°';
ax.YLabel.String = '\fontname{宋体}偏移\fontname{Times New Roman}/μm';
% ax.XLabel.Interpreter = 'latex';
% ax.YLabel.Interpreter = 'latex';


dis_y_err = zeros(1,13);
dis_y_err_15 = zeros(1,13);
for in = 1:13
    dis_y_err(in) = max(distortion(:,in)) - min(distortion(:,in));
    dis_y_err_15(in) = max(distortion(3:11,in)) - min(distortion(3:11,in));
end
axx = plot(alp(1,:),dis_y_err*1e3,'-','linewidth',1);
ax15 = plot(alp(1,:),dis_y_err_15*1e3,'-','linewidth',1);
   
[lgd,icons,plots,txt]  = legend([axx ax15],'||\beta||<20.8°','||\beta||<15°');
ax.Legend.Location = 'best';
ax.Legend.FontSize = 9;
ax.Legend.LineWidth = 0.5;
% ax.Legend.Interpreter = 'latex';
path = pwd;
filename = [path '\figure\beta引起的最大畸变图'];
print(filename,'-dpng','-r600');
filename = [path(1:end-6) 'latex\figures\max_distortion_by_beta'];
print(filename,'-depsc','-r600');
% print('F:\code\论文\graduation-thesis\matlab\figure\beta引起的最大畸变图','-dpng','-r600');





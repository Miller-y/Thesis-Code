%% 画 beta 引起的怕偏差变化
load('data\Distortion.mat')
plot_Distortion;
% 创建figure对象
Fig =figure();
Fig.Units = 'centimeters'; 
Fig.Position = [30 15 12 7];
Fig.Renderer = 'painters';
Fig.Colormap = lines(13);
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
ax.ColorOrder = lines(13);
ax.XLimMode = 'auto';
ax.YLimMode = 'auto';
ax.XLim = [-22 22];
% ax.YLim = [-5 20];
ax.XLabel.String = '\beta/°';
ax.YLabel.String = '\fontname{宋体}畸变\fontname{Times New Roman}/mm';
% ax.XLabel.Interpreter = 'latex';
% ax.YLabel.Interpreter = 'latex';

dis_y_err = zeros(1,13);
ac = [];
for in = 1:13
    ac(end+1) = plot(beta(:,in),distortion(:,in),'-','linewidth',1);
    dis_y_err(in) = max(distortion(:,in)) - min(distortion(:,in));
%     hold on;
end
   
[lgd,icons,plots,txt]  = legend(ac,'\alpha=-20.8°',...
    '\alpha=-17.6°',...
    '\alpha=-14.2°',...
    '\alpha=-10.8°',...
    '\alpha=-7.2°',...
    '\alpha=-3.6°',...
    '\alpha=0°',...
    '\alpha=3.6°',...
    '\alpha=7.2°',...
    '\alpha=10.8°',...
    '\alpha=14.2°',...
    '\alpha=17.6°',...
    '\alpha=20.8°');

ax.Legend.Location = 'bestoutside';
ax.Legend.FontSize = 9;
ax.Legend.LineWidth = 0.5;
% ax.Legend.Interpreter = 'latex';

path = pwd;
filename = [path '\figure\distortion_by_beta'];
print(filename,'-dpng','-r600');
filename = [path(1:end-6) 'latex\figures\distortion_by_beta'];
print(filename,'-depsc','-r600');
% print('F:\code\论文\graduation-thesis\matlab\figure\beta引起的误差图','-dpng','-r600');






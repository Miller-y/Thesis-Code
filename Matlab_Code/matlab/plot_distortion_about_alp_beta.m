%% 画 Distortion as a function of α and β

% 创建figure对象
Fig =figure();
Fig.Units = 'centimeters'; 
Fig.Position = [30 15 9 7];
Fig.Renderer = 'painters';
% Fig.Colormap = tab10(30);
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
% ax.ColorOrder = tab10(30);
ax.XLimMode = 'auto';
ax.YLimMode = 'auto';
ax.ZLimMode = 'auto';
ax.XLim = [-22 22];
ax.YLim = [-22 22];
ax.ZLim = [-0.3 0.4];
ax.XLabel.String = '\alpha/°';
ax.YLabel.String = '\beta/°';
ax.ZLabel.String = '\fontname{宋体}畸变\fontname{Times New Roman}/mm';
ax.View = [45 45];
ax.View = [15 35];
% view(3)
s = surf(alp,beta,distortion);

path = pwd;
filename = [path '\figure\distortion_about_alp_beta'];
print(filename,'-dpng','-r600');
filename = [path(1:end-6) 'latex\figures\distortion_about_alp_beta'];
print(filename,'-depsc','-r600');
% print('F:\code\论文\graduation-thesis\matlab\figure\distortion_about_alp_beta','-dpng','-r600');

% view(ax,);
% ax.XLabel.Interpreter = 'latex';
% ax.YLabel.Interpreter = 'latex';


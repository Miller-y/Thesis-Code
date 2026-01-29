%% 画旋转外参的迭代残差变化
load('data\iterate_err.mat')
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
% ax.Position = [0.1344,0.13,0.806,0.82];
ax.LineWidth = 1;
ax.FontName = 'Times New Roman';
% ylabel('\fontname{宋体}我是帅哥\fontname{Times New Roman}/wssg');
ax.FontSize = 9;
ax.NextPlot = 'add';
ax.ColorOrder = tab10(30);
ax.XLimMode = 'auto';
ax.YLimMode = 'auto';
% ax.YDir = 'reverse';
% ax.XLim = [-22 22];
% ax.YLim = [0 4];
% ax.YTick = [1 2 3];
ax.XLabel.String = '\fontname{宋体}迭代次数';
ax.YLabel.String = '\fontname{宋体}残差';
% ax.XLabel.Interpreter = 'latex';
% ax.YLabel.Interpreter = 'latex';

 plot(iterate_err,'o-','linewidth',1);
 
 % 创建 textarrow
annotation('textarrow',[0.701458333333334 0.683819444444445],...
    [0.321222222222223 0.194340277777778],...
    'String','\fontname{Times New Roman}1.84e-7');

%  text(10,0.01,'\leftarrow \n s','Fontsize',9);
% [lgd,icons,plots,txt]  = legend(ac,'1\fontname{宋体}号线阵相机',...
%     '1\fontname{宋体}号线阵相机',...
%     '1\fontname{宋体}号线阵相机');
% 
% ax.Legend.Location = 'bestoutside';
% ax.Legend.FontSize = 9;
% ax.Legend.LineWidth = 0.5;
% % ax.Legend.Interpreter = 'latex';

path = pwd;
filename = [path '\figure\旋转迭代残差变化'];
print(filename,'-dpng','-r600');
filename = [path(1:end-6) 'latex\figures\iterate_R_err'];
print(filename,'-depsc','-r600');



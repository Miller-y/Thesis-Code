%% 画旋转外参的迭代残差变化
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
% ax.YDir = 'reverse';
ax.XLim = [2360 2610];
% ax.YLim = [500 3500];
% ax.YTick = [1 2 3];
ax.XLabel.String = '\fontname{宋体}像素坐标';
ax.YLabel.String = '\fontname{宋体}灰度值';
% ax.XLabel.Interpreter = 'latex';
% ax.YLabel.Interpreter = 'latex';

pixel = [2360:2610];
gray  = ones(1,length(pixel))*10+round(10*rand(1,length(pixel)));
x = -11:12;
y = round(2410*7*normpdf(x,0,3));

gray(2481-2360:2481+23-2360) = 10+y;

plot(pixel,gray,'o','linewidth',1,'MarkerSize',3);
 
%  text(10,0.01,'\leftarrow \n s','Fontsize',9);
% [lgd,icons,plots,txt]  = legend(ac,'1\fontname{宋体}号线阵相机',...
%     '1\fontname{宋体}号线阵相机',...
%     '1\fontname{宋体}号线阵相机');
% 
% ax.Legend.Location = 'bestoutside';
% ax.Legend.FontSize = 9;
% ax.Legend.LineWidth = 0.5;
% % ax.Legend.Interpreter = 'latex';
% 

filename = 'figure\pixel_actually';
print(filename,'-dpng','-r600');
filename = '..\latex\figures\pixel_actually';
print(filename,'-depsc','-r600');



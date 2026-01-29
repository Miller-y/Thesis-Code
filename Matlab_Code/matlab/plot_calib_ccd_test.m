%% 基于最大似然估计相机标定方法实验相关图
load('data\BA.mat')
%% 位移误差
% 创建figure对象
Fig =figure();
Fig.Units = 'centimeters'; 
Fig.Position = [30 15 10 6];
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
% ax.LabelFontSizeMultiplier = 1;
ax.NextPlot = 'add';
ax.ColorOrder = tab10(30);
ax.XLimMode = 'auto';
ax.YLimMode = 'auto';
% ax.XLim = [-22 22];
ax.YLim = [-50 20];
ax.XLabel.FontSize
ax.XLabel.String = '\fontname{宋体}线阵相机不同位姿对应的编号';
ax.YLabel.String = '\fontname{宋体}误差\fontname{Times New Roman}/mm';
% ax.XLabel.Interpreter = 'latex';
% ax.YLabel.Interpreter = 'latex';

% 之前ccd位置的时候调整了下坐标系
err_x_ba = T_set(3,:) - T_BA(3,:);
err_z_ba = T_set(1,:) - T_BA(1,:);
err_x_noba = T_set(3,:) - T_before(3,:);
err_z_noba = T_set(1,:) - T_before(1,:);

axx = [];
axx(end+1) = plot(err_x_ba,'*-','linewidth',1,'Color',ax.ColorOrder(1,:));
axx(end+1) = plot(err_z_ba,'*-','linewidth',1,'Color',ax.ColorOrder(2,:));
axx(end+1) = plot(err_x_noba,'o-','linewidth',1,'Color',ax.ColorOrder(1,:));
axx(end+1) = plot(err_z_noba,'o-','linewidth',1,'Color',ax.ColorOrder(2,:));

   
[lgd,icons,plots,txt]  = legend(axx,...
    '\fontname{宋体}优化后X轴位移误差',...
    '\fontname{宋体}优化后Z轴位移误差',...
    '\fontname{宋体}优化前X轴位移误差',...
    '\fontname{宋体}优化前Z轴位移误差');
ax.Legend.NumColumns = 2;
ax.Legend.Location = 'southwest';'bestoutside';
ax.Legend.FontSize = 9;
ax.Legend.LineWidth = 0.5;

% ax.Legend.Interpreter = 'latex';
path = pwd;
filename = [path '\figure\优化前后平移误差对比'];
% print(filename,'-dpng','-r600');
filename = [path(1:end-6) 'latex\figures\t_compare_op'];
print(filename,'-depsc','-r600');
% print('F:\code\论文\graduation-thesis\matlab\figure\优化前后平移误差对比','-dpng','-r600');


%% 姿态误差
% 创建figure对象
Fig =figure();
Fig.Units = 'centimeters'; 
Fig.Position = [30 15 10 6];
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
% ax.LabelFontSizeMultiplier = 1;
ax.NextPlot = 'add';
ax.ColorOrder = tab10(30);
ax.XLimMode = 'auto';
ax.YLimMode = 'auto';
% ax.XLim = [-22 22];
ax.YLim = [0 0.032];
ax.XLabel.FontSize;
ax.XLabel.String = '\fontname{宋体}线阵相机不同位姿对应的编号';
ax.YLabel.String = '\fontname{宋体}误差\fontname{Times New Roman}/rad';
% ax.XLabel.Interpreter = 'latex';
% ax.YLabel.Interpreter = 'latex';
R_err_noba = [];
R_err_ba = [];
for i = 1:length(R_set(1,:))
    tmpR_s =  rotationVectorToMatrix( -R_set(:,i));
    tmpR_b =  rotationVectorToMatrix( -R_before(:,i));
    tmpR_ba =  rotationVectorToMatrix( -R_BA(:,i));
    R_err_ba(end+1) = norm(rotationMatrixToVector(tmpR_s'*tmpR_ba));
    R_err_noba(end+1) = norm(rotationMatrixToVector(tmpR_s'*tmpR_b));
end
axx = [];
axx(end+1) = plot(R_err_ba,'*-','linewidth',1,'Color',ax.ColorOrder(1,:));
axx(end+1) = plot(R_err_noba,'o-','linewidth',1,'Color',ax.ColorOrder(2,:));

[lgd,icons,plots,txt]  = legend(axx,...
    '\fontname{宋体}优化后姿态误差',...
    '\fontname{宋体}优化前姿态误差');
ax.Legend.Location = 'northwest';
ax.Legend.FontSize = 9;
ax.Legend.LineWidth = 0.5;
% ax.Legend.Interpreter = 'latex';
path = pwd;
filename = [path '\figure\优化前后姿态误差对比'];
% print(filename,'-dpng','-r600');
filename = [path(1:end-6) 'latex\figures\R_compare_op'];
print(filename,'-depsc','-r600');
% print('F:\code\论文\graduation-thesis\matlab\figure\优化前后姿态误差对比','-dpng','-r600');



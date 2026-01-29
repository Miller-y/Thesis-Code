%% lops速度和定位误差
load('data\lops_i_zero.mat')

lops_err = t_wb_real(:,1:10:length(t_wb_real)/2)*1000 - t_wb_lops(:,1:400);
lops_err_norm = zeros(1,length(lops_err));
for i =  1 :length(lops_err)
    lops_err_norm(i) = norm(lops_err(:,i));
end
v_wb_real = -v_wb_real/v_wb_real(1,2000)*2.5;

%% 画lops速度和定位误差
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
ax.NextPlot = 'add';
ax.ColorOrder = tab10(10);
ax.XLimMode = 'auto';
ax.YLimMode = 'auto';
% ax.XLim = [0 6];
% % ax.YLim = [-2 22];
% ax.XLabel.String = '\fontname{宋体}速度\fontname{Times New Roman}/m/s';
ax.XLabel.String = '\fontname{宋体}时间\fontname{Times New Roman}/m/s';
ax.YLabel.String = '\fontname{宋体}定位误差\fontname{Times New Roman}/mm';

% ax.XLabel.Interpreter = 'latex';
% ax.YLabel.Interpreter = 'latex';
color = tab10(10);

% plot(t_imu(1:4000),v_wb_real(1,1:4000)*1000);
% plot(t_lops(1:400),t_wb_real(1,1:10:length(t_wb_real)/2)*1000 - t_wb_lops(1,1:400));
% hold on ;
% plot(t_lops(1:400),t_wb_real(2,1:10:length(t_wb_real)/2)*1000 - t_wb_lops(2,1:400));
% plot(t_lops(1:400),t_wb_real(3,1:10:length(t_wb_real)/2)*1000 - t_wb_lops(3,1:400));
% plot(t_imu(1:4000),v_wb_real(1,1:4000)*1000);

% plot(-v_wb_real(1,1:10:2000),lops_err_norm(1:200),'-','linewidth',1);


yyaxis left;
axx = plot(t_lops(1:200),lops_err_norm(1:200),'-','linewidth',1);
ax.YLabel.Color = 'k';

yyaxis right;
ax15 = plot(t_imu(1:2000),-v_wb_real(1,1:2000),'-','linewidth',1,'Color',color(4,:));
ax.YAxis(2).Color  = color(4,:);
ylabel('\fontname{宋体}速度\fontname{Times New Roman}/m/s','Color','k')
% ylim([-32 32])
legend([axx ax15],'\fontname{宋体}定位误差','\fontname{宋体}移动速度');
ax.Legend.FontSize = 9;
ax.Legend.LineWidth = 0.5;
ax.Legend.Location = 'northwest';
% path = pwd;
filename = 'figure\lops_move_test';
print(filename,'-dpng','-r600');
filename = '..\latex\figures\lops_move_test';
print(filename,'-depsc','-r600');

%% 画光学传感器分布

% 创建figure对象
Fig =figure();
Fig.Units = 'centimeters'; 
Fig.Position = [30 15 11 7];
Fig.Renderer = 'painters';
Fig.Colormap = tab10(30);
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
ax.ColorOrder = tab10(30);
axis equal;
ax.XLimMode = 'auto';
ax.YLimMode = 'auto';
ax.ZLimMode = 'auto';
ax.XLim = [-100 100];
ax.YLim = [-100 100];
ax.ZLim = [-20 60];
ax.XLabel.String = 'X/mm';
ax.YLabel.String = 'Y/mm';
% ax.YLabel.Rotation = 75;
ax.ZLabel.String = 'Z/mm';

% ax.View = [180 90];
ax.View = [-105 12];
ax.View = [[-109.139306358381 23.3029766367079]];

% quiver3(0,0,0,1,0,0,50,'color',[0.843137254901961,0.149019607843137,0.172549019607843],'linewidth',1);
% quiver3(0,0,0,0,1,0,50,'color',[0.152941176470588,0.631372549019608,0.278431372549020],'linewidth',1);
% quiver3(0,0,0,0,0,1,50,'color','b','linewidth',1);

len = 30;lw = 2;
axx = plot_frame(rotz(0),[0,0,20],len,lw);
plot_frame(rotz(0),[70,0,0],len,lw);
plot_frame(rotz(120), rotz(120)*[70,0,0]',len,lw);
plot_frame(rotz(-120), rotz(-120)*[70,0,0]',len,lw);
scale = 1;
color = tab10(10);
cnt = 1;
plot_CCD(rotz(0),[70,0,0]',scale,color(cnt,:)); %cnt = cnt+2;
plot_CCD(rotz(120), rotz(120)*[70,0,0]',scale,color(cnt,:)); %cnt = cnt+1;
plot_CCD(rotz(-120), rotz(-120)*[70,0,0]',scale,color(cnt,:)); %cnt = cnt+1;
text(70,15,-10,'$\bf{C}_1$','FontSize',9,'Interpreter','latex')
text(-35.0000,60.6218,-10,'$\bf{C}_2$','FontSize',9,'Interpreter','latex')
text(-35.0000,-60.6218,-10,'$\bf{C}_3$','FontSize',9,'Interpreter','latex')
text(0,0,10,'$\bf{S}$','FontSize',9,'Interpreter','latex')
 legend1= legend(axx,'X','Y','Z');
set(legend1,...
    'Position',[0.530957878503889 0.682660461556835 0.150793650320598 0.19383259399872]);

path = pwd;
filename = [path '\figure\光学传感器分布'];
% print(filename,'-dpng','-r600');
filename = [path(1:end-6) 'latex\figures\op_sensor_ccd'];
print(filename,'-depsc','-r600');

% print('F:\code\论文\graduation-thesis\matlab\figure\光学传感器分布','-dpng','-r600');



% 创建figure对象
Fig =figure();
Fig.Units = 'centimeters'; 
Fig.Position = [30 15 7.5 7];
Fig.Renderer = 'painters';
Fig.Colormap = tab10(30);
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
ax.ColorOrder = tab10(30);
axis equal;
ax.XLimMode = 'auto';
ax.YLimMode = 'auto';
ax.ZLimMode = 'auto';
ax.XLim = [-100 100];
ax.YLim = [-100 100];
ax.ZLim = [-10 60];
ax.XLabel.String = 'X/mm';
ax.YLabel.String = 'Y/mm';
% ax.YLabel.Rotation = 75;
ax.ZLabel.String = 'Z/mm';

% ax.View = [180 90];
ax.View = [-105 12];
ax.View = [-90 90];

% quiver3(0,0,0,1,0,0,50,'color',[0.843137254901961,0.149019607843137,0.172549019607843],'linewidth',1);
% quiver3(0,0,0,0,1,0,50,'color',[0.152941176470588,0.631372549019608,0.278431372549020],'linewidth',1);
% quiver3(0,0,0,0,0,1,50,'color','b','linewidth',1);

len = 30;lw = 2;
axx = plot_frame(rotz(0),[0,0,30],len,lw);
plot_frame(rotz(0),[70,0,10],len,lw);
plot_frame(rotz(120), rotz(120)*[70,0,10]',len,lw);
plot_frame(rotz(-120), rotz(-120)*[70,0,10]',len,lw);
scale = 1;
color = tab10(10);
cnt = 1;
plot_CCD(rotz(0),[70,0,10]',scale,color(cnt,:)); %cnt = cnt+2;
plot_CCD(rotz(120), rotz(120)*[70,0,10]',scale,color(cnt,:)); %cnt = cnt+1;
plot_CCD(rotz(-120), rotz(-120)*[70,0,10]',scale,color(cnt,:)); %cnt = cnt+1;
text(70,-5,0,'$\bf{C}_1$','FontSize',9,'Interpreter','latex')
text(-26.3397,65.6218,0,'$\bf{C}_2$','FontSize',9,'Interpreter','latex')
text(-43.6603,-55.6218,0,'$\bf{C}_3$','FontSize',9,'Interpreter','latex')
text(0,0,20,'$\bf{S}$','FontSize',9,'Interpreter','latex')
legend1 = legend(axx,'X','Y','Z');
set(legend1,...
    'Position',[0.694518484564495 0.703827128223502 0.150793650320598 0.19383259399872]);

path = pwd;
filename = [path '\figure\光学传感器分布_俯视'];
print(filename,'-dpng','-r600');
filename = [path(1:end-6) 'latex\figures\op_sensor_ccd_up'];
print(filename,'-depsc','-r600');
% print('F:\code\论文\graduation-thesis\matlab\figure\光学传感器分布_俯视','-dpng','-r600');
% 

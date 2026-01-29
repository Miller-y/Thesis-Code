%% 光学定位传感器俯视图
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
ax.Box = 'off';
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
ax.XLabel.String = 'X';
ax.YLabel.String = 'Y';
ax.XTick = [];
ax.YTick = [];
% ax.YLabel.Rotation = 75;
ax.ZLabel.String = 'Z/mm';

% ax.View = [180 90];
ax.View = [-105 12];
ax.View = [-90 90];

% quiver3(0,0,0,1,0,0,50,'color',[0.843137254901961,0.149019607843137,0.172549019607843],'linewidth',1);
% quiver3(0,0,0,0,1,0,50,'color',[0.152941176470588,0.631372549019608,0.278431372549020],'linewidth',1);
% quiver3(0,0,0,0,0,1,50,'color','b','linewidth',1);
o = [70,0,10]';
len = 90;
% x = rotz(0)*[len,0,0]';
y = rotz(0)*[0,len,0]';
% z = rotz(0)*[0,0,len]';
line([o(1),o(1)-40],[o(2),o(2)],[o(3),o(3)],'color','k','LineStyle','-.','linewidth',0.5);
line([o(1),o(1)+y(1)],[o(2)-y(2),o(2)+y(2)],[o(3),o(3)+y(3)],'color','k','LineStyle','-.','linewidth',0.5);
% line([o(1),o(1)+z(1)],[o(2),o(2)+z(2)],[o(3),o(3)+z(3)],'color','k','linewidth',1);
o = rotz(120)*[70,0,10]';
line([o(1),o(1)],[-90,90],[o(3),o(3)],'color','k','LineStyle','-.','linewidth',0.5);
o = rotz(-120)*[70,0,10]';
x = [70,o(2),o(3)];
% line([o(1),x(1)],[o(2),x(2)],[o(3),x(3)],'color','k','LineStyle','-.','linewidth',0.5);
% 创建 doublearrow
annotation(Fig,'doublearrow',[0.52 0.3],...
    [0.69 0.69]);
annotation(Fig,'doublearrow',[0.53 0.75],...
    [0.69 0.69]);
annotation(Fig,'doublearrow',[0.285 0.285],...
    [0.8 0.4]);
annotation(Fig,'doublearrow',[0.75 0.75],...
    [0.8 0.4]);

text(50,-25,0,'$t_y$','FontSize',9,'Interpreter','latex')
text(50,35,0,'$t_y$','FontSize',9,'Interpreter','latex')
text(20,-63,0,'$d$','FontSize',9,'Interpreter','latex')
text(20,69,0,'$d$','FontSize',9,'Interpreter','latex')

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
text(-95,95,0,'$\bf{W}$','FontSize',9,'Interpreter','latex')
text(75,-5,0,'$\bf{C}_1$','FontSize',9,'Interpreter','latex')
text(-50,68,'$\bf{C}_2$','FontSize',9,'Interpreter','latex')
text(-46,-54,0,'$\bf{C}_3$','FontSize',9,'Interpreter','latex')
text(0,-3,20,'$\bf{S}$','FontSize',9,'Interpreter','latex')
legend1 = legend(axx(1:2),'X','Y');
% set(legend1,...
%     'Position',[0.694518484564495 0.703827128223502 0.150793650320598 0.19383259399872]);
set(legend1,...
    'Position',[0.654029632348411 0.170689938000012 0.208480564528556 0.115530300095225]);

path = pwd;
% filename = [path '\figure\光学传感器分布_俯视'];
% print(filename,'-dpng','-r600');
filename = [path(1:end-6) 'latex\figures\op_sensor_ccd_up_detail'];
print(filename,'-depsc','-r600');
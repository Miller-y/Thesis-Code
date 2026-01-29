
load('xjb_new.mat');
zupt;

% 创建figure对象
Fig =figure();
Fig.Units = 'centimeters'; 
Fig.Position = [30 15 10 7.5];
Fig.Renderer = 'painters';
Fig.Colormap = tab10(20);
Fig.Name = '1';
% Fig.Color = 'w';
% 创建axes对象, 设定坐标轴属性
clear ax;
ax = axes();
ax.Parent = Fig;
ax.XGrid = 'off';
ax.YGrid = 'off';
ax.GridLineStyle = '-.';
ax.Visible = 'on';
ax.Box = 'on';
% ax.Position = [0.1344,0.13,0.806,0.82];
ax.LineWidth = 1;
ax.FontName = 'Times New Roman';
ax.FontSize = 9;
ax.NextPlot = 'add';
ax.ColorOrder = tab10(20);
ax.XLimMode = 'auto';
ax.YLimMode = 'auto';
ax.XLim = [22 50];
ax.YLim = [-5 20];
ax.XLabel.String = '\fontname{宋体}时间\fontname{Times New Roman}/s';
ax.YLabel.String = '\fontname{宋体}加速度\fontname{Times New Roman}/m/s^2';
% ax.XLabel.Interpreter = 'latex';
% ax.YLabel.Interpreter = 'latex';
ls = 1:10:length(acc);%(22000:45000);


% ac = plot((1:length(ls))/1000,g(ls),'linewidth',0.5);
zvd = zeros(1,length(acc));
for i = 1:length(c_list)
    if(length( c_list{i}) >100)
%         temp = c_list{i}(1:600:end);
%         zp =  plot(temp/1000,ones(1,length(temp))*10,'ko','linewidth',1,...
%             'markerfacecolor','k','markersize',3);
        zp = rectangle('position',[c_list{i}(1)/1000 -3 (c_list{i}(end)-c_list{i}(1))/1000 15],...
           'EdgeColor',Fig.Colormap(4,:),'linewidth',1.5,'FaceColor','none' ,'linestyle',':');
%        alpha(0.5);
    end
end

acx = plot(ls/1000,acc(ls,1)/10*9.8,'-.','linewidth',1);
acy = plot(ls/1000,acc(ls,2)/10*9.8,'--','linewidth',1);
acz = plot(ls/1000,acc(ls,3)/10*9.8,'-','linewidth',1);
annotation('textarrow',[0.44 0.48],...
    [0.74 0.67],'String',{'静止区间'},'FontName','宋体');
% zvd(list) = 10;
% zp = plot((1:length(ls))/1000,zvd(ls),'linewidth',1);
% [lgd,icons,plots,txt]  = legend([ac,zp],'Magnitude of Acceleration ','Zero-Velocity Detection');
[lgd,icons,plots,txt]  = legend([acx,acy,acz],'X\fontname{宋体}轴',...
    'Y\fontname{宋体}轴','Z\fontname{宋体}轴');
ax.Legend.Location = 'northeast';
ax.Legend.FontSize = 9;
ax.Legend.LineWidth = 0.5;
% ax.Legend.Interpreter = 'latex';
% ax.Legend.Orientation = 'horizontal';
% ax.Legend.NumColumns =2;

filename = '..\figure\imu_zvd';
print(filename,'-dpng','-r600');
filename = '..\..\latex\figures\imu_zvd';
print(filename,'-depsc','-r600');

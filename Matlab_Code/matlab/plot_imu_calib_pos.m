%% 画imu标定时的姿态和位置

% 创建figure对象
Fig =figure();
Fig.Units = 'centimeters'; 
Fig.Position = [30 15 12 7.2];
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
ax.XLim = [-350 300];
% ax.YLim = [-350 350];
ax.ZLim = [-100 100];
ax.XLabel.String = 'X/mm';
ax.YLabel.String = 'Y/mm';
% ax.YLabel.Rotation = 75;
ax.ZLabel.String = 'Z/mm';

% ax.View = [180 90];
ax.View = [-165 42];

path = pwd;
load( [path '\data\imu_cabli_pose_use.mat'])
scale = 7;
color = tab10(42);
cnt = 1;
R = [];
for i = 1:length(imuR)
    if(norm(imuR(:,i)) ~= 0)
        R(:,:,i) = axang2rotm([imuR(:,i)'/norm(imuR(:,i)) norm(imuR(:,i))]);
    else
        R(:,:,i)= axang2rotm([1,0,0,0]);
    end
    t = imuT([2,1,3],i)*50;
    t(3) = imuT(3,i);
    plot_imu( R(:,:,i),t,scale,color(cnt,:)); cnt = cnt+1;
end

% path = pwd;
% filename = [path '\figure\imu_calib_pos'];
% print(filename,'-dpng','-r600');
% filename = [path(1:end-6) 'latex\figures\imu_calib_pos'];
% print(filename,'-depsc','-r600');

%% 画单个imu
function p = plot_imu(R,t,scale,EdgeColor)
%% imu 几何关系
h = 1;
l = 4;
w = 4;
start_point = -[l,w,h]/2*scale; %ccd一顶点
final_point = [l,w,h]/2*scale; %ccd一顶点
%存数据
global T;
global R_buf;
T(end+1,:) = t';
rv = rotm2axang(roty(-90)*R);
R_buf(end+1,:) = rv(1:3)*rv(4);
% start_point = R*start_point+t;
% final_point = R*final_point+t; 
% start_point = start_point';
% final_point = final_point';
%% 根据起点和终点，计算长方体的8个的顶点
vertexIndex=[0 0 0;0 0 1;0 1 0;0 1 1;1 0 0;1 0 1;1 1 0;1 1 1];
cuboidSize=final_point-start_point;             %方向向量
vertex=repmat(start_point,8,1)+vertexIndex.*repmat(cuboidSize,8,1);
for i = 1:length(vertex)
   vertex(i,:) =  (R*vertex(i,:)'+t)';
end
%% 定义6个平面分别对应的顶点
facet=[1 2 4 3;1 2 6 5;1 3 7 5;2 4 8 6;3 4 8 7;5 6 8 7];
%% 定义8个顶点的颜色，绘制的平面颜色根据顶点的颜色进行插补
color=[0;0;0;0;1;1;1;1];
%% 绘制并展示图像
clear S
S.Vertices = vertex;
S.Faces = facet;
% S.FaceVertexCData = color;
S.FaceColor = 'none';%[112 128 144]/255;
S.EdgeColor = EdgeColor; %[112 128 144]/255;
S.FaceAlpha = 0.9;
S.LineWidth = 0.5;
patch(S)
hold on;
start_point = [-l/4,-w/4,h/2]*scale; %ccd一顶点
final_point = [l/2,w/2,h]/2*scale; %ccd一顶点
vertexIndex=[0 0 0;0 0 1;0 1 0;0 1 1;1 0 0;1 0 1;1 1 0;1 1 1];
cuboidSize=final_point-start_point;             %方向向量
vertex=repmat(start_point,8,1)+vertexIndex.*repmat(cuboidSize,8,1);
for i = 1:length(vertex)
   vertex(i,:) =  (R*vertex(i,:)'+t)';
end
S.Vertices = vertex;
S.EdgeColor = EdgeColor; %[220 220 220]/255;
S.FaceColor = 'none';%[220 220 220]/255;
patch(S)
% p = patch('Vertices',vertex,'Faces',facet,'FaceVertexCData',color,'FaceColor','interp','FaceAlpha',0.5);

end
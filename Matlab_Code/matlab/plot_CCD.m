%% 根据三维空间中位姿画CCD
%输入R：             旋转矩阵;
%输入t：             平移向量;
%scale :            缩放,默认单位是mm
%输出：                        长方体
function p = plot_CCD(R,t,scale,EdgeColor)
%% ccd 几何关系
h = 4;
l = 40;
w = 7;
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
start_point = [-(l-10)/2,-w/4,h/2]*scale; %ccd一顶点
final_point = [(l-10),w/2,h]/2*scale; %ccd一顶点
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
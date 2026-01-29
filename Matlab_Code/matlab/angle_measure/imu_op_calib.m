%% imu安装误差标定实验
%% 数据处理， imu数据由c++静止仿真得到
load('..\data\imu_op_calib.mat')

% 根据光学传感器计算旋转轴 1
% rotm2eul(X_R_1'*X_R_2)/pi*180;
% rotm2eul(X_R_2'*X_R_3)/pi*180;
rv_x_1 = rotm2axang(X_R_1'*X_R_2);
rv_x_2 = rotm2axang(X_R_2'*X_R_3);
rv_s_x = (rv_x_1(1:3) + rv_x_2(1:3))/2;
% [R12 T12] = svd_cal_RT_WS(X_M_S1,X_M_S2);
% [R23 T23] = svd_cal_RT_WS(X_M_S2,X_M_S3);
% rotm2axang(R12)
% rotm2axang(R23)
% 根据光学传感器计算旋转轴 2
rv_y_1 = rotm2axang(Y_R_1'*Y_R_2);
rv_y_2 = rotm2axang(Y_R_2'*Y_R_3);
rv_s_y = (rv_y_1(1:3) + rv_y_2(1:3))/2;

% 实际旋转轴
rv_real_x = [2,2,1]/3;
rv_real_y = [2,-2,1]/3;
% 实际旋转矩阵
X_R_1_real = axang2rotm([rv_real_x -15/180*pi]);
X_R_2_real = axang2rotm([rv_real_x 0/180*pi]);
X_R_3_real = axang2rotm([rv_real_x 15/180*pi]);
Y_R_1_real = axang2rotm([rv_real_y -15/180*pi]);
Y_R_2_real = axang2rotm([rv_real_y 0/180*pi]);
Y_R_3_real = axang2rotm([rv_real_y 15/180*pi]);

load('..\data\imu_op_calib_imu_noise.mat')
% acc噪声数据分组
acc = zeros(3,401,300);
acc1 = zeros(3,401,100);
acc2 = zeros(3,401,100);
acc3 = zeros(3,401,100);
for i = 0 : length(acc_data)/3-1 %将900个分组三分
    acc(:,:,i+1) = acc_data(i*3+1:i*3+3,:);
end

for i = 0 : length(acc(1,1,:))/3-1  %300个再分组三分
    acc1(:,:,i+1) = acc(:,:,i*3+1); % 第一次测量角速度
    acc2(:,:,i+1) = acc(:,:,i*3+2); % 第二次测量角速度
    acc3(:,:,i+1) = acc(:,:,i*3+3); % 第三次测量角速度
end

acc1_mean = zeros(3,length(acc1(1,1,:)));
acc2_mean = zeros(3,length(acc1(1,1,:)));
acc3_mean = zeros(3,length(acc1(1,1,:)));
for i = 1:length(acc1(1,1,:)) % 计算每组的平均值，每组100个
    acc1_mean(:,i) = mean(acc1(:,:,i)')';
    acc2_mean(:,i) = mean(acc2(:,:,i)')';
    acc3_mean(:,i) = mean(acc3(:,:,i)')';
end
err_buf = zeros(100,1);
% 相当于100次实验
for i = 1:length(acc1_mean(1,:))
% for i = 8:8
    % 随机生成旋转误差
    rv_sb = [rand() rand() rand()] - [1 1  1]/2;
    rv_sb= rv_sb/norm(rv_sb);
    angle_sb = (rand())+0.5;
    R_SB = axang2rotm([rv_sb angle_sb]);
    
    % 相当于计算第1个旋转轴
    R_WB_1 = X_R_1_real*R_SB;
    R_WB_2 = X_R_2_real*R_SB;
    R_WB_3 = X_R_3_real*R_SB;
    a1_B = R_WB_1'*acc1_mean(:,i);
    a2_B = R_WB_2'*acc2_mean(:,i);
    a3_B = R_WB_3'*acc3_mean(:,i);
    rv_b_x = cal_axis(a1_B,a2_B,a3_B);
    tmp = R_SB*rv_b_x';
    if(tmp(1) < 0) % 仿真用，实际根据旋转轴方向大概判断
        rv_b_x = -rv_b_x;
    end
    
    % 相当于计算第2个旋转轴
    R_WB_1 = Y_R_1_real*R_SB;
    R_WB_2 = Y_R_2_real*R_SB;
    R_WB_3 = Y_R_3_real*R_SB;
    a1_B = R_WB_1'*acc1_mean(:,i);
    a2_B = R_WB_2'*acc2_mean(:,i);
    a3_B = R_WB_3'*acc3_mean(:,i);
    rv_b_y = cal_axis(a1_B,a2_B,a3_B);
    tmp = R_SB*rv_b_y';
    if(tmp(1) < 0) % 仿真用，实际根据旋转轴方向大概判断
        rv_b_y = -rv_b_y;
    end
    V1 = cal_V(rv_s_x, rv_b_x);
    V2 = cal_V(rv_s_y, rv_b_y);
    V = [V1;V2];
    [U,S,VV] = svd(V);
    q_sb_ans = VV(:,4)';
    R_sb_ans = quat2rotm(q_sb_ans);
    r_err = rotm2axang(R_sb_ans'*R_SB);
    err_buf(i) = r_err(4);
    if(r_err(4) > 1)
       [rv_sb angle_sb] ;
       R_SB;
       rotm2axang(R_sb_ans);
    end
end

max(abs(err_buf))
rms(err_buf)
%% 画图
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
% ax.XLim = [0 4];
% % ax.YLim = [-2 22];
ax.XLabel.String = '\fontname{宋体}实验次数';
ax.YLabel.String = '\fontname{宋体}误差\fontname{Times New Roman}/rad';

% ax.XLabel.Interpreter = 'latex';
% ax.YLabel.Interpreter = 'latex';

ax15 = plot(err_buf,'-','linewidth',1);


filename = '..\figure\imu_op_calib_err';
print(filename,'-dpng','-r600');
filename = '..\..\latex\figures\imu_op_calib_err';
print(filename,'-depsc','-r600');



%% 使用SVD计算旋转和平移
% 论文 Least-Squares Fitting of Two 3-D Point Sets
function [R,T] = svd_cal_RT_WS(M_W,M_S)
p11 = M_S(:,1);p12 = M_S(:,2);p13 = M_S(:,3);
p21 = M_W(:,1);p22 = M_W(:,2);p23 = M_W(:,3);
p1 = (p11+p12+p13)/3;
p2 = (p21+p22+p23)/3;
p11_ = p11-p1;
p12_ = p12-p1;
p13_ = p13-p1;
p21_ = p21-p2;
p22_ = p22-p2;
p23_ = p23-p2;


W = p21_*p11_';
W = W + p22_*p12_';
W = W + p23_*p13_';
[U,S,V] = svd(W);
R = U*V';
if(det(R) < 0)
    % NOTE: 不是R = -R
    V(:,3) = - V(:,3);
    %     R = V*U';
    R = U*V';
end
T = p2 - R*p1;
end
%% 根据三个加速度计算旋转轴
function rv = cal_axis(a1,a2,a3)

a_err(1,:) = a2-a1;
a_err(2,:) = a3-a2;

rv = cross(a_err(1,:),a_err(2,:))/norm(cross(a_err(1,:),a_err(2,:)));
end

%% 计算论文中矩阵V
function V = cal_V(vs, vb)
V = zeros(4);
% L = zeros(4);
x = vs(1); 
y = vs(2); 
z =  vs(3);
vl = [
    0 -x -y -z
    x 0  -z y
    y z 0 -x
    z -y x 0];
x = vb(1); 
y = vb(2); 
z =  vb(3);
vr = [
    0 -x -y -z
    x 0  z -y
    y -z 0 x
    z y -x 0];
V = vl-vr;
% V(1,2) = -vs(1)+vb(1); V(1,3) = -vs(2)+vb(2); V(1,4) = -vs(3)+vb(3);
% V(2,1) = vs(1)-vb(1); V(2,3) = -vs(3)-vb(3); V(2,4) = vs(2)+vb(2);
% V(3,1) = vs(2)-vb(2); V(3,2) = vs(3)+vb(3); V(3,4) = -vs(1)-vb(1);
% V(4,1) = vs(3)-vb(3); V(4,2) = -vs(2)-vb(2); V(4,3) = vs(1)+vb(1);
% V
end









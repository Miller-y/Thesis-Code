%% 清理环境
clc; clear; close all;

%% ========================================================
%  参数设置与噪声函数定义
% ========================================================
% 空间定义 (符合论文: X右, Y上, Z深)
Space_X = 3.5; Space_Y = 3.5; Space_Z = 3.5;
Fs = 60;   % 采样率 60Hz
Vel = 0.1; % 速度 0.1 m/s

% --- 核心噪声函数：模拟手持长杆的不稳定性 ---
% jitter_amp: 高频颤抖 (手抖)
% drift_amp:  低频漂移 (手臂摆动/走位不稳/长杆杠杆效应)
% drift_freq: 漂移平滑度 (数值越小越平滑，模拟身体晃动周期)
get_human_noise = @(N, jitter_amp, drift_amp, drift_win) ...
    (randn(N, 1) * jitter_amp) + ... % 高频
    (smoothdata(randn(N, 1), 'gaussian', drift_win) * drift_amp); % 低频

%% ========================================================
%  场景 1: 从下向上的斜直线 (Diagonal Line)
% ========================================================
figure('Name', 'Scenario 1: Diagonal Line', 'Color', 'w');

% 1. 理想路径：空间大对角线 (左前下 -> 右后上)
% 模拟人想把球从低处举高推向远处
P_start = [0.4, 0.4, 0.4]; 
P_end   = [2.2, 2.2, 2.2]; 

Dist = norm(P_end - P_start);
N = ceil((Dist / Vel) * Fs);
t = linspace(0, 1, N)';

path_ideal = P_start + (P_end - P_start) .* t;

% 2. 添加"严重"的抖动
% 垂直方向(Y)抖动最大，因为对抗重力举着2米杆子很累
noise_y = get_human_noise(N, 0.008, 0.6, Fs*1.5); 
% 深度方向(Z)其次，因为站在外部很难判断深度
noise_z = get_human_noise(N, 0.005, 0.4, Fs*2.0);
% 水平方向(X)相对较稳
noise_x = get_human_noise(N, 0.005, 0.3, Fs*2.0);

path_real = path_ideal + [noise_x, noise_y, noise_z];

% 3. 绘图 (映射 plot3(X, Z, Y) 以符合视觉习惯)
plot3(path_real(:,1), path_real(:,3), path_real(:,2), 'b-', 'LineWidth', 1.2);
grid on; box on; axis equal;
xlim([0 Space_X]); ylim([0 Space_Z]); zlim([0 Space_Y]);
xlabel('X (Horizontal)'); ylabel('Z (Depth)'); zlabel('Y (Height)');
title({'Ground Truth: Diagonal Line', '(Heavy Vertical Jitter & Fatigue)'});
view(-35, 20); % 调整视角以看清斜率

%% ========================================================
%  场景 2: 严重变形的矩形 (Distorted Rectangle)
% ========================================================
figure('Name', 'Scenario 2: Distorted Rectangle', 'Color', 'w');

% 关键点 (不规则分布，模拟人站位不正)
corners = [0.5, 0.5;  0.4, 2.1;  2.2, 2.0;  2.1, 0.6;  0.5, 0.5]; 
% 基础高度 (人想保持在1.5m，但实际上会有很大起伏)
base_h = 1.5; 

traj_rect = [];
for i = 1:4
    p1 = corners(i,:); p2 = corners(i+1,:);
    seg_len = norm(p2-p1);
    seg_N = ceil((seg_len/Vel)*Fs);
    t_seg = linspace(0, 1, seg_N)';
    
    % 基础线性插值
    seg_x = p1(1) + (p2(1)-p1(1))*t_seg;
    seg_z = p1(2) + (p2(2)-p1(2))*t_seg;
    
    % --- 变形处理 ---
    % 1. 弓形弯曲 (Bowing): 每一边都不是直的，而是向外或向内弯
    bend_dir = (-1)^i; % 交替弯曲
    bend_amount = sin(t_seg*pi) * 0.25 * bend_dir; % 最大弯曲25cm
    
    % 计算垂直于运动方向的向量，叠加弯曲
    vec = p2-p1; perp_vec = [-vec(2), vec(1)] / norm(vec);
    seg_x = seg_x + perp_vec(1)*bend_amount;
    seg_z = seg_z + perp_vec(2)*bend_amount;
    
    % 2. 高度剧烈起伏 (马鞍面效应)
    % 模拟走到远处(Z大)时，手会不自觉下垂
    h_sag = (seg_z - 1.25) * -0.2; 
    seg_y = base_h + h_sag + get_human_noise(seg_N, 0.005, 0.3, Fs);
    
    % 3. 随机游走噪声
    seg_x = seg_x + get_human_noise(seg_N, 0.005, 0.15, Fs);
    seg_z = seg_z + get_human_noise(seg_N, 0.005, 0.15, Fs);
    
    traj_rect = [traj_rect; seg_x, seg_y, seg_z];
end

% 整体平滑，模拟转弯处的圆角
traj_rect = smoothdata(traj_rect, 'gaussian', Fs);

plot3(traj_rect(:,1), traj_rect(:,3), traj_rect(:,2), 'r-', 'LineWidth', 1.2);
grid on; box on; axis equal;
xlim([0 Space_X]); ylim([0 Space_Z]); zlim([0 Space_Y]);
xlabel('X (Horizontal)'); ylabel('Z (Depth)'); zlabel('Y (Height)');
title({'Ground Truth: Distorted Rectangle', '(Curved Sides & Unstable Height)'});
view(-35, 30);

%% ========================================================
%  场景 3: 类似土豆的圆形 (Potato/Oval)
% ========================================================
figure('Name', 'Scenario 3: Potato Circle', 'Color', 'w');

% 基础设置
R_base = 0.9;
Center = [1.25, 1.25];
H_base = 1.3;
N_circ = ceil((2*pi*R_base / Vel) * Fs);
theta = linspace(0, 2*pi, N_circ)';

% --- 变形处理 ---
% 1. 半径突变: 模拟画圆时手伸不直，变成椭圆或土豆
% 叠加几个低频正弦波，创造"长短轴"不一致的效果
radius_noise = 0.2 * sin(theta) + 0.15 * cos(3*theta); 
R_real = R_base + radius_noise + get_human_noise(N_circ, 0.002, 0.1, Fs*5);

% 2. 角度不均匀: 速度忽快忽慢
theta_real = theta + 0.1 * sin(5*theta);

X_circ = Center(1) + R_real .* cos(theta_real);
Z_circ = Center(2) + R_real .* sin(theta_real);

% 3. 高度严重漂移 (Slinky effect)
% 模拟起步高，转一圈回来手累了，高度降低
Y_circ = H_base + 0.4 * cos(theta) + linspace(0, -0.3, N_circ)'; 
Y_circ = Y_circ + get_human_noise(N_circ, 0.005, 0.2, Fs);

% 4. 不闭合 (Open Loop)
% 强制让终点偏离起点 (漂移)
drift_x = linspace(0, 0.3, N_circ)';
drift_z = linspace(0, -0.2, N_circ)';
X_circ = X_circ + drift_x;
Z_circ = Z_circ + drift_z;

plot3(X_circ, Z_circ, Y_circ, 'g-', 'LineWidth', 1.2);
grid on; box on; axis equal;
xlim([0 Space_X]); ylim([0 Space_Z]); zlim([0 Space_Y]);
xlabel('X (Horizontal)'); ylabel('Z (Depth)'); zlabel('Y (Height)');
title({'Ground Truth: Potato Trajectory', '(Variable Radius & Open Loop)'});
view(-35, 30);
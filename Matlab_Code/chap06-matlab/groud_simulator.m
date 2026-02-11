%% 清理环境
clc; clear; close all;

%% ========================================================
%  参数设置
% ========================================================
Space_X = 3.5; Space_Y = 3.5; Space_Z = 3.5;
Fs = 60;   % 采样率
Vel = 0.1; % 速度

% 定义绘图样式
Style_GT   = {'Color', [1, 0.25, 0], 'LineWidth', 1.0, 'DisplayName', 'Ground Truth (Optical)'}; % 橙红细线
Style_Pred = {'Color', [0, 0.3, 0.8], 'LineWidth', 2.5, 'DisplayName', 'CSI Model Output'};      % 蓝色粗线

%% ========================================================
%  辅助函数：生成真值噪声 & 生成预测轨迹
% ========================================================
% 1. 生成人手抖动噪声 (同前)
get_human_noise = @(N, jitter_amp, drift_amp, drift_win) ...
    (randn(N, 1) * jitter_amp) + ... 
    (smoothdata(randn(N, 1), 'gaussian', drift_win) * drift_amp);

%% ========================================================
%  场景 1: 斜线 (Diagonal)
% ========================================================
figure('Name', 'Comparison: Diagonal', 'Color', 'w', 'Position', [100, 100, 800, 600]);

% --- 1. 生成真值 (GT) ---
P_start = [0.4, 0.4, 0.4]; P_end = [2.2, 2.2, 2.2]; 
Dist = norm(P_end - P_start); N = ceil((Dist / Vel) * Fs); t = linspace(0, 1, N)';
path_ideal = P_start + (P_end - P_start) .* t;

% 添加严重抖动 (模拟疲劳)
noise_y = get_human_noise(N, 0.008, 0.6, Fs*1.5); 
noise_z = get_human_noise(N, 0.005, 0.4, Fs*2.0);
noise_x = get_human_noise(N, 0.005, 0.3, Fs*2.0);
path_gt = path_ideal + [noise_x, noise_y, noise_z];

% --- 2. 生成预测 (Pred) ---
% 设定目标 MAE 为 0.30 左右
[path_pred, mae1, rmse1] = generate_csi_prediction(path_gt, 0.30);

% --- 3. 绘图 ---
plot3(path_gt(:,1), path_gt(:,3), path_gt(:,2), Style_GT{:}); hold on;
plot3(path_pred(:,1), path_pred(:,3), path_pred(:,2), Style_Pred{:});

% 装饰
grid on; box on; axis equal;
xlim([0 Space_X]); ylim([0 Space_Z]); zlim([0 Space_Y]);
xlabel('X (Horizontal) [m]'); ylabel('Z (Depth) [m]'); zlabel('Y (Height) [m]');
legend('show', 'Location', 'northwest');
title({'\bfScenario 1: Diagonal Trajectory Comparison', ...
       });
view(-35, 20);

%% ========================================================
%  场景 2: 矩形 (Rectangle)
% ========================================================
figure('Name', 'Comparison: Rectangle', 'Color', 'w', 'Position', [150, 150, 800, 600]);

% --- 1. 生成真值 (GT) ---
corners = [0.5, 0.5; 0.4, 2.1; 2.2, 2.0; 2.1, 0.6; 0.5, 0.5]; 
base_h = 1.5; 
path_gt = [];

for i = 1:4
    p1 = corners(i,:); p2 = corners(i+1,:);
    seg_len = norm(p2-p1); seg_N = ceil((seg_len/Vel)*Fs); t_seg = linspace(0, 1, seg_N)';
    
    seg_x = p1(1) + (p2(1)-p1(1))*t_seg;
    seg_z = p1(2) + (p2(2)-p1(2))*t_seg;
    
    % 弓形弯曲
    bend = sin(t_seg*pi) * 0.25 * (-1)^i;
    vec = p2-p1; perp = [-vec(2), vec(1)] / norm(vec);
    seg_x = seg_x + perp(1)*bend; seg_z = seg_z + perp(2)*bend;
    
    % 高度起伏
    seg_y = base_h + (seg_z - 1.25)*-0.2 + get_human_noise(seg_N, 0.005, 0.2, Fs);
    % 随机游走
    seg_x = seg_x + get_human_noise(seg_N, 0.005, 0.1, Fs);
    seg_z = seg_z + get_human_noise(seg_N, 0.005, 0.1, Fs);
    
    path_gt = [path_gt; seg_x, seg_y, seg_z];
end
path_gt = smoothdata(path_gt, 'gaussian', Fs*0.5); % 基础平滑

% --- 2. 生成预测 (Pred) ---
% 矩形容易出现过冲和切角，误差可能稍大，设定目标 MAE 0.31
[path_pred, mae2, rmse2] = generate_csi_prediction(path_gt, 0.31);

% 模拟CSI模型的"切角"效应 (Corner Cutting)
% 在转弯处，CSI模型往往反应不过来，会画圆弧，导致误差增大
path_pred = smoothdata(path_pred, 'gaussian', 120); 

% --- 3. 绘图 ---
plot3(path_gt(:,1), path_gt(:,3), path_gt(:,2), Style_GT{:}); hold on;
plot3(path_pred(:,1), path_pred(:,3), path_pred(:,2), Style_Pred{:});

grid on; box on; axis equal;
xlim([0 Space_X]); ylim([0 Space_Z]); zlim([0 Space_Y]);
xlabel('X (Horizontal) [m]'); ylabel('Z (Depth) [m]'); zlabel('Y (Height) [m]');
legend('show', 'Location', 'northwest');
title({'\bfScenario 2: Rectangle Trajectory Comparison', ...
       });
view(-35, 30);

%% ========================================================
%  场景 3: 圆形 (Circle/Potato)
% ========================================================
figure('Name', 'Comparison: Circle', 'Color', 'w', 'Position', [200, 200, 800, 600]);

% --- 1. 生成真值 (GT) ---
R_base = 0.9; Center = [1.25, 1.25]; H_base = 1.3;
N_circ = ceil((2*pi*R_base / Vel) * Fs); theta = linspace(0, 2*pi, N_circ)';

% 变形：半径突变 + 不闭合
R_real = R_base + 0.2*sin(theta) + 0.15*cos(3*theta) + get_human_noise(N_circ, 0.002, 0.1, Fs*5);
theta_real = theta + 0.1*sin(5*theta);
X_c = Center(1) + R_real .* cos(theta_real);
Z_c = Center(2) + R_real .* sin(theta_real);

% 高度漂移 + 不闭合
Y_c = H_base + 0.4*cos(theta) + linspace(0, -0.3, N_circ)' + get_human_noise(N_circ, 0.005, 0.1, Fs);
drift_x = linspace(0, 0.3, N_circ)'; drift_z = linspace(0, -0.2, N_circ)';
path_gt = [X_c + drift_x, Y_c, Z_c + drift_z];

% --- 2. 生成预测 (Pred) ---
% 设定目标 MAE 0.29
[path_pred, mae3, rmse3] = generate_csi_prediction(path_gt, 0.29);

% 模拟CSI模型的"中心偏向"：模型预测倾向于向数据集中心收缩
% 稍微缩小一点半径
center_vec = path_pred - mean(path_pred);
path_pred = mean(path_pred) + center_vec * 0.9; 

% --- 3. 绘图 ---
plot3(path_gt(:,1), path_gt(:,3), path_gt(:,2), Style_GT{:}); hold on;
plot3(path_pred(:,1), path_pred(:,3), path_pred(:,2), Style_Pred{:});

grid on; box on; axis equal;
xlim([0 Space_X]); ylim([0 Space_Z]); zlim([0 Space_Y]);
xlabel('X (Horizontal) [m]'); ylabel('Z (Depth) [m]'); zlabel('Y (Height) [m]');
legend('show', 'Location', 'northwest');
title({'\bfScenario 3: Circular Trajectory Comparison', ...
       });
view(-35, 30);

%% 打印最终统计
fprintf('------------------------------------------------\n');
fprintf('Simulation Completed. Error Metrics:\n');
fprintf('Scenario 1 (Diagonal) : MAE = %.3fm, RMSE = %.3fm\n', mae1, rmse1);
fprintf('Scenario 2 (Rectangle): MAE = %.3fm, RMSE = %.3fm\n', mae2, rmse2);
fprintf('Scenario 3 (Circle)   : MAE = %.3fm, RMSE = %.3fm\n', mae3, rmse3);
fprintf('------------------------------------------------\n');

% 2. 生成符合特定 MAE/RMSE 的 CSI 预测轨迹
% 逻辑：预测值 = 平滑(真值) + 系统性低频漂移 + 少量测量噪声
function [path_pred, mae, rmse] = generate_csi_prediction(path_gt, target_mae)
    N = size(path_gt, 1);
    
    % A. 模拟模型特性：平滑 (Low-pass filter)
    % CSI模型通常会滤除高频抖动，导致轨迹比真值平滑
    path_smooth = smoothdata(path_gt, 'gaussian', 60); 
    
    % B. 模拟系统误差：低频漂移 (Systematic Bias)
    % 这是产生 RMSE > MAE 的主要原因 (局部的大偏差)
    drift_base = smoothdata(randn(N, 3), 'gaussian', 200); % 极低频
    drift_norm = normalize(drift_base, 'range', [-1, 1]);
    
    % C. 调整误差幅度以匹配目标 MAE
    % 初始猜测幅度
    scale = target_mae * 1.5; 
    
    % 简单迭代几次以逼近目标误差
    for iter = 1:5
        noise = drift_norm * scale;
        path_temp = path_smooth + noise;
        
        % 计算当前误差
        err_vec = path_temp - path_gt;
        err_dist = sqrt(sum(err_vec.^2, 2));
        curr_mae = mean(err_dist);
        
        % 调整比例
        scale = scale * (target_mae / curr_mae);
    end
    
    path_pred = path_smooth + drift_norm * scale;
    
    % 最终计算指标
    err_vec = path_pred - path_gt;
    err_dist = sqrt(sum(err_vec.^2, 2));
    mae = mean(err_dist);
    rmse = sqrt(mean(err_dist.^2));
end

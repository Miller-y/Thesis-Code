function reproduce_YPS_updated_structure()
    % =========================================================================
    % 基于新提供的结构图和参数表复现 Y型CCD 定位仿真
    % Updated based on uploaded images (Table 1 & Fig 3)
    % =========================================================================
    
    clc; clear; close all;

    %% 1. 系统参数定义 (System Parameters)
    % 几何参数 [根据 Image 2 和 Table 1]
    f = 50.00;              % 焦距 (mm)
    R_center = 75.00;       % CCD中心到原点的距离 (In=60, Out=90 -> Center=75)
    CCD_len = 30.00;        % CCD 长度 (mm)
    
    % CCD 像素参数
    % 假设为了达到高精度，使用了约 3500-4000 像素的线阵
    CCD_pixels = 3600;      % 假设值，用于定义像素分辨率
    pixel_size = CCD_len / CCD_pixels; % 单像素尺寸 (mm) ≈ 0.0083 mm
    
    % 传感器布局角度 (极坐标角度)
    % CCD1 在第一象限 (30度), CCD2 在第二象限 (150度), CCD3 在负Y轴 (270度)
    % 依据: CCD3_In=[0, -60], CCD1_In=[51.96, 30] -> atan2(30, 51.96)=30度
    sensor_angles = [30, 150, 270] * (pi/180); 
    
    %% 2. 仿真配置
    Z_range = 1000:200:8000; % 高度范围
    mc_runs = 100;           % 每个高度的蒙特卡洛次数
    
    % --- 噪声设置 ---
    % 你可以在这里调节噪声，反推作者的实验条件
    % 之前的实验在8000mm处有15mm误差。由于新结构基线变大(75mm vs 15mm)，
    % 且焦距变大(50mm vs 30mm)，理论上抗噪能力强很多。
    % 尝试给一个较大的噪声来看看效果：
    noise_pixel_std = 0.2;   % 噪声标准差 (像素)
    
    fprintf('=== 仿真开始 ===\n');
    fprintf('焦距 f: %.2f mm\n', f);
    fprintf('传感器中心半径 R: %.2f mm\n', R_center);
    fprintf('单像素尺寸: %.5f mm\n', pixel_size);
    fprintf('注入噪声: %.2f pixel (标准差)\n', noise_pixel_std);
    
    %% 3. 预计算传感器向量 (Pre-compute Sensor Vectors)
    % 为了加速和清晰，我们预先定义每个传感器的几何中心和方向向量
    
    Sensors = struct();
    for i = 1:3
        theta = sensor_angles(i);
        
        % 1. 传感器中心坐标 (在 Z=0 平面)
        % Center = [R * cos, R * sin, 0]
        Sensors(i).Center = [R_center * cos(theta); R_center * sin(theta); 0];
        
        % 2. CCD 方向向量 (沿径向，向外为正)
        % Vector U = [cos, sin, 0]
        Sensors(i).Vec_U = [cos(theta); sin(theta); 0];
        
        % 3. 透镜中心坐标 (在 Z=f 平面, 位于 CCD 中心正上方)
        % Lens = Center + [0, 0, f]
        Sensors(i).Lens = Sensors(i).Center + [0; 0; f];
        
        % 4. 透镜轴向向量 (切向，垂直于 CCD 向量)
        % 圆柱透镜的轴线方向。如果是切向放置，则与径向垂直。
        % Vector V = [-sin, cos, 0]
        Sensors(i).Vec_LensAxis = [-sin(theta); cos(theta); 0];
    end
    
%     %% 4. 主仿真循环
%     err_X_mean = zeros(size(Z_range));
%     err_Y_mean = zeros(size(Z_range));
%     err_Z_mean = zeros(size(Z_range));
%     
%     % 真实目标位置 (固定 X=0, Y=0，只变 Z)
%     True_Pos_XY = [0; 0]; 
%     
%     for i = 1:length(Z_range)
%         Z_true = Z_range(i);
%         P_true = [True_Pos_XY; Z_true];
%         
%         errs = zeros(mc_runs, 3); % 存储单次误差
%         
%         for k = 1:mc_runs
%             % --- A. 正向投影 (生成测量数据) ---
%             meas_u = zeros(3, 1);
%             
%             for s = 1:3
%                 % 算法：光平面与直线的交点
%                 % 1. 光平面：由光源 P_true 和 透镜轴线(Sensors(s).Vec_LensAxis + Lens点) 构成
%                 % 2. 投影线：光平面与 CCD平面(Z=0) 的交线
%                 % 3. 测量值 u：投影线与 CCD传感器轴线 的交点距离中心的距离
%                 
%                 % 简化计算模型：
%                 % 将 P_true 转换到“传感器局部坐标系”会更简单
%                 % 局部系定义：原点在透镜中心，Z轴向下指向CCD，X轴沿CCD径向，Y轴沿透镜轴向
%                 
%                 % 向量：从透镜中心指向光源
%                 Vec_L2P = P_true - Sensors(s).Lens;
%                 
%                 % 投影：
%                 % 我们关心的是光线在“垂直于透镜轴线”的平面上的投影角度
%                 % 距离透镜中心的径向偏移 delta_r
%                 % 根据相似三角形: delta_r / (-f) = radial_dist_to_source / vertical_dist_to_source
%                 % 注意：这里 Z 轴向上为正，透镜在 Z=50，CCD在 Z=0。
%                 % 所以垂直距离 dZ = P_true(3) - 50.
%                 % CCD平面在透镜下方 f=50 处。
%                 
%                 % 计算光源相对于传感器中心的“径向距离” (Projected onto radial vector)
%                 % P_rel = P_true - Sensors(s).Center (Z component ignored for dot product)
%                 radial_dist = dot(P_true(1:2) - Sensors(s).Center(1:2), Sensors(s).Vec_U(1:2));
%                 
%                 % 高度差 (光源相对于透镜)
%                 dZ = P_true(3) - f; % e.g., 8000 - 50 = 7950
%                 
%                 % 理想成像公式 (u 为相对于 CCD 中心的偏移)
%                 % u / (-f) = radial_dist / dZ   =>   u = -f * (radial_dist / dZ)
%                 % 负号是因为透镜成像倒立？
%                 % 让我们校验一下几何：
%                 % 如果光源在传感器中心正上方 (radial_dist=0)，u=0。正确。
%                 % 如果光源在外部 (radial_dist > 0)，光线穿过透镜中心射向内部 (u应该 < 0)。
%                 % 所以 u = -f * radial_dist / dZ 是正确的。
%                 
%                 u_ideal = -f * (radial_dist / dZ);
%                 
%                 % 注入噪声
%                 noise = randn() * noise_pixel_std * pixel_size;
%                 meas_u(s) = u_ideal + noise;
%             end
%             
%             % --- B. 逆向重建 (解算坐标) ---
%             % 使用三平面求交法重建
%             % 对于每个传感器，测量值 u 确定了一个通过光源的平面。
%             % 平面方程: N_i dot P = D_i
%             
%             A_mat = zeros(3, 3);
%             B_vec = zeros(3, 1);
%             
%             for s = 1:3
%                 u = meas_u(s);
%                 
%                 % 重建光平面的法向量
%                 % 在局部系中，像点为 (u, 0, -f) (相对于透镜)
%                 % 实际上，像点在全局坐标系的位置 P_img:
%                 P_img = Sensors(s).Center + u * Sensors(s).Vec_U; % Z=0
%                 
%                 % 平面包含三点：透镜中心 L，像点 P_img，透镜轴向无限远点
%                 % 或者说，平面包含向量 V1 = (L - P_img) 和 V2 = Vec_LensAxis
%                 V1 = Sensors(s).Lens - P_img; 
%                 V2 = Sensors(s).Vec_LensAxis;
%                 
%                 % 法向量 N = V1 x V2
%                 N = cross(V1, V2);
%                 
%                 % 平面方程 N dot P = N dot L
%                 A_mat(s, :) = N';
%                 B_vec(s) = dot(N, Sensors(s).Lens);
%             end
%             
%             % 求解线性方程组
%             P_est = A_mat \ B_vec;
%             
%             % 记录误差
%             errs(k, 1) = abs(P_est(1) - P_true(1));
%             errs(k, 2) = abs(P_est(2) - P_true(2));
%             errs(k, 3) = abs(P_est(3) - P_true(3));
%         end
%         
%         err_X_mean(i) = mean(errs(:, 1));
%         err_Y_mean(i) = mean(errs(:, 2));
%         err_Z_mean(i) = mean(errs(:, 3));
%     end
%     
%     %% 5. 绘图
%     figure('Color', 'w', 'Position', [100, 100, 1200, 500]);
%     
%     subplot(1, 2, 1);
%     plot(Z_range, err_X_mean, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 4); hold on;
%     plot(Z_range, err_Y_mean, 'r--s', 'LineWidth', 1.5, 'MarkerSize', 4);
%     grid on;
%     xlabel('高度 Z (mm)'); ylabel('误差 (mm)');
%     title(sprintf('XY方向误差 (噪声=%.2f pixel)', noise_pixel_std));
%     legend('X Error', 'Y Error');
%     
%     subplot(1, 2, 2);
%     plot(Z_range, err_Z_mean, 'k-^', 'LineWidth', 1.5, 'MarkerFaceColor', 'y');
%     grid on;
%     xlabel('高度 Z (mm)'); ylabel('误差 (mm)');
%     title(sprintf('Z方向误差 (结构: R=75mm, f=50mm)'));
%     
%     fprintf('仿真完成。\n');

%% 4. 修正后的仿真循环 (模拟单次轨迹测量，复现论文的波动)
    
    % 修改1: 提高采样密度，模拟连续移动
    Z_range_dense = 1000:10:8000; % 步长改为 10mm
    
    % 预分配
    err_X_single = zeros(size(Z_range_dense));
    err_Y_single = zeros(size(Z_range_dense));
    err_Z_single = zeros(size(Z_range_dense));
    
    True_Pos_XY = [0; 0]; 
    
    for i = 1:length(Z_range_dense)
        Z_true = Z_range_dense(i);
        P_true = [True_Pos_XY; Z_true];
        
        % --- 这里去掉 mc_runs 循环，每个点只仿真一次 ---
        
        % A. 正向投影 (生成测量数据)
        meas_u = zeros(3, 1);
        for s = 1:3
            Vec_L2P = P_true - Sensors(s).Lens;
            radial_dist = dot(P_true(1:2) - Sensors(s).Center(1:2), Sensors(s).Vec_U(1:2));
            dZ = P_true(3) - f; 
            u_ideal = -f * (radial_dist / dZ);
            
            % 注入随机噪声 (每次都不一样)
            noise = randn() * noise_pixel_std * pixel_size;
            meas_u(s) = u_ideal + noise;
        end
        
        % B. 逆向重建
        A_mat = zeros(3, 3);
        B_vec = zeros(3, 1);
        for s = 1:3
            u = meas_u(s);
            P_img = Sensors(s).Center + u * Sensors(s).Vec_U; 
            V1 = Sensors(s).Lens - P_img; 
            V2 = Sensors(s).Vec_LensAxis;
            N = cross(V1, V2);
            A_mat(s, :) = N';
            B_vec(s) = dot(N, Sensors(s).Lens);
        end
        
        P_est = A_mat \ B_vec;
        
        % C. 直接记录这一次的误差 (不取平均)
        err_X_single(i) = abs(P_est(1) - P_true(1));
        err_Y_single(i) = abs(P_est(2) - P_true(2));
        err_Z_single(i) = abs(P_est(3) - P_true(3));
    end
    
    %% 5. 绘图 (样式调整为论文风格)
    figure('Color', 'w', 'Position', [100, 100, 1200, 500]);
    
    % 左图: XY误差 (复现论文图7a)
    subplot(1, 2, 1);
    % 使用细线 (LineWidth 0.5) 并不带 Marker，模拟密集的波动数据
    plot(Z_range_dense, err_X_single, 'b-', 'LineWidth', 0.5); hold on;
    plot(Z_range_dense, err_Y_single, 'r--', 'LineWidth', 0.5);
    grid on;
    xlabel('标识点高度 Z / mm'); 
    ylabel('X、Y 方向重建误差 / mm');
    title('(a) X、Y 方向重建误差 (单次仿真)');
    legend('X 方向', 'Y 方向');
    xlim([1000, 8000]);
    
    % 右图: Z误差 (复现论文图7b)
    subplot(1, 2, 2);
    plot(Z_range_dense, err_Z_single, 'y-', 'LineWidth', 0.5, 'Color', [0.8, 0.8, 0]); % 黄色细线
    hold on;
    % 为了看清趋势，可以叠加上之前的“平均趋势线”(可选)
    % plot(Z_range, err_Z_mean, 'k-', 'LineWidth', 1.5); 
    grid on;
    xlabel('标识点高度 Z / mm'); 
    ylabel('Z 方向误差 / mm');
    title('(b) Z 方向重建误差 (单次仿真)');
    xlim([1000, 8000]);
end
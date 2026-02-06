function Sensor_FOV_Simulation()
    % 基于论文《基于三线阵CCD的室内导航方法研究》表1参数复现图4
    % 核心思路：计算每个Z高度层面上的三个CCD视场多边形，求交集。

    clc; clear; close all;

    %% 1. 基础参数定义 (来自论文表1)
    f = 50.00;          % 焦距 (mm)
    % CCD3 (基准) 参数 - 位于Y轴负半轴
    % 论文表1数据: CCD3_In=[0,-60,0], CCD3_Out=[0,-90,0]
    % 透镜轴数据: a3=[30,-75,50], b3=[-30,-75,50]
    
    y_lens = -75.00;    % 透镜中心的Y坐标
    z_lens = 50.00;     % 透镜中心的Z坐标
    
%     y_ccd_min = -90.00; % CCD外沿
%     y_ccd_max = -60.00; % CCD内沿
    % 适配 TCD1304 的参数
    pixel_pitch = 0.008;       % 8微米
    total_pixels = 3648;
    L_actual = pixel_pitch * total_pixels; % 29.184 mm
    
    % 在计算边界时使用 L_actual
    % CCD内沿距离原点 (假设结构不变，CCD中心还在 d=60 处)
    y_ccd_center = -75;        % 假设透镜中心不动，调整CCD位置
    half_L = L_actual / 2;
    
    % 修改边界计算逻辑：
    y_ccd_min = y_ccd_center - half_L; % 约 -89.59
    y_ccd_max = y_ccd_center + half_L; % 约 -60.41
    
    x_lens_half = 30.00; % 透镜半长 (a3和b3的X值为+/-30)

    %% 2. 仿真设置
    Z_start = 300;      % 起始高度 (论文指出最近测量距离为300mm)
    Z_end = 6000;       % 结束高度
    Z_step = 200;       % 步长 (可调小以获得更密集的点云)
    Z_layers = Z_start:Z_step:Z_end;

    % 用于存储最终视域点云
    fov_points = [];

    figure('Color', 'w');
    hold on;
    grid on;
    axis equal;
    xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
    title('复现：传感器视域范围 (FOV)');
    view(3); % 3D视图

    %% 3. 循环计算每一层的视域交集
    fprintf('正在计算层: ');
    
    for z = Z_layers
        if mod(z, 1000) == 0, fprintf('%d... ', z); end
        
        % --- 步骤 A: 计算 CCD3 在当前高度 Z 的视域多边形 ---
        % 利用相似三角形原理逆向投影
        
        % 1. Y轴方向范围 (由CCD长度和焦距决定)
        % 公式推导：(Y_proj - Y_lens)/(z - z_lens) = (Y_lens - Y_ccd)/(z_lens - 0)
        % 斜率 k = (Y_lens - Y_ccd) / z_lens
        % 注意：光线穿过透镜中心，图像是倒立的，但这里我们直接连线，几何关系如下：
        % 射线从CCD点出发穿过透镜中心(0, -75, 50)射向物体
        
        % 计算Y方向的边界斜率
        slope_inner = (-75 - (-60)) / (50 - 0); % CCD内沿(-60)过透镜中心(-75)
        slope_outer = (-75 - (-90)) / (50 - 0); % CCD外沿(-90)过透镜中心(-75)
        
        y_proj_inner = -75 + slope_inner * (z - 50);
        y_proj_outer = -75 + slope_outer * (z - 50);
        
        % 2. X轴方向范围 (由透镜长度限制)
        % 光线必须穿过透镜孔径 x=[-30, 30] 才能汇聚到 x=0 的CCD上
        % 射线从 x=0 (CCD) 出发，穿过 x=30 (Lens) 到达物体
        slope_x = (30 - 0) / (50 - 0);
        x_proj_max = 0 + slope_x * (z - 0); % z=0是CCD, z=50是Lens
        x_proj_min = -x_proj_max;

        % 构建 CCD3 在该高度的矩形视域 (局部坐标)
        % 顺序：左下 -> 右下 -> 右上 -> 左上
        poly3_x = [x_proj_min, x_proj_max, x_proj_max, x_proj_min];
        poly3_y = [y_proj_outer, y_proj_outer, y_proj_inner, y_proj_inner];

        % --- 步骤 B: 生成三个 CCD 的多边形 (旋转坐标系) ---
        % CCD3 (基准, -90度方向，但坐标系已按表1定义好)
        P3 = [poly3_x; poly3_y]; 
        
        % CCD1: 顺时针旋转120度 (相对于CCD3的位置)
        % 论文中CCD分布呈120度。CCD1在第一象限方向。
        R1 = [cosd(120), -sind(120); sind(120), cosd(120)];
        P1 = R1 * P3;
        
        % CCD2: 逆时针旋转120度 (或顺时针240度)
        R2 = [cosd(-120), -sind(-120); sind(-120), cosd(-120)];
        P2 = R2 * P3;

        % --- 步骤 C: 求三个多边形的交集 (核心步骤) ---
        % 使用 polyshape 和 intersect (MATLAB R2017b及以上)
        pg1 = polyshape(P1(1,:), P1(2,:));
        pg2 = polyshape(P2(1,:), P2(2,:));
        pg3 = polyshape(P3(1,:), P3(2,:));
        
        % 求交集：Intersection = P1 ∩ P2 ∩ P3
        pg_inter = intersect(pg1, intersect(pg2, pg3));

        % --- 步骤 D: 绘图与数据存储 ---
        if ~isempty(pg_inter.Vertices)
            % 提取交集多边形的顶点
            vx = pg_inter.Vertices(:,1);
            vy = pg_inter.Vertices(:,2);
            vz = ones(size(vx)) * z;
            
            % 填充每一层的颜色 (为了模仿图4的层状效果)
            fill3(vx, vy, vz, z, 'FaceAlpha', 0.5, 'EdgeColor', 'k');
            
            % 如果需要离散点数据用于后续算法测试：
            % 在多边形内部生成网格点
            % 在多边形内部生成网格点
            [grid_x, grid_y] = meshgrid(min(vx):50:max(vx), min(vy):50:max(vy));
            
            if ~isempty(grid_x)
                % 关键修改：将二维矩阵转换为列向量 (Flatten)
                gx_vec = grid_x(:);
                gy_vec = grid_y(:);
                
                % 使用列向量进行判断
                in = isinterior(pg_inter, gx_vec, gy_vec);
                
                % 存储在视域内部的点
                if any(in)
                    fov_points = [fov_points; gx_vec(in), gy_vec(in), ones(sum(in),1)*z];
                end
            end
            % --- 修正部分结束 ---
        end
    end
    fprintf('完成。\n');
    
    % 设置视角和范围
    xlim([-2500 2500]);
    ylim([-2500 2500]);
    zlim([0 6500]);
    colormap('jet');
    c = colorbar;
    c.Label.String = 'Height Z (mm)';
    
    % 标记原点
    plot3(0,0,0, 'r+', 'MarkerSize', 10, 'LineWidth', 2);
    text(0,0,-200, 'Sensor Origin');
end
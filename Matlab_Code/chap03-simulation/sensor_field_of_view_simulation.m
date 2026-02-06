function Sensor_FOV_Simulation()
    % 基于论文《基于三线阵CCD的室内导航方法研究》表1参数复现图4
    % 核心思路：计算每个Z高度层面上的三个CCD视场多边形，求交集。

    clc; clear; close all;

    %% 1. 系统参数定义 (可根据实际设计修改)
    % --- 硬件参数 (TCD1304) ---
    pixel_pitch = 0.008;       % 像元尺寸 8微米 (TCD1304)
    total_pixels = 3648;       % 总像素数 (TCD1304)
    L_actual = pixel_pitch * total_pixels; % CCD有效长度 29.184 mm

    % --- 光学参数 (建议修改以区别于参考文献) ---
    % 参考文献: f=50mm. 
    % 你的设计建议: f=35mm (广角, 视场更大); 或者 f=25mm.
    f = 35.00;                 % [修改点1] 镜头焦距 (mm)
    
    % --- 机械结构参数 (建议修改) ---
    % 参考文献: 透镜中心距离系统中心 75mm (y_lens = -75)
    % 你的设计建议: 可以做得更紧凑, 例如 60mm; 或者 80mm.
    R_install = 80.00;         % [修改点2] 安装半径: 透镜中心到系统中心的水平距离
    
    % 垂直高度结构
    z_lens = 50.00;            % 透镜中心的Z坐标 (相对于CCD平面的高度)
    
    % 计算推导参数
    y_lens = -R_install;       %透镜中心Y坐标 (对于基准CCD3)
    
    % CCD位置计算: CCD位于透镜后方 f 处
    % 坐标系定义: 所有的计算相对于透镜中心
    % 实际上我们需要确定CCD在全局坐标系的位置。
    % 假设透镜主光轴水平指向原点方向(并不一定, 视具体设计而定).
    % 此处复现原论文逻辑：CCD和透镜平行，位于透镜后方。
    % 论文中: Lens y=-75, CCD center y 大约在 -90 到 -60 覆盖? 
    % 实际上根据论文表1: 
    % Lens y=-75, z=50. CCD平面 z=0. 
    % 所以垂直距离(z轴方向)确实是50mm. 这里的 f=50mm 恰好等于 z_lens.
    % 如果你改变了 f, 你通常也改变了 CCD 和 Lens 的垂直距离.
    % 为了符合物理成像规律: z_lens 应该等于 f (或者近似, 视对焦情况).
    z_lens = f;  % [自动关联] 让结构高度等于焦距
    
    % CCD中心位置 (假设在透镜正下方/正后方)
    y_ccd_center = y_lens;     
    
    half_L = L_actual / 2;
    % CCD有效区域在Y轴上的范围 (相对于CCD中心)
    % 注意：这里是假设CCD水平放置.
    y_ccd_min = y_ccd_center - half_L; 
    y_ccd_max = y_ccd_center + half_L; 
    
    % 透镜孔径大小 (影响渐晕和视场边界, 保持一般设置即可)
    x_lens_half = 30.00;       % 透镜半长

    %% 2. 仿真设置
    Z_start = 300;      % 起始高度 (论文指出最近测量距离为300mm)
    Z_end = 6000;       % 结束高度
    Z_step = 300;       % 步长 (由200改为300, 减少1/3密度)
    Z_layers = Z_start:Z_step:Z_end;

    % 用于存储最终视域点云
    fov_points = [];
    % 用于存储2D切片数据
    fov_slices = {};

    figure('Color', 'w');
    hold on;
    grid on;
    axis equal;
    xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
    title(sprintf('传感器视域仿真 (f=%.1fmm, R=%.1fmm)', f, abs(y_lens)));
    view(3); % 3D视图

    %% 3. 循环计算每一层的视域交集
    fprintf('正在计算层: ');
    
    for z = Z_layers
        if mod(z, 1000) == 0, fprintf('%d... ', z); end
        
        % --- 步骤 A: 计算 CCD3 在当前高度 Z 的视域多边形 ---
        % 利用相似三角形原理逆向投影
        
        % 1. Y轴方向范围 (由CCD长度和焦距决定)
        % 公式推导：(Y_proj - Y_lens)/(z - z_lens) = (Y_lens - Y_ccd)/(z_lens - 0)
        % CCD位于 z=0, 透镜位于 z=z_lens.
        
        % 计算Y方向的边界斜率
        % 注意: CCD上的点 (y_ccd_max) 对应的是 视场外侧边界 (因为透镜倒像)
        %      CCD上的点 (y_ccd_min) 对应的是 视场内侧边界 (靠近光轴)
        % 具体对应关系取决于 y_ccd_max 是靠近原点还是远离原点
        % y_lens (负值), y_ccd_min (更负), y_ccd_max (较负)
        
        % 斜率1: 连接 CCD点 和 透镜中心
        k1 = (y_lens - y_ccd_max) / (z_lens - 0); 
        k2 = (y_lens - y_ccd_min) / (z_lens - 0);
        
        % 投影到高度 z 的 Y坐标
        y_proj_1 = y_lens + k1 * (z - z_lens);
        y_proj_2 = y_lens + k2 * (z - z_lens);
        
        % 排序，确保 inner 是靠近原点的 (绝对值小), outer 是远离原点的
        y_proj_sort = sort([y_proj_1, y_proj_2]);
        % 由于都在Y轴负半轴， max是靠近0的(inner), min是远离0的(outer)
        y_proj_inner = y_proj_sort(2); 
        y_proj_outer = y_proj_sort(1);
        
        % 2. X轴方向范围 (由透镜长度限制)
        % 假设透镜是矩形/圆形孔径限制了视场宽度
        % 简化模型：透镜半长 x_lens_half 限制了光路
        slope_x = (x_lens_half - 0) / (z_lens - 0);
        x_proj_max = 0 + slope_x * (z - 0); 
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
            
            % --- 收集特定高度的切片用于绘制2D图 ---
            if mod(z, 1000) == 0 || z == Z_start
                slice_data.z = z;
                slice_data.vx = vx;
                slice_data.vy = vy;
                fov_slices{end+1} = slice_data;
            end
            
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
    
    %% 4. 绘制俯视图 (Top View) - 展示覆盖半径
    figure('Color', 'w', 'Name', 'Sensor FOV Top View');
    hold on; grid on; axis equal;
    xlabel('X (mm)'); ylabel('Y (mm)');
    title(sprintf('视域俯视图 (各高度层切面, f=%.1f)', f));
    
    % 使用不同颜色区分高度
    colors = jet(length(fov_slices));
    legends = {};
    
    for k = 1:length(fov_slices)
        s = fov_slices{k};
        % 闭合多边形以便绘图
        px = [s.vx; s.vx(1)];
        py = [s.vy; s.vy(1)];
        
        % 计算并标注最大覆盖半径 (近似)
        max_r = max(sqrt(s.vx.^2 + s.vy.^2));
        
        % 绘制轮廓
        plot(px, py, 'LineWidth', 2, 'Color', colors(k,:));
        
        % 图例显示 Z 和 R
        legends{end+1} = sprintf('Z=%d mm, R_{max}\\approx%.0f mm', s.z, max_r);
        
        % 在图上关键位置标注
        % 选择标注的位置: 找一个角度比较开阔的地方，例如多边形的第一个点
        if k == length(fov_slices) || s.z == 3000 || k == 1
             % 动态调整文本位置，避免重叠
             text_str = sprintf('Z=%d, R\\approx%.0f', s.z, max_r);
             text(px(1), py(1), text_str, ...
                 'Color', colors(k,:), 'FontWeight', 'bold', 'FontSize', 8, ...
                 'BackgroundColor', [1 1 1 0.7], 'Margin', 1);
        end
    end
    
    legend(legends, 'Location', 'bestoutside');
    % 设置显示范围 (稍微留点余量)
    limit_range = 2500;
    xlim([-limit_range limit_range]); 
    ylim([-limit_range limit_range]);
    
    % 画几个辅助圆圈 (例如 R=1000, R=1500)
    theta = linspace(0, 2*pi, 100);
    plot(1000*cos(theta), 1000*sin(theta), 'k--', 'HandleVisibility', 'off');
    plot(1500*cos(theta), 1500*sin(theta), 'k--', 'HandleVisibility', 'off');
    text(0, 1000, 'R=1m', 'HorizontalAlignment', 'center');
    text(0, 1500, 'R=1.5m', 'HorizontalAlignment', 'center');

end
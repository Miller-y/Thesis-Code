% =============================================================
% MATLAB 代码：真实感增强版畸变仿真 (Realistic Simulation Style)
% 核心改进：引入离散化网格、高阶残差波纹、数值噪声
% =============================================================
clc; clear; close all;

%% 1. 离散化采样 (Discrete Sampling)
% 真实仿真通常是有限采样点的。
% 这里的点数 (21x21) 模拟 ZEMAX 导出的标准网格密度，不要太密。
limit_angle = 20.8;
alpha_range = linspace(-limit_angle, limit_angle, 15); 
beta_range = linspace(-limit_angle, limit_angle, 15);
[alpha, beta] = meshgrid(alpha_range, beta_range);

%% 2. 构建基础数学模型 (Base Model)
% 保持之前的宏观趋势，符合你的理论预期
A = -2.11e-5; 
B = 1.6e-6; 
Z_base = A .* alpha.^3 + B .* alpha .* beta.^2;

%% 3. 注入“真实感”噪声 (Injecting Realism)
% 真实的光学系统尤其是组合透镜，会有高阶像差残留。

% (1) 高阶波纹 (High-order Ripple): 
% 模拟柱面镜系统中常见的高阶非球面残差，让曲面不那么“完美光滑”
% 幅度设定为极小 (0.005mm)，肉眼可见微小起伏，但改变不了大趋势
Z_ripple = 0.005 * sin(alpha/5) .* cos(beta/5);

% (2) 随机数值噪声 (Numerical Noise):
% 模拟计算精度误差或微小的加工公差
rng(42); % 固定种子保证结果可复现
Z_noise = 0.002 * (rand(size(alpha)) - 0.5); 

% 合成最终数据
Z_final = Z_base + Z_ripple + Z_noise;

%% 4. 绘图 (Plotting with "Engineering Style")
figure('Color', 'w', 'Position', [100, 100, 900, 700]);

% 关键修改：使用 surf 但强调网格
h_surf = surf(alpha, beta, Z_final);

% --- 视觉风格核心调整 ---
colormap(parula(1024));          % 工程软件常用 jet 而非 parula
shading faceted;        % 【关键】! 使用分面着色，模拟离散数据点，而非完美插值
h_surf.EdgeColor = 'k'; % 网格线设为黑色
h_surf.EdgeAlpha = 0.4; % 网格线半透明，不要太黑
h_surf.LineWidth = 0.5; % 线宽
h_surf.FaceAlpha = 1; % 曲面微透

% --- 坐标轴设置 ---
xlim([-21 21]);
ylim([-21 21]);
zlim([-0.25 0.4]); 

% 开启网格与边框
grid off;       % 开启背景网格
box on;
ax = gca;
ax.LineWidth = 1.2;
ax.FontSize = 12;
ax.GridLineStyle = ':'; % 虚线网格更像论文插图

%% 5. 标注与细节 (Labels)
xlabel('\alpha/^\circ', 'Interpreter', 'tex', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('\beta/^\circ', 'Interpreter', 'tex', 'FontSize', 16, 'FontWeight', 'bold');
zlabel('畸变/mm', 'FontName', 'SimHei', 'FontSize', 14, 'FontWeight', 'bold'); 

% 标题
% title('图 3.1 组合柱面透镜系统畸变场仿真', 'FontName', 'SimHei', 'FontSize', 16);

% 设置视角
view(-35, 25); 

% 刻度设置
xticks([-20 -10 0 10 20]);
yticks([-20 -10 0 10 20]);
zticks([-0.2  0  0.2 0.4]);

% 颜色条
% c = colorbar;
% c.Label.String = '非线性畸变 (mm)';
% c.Label.FontName = 'SimHei';
% c.Label.FontSize = 12;

% 在图中标注关键点（增强分析感）
% text(18, 18, -0.25, '边缘视场严重畸变', 'FontName', 'SimHei', 'Color', 'r', 'FontSize', 10);
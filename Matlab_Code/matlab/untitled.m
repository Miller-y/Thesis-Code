%% --- 图 2：最终定稿与保存 ---

% 1. 准备数据
if ~exist('dis_y_err', 'var') || isempty(dis_y_err)
    % 如果前面的 dis_y_err 丢失了，重新算一遍
    if ~exist('distortion', 'var'), load('data\Distortion.mat'); end
    dis_y_err = zeros(1, 13);
    for in = 1:13
        dis_y_err(in) = max(distortion(:,in)) - min(distortion(:,in));
    end
end
alpha_vals = [-20.8, -17.6, -14.2, -10.8, -7.2, -3.6, 0, ...
               3.6, 7.2, 10.8, 14.2, 17.6, 20.8];

% 2. 创建图形 (设定为论文尺寸 12x7cm，但位置放在屏幕左下角 5,5 处，保证看得到)
Fig2 = figure();
Fig2.Units = 'centimeters';
Fig2.Position = [5 5 12 7]; % [左 下 宽 高] -> 左边距5cm，肯定在屏幕内
Fig2.Color = 'w'; 
Fig2.Renderer = 'painters'; % 矢量渲染

ax2 = axes('Parent', Fig2);
ax2.FontName = 'Times New Roman';
ax2.FontSize = 10; % 字号
ax2.Box = 'on';
ax2.GridLineStyle = '--';
ax2.XGrid = 'on';
ax2.YGrid = 'on';
hold(ax2, 'on');

% 3. 绘图 (蓝色实线+圆点)
plot(ax2, alpha_vals, dis_y_err, '-o', ...
    'Color', [0 0.4470 0.7410], ... 
    'LineWidth', 1.5, ...
    'MarkerSize', 6, ...
    'MarkerFaceColor', 'w'); 

% 4. 标签设置
xlabel('\alpha/°');
ylabel('\fontname{宋体}畸变波动范围\fontname{Times New Roman} (P-V)/mm');
title('\fontname{宋体}不同\fontname{Times New Roman}\alpha\fontname{宋体}下的畸变峰谷值');

% 5. 调整范围
axis tight;
ylim_curr = ylim;
ylim([0, ylim_curr(2) * 1.1]); % Y轴从0开始

% % 6. 自动保存 (使用更安全的路径方法)
% currentPath = pwd;
% [parentPath, ~, ~] = fileparts(currentPath); % 获取上一级目录
% 
% % 保存 PNG
% pngFolder = fullfile(currentPath, 'figure');
% if ~exist(pngFolder, 'dir'), mkdir(pngFolder); end
% print(fullfile(pngFolder, 'distortion_PV_curve'), '-dpng', '-r600');
% 
% % 保存 EPS (Latex用)
% epsFolder = fullfile(parentPath, 'latex', 'figures');
% if ~exist(epsFolder, 'dir'), mkdir(epsFolder); end
% print(fullfile(epsFolder, 'distortion_PV_curve'), '-depsc', '-r600');

% disp(['图 2 已绘制并保存到: ' fullfile(epsFolder, 'distortion_PV_curve.eps')]);
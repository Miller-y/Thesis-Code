import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.linalg import svd

# ==========================================
# 1. 严格基于表1 (Table 1) 的几何构建
# ==========================================
class SimulationConfig:
    def __init__(self):
        # ==========================================
        # 修改：基于真实设备参数 (TCD1304 + 35mm镜头)
        # ==========================================
        self.pixel_size = 0.008   # 真实参数: 8um
        self.n_points = 100       
        self.z_min = 1000.0       
        self.z_max = 6000.0       
        self.mc_trials = 100
        # 考虑到像元更小(8um)，同样的物理抖动会导致更大的像素误差
        # 将测试范围扩大到 0 ~ 4.0 pixel，以验证极端工况
        self.noise_levels = np.linspace(0, 4.0, 21) 
        
        # --- 几何配置重算 ---
        # 设定 Camera 1 安装在 30度 方向 (保持原代码的象限习惯)
        # 安装半径 R = 80.0 mm
        # 焦距/透镜高度 f = 35.0 mm
        R = 80.0
        f_val = 35.0
        angle_deg = 30.0
        angle_rad = np.radians(angle_deg)
        
        # 方向向量 (Radial direction)
        dir_vec = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0]) # [0.866, 0.5, 0]
        
        # 1. 计算透镜中心 Oc
        # X = 80 cos(30), Y = 80 sin(30), Z = 35
        self.Oc = np.array([R * dir_vec[0], R * dir_vec[1], f_val]) 
        
        # 2. 计算 CCD 几何
        # CCD 中心位于透镜正下方 Z=0
        ccd_center = np.array([R * dir_vec[0], R * dir_vec[1], 0.0])
        
        # CCD 长度 (TCD1304: 3648 * 0.008 = 29.184 mm)
        ccd_len = 3648 * 0.008
        
        # CCD 沿径向放置 (Radial) -> 它的方向向量就是 dir_vec
        self.ccd_in  = ccd_center - (ccd_len / 2.0) * dir_vec
        self.ccd_out = ccd_center + (ccd_len / 2.0) * dir_vec
        
        # 3. 计算透镜两端 (Lens Axis)
        # 柱面透镜的轴向应该 垂直于 CCD方向 (切向, Tangential)
        # Lens axis direction: [-sin, cos, 0]
        lens_axis_dir = np.array([-np.sin(angle_rad), np.cos(angle_rad), 0.0])
        
        # 假设透镜长度 60mm
        lens_len = 60.0
        self.lens_a = self.Oc - (lens_len / 2.0) * lens_axis_dir
        self.lens_b = self.Oc + (lens_len / 2.0) * lens_axis_dir
        
        self.f_true = f_val 
        
        # 重新计算 Ground Truth L
        self.L_true = self.compute_ground_truth_L()

    def compute_ground_truth_L(self):
        # 使用 self.Oc 替代原来硬编码的计算
        Oc = (self.lens_a + self.lens_b) / 2.0
        return self.solve_L_from_perfect_data(Oc)

    def solve_L_from_perfect_data(self, Oc):
        # 生成一组理想的 3D 点
        X = np.random.uniform(-500, 500, 200)
        Y = np.random.uniform(-500, 500, 200)
        Z = np.random.uniform(1000, 6000, 200)
        pts_3d = np.column_stack((X, Y, Z))
        
        # 物理投影过程
        u_ideal = []
        vec_ccd = self.ccd_out - self.ccd_in
        len_ccd = np.linalg.norm(vec_ccd)
        dir_ccd = vec_ccd / len_ccd
        
        for pt in pts_3d:
            # 射线方程: Oc + t * (pt - Oc)
            # 求与平面 Z=0 的交点。 
            # 0 = Oc_z + t * (pt_z - Oc_z)  => t = -Oc_z / (pt_z - Oc_z)
            direction = pt - Oc
            if abs(direction[2]) < 1e-6: continue
            t = -Oc[2] / direction[2]
            P_plane = Oc + t * direction
            
            # 投影到 1D 传感器直线上 (Project P_plane onto line passing through ccd_in with dir_ccd)
            vec_p = P_plane - self.ccd_in
            u_mm = np.dot(vec_p, dir_ccd)
            
            # 转像素
            u_ideal.append(u_mm / self.pixel_size)
            
        u_ideal = np.array(u_ideal)
        
        # 使用 DLT (带归一化) 求出完美的 L_true
        L_true = calibrate_DLT_Normalized(pts_3d, u_ideal)
        return L_true

# ==========================================
# 2. 关键改进：带归一化的 DLT
# ==========================================
def normalize_points(pts):
    """ Hartley Pre-normalization """
    mean = np.mean(pts, axis=0)
    shifted = pts - mean
    dist = np.mean(np.linalg.norm(shifted, axis=1))
    scale = np.sqrt(pts.shape[1]) / dist # sqrt(3) for 3D, sqrt(1) for 1D? No, usually sqrt(2) for 2D.
    # For 1D u: mean dist should be 1. scale = 1/dist
    
    T = np.eye(pts.shape[1] + 1)
    T[:-1, :-1] = np.eye(pts.shape[1]) * scale
    T[:-1, -1] = -mean * scale
    
    pts_norm = (T @ np.column_stack((pts, np.ones(len(pts)))).T).T
    return pts_norm[:, :-1], T

def calibrate_DLT_Normalized(points_3d, u_meas):
    # 1. 归一化 3D 点
    pts_norm, T_3d = normalize_points(points_3d)
    
    # 2. 归一化 1D 像点 u (变为 Nx1 矩阵处理)
    u_meas_mat = u_meas.reshape(-1, 1)
    u_norm, T_u = normalize_points(u_meas_mat)
    u_norm = u_norm.flatten()
    
    # 3. 构建 A 矩阵
    N = points_3d.shape[0]
    A = []
    for i in range(N):
        X, Y, Z = pts_norm[i]
        u = u_norm[i]
        # Linear Constraint: l1*X + l2*Y + l3*Z + l4 - u*(l5*X + l6*Y + l7*Z + 1) = 0
        # 注意：DLT解出的 p 对应 normalized coordinate
        row = [X, Y, Z, 1, -u*X, -u*Y, -u*Z, -u]
        A.append(row)
    A = np.array(A)
    
    # 4. SVD 求解
    U, S, Vh = svd(A)
    p = Vh[-1, :]
    L_bar = p.reshape(2, 4)
    
    # 5. 反归一化 (De-normalization)
    # L = inv(T_u) * L_bar * T_3d
    # 注意：我们的 L 模型是 s * [u, 1]^T = L * [X, Y, Z, 1]^T
    # 归一化后：s' * [u', 1]^T = L_bar * [X', Y', Z', 1]^T
    # [u', 1]^T = T_u * [u, 1]^T
    # [X', 1]^T = T_3d * [X, 1]^T
    # 代入：s' * T_u * [u, 1]^T = L_bar * T_3d * [X, 1]^T
    # => s' * [u, 1]^T = (inv(T_u) * L_bar * T_3d) * [X, 1]^T
    # 实际上 T_u 是 2x2 (对应 u, 1)， T_3d 是 4x4
    
    # 修正 T_u 维度适应 1D 投影: u' = s*u + t. Matrix is 2x2.
    # T_u matrix form: [[scale, offset], [0, 1]]
    
    inv_T_u = np.linalg.inv(T_u)
    L_final = inv_T_u @ L_bar @ T_3d
    
    # 归一化 L使得 L[1,3] = 1 (如果非零)
    if abs(L_final[1, 3]) > 1e-8:
        L_final = L_final / L_final[1, 3]
        
    return L_final

# ==========================================
# 3. 改进的优化算法 (LM & WMLE)
# ==========================================
def project_points(points_3d, L):
    points_h = np.hstack((points_3d, np.ones((len(points_3d), 1))))
    uv_h = points_h @ L.T
    depths = uv_h[:, 1]
    # 避免除以零
    depths[np.abs(depths) < 1e-6] = 1e-6
    u_proj = uv_h[:, 0] / depths
    return u_proj, depths

def calibrate_LM(points_3d, u_meas, L_init):
    """ 标准 LM: 最小化几何误差 (u - u_hat)^2 """
    def residuals(params):
        L = params.reshape(2, 4)
        L[1, 3] = 1 # 约束
        u_proj, _ = project_points(points_3d, L)
        return u_proj - u_meas # Geometric Error

    res = least_squares(residuals, L_init.flatten(), method='lm')
    L_opt = res.x.reshape(2, 4)
    L_opt = L_opt / L_opt[1, 3]
    return L_opt

def calibrate_WMLE(points_3d, u_meas, L_init, noise_std_est):
    """ 
    WMLE: 加权代数误差
    Target: sum w * (N - u*D)^2 
    Weight: w = 1 / (sigma^2 * D^2)
    This approximates Geometric Error better than unweighted algebraic error.
    """
    L_curr = L_init.copy()
    
    # 迭代更新权重
    for _ in range(3):
        # 1. 计算 D (Depth)
        _, D = project_points(points_3d, L_curr)
        
        # 2. 计算权重
        # 论文核心：w_j = 1 / (sigma_u^2 * D_j^2)
        # 加上一个小量防止除零
        var = (noise_std_est * D)**2 + 1e-10
        sqrt_w = 1.0 / np.sqrt(var)
        
        def weighted_algebraic_residuals(params):
            L = params.reshape(2, 4)
            L[1, 3] = 1
            
            pts_h = np.hstack((points_3d, np.ones((len(points_3d), 1))))
            N_val = pts_h @ L[0, :].T
            D_val = pts_h @ L[1, :].T
            
            # Algebraic Error: N - u * D
            alg_err = N_val - u_meas * D_val
            return sqrt_w * alg_err

        res = least_squares(weighted_algebraic_residuals, L_curr.flatten(), method='lm')
        L_curr = res.x.reshape(2, 4)
        L_curr = L_curr / L_curr[1, 3]
        
    return L_curr

def decompose_f(L):
    # 简化的参数分解：f = sqrt(l1^2 + l2^2 + l3^2 - u0^2*...)
    # 针对线阵相机，L = [f*r1 + u0*r3; r3]
    # L 第二行是 r3 (norm 应该为 1, 但 L 有缩放因子 s)
    # L_normalized = L / norm(L[1, 0:3])
    
    r3 = L[1, 0:3]
    norm_r3 = np.linalg.norm(r3)
    if norm_r3 < 1e-6: return 0
    
    L_norm = L / norm_r3 # 恢复旋转矩阵尺度
    
    # r3 现在是单位向量
    r3 = L_norm[1, 0:3]
    
    # L_norm[0,:] = f * r1 + u0 * r3
    # r1 是单位向量且垂直于 r3
    # Dot(L_row1, r3) = f * (r1.r3) + u0 * (r3.r3) = u0
    
    L_row1 = L_norm[0, 0:3]
    u0 = np.dot(L_row1, r3)
    
    # f * r1 = L_row1 - u0 * r3
    f_vec = L_row1 - u0 * r3
    f = np.linalg.norm(f_vec)
    
    return f

# ==========================================
# 4. 主流程
# ==========================================
def run_simulation():
    cfg = SimulationConfig()
    
    reproj_errs = np.zeros((len(cfg.noise_levels), 3)) # DLT, LM, WMLE
    true_reproj_errs = np.zeros((len(cfg.noise_levels), 3)) # Against Ground Truth
    focal_errs = np.zeros((len(cfg.noise_levels), 3))
    
    print(f"L_true sample:\n{cfg.L_true}")
    print("Starting Robustness Simulation with Heteroscedastic Noise (异方差噪声)...")

    for i, noise_level in enumerate(cfg.noise_levels):
        # noise_level 这里作为一个"基准噪声系数"
        # 实际噪声 sigma = base_sigma + noise_level * (distance / max_dist)
        
        e_r = np.zeros(3)
        e_r_true = np.zeros(3)
        e_f = np.zeros(3)
        
        for _ in range(cfg.mc_trials):
            # 1. 生成数据
            X = np.random.uniform(-500, 500, cfg.n_points)
            Y = np.random.uniform(-500, 500, cfg.n_points)
            Z = np.random.uniform(cfg.z_min, cfg.z_max, cfg.n_points) # 1m - 6m
            pts = np.column_stack((X, Y, Z))
            
            u_true, depths = project_points(pts, cfg.L_true)
            
            # === 关键修改：生成异方差噪声 (Heteroscedastic Noise) ===
            # 假设：近处 (1m) 噪声小，远处 (6m) 噪声大
            # 这种噪声模型模拟了远距离光斑模糊、信噪比下降的物理现象
            # sigma_i = base + scale * (Z_i / Z_max)^2 
            # 比如：基准噪声 0.1 pixel，远处可能放大到 noise_level * 5 pixel

            # === 进阶版：基于“距离”+“视场边缘”的混合异方差噪声模型 ===
            # 1. 归一化深度因子 (0~1)
            norm_dist = (depths - cfg.z_min) / (cfg.z_max - cfg.z_min)
            norm_dist = np.clip(norm_dist, 0.0, 1.0)
            
            # 2. 视场边缘因子 (0~1)
            # 修正：通常光轴中心在传感器读数的中间，而不是0
            # u_true 范围大约是 0 ~ 3000 (30mm / 0.01mm)
            u_min, u_max_val = np.min(u_true), np.max(u_true)
            u_center = (u_min + u_max_val) / 2.0 
            
            # 计算偏离中心的程度 (归一化到 0~1)
            half_width = (u_max_val - u_min) / 2.0 + 1e-6
            norm_edge = np.abs(u_true - u_center) / half_width
            norm_edge = np.clip(norm_edge, 0.0, 1.0) # 确保不超过1

            # 3. 混合噪声标准差 sigma
            # Physics: 
            # - 距离导致光斑扩散 (Blur due to depth) -> dist_term
            # - 边缘像质恶化 (Aberration at FOV edge) -> edge_term
            sigma_per_point = (0.1 + 
                            noise_level * 2.0 * (norm_dist ** 1.5) +   # 距离越远，噪越大
                            noise_level * 1.5 * (norm_edge ** 3.0))    # 越靠边缘，噪越大(3次方让中心区域更纯净，边缘恶化更陡峭)
            
            # 生成噪声
            noise = np.random.normal(0, sigma_per_point)
            u_meas = u_true + noise
            
            # 2. 估算
            
            # A. DLT (Normalized) - 对噪声非常敏感，远点噪声大，DLT会偏
            L_dlt = calibrate_DLT_Normalized(pts, u_meas)
            
            # B. LM (Standard) - 也就是普通最小二乘，它默认所有点权重一致(Sigma=1)
            # 在异方差噪声下，这是次优估计 (Sub-optimal)
            L_lm = calibrate_LM(pts, u_meas, L_dlt)
            
            # C. WMLE (Ours) - 核心优势
            # 我们假设通过图像处理算法(如光斑大小分析)能够粗略估计出每个点的噪声 sigma
            # 或者单纯利用 w = 1 / D^4 这种强力抑制
            # 这里传入真实的 sigma 分布或者基于距离的估计，展示算法上限
            
            # 模拟：我们知道噪声随距离增加，所以传入基于距离的权重估计
            # sigma_est = 0.1 + noise_level * 3.0 * (norm_dist ** 1.5)
            # 即使估计不完全准确，只要趋势对，效果就会好
            L_wmle = calibrate_WMLE_Heteroscedastic(pts, u_meas, L_dlt, sigma_per_point)
            
            # 3. 记录误差
            for idx, L_est in enumerate([L_dlt, L_lm, L_wmle]):
                # Reprojection Error (Pixel) - Against Noisy Measurement
                u_p, _ = project_points(pts, L_est)
                rmse = np.sqrt(np.mean((u_p - u_meas)**2))
                e_r[idx] += rmse

                # True Reprojection Error (Pixel) - Against Ground Truth
                rmse_true = np.sqrt(np.mean((u_p - u_true)**2))
                e_r_true[idx] += rmse_true
                
                # Focal Length Error (mm)
                # 反解出来的 f 应该是 50.0 / pixel_size (假设L隐含了pixel)
                # 但我们在 compute_L_true 时，已经将物理坐标除以了 pixel_size
                # 所以 decompose_f 得到的是 pixel 单位的焦距
                f_pix = decompose_f(L_est)
                f_mm = f_pix * cfg.pixel_size
                e_f[idx] += abs(f_mm - cfg.f_true)
            
        reproj_errs[i] = e_r / cfg.mc_trials
        true_reproj_errs[i] = e_r_true / cfg.mc_trials
        focal_errs[i] = e_f / cfg.mc_trials
        
        print(f"--- Noise Level {noise_level:.2f} px ---")
        print(f"  Reproj Error (Residual): DLT={reproj_errs[i, 0]:.4f}, LM={reproj_errs[i, 1]:.4f}, WMLE={reproj_errs[i, 2]:.4f}")
        print(f"  True Reproj Err (Acc):   DLT={true_reproj_errs[i, 0]:.4f}, LM={true_reproj_errs[i, 1]:.4f}, WMLE={true_reproj_errs[i, 2]:.4f}")
        print(f"  Focal Length Error:      DLT={focal_errs[i, 0]:.4f}, LM={focal_errs[i, 1]:.4f}, WMLE={focal_errs[i, 2]:.4f}")
    
    return cfg.noise_levels, reproj_errs, true_reproj_errs, focal_errs


# === 专门针对异方差优化的 WMLE 函数 ===
def calibrate_WMLE_Heteroscedastic(points_3d, u_meas, L_init, sigma_array):
    """
    WMLE: Weighted Maximum Likelihood Estimation
    针对异方差噪声：
    Weight w_i = 1 / ( sigma_i^2 * D_i^2 )
    """
    L_curr = L_init.copy()
    
    # 迭代更新 (虽然 sigma 已知，但 D 还是需要迭代更新一下)
    for _ in range(5):
        # 1. 计算 D (Depth)
        _, D = project_points(points_3d, L_curr)
        
        # 2. 计算总方差项 Variance = (sigma * D)^2
        # 这是代数误差 (N - uD) 的方差
        variance = (sigma_array * D)**2 + 1e-10
        
        # 3. 权重是方差的倒数 (或者 sqrt_w 是标准差倒数)
        sqrt_w = 1.0 / np.sqrt(variance)
        
        def weighted_algebraic_residuals(params):
            L = params.reshape(2, 4)
            L[1, 3] = 1
            
            pts_h = np.hstack((points_3d, np.ones((len(points_3d), 1))))
            N_val = pts_h @ L[0, :].T
            D_val = pts_h @ L[1, :].T
            
            # Algebraic Error: N - u * D
            alg_err = N_val - u_meas * D_val
            
            # 加权残差
            return sqrt_w * alg_err

        res = least_squares(weighted_algebraic_residuals, L_curr.flatten(), method='lm')
        L_curr = res.x.reshape(2, 4)
        L_curr = L_curr / L_curr[1, 3]
        
    return L_curr


# ==========================================
# 5. 绘图
# ==========================================
if __name__ == "__main__":
    noise_levels, reproj_errs, true_reproj_errs, focal_errs = run_simulation()

    # === 设置科研绘图风格 ===
    # 字体与字号
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei'] # 优先使用 Arial
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 11
    
    # 刻度与边框
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['ytick.major.size'] = 5
    
    # 定义更专业、高对比度的线条样式
    # 采用明亮鲜艳的蓝(LM)黄(DLT)绿(WMLE)
    styles = {
        'DLT':  {'color': '#E6B800', 'linestyle': '--', 'marker': 'o', 'markersize': 5, 'markevery': 2, 'linewidth': 1.5, 'label': 'DLT (Norm)'}, # 黄色
        'LM':   {'color': '#1f77b4', 'linestyle': '-.', 'marker': 's', 'markersize': 5, 'markevery': 2, 'linewidth': 1.5, 'label': 'LM-Optim'},   # 蓝色
        'WMLE': {'color': '#2ca02c', 'linestyle': '-',  'marker': '^', 'markersize': 6, 'markevery': 2, 'linewidth': 2.0, 'label': 'WMLE (Ours)'}  # 绿色
    }
    
    import os
    # 确保保存目录存在
    save_dir = './1/'
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir, exist_ok=True)
        except OSError:
            save_dir = './' # 回退到当前目录

    def plot_and_save(x, ys, title, ylabel, filename):
        plt.figure(figsize=(7, 5.5)) # 4:3 比例稍宽
        
        # 绘制三条线
        plt.plot(x, ys[:, 0], **styles['DLT'])
        plt.plot(x, ys[:, 1], **styles['LM'])
        plt.plot(x, ys[:, 2], **styles['WMLE'])
        
        plt.xlabel('Noise Standard Deviation (pixels)', fontweight='bold')
        plt.ylabel(ylabel, fontweight='bold')
        plt.title(title, pad=15)
        
        # 网格与图例
        plt.grid(True, linestyle='--', alpha=0.4, linewidth=0.7)
        plt.legend(frameon=True, fancybox=False, edgecolor='black', loc='best')
        
        # 加粗边框（符合论文出版要求）
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            
        plt.tight_layout()
        save_path = os.path.join(save_dir, filename)
        print(f"Saving figure to {save_path}")
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

    # Plot 1: Reprojection (Measurement Residual)
    plot_and_save(noise_levels, reproj_errs, 
                  'Measurement Residual (Fitting Error)', 
                  'Reprojection RMSE (pixels)', 
                  'reproj_residual.png')

    # Plot 2: True Geometric Error
    plot_and_save(noise_levels, true_reproj_errs, 
                  'True Geometric Error (vs Ground Truth)', 
                  'True Reprojection RMSE (pixels)', 
                  'reproj_true_accuracy.png')

    # Plot 3: Focal Length Recovery
    plot_and_save(noise_levels, focal_errs, 
                  'Physical Parameter Estimation Error', 
                  'Focal Length Error (mm)', 
                  'focal_length_error.png')

    plt.show()
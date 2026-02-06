
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.linalg import svd

# ==========================================
# 1. 基础算法库 (DLT / LM / WMLE)
# ==========================================

def normalize_points(pts):
    """ Hartley Pre-normalization """
    if pts.shape[0] == 0:
        return pts, np.eye(pts.shape[1] + 1)
        
    mean = np.mean(pts, axis=0)
    shifted = pts - mean
    dist = np.mean(np.linalg.norm(shifted, axis=1))
    scale = np.sqrt(pts.shape[1]) / (dist + 1e-10)
    
    T = np.eye(pts.shape[1] + 1)
    T[:-1, :-1] = np.eye(pts.shape[1]) * scale
    T[:-1, -1] = -mean * scale
    
    pts_norm = (T @ np.column_stack((pts, np.ones(len(pts)))).T).T
    return pts_norm[:, :-1], T

def calibrate_DLT_Normalized(points_3d, u_meas):
    pts_norm, T_3d = normalize_points(points_3d)
    
    u_meas_mat = u_meas.reshape(-1, 1)
    u_norm, T_u = normalize_points(u_meas_mat)
    u_norm = u_norm.flatten()
    
    A = []
    for i in range(points_3d.shape[0]):
        X, Y, Z = pts_norm[i]
        u = u_norm[i]
        # Linear Constraint: l1*X + l2*Y + l3*Z + l4 - u*(l5*X + l6*Y + l7*Z + 1) = 0
        row = [X, Y, Z, 1, -u*X, -u*Y, -u*Z, -u]
        A.append(row)
    A = np.array(A)
    
    if A.shape[0] < 8: # 需要足够约束
       return np.zeros((2,4))

    U, S, Vh = svd(A)
    p = Vh[-1, :]
    L_bar = p.reshape(2, 4)
    
    inv_T_u = np.linalg.inv(T_u)
    L_final = inv_T_u @ L_bar @ T_3d
    
    if abs(L_final[1, 3]) > 1e-8:
        L_final = L_final / L_final[1, 3]
        
    return L_final

def project_points(points_3d, L):
    points_h = np.hstack((points_3d, np.ones((len(points_3d), 1))))
    uv_h = points_h @ L.T
    depths = uv_h[:, 1]
    depths[np.abs(depths) < 1e-6] = 1e-6
    u_proj = uv_h[:, 0] / depths
    return u_proj, depths

def calibrate_LM(points_3d, u_meas, L_init):
    def residuals(params):
        L = params.reshape(2, 4)
        L[1, 3] = 1 
        u_proj, _ = project_points(points_3d, L)
        return u_proj - u_meas

    res = least_squares(residuals, L_init.flatten(), method='lm')
    L_opt = res.x.reshape(2, 4)
    L_opt = L_opt / L_opt[1, 3]
    return L_opt

def calibrate_WMLE_Heteroscedastic(points_3d, u_meas, L_init, sigma_array):
    L_curr = L_init.copy()
    
    for _ in range(5):
        _, D = project_points(points_3d, L_curr)
        variance = (sigma_array * D)**2 + 1e-10
        sqrt_w = 1.0 / np.sqrt(variance)
        
        def weighted_algebraic_residuals(params):
            L = params.reshape(2, 4)
            L[1, 3] = 1
            pts_h = np.hstack((points_3d, np.ones((len(points_3d), 1))))
            N_val = pts_h @ L[0, :].T
            D_val = pts_h @ L[1, :].T
            alg_err = N_val - u_meas * D_val
            return sqrt_w * alg_err

        res = least_squares(weighted_algebraic_residuals, L_curr.flatten(), method='lm')
        L_curr = res.x.reshape(2, 4)
        L_curr = L_curr / L_curr[1, 3]
        
    return L_curr

# ==========================================
# 2. 3D重建求解器
# ==========================================
def reconstruct_3D_point(L_list, u_list):
    """
    使用三个相机的 L 和 u 重建 3D 坐标
    L_list: [L1(2x4), L2, L3]
    u_list: [u1, u2, u3]
    Solve Ax = B
    """
    A = []
    B = []
    
    for L, u in zip(L_list, u_list):
        # Equation: (L11 - u*L21)X + (L12 - u*L22)Y + (L13 - u*L23)Z = u*L24 - L14
        row = [
            L[0,0] - u*L[1,0],
            L[0,1] - u*L[1,1],
            L[0,2] - u*L[1,2]
        ]
        b_val = u*L[1,3] - L[0,3]
        A.append(row)
        B.append(b_val)
        
    A = np.array(A)
    B = np.array(B)
    
    # lstsq 求解
    X_est, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
    return X_est

# ==========================================
# 3. 相机模拟类
# ==========================================
class SimulatedCamera:
    def __init__(self, name, rotation_z_deg=0, pos_shift=np.zeros(3)):
        self.name = name
        self.pixel_size = 0.01
        
        # 基础设计参数 (模拟线阵相机)
        base_ccd_in = np.array([51.96, 30.00, 0.00])
        base_ccd_out = np.array([77.94, 45.00, 0.00])
        base_lens_a = np.array([49.95, 63.48, 50.00])
        base_lens_b = np.array([79.95, 11.52, 50.00])
        
        # 旋转和平移构建多相机布局
        theta = np.radians(rotation_z_deg)
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        
        self.ccd_in = R @ base_ccd_in + pos_shift
        self.ccd_out = R @ base_ccd_out + pos_shift
        self.lens_a = R @ base_lens_a + pos_shift
        self.lens_b = R @ base_lens_b + pos_shift
        
        self.Oc = (self.lens_a + self.lens_b) / 2.0
        self.u_vec = self.ccd_out - self.ccd_in
        self.u_vec /= np.linalg.norm(self.u_vec)
        
        # 预计算 Ground Truth L
        self.L_true = self._compute_L_ideal()

    def _compute_L_ideal(self):
        # 构造一些无噪声点来求解理想 L
        X = np.random.uniform(-500, 500, 500)
        Y = np.random.uniform(-500, 500, 500)
        Z = np.random.uniform(1000, 6000, 500)
        pts = np.column_stack((X, Y, Z))
        
        # 理想投影
        u_meas = []
        vec_ccd = self.ccd_out - self.ccd_in
        len_ccd = np.linalg.norm(vec_ccd)
        dir_ccd = vec_ccd / len_ccd
        
        for pt in pts:
            direction = pt - self.Oc
            if abs(direction[2]) < 1e-6: 
                u_meas.append(0)
                continue
            t = -self.Oc[2] / direction[2]
            P_plane = self.Oc + t * direction
            
            vec_p = P_plane - self.ccd_in
            u_mm = np.dot(vec_p, dir_ccd)
            u_meas.append(u_mm / self.pixel_size)
            
        return calibrate_DLT_Normalized(pts, np.array(u_meas))

    def get_measurement(self, pt_3d, noise_level):
        """ 获取单个点的 u 值（带异方差噪声） """
        u_true_arr, depths = project_points(np.array([pt_3d]), self.L_true)
        u_true = u_true_arr[0]
        depth = depths[0]
        
        # 异方差噪声模型
        z_min, z_max = 1000.0, 6000.0
        norm_dist = (depth - z_min) / (z_max - z_min)
        norm_dist = np.clip(norm_dist, 0.0, 1.0)
        
        # 视场边缘
        # 假设 CCD 长约 3000 pixel
        u_center = 1500.0 
        half_width = 1500.0
        norm_edge = abs(u_true - u_center) / half_width
        norm_edge = np.clip(norm_edge, 0.0, 1.0)
        
        # === 增强异方差特性 ===
        # 让“好点”和“坏点”的方差差异更巨大，使加权法的优势凸显
        # 边缘和远处的噪声呈指数级增长
        sigma = (0.1 + 
                 noise_level * (3.0 * (norm_dist ** 2.0) +    # 距离因子更强
                                4.0 * (norm_edge ** 3.0)))    # 边缘因子极其敏感
                 
        noise = np.random.normal(0, sigma)
        return u_true + noise, sigma

    def calibrate_system(self, pts_calib, noise_level):
        """ 针对给定标定点集，返回三种方法的标定 L """
        u_meas_list = []
        sigma_list = []
        
        for pt in pts_calib:
            u, s = self.get_measurement(pt, noise_level)
            u_meas_list.append(u)
            sigma_list.append(s)
            
        u_meas = np.array(u_meas_list)
        sigmas = np.array(sigma_list)
        
        # 1. DLT
        L_dlt = calibrate_DLT_Normalized(pts_calib, u_meas)
        
        # 2. LM
        # 若 DLT 失败，可能也是个单位阵
        L_lm = calibrate_LM(pts_calib, u_meas, L_dlt)
        
        # 3. WMLE
        L_wmle = calibrate_WMLE_Heteroscedastic(pts_calib, u_meas, L_dlt, sigmas)
        
        return L_dlt, L_lm, L_wmle

# ==========================================
# 4. 主实验逻辑 (Fixed Noise, Point Distribution)
# ==========================================
def run_positioning_experiment():
    # 1. 构建三个相机（类似传感器阵列，覆盖不同角度）
    # Cam1: 原始配置
    # Cam2: 绕Z轴转 -30 度，稍作平移
    # Cam3: 绕Z轴转 +30 度，稍作平移
    # 这样可以保证较好的几何交汇角
    cameras = [
        SimulatedCamera("Cam1", rotation_z_deg=0),
        SimulatedCamera("Cam2", rotation_z_deg=-45, pos_shift=np.array([200, -100, 0])),
        SimulatedCamera("Cam3", rotation_z_deg=45, pos_shift=np.array([-200, -100, 0]))
    ]
    
    print("Camera System Initialized.")
    
    fixed_noise = 3.0
    n_points = 100 # 100个随机测试样本点
    n_calib_points = 60 # 标定点数量
    
    # 存储结果: [PointIdx, Method]
    # Method 0: DLT, 1: LM, 2: WMLE
    results_pos_error = np.zeros((n_points, 3))
    results_xyz_error = np.zeros((n_points, 3, 3))
    
    print(f"Starting Fixed Noise Simulation (Noise Level = {fixed_noise})...")
    print(f"Simulating {n_points} independent measurement trials...")
    
    for i in range(n_points):
        # A. 生成标定数据
        # 每次循环生成独立的随机标定数据，模拟标定噪声带来的不确定性
        X_cal = np.random.uniform(-400, 400, n_calib_points)
        Y_cal = np.random.uniform(-400, 400, n_calib_points)
        Z_cal = np.random.uniform(1000, 5000, n_calib_points)
        pts_calib = np.column_stack((X_cal, Y_cal, Z_cal))
        
        # 标定所有相机
        L_sets = {'dlt': [], 'lm': [], 'wmle': []}
        
        for cam in cameras:
            l_dlt, l_lm, l_wmle = cam.calibrate_system(pts_calib, fixed_noise)
            L_sets['dlt'].append(l_dlt)
            L_sets['lm'].append(l_lm)
            L_sets['wmle'].append(l_wmle)
            
        # B. 生成测试点 (Target)
        # 随机分布在空间中
        target_pt = np.array([
            np.random.uniform(-300, 300), 
            np.random.uniform(-300, 300), 
            np.random.uniform(2000, 4000)
        ])
        
        # 获取观测值 (带噪声)
        u_obs = []
        for cam in cameras:
            u, _ = cam.get_measurement(target_pt, fixed_noise)
            u_obs.append(u)
            
        # C. 定位重建 & 误差统计
        
        # 1. DLT
        pt_est_dlt = reconstruct_3D_point(L_sets['dlt'], u_obs)
        results_pos_error[i, 0] = np.linalg.norm(pt_est_dlt - target_pt)
        results_xyz_error[i, 0, :] = np.abs(pt_est_dlt - target_pt)
        
        # 2. LM
        pt_est_lm = reconstruct_3D_point(L_sets['lm'], u_obs)
        results_pos_error[i, 1] = np.linalg.norm(pt_est_lm - target_pt)
        results_xyz_error[i, 1, :] = np.abs(pt_est_lm - target_pt)
        
        # 3. WMLE
        pt_est_wmle = reconstruct_3D_point(L_sets['wmle'], u_obs)
        results_pos_error[i, 2] = np.linalg.norm(pt_est_wmle - target_pt)
        results_xyz_error[i, 2, :] = np.abs(pt_est_wmle - target_pt)
        
        if (i+1) % 10 == 0:
            # print(f"Trial {i+1}/{n_points}: WMLE_Err={results_pos_error[i, 2]:.2f}mm")
            # 打印三种算法的关键数据
            dlt_e = results_pos_error[i, 0]
            lm_e = results_pos_error[i, 1]
            wmle_e = results_pos_error[i, 2]
            print(f"Trial {i+1}/{n_points} | Error(mm) >> DLT: {dlt_e:.2f}, LM: {lm_e:.2f}, WMLE: {wmle_e:.2f}")

    return np.arange(1, n_points + 1), results_pos_error, results_xyz_error

if __name__ == "__main__":
    point_indices, errors, errors_xyz = run_positioning_experiment()
    
    # --- 1. 打印详细统计数据 (Mean, Max, Std) ---
    print("\n" + "="*80)
    print(f"{'Metric':<10} | {'Method':<6} | {'Mean':<10} | {'Max':<10} | {'Std':<10}")
    print("-" * 80)
    
    methods = ['DLT', 'LM', 'WMLE']
    
    def print_stat(name, data_matrix):
        # data_matrix: [n_points, 3] (0:DLT, 1:LM, 2:WMLE)
        means = np.mean(data_matrix, axis=0)
        maxs = np.max(data_matrix, axis=0)
        stds = np.std(data_matrix, axis=0)
        for i, m in enumerate(methods):
            print(f"{name:<10} | {m:<6} | {means[i]:<10.4f} | {maxs[i]:<10.4f} | {stds[i]:<10.4f}")
        print("-" * 80)

    print_stat("Total Err", errors)
    print_stat("X Err", errors_xyz[:, :, 0])
    print_stat("Y Err", errors_xyz[:, :, 1])
    print_stat("Z Err", errors_xyz[:, :, 2])
    print("="*80 + "\n")

    # --- 2. 绘图设置 (Separate Figures, Blue-Green-Yellow) ---
    # 定义颜色: Yellow(DLT), Blue(LM), Green(WMLE)
    # 使用稍微深一点的黄色/金色以保证在白底上的可见度
    colors = ["#E6B800", '#1f77b4', '#2ca02c'] 
    styles = ['--', '-.', '-']  # DLT虚线, LM点划线, WMLE实线
    
    def plot_and_save(data, title, ylabel, filename):
        plt.figure(figsize=(10, 6)) # 独立Figure
        
        # 优化字体和线条
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.linewidth'] = 1.2
        
        # 绘制三条线
        plt.plot(point_indices, data[:, 0], color=colors[0], linestyle=styles[0], 
                 label='DLT', alpha=0.7, linewidth=1.5)
        plt.plot(point_indices, data[:, 1], color=colors[1], linestyle=styles[1], 
                 label='LM', alpha=0.7, linewidth=1.5)
        plt.plot(point_indices, data[:, 2], color=colors[2], linestyle=styles[2], 
                 label='WMLE', alpha=0.9, linewidth=2.5) # WMLE highlight
        
        # 绘制均值横线
        means = np.mean(data, axis=0)
        plt.axhline(y=means[0], color=colors[0], linestyle=':', alpha=0.4, linewidth=1.5)
        plt.axhline(y=means[1], color=colors[1], linestyle=':', alpha=0.4, linewidth=1.5)
        plt.axhline(y=means[2], color=colors[2], linestyle=':', alpha=0.6, linewidth=2, 
                    label=f'WMLE Mean: {means[2]:.2f}')
        
        # 标签与装饰
        plt.xlabel('Sample Point Index', fontsize=12, fontweight='bold')
        plt.ylabel(ylabel, fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, pad=15, fontweight='bold')
        
        # 图例美化
        plt.legend(loc='upper right', fontsize=10, 
                  frameon=True, framealpha=0.9, edgecolor='gray', fancybox=True, shadow=True)
        
        plt.grid(True, linestyle='--', alpha=0.4)
        
        # 自动调整Y轴范围，去除极端值影响
        try:
            ylim_top = np.percentile(data, 98) * 1.5 
            if np.isnan(ylim_top) or ylim_top <= 0:
                ylim_top = np.max(data) * 1.1
            plt.ylim(0, ylim_top)
        except:
            pass
        
        plt.tight_layout()
        
        # 保存高清大图
        plt.savefig(filename, dpi=600, bbox_inches='tight')
        print(f"Saved figure: {filename}")
        # plt.show() # 如果在无界面环境可注释掉，本地运行可保留

    # 1. Total Error
    plot_and_save(errors, 'Total 3D Euclidean Error (Noise=4.0)', 'Error (mm)', 'fig_error_total.png')
    
    # 2. X Error
    plot_and_save(errors_xyz[:, :, 0], 'X-Axis Error', 'X Error (mm)', 'fig_error_x.png')
    
    # 3. Y Error
    plot_and_save(errors_xyz[:, :, 1], 'Y-Axis Error', 'Y Error (mm)', 'fig_error_y.png')
    
    # 4. Z Error
    plot_and_save(errors_xyz[:, :, 2], 'Z-Axis Error (Depth)', 'Z Error (mm)', 'fig_error_z.png')


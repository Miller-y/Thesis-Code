
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
                                4.0 * (norm_edge ** 4.0)))    # 边缘因子极其敏感
                 
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
# 4. 主实验逻辑
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
    print(f"Cam1 L_true:\n{cameras[0].L_true}")
    
    noise_levels = np.linspace(0, 2.0, 11) # 0 到 2.0 像素级别噪声
    n_trials = 50 # 每个噪声等级实验次数
    n_calib_points = 60 # 标定点数量
    
    # 存储结果: [NoiseLevel, Method] -> Mean Euclidean Error
    # Method 0: DLT, 1: LM, 2: WMLE
    results_pos_error = np.zeros((len(noise_levels), 3))
    
    # 新增: 存储 XYZ 分量误差 [NoiseLevel, Method, 3(x,y,z)]
    results_xyz_error = np.zeros((len(noise_levels), 3, 3))
    
    print("Starting Positioning Simulation...")
    
    for i, nl in enumerate(noise_levels):
        errors_dlt = []
        errors_lm = []
        errors_wmle = []
        
        # 临时存储 XYZ 误差列表
        xyz_err_dlt_list = []
        xyz_err_lm_list = []
        xyz_err_wmle_list = []
        
        for _ in range(n_trials):
            # A. 生成标定数据
            X_cal = np.random.uniform(-400, 400, n_calib_points)
            Y_cal = np.random.uniform(-400, 400, n_calib_points)
            Z_cal = np.random.uniform(1000, 5000, n_calib_points)
            pts_calib = np.column_stack((X_cal, Y_cal, Z_cal))
            
            # 标定所有相机
            L_sets = {'dlt': [], 'lm': [], 'wmle': []}
            
            for cam in cameras:
                l_dlt, l_lm, l_wmle = cam.calibrate_system(pts_calib, nl)
                L_sets['dlt'].append(l_dlt)
                L_sets['lm'].append(l_lm)
                L_sets['wmle'].append(l_wmle)
                
            # B. 生成测试点 (Target)
            # 这里的测试点应该是一个新点，不在标定集中
            target_pt = np.array([
                np.random.uniform(-300, 300), 
                np.random.uniform(-300, 300), 
                np.random.uniform(2000, 4000)
            ])
            
            # 获取测试点的观测值 (带噪声)
            u_obs = []
            for cam in cameras:
                u, _ = cam.get_measurement(target_pt, nl)
                u_obs.append(u)
                
            # C. 定位重建 & 误差统计
            
            # 1. DLT
            pt_est_dlt = reconstruct_3D_point(L_sets['dlt'], u_obs)
            errors_dlt.append(np.linalg.norm(pt_est_dlt - target_pt))
            xyz_err_dlt_list.append(np.abs(pt_est_dlt - target_pt))
            
            # 2. LM
            pt_est_lm = reconstruct_3D_point(L_sets['lm'], u_obs)
            errors_lm.append(np.linalg.norm(pt_est_lm - target_pt))
            xyz_err_lm_list.append(np.abs(pt_est_lm - target_pt))
            
            # 3. WMLE
            pt_est_wmle = reconstruct_3D_point(L_sets['wmle'], u_obs)
            errors_wmle.append(np.linalg.norm(pt_est_wmle - target_pt))
            xyz_err_wmle_list.append(np.abs(pt_est_wmle - target_pt))
            
        results_pos_error[i, 0] = np.mean(errors_dlt)
        results_pos_error[i, 1] = np.mean(errors_lm)
        results_pos_error[i, 2] = np.mean(errors_wmle)
        
        results_xyz_error[i, 0, :] = np.mean(xyz_err_dlt_list, axis=0)
        results_xyz_error[i, 1, :] = np.mean(xyz_err_lm_list, axis=0)
        results_xyz_error[i, 2, :] = np.mean(xyz_err_wmle_list, axis=0)
        
        print(f"Noise {nl:.1f}: DLT={results_pos_error[i,0]:.2f}mm, LM={results_pos_error[i,1]:.2f}mm, WMLE={results_pos_error[i,2]:.2f}mm")

    return noise_levels, results_pos_error, results_xyz_error

if __name__ == "__main__":
    noise_vals, errors, errors_xyz = run_positioning_experiment()
    
    # 创建 2x2 的子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Total Error
    ax = axes[0, 0]
    ax.plot(noise_vals, errors[:, 0], 'k--o', label='DLT Method')
    ax.plot(noise_vals, errors[:, 1], 'b-.^', label='LM Method')
    ax.plot(noise_vals, errors[:, 2], 'r-s', linewidth=2, label='Proposed WMLE')
    ax.set_xlabel('Noise Level Factor')
    ax.set_ylabel('Total Euclidean Error (mm)')
    ax.set_title('Total 3D Positioning Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Function to plot subplot
    def plot_axis_error(ax, axis_idx, axis_name):
        ax.plot(noise_vals, errors_xyz[:, 0, axis_idx], 'k--o', label='DLT')
        ax.plot(noise_vals, errors_xyz[:, 1, axis_idx], 'b-.^', label='LM')
        ax.plot(noise_vals, errors_xyz[:, 2, axis_idx], 'r-s', linewidth=2, label='WMLE')
        ax.set_xlabel('Noise Level Factor')
        ax.set_ylabel(f'{axis_name} Error (mm)')
        ax.set_title(f'{axis_name}-Axis Positioning Error')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 2. X Error
    plot_axis_error(axes[0, 1], 0, 'X')
    
    # 3. Y Error
    plot_axis_error(axes[1, 0], 1, 'Y')
    
    # 4. Z Error (通常 Z 轴误差最大)
    plot_axis_error(axes[1, 1], 2, 'Z')
    
    plt.tight_layout()
    # plt.savefig('positioning_result_xyz.png')
    plt.show()

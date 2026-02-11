
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.linalg import svd
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. 基础算法库
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
        row = [X, Y, Z, 1, -u*X, -u*Y, -u*Z, -u]
        A.append(row)
    A = np.array(A)
    
    if A.shape[0] < 8: 
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

def reconstruct_3D_point(L_list, u_list):
    """
    Simulate the internal algorithm of the YOPS sensor.
    L_list: [L1, L2, L3]
    u_list: [u1, u2, u3]
    """
    A = []
    B = []
    
    for L, u in zip(L_list, u_list):
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
    
    X_est, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
    return X_est

# ==========================================
# 2. 模拟组件
# ==========================================
class SimulatedCamera:
    def __init__(self, name, rotation_z_deg=0, pos_shift=np.zeros(3)):
        self.name = name
        self.pixel_size = 0.008 # 8um
        
        # 基础位置 (极坐标 R=50mm)
        # 先定义相机中心在 XY 平面上的位置
        theta_rad = np.radians(rotation_z_deg)
        R = 50.0
        # Cam Position (Lens Center)
        cx = R * np.cos(theta_rad)
        cy = R * np.sin(theta_rad)
        cz = 0.0
        self.Oc = np.array([cx, cy, cz]) + pos_shift
        
        # 定义对准点 (Look At Point) - 解决中心奇异性问题
        # 假设传感器聚焦于 Z=1500mm 处
        look_at_pt = np.array([0.0, 0.0, 1500.0]) + pos_shift
        
        # 构建相机坐标系
        # Z_cam (Forward) 指向对准点
        Z_cam = look_at_pt - self.Oc
        Z_cam /= np.linalg.norm(Z_cam)
        
        # 定义 Up 向量 / Sensor 向量
        # 关键修正: 为了感知深度(Z)，传感器必须能检测到视差(Parallax)。
        # 对于位于 X 轴(Cam0)的相机，视差由 X 方向的偏移产生。
        # 如果 CCD 沿 Y 轴(切向)，则对 X 方向的偏移不敏感。
        # 因此，CCD 必须沿 径向 (Radial) 放置，或者至少包含径向分量。
        # 这里我们将 CCD 方向设置为 径向 (Radial)。
        # Radial vector: [cos, sin, 0]
        
        radial_vec = np.array([np.cos(theta_rad), np.sin(theta_rad), 0.0])
        
        # 确保 X_cam (CCD direction) 垂直于 Z_cam 并位于径向平面内
        # Y_cam = Z_cam x radial_vec (Should be Tangential-ish)
        Y_cam_tmp = np.cross(Z_cam, radial_vec)
        
        # 如果 Z_cam 和 radial_vec 平行（例如看这里），cross为0
        # Z_cam ~ [0,0,1], radial ~ [1,0,0]. Cross ~ [0,1,0]. Fine.
        
        Y_cam_tmp /= np.linalg.norm(Y_cam_tmp)
        
        # X_cam (Sensor Axis) = Y_cam x Z_cam
        # 这将使 Sensor 尽可能贴近 Radial 方向，同时垂直于光轴
        X_cam = np.cross(Y_cam_tmp, Z_cam)
        
        # Note: Depending on the specific sensor mounting in the thesis, 
        # it could be Radial or Tangential. But for Z-estimation from a compact cluster,
        # Radial is required for triangulation. Tangential gives zero baseline for Z (on axis).
        
        # CCD Geometry
        f = 35.0 # Focal length
        sensor_len = 3648 * 0.008
        
        # CCD Center is BEHIND the lens (-Z_cam * f)
        ccd_center = self.Oc - Z_cam * f
        
        self.ccd_in = ccd_center - X_cam * (sensor_len / 2.0)
        self.ccd_out = ccd_center + X_cam * (sensor_len / 2.0)
        
        # 计算 L_true (Ground Truth Intrinsics)
        self.L_true = self._compute_L_ideal()

    def _compute_L_ideal(self):
        # 构造大范围理想点求解
        X = np.random.uniform(-1000, 1000, 500)
        Y = np.random.uniform(-1000, 1000, 500)
        Z = np.random.uniform(500, 5000, 500)
        pts = np.column_stack((X, Y, Z))
        
        u_meas = []
        vec_ccd = self.ccd_out - self.ccd_in
        dir_ccd = vec_ccd / np.linalg.norm(vec_ccd)
        
        for pt in pts:
            direction = pt - self.Oc
            # 投影到包含 line(ccd_in, ccd_out) 和 Oc 的平面上? 
            # 简化模型：标准针孔投影 + 1D 截取
            # 我们假设它是标准的线性相机模型
            
            # 使用几何法求交点 (Central Projection)
            # Plane defined by Oc and CCD line vector? No.
            # Typical Linear Camera: Projects point onto a plane passing through CCD line and Optical Center.
            # Then measures distance on the line.
            
            # Method 2: Use Generic Projection
            # Let's define a local coordinate system for the camera
            # and project
            
            # 简易做法：3D空间求交
            # 平面方程 P_plane: contains ccd_in, ccd_out, Oc
            #  Wait, a linear camera integrates light over a "plane" of sensitvity (cylindrical lens) 
            #  OR it is a pinhole satisfying a line sensor.
            #  Assuming Pinhole model masked by a line.
            
            # Plane of the linear sensor: 
            # Normal vector = cross( (ccd_out-ccd_in), (Oc - ccd_in) )
            
            # Point P projected onto line?
            # Ideally: intersection of ray (Oc->P) with line (ccd_in->ccd_out).
            # But in 3D, lines generally don't intersect.
            # Linear cameras usually have a cylindrical lens that focuses a plane onto a line.
            # The coordinate 'u' corresponds to the angle around the axis of the cylindrical lens.
            # 
            # Existing code `single_point_positioning_point.py` uses:
            # t = -self.Oc[2] / direction[2]  --> Intersect with Z=0 plane (if camera is vertical)
            # This assumes measurement plane is Z=const.
            
            # Let's stick to the method that works for general orientation:
            # Project vector (P - Oc) onto the sensitive direction.
            # But strict 'u' on a line is defined by intersection of plane (Oc, P, Axis_perp_to_CCD) with CCD line?
            
            # Let's use the explicit geometry from existing `single_point_positioning.py`:
            # "Project P_plane onto line passing through ccd_in with dir_ccd"
            # It assumes the sensor measures the orthogonal projection of the intersection point on the plane.
            
            # Let's implement calculate intersection with the Plane containing CCD line and orthogonal to Optical Axis?
            # Or simply:
            # 1. Define Image Plane (Vector u, Vector v).
            # 2. Project P to Image Plane (u,v).
            # 3. Discard 'v', keep 'u'. This is the standard linear camera model.
            
            # Z_cam is defined as Oc - CCD_center (+Z direction)
            # The Scene is at +Z.
            # So Z_cam is pointing TOWARDS the scene.
            Z_cam = self.Oc - (self.ccd_in + self.ccd_out)/2.0 
            Z_cam /= np.linalg.norm(Z_cam)
            
            X_cam = self.ccd_out - self.ccd_in # U direction
            X_cam /= np.linalg.norm(X_cam)
            
            Y_cam = np.cross(Z_cam, X_cam)
            
            # Project P to local frame
            P_local = pt - self.Oc
            
            x_local = np.dot(P_local, X_cam)
            # y_local = np.dot(P_local, Y_cam)
            z_local = np.dot(P_local, Z_cam) # Forward axis
            
            if z_local < 1e-3: 
                u_meas.append(0)
                continue
                
            # f = distance from Oc to CCD center
            f = 35.0
            
            # Pinhole model: u = f * x / z
            # Note: This is "Frontal" projection model (virtual image plane).
            # Physical sensor is behind, causing inversion x -> -x.
            # But DLT handles the sign automatically. 
            # We just need a consistent geometric mapping.
            # Let's use the non-inverted (virtual plane) scaling for simplicity in simulation
            # as long as consistency is maintained in 'get_measurement'.
            u_mm = f * (x_local / z_local)
            
            # Convert to pixels
            # Center of CCD (0 position) corresponds to intersection of Optical Axis.
            # u_pix ranges from -L/2 to L/2
            u_pix = u_mm / self.pixel_size
            u_meas.append(u_pix)

        return calibrate_DLT_Normalized(pts, np.array(u_meas))

    def get_measurement(self, pt_3d, noise_sigma=1.0):
        u_true_arr, depths = project_points(np.array([pt_3d]), self.L_true)
        u_true = u_true_arr[0]
        
        # Add noise
        noise = np.random.normal(0, noise_sigma)
        return u_true + noise

# ==========================================
# 3. 实验流程
# ==========================================

def run_simulation():
    print("=== 开始 Record1.md 仿真实验 ===")
    
    # 1. 硬件准备: 搭建 YOPS 传感器 (Y型分布)
    # 120度间隔
    cameras = [
        SimulatedCamera("Cam_0", rotation_z_deg=0),
        SimulatedCamera("Cam_120", rotation_z_deg=120),
        SimulatedCamera("Cam_240", rotation_z_deg=240)
    ]
    
    # 2. 标定阶段 (Calibration) - "Stage 4 requirement"
    print("\n[Stage 1] 执行传感器标定...")
    n_calib = 100
    X_c = np.random.uniform(-600, 600, n_calib)
    Y_c = np.random.uniform(-600, 600, n_calib)
    Z_c = np.random.uniform(500, 3000, n_calib)
    pts_calib = np.column_stack((X_c, Y_c, Z_c))
    
    L_calibrated = []
    calibration_noise = 0.5 # separate noise for calibration
    
    for cam in cameras:
        u_meas = []
        sigmas = []
        for pt in pts_calib:
            u = cam.get_measurement(pt, noise_sigma=calibration_noise)
            u_meas.append(u)
            sigmas.append(calibration_noise) # Uniform weight for initial setup
            
        u_meas = np.array(u_meas)
        sigmas = np.array(sigmas)
        
        # 使用 WMLE 进行标定 (如 Record1 所述 "WMLE标定后...")
        # 先 DLT 初值
        L_dlt = calibrate_DLT_Normalized(pts_calib, u_meas)
        # 再 WMLE
        L_wmle = calibrate_WMLE_Heteroscedastic(pts_calib, u_meas, L_dlt, sigmas)
        L_calibrated.append(L_wmle)
        
    print("标定完成。获得 3 个相机的内参矩阵 L。")

    # 3. 实验布设 (Experiment Setup) - Stage 2
    # 空间规划: 2.5m x 2.5m x 2.5m (XY [-1250, 1250], Z [500, 2500])
    # Z layers: 500, 1500, 2500
    # Grid: 3x3
    
    z_layers = [500, 1500, 2500]
    xy_grid = [-1250, 0, 1250]
    
    test_points = []
    layer_indices = [] # 0, 1, 2 for plot colors
    
    for idx, z in enumerate(z_layers):
        for x in xy_grid:
            for y in xy_grid:
                test_points.append([x, y, z])
                layer_indices.append(idx)
                
    test_points = np.array(test_points)
    print(f"\n[Stage 2] 生成测试点集: 共 {len(test_points)} 个点 (分层采样: {z_layers})")

    # 4. 数据采集 (Data Acquisition) - Stage 3
    # Stop-and-Go 模式仿真
    # 假设传感器噪声 (Sensor Noise) 
    measurement_noise = 1.2 # pixels
    
    reconstructed_points = []
    errors = []
    
    print(f"\n[Stage 3] 开始测量采集 (Noise Sigma = {measurement_noise} pix)...")
    
    for i, pt_true in enumerate(test_points):
        # Sensor 采集 (u1, u2, u3)
        u_readings = []
        for cam in cameras:
            u = cam.get_measurement(pt_true, noise_sigma=measurement_noise)
            u_readings.append(u)
            
        # Laser Tracker 采集 (Ground Truth)
        # 仿真中 pt_true 即为转换后的 GT
        
        # 数据处理: 重建坐标
        pt_recon = reconstruct_3D_point(L_calibrated, u_readings)
        reconstructed_points.append(pt_recon)
        
        # 计算误差
        err_vec = pt_recon - pt_true
        err_norm = np.linalg.norm(err_vec)
        errors.append(err_norm)
        
        # print(f"Point {i+1}: True={pt_true}, Recon={pt_recon.astype(int)}, Err={err_norm:.2f} mm")

    reconstructed_points = np.array(reconstructed_points)
    errors = np.array(errors)

    # 5. 结果分析 (Result Analysis) - Stage 4
    # 计算总体指标
    rmse_all = np.sqrt(np.mean(errors**2))
    
    # 计算分量误差
    diff = reconstructed_points - test_points # (N, 3)
    diff_sq = diff**2
    rmse_x_all = np.sqrt(np.mean(diff_sq[:, 0]))
    rmse_y_all = np.sqrt(np.mean(diff_sq[:, 1]))
    rmse_z_all = np.sqrt(np.mean(diff_sq[:, 2]))
    max_err_all = np.max(errors)

    print("\n[Stage 4] 结果分析 (统计数据表):")
    print("-" * 75)
    print(f"{'Layer':<12} | {'RMSE_all':<10} | {'RMSE_X':<10} | {'RMSE_Y':<10} | {'RMSE_Z':<10} | {'Max_Err':<10}")
    print("-" * 75)

    # 这里的 z_layers 已经在上面定义为 [500, 1500, 2500]
    layer_names = ["Near (500)", "Mid (1500)", "Far (2500)"]
    
    for z_target, name in zip(z_layers, layer_names):
        # 筛选该层的数据索引
        # 由于浮点数比较，使用一个小的阈值
        indices = [i for i, pt in enumerate(test_points) if abs(pt[2] - z_target) < 1.0]
        
        if not indices:
            continue
            
        layer_diff = diff[indices] # (9, 3)
        layer_errors = errors[indices] # (9,)
        
        l_rmse_all = np.sqrt(np.mean(layer_errors**2))
        l_rmse_x = np.sqrt(np.mean(layer_diff[:, 0]**2))
        l_rmse_y = np.sqrt(np.mean(layer_diff[:, 1]**2))
        l_rmse_z = np.sqrt(np.mean(layer_diff[:, 2]**2))
        l_max_err = np.max(layer_errors)
        
        print(f"{name:<12} | {l_rmse_all:<10.4f} | {l_rmse_x:<10.4f} | {l_rmse_y:<10.4f} | {l_rmse_z:<10.4f} | {l_max_err:<10.4f}")

    print("-" * 75)
    print(f"{'Overall':<12} | {rmse_all:<10.4f} | {rmse_x_all:<10.4f} | {rmse_y_all:<10.4f} | {rmse_z_all:<10.4f} | {max_err_all:<10.4f}")
    print("-" * 75)
    
    # --- Plotting ---
    fig = plt.figure(figsize=(14, 6))
    
    # Subplot 1: 3D Scatter View
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    # 绘制边框 2.5m x 2.5m x 2.5m
    # 简单画几个角点示意
    corners = np.array([
        [-1250, -1250, 500],
        [1250, -1250, 500],
        [1250, 1250, 500],
        [-1250, 1250, 500],
        [-1250, -1250, 2500],
        [1250, -1250, 2500],
        [1250, 1250, 2500],
        [-1250, 1250, 2500]
    ])
    # ax1.scatter(corners[:,0], corners[:,1], corners[:,2], c='k', marker='.', s=1, alpha=0.1)

    # 绘制测试点
    colors = ['r', 'g', 'b'] # distinct colors for layers
    for i, pt in enumerate(test_points):
        layer = layer_indices[i]
        c = colors[layer]
        ax1.scatter(pt[0], pt[1], pt[2], c=c, marker='o', s=30, label=f'Z={z_layers[layer]}' if i % 9 == 0 else "")
        # 画误差线 (放大 50 倍显示)
        pt_rec = reconstructed_points[i]
        scale = 50.0
        err_vis = (pt_rec - pt) * scale
        ax1.plot([pt[0], pt[0]+err_vis[0]], [pt[1], pt[1]+err_vis[1]], [pt[2], pt[2]+err_vis[2]], c='k', alpha=0.5)

    ax1.set_title("3D Distribution of Test Points\n(Line indicates Error x50)")
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_zlabel("Z (mm)")
    # ax1.legend()
    # Ensure aspect ratio roughly equal visually
    ax1.set_box_aspect([1,1,1])

    # Subplot 2: Z-axis Error Curve
    ax2 = fig.add_subplot(1, 2, 2)
    
    # Group errors by Z
    z_error_map = {z: [] for z in z_layers}
    for i, pt in enumerate(test_points):
        z = pt[2]
        z_error_map[z].append(errors[i])
        
    means = [np.mean(z_error_map[z]) for z in z_layers]
    stds = [np.std(z_error_map[z]) for z in z_layers]
    
    ax2.errorbar(z_layers, means, yerr=stds, fmt='-o', capsize=5)
    ax2.set_xlabel("Distance Z (mm)")
    ax2.set_ylabel("Position Error (mm)")
    ax2.set_title("Error vs Distance (Mean ± Std)")
    ax2.set_ylim(bottom=0)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("record1_simulation_results.png")
    print("\n图表已保存至: record1_simulation_results.png")
    # plt.show()

if __name__ == "__main__":
    run_simulation()

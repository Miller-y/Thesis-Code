import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ast

# ================= 配置区域 =================
# 输入文件名
INPUT_FILE = 'data/csi_data_104_handled.xlsx'

# 时间段定义
TIME_RANGES = {
    'Position 1':            ('22:21:09', '22:21:19'),
    'Position 2':            ('22:21:52', '22:22:02'),
    'Position 3':            ('22:21:31', '22:21:39'),
}

# 可视化截取的帧数
VIS_FRAMES = 200
# ===========================================

def parse_cleaned_csi(csi_str):
    """
    解析已清洗的CSI数据字符串。
    格式假设: "[imag, real, imag, real, ...]" (虚部在前, 实部在后)
    """
    try:
        data_list = ast.literal_eval(csi_str)
        complex_data = []
        
        if len(data_list) % 2 != 0:
            return None
            
        for i in range(0, len(data_list), 2):
            imag = data_list[i]
            real = data_list[i+1]
            complex_data.append(complex(real, imag))
            
        return np.array(complex_data)
    except Exception as e:
        return None

def main():
    print(f"1. 正在读取文件: {INPUT_FILE} ...")
    try:
        df = pd.read_excel(INPUT_FILE)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {INPUT_FILE}。请确保文件名正确且在当前目录下。")
        return

    df['time'] = df['time'].astype(str)

    # 存储不同位置的矩阵数据
    results = {}

    print(f"2. 正在解析CSI数据并截取前 {VIS_FRAMES} 帧...")
    
    for label, (start_time, end_time) in TIME_RANGES.items():
        # 1. 筛选时间
        mask = (df['time'] >= start_time) & (df['time'] <= end_time)
        subset = df[mask].copy()
        
        if len(subset) == 0:
            print(f"   警告: 时间段 {label} 未找到数据！")
            continue
            
        # 2. 解析
        parsed_series = subset['csidata'].apply(parse_cleaned_csi)
        valid_csi = parsed_series.dropna()
        
        if len(valid_csi) == 0:
            continue
            
        try:
            # 堆叠成矩阵 [N_frames, N_subcarriers]
            csi_matrix = np.vstack(valid_csi.values)
        except ValueError:
            # 简单修复长度不一致
            lens = valid_csi.apply(len)
            mode_len = lens.mode()[0]
            valid_csi = valid_csi[lens == mode_len]
            csi_matrix = np.vstack(valid_csi.values)
            
        # 3. 计算幅度并截取
        amplitude = np.abs(csi_matrix)
        
        # 截取前 N 帧，如果不足则取全部
        n_frames = amplitude.shape[0]
        limit = min(n_frames, VIS_FRAMES)
        amplitude_subset = amplitude[:limit, :]
        
        results[label] = amplitude_subset
        print(f"   -> {label}: 获取矩阵形状 {amplitude_subset.shape}")

    if not results:
        print("未提取到任何数据。")
        return

    # ================= 绘图部分 =================
    print("3. 正在生成3D幅度热图...")
    import os

    # 输出目录
    output_dir = '../../Thesis-figures/chap04/ESP32C6/3d_comparison_104'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    for i, (label, matrix) in enumerate(results.items()):
        # 为每一张图创建一个独立的 Figure
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # 准备坐标轴数据
        frames = range(matrix.shape[0])
        subcarriers = range(matrix.shape[1])
        X, Y = np.meshgrid(subcarriers, frames)
        Z = matrix
        
        # 绘制表面图 (cmap='viridis' 是经典热图配色)
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
        
        ax.set_title(f'{label} Amplitude')
        ax.set_xlabel('Subcarrier Index')
        ax.set_ylabel('Time (Frame)')
        ax.set_zlabel('Amplitude')
        
        # 设置视角，让展示更清楚
        ax.view_init(elev=30, azim=-60)
        
        # 添加颜色条 (可选)
        # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

        plt.tight_layout()
        
        # 构造文件名 (去除空格等特殊字符)
        safe_label = label.replace(" ", "_")
        output_img = os.path.join(output_dir, f'3d_comparison_104_{safe_label}.jpg')
        
        plt.savefig(output_img, dpi=600)
        print(f"图片已保存至: {output_img}")
        
        # 关闭当前图形，避免内存累积
        plt.close(fig)

if __name__ == '__main__':
    main()

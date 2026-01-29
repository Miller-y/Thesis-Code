import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

# ================= 配置区域 =================
# 输入文件名
INPUT_FILE = 'csi_data_cleaned_fixed.xlsx'

# 目标子载波索引 (从0开始计数)
# 第33个子载波 -> 索引 32
TARGET_SUBCARRIER_INDEX = 32 

# 时间段定义
TIME_RANGES = {
    'No Target':             ('14:40:45', '14:40:55'),
    'Position 1 (Left-Top)': ('14:41:06', '14:41:16'),
    'Position 2 (Right-Top)':('14:41:36', '14:41:46'),
    'Position 3 (Bot-Right)':('14:41:52', '14:42:02'),
    'Position 4 (Center)':   ('14:42:07', '14:42:17')
}
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

    # 存储每个状态下该子载波的数据列表
    results = {}

    print(f"2. 正在提取第 {TARGET_SUBCARRIER_INDEX + 1} 个子载波 (Index {TARGET_SUBCARRIER_INDEX}) 的数据...")
    
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
            
        # 3. 检查索引是否有效
        n_subcarriers = csi_matrix.shape[1]
        if TARGET_SUBCARRIER_INDEX >= n_subcarriers:
            print(f"   错误: 请求索引 {TARGET_SUBCARRIER_INDEX}，但数据只有 {n_subcarriers} 个子载波。")
            return

        # 4. 提取特定子载波的所有帧数据 (复数数组)
        subcarrier_data = csi_matrix[:, TARGET_SUBCARRIER_INDEX]
        
        results[label] = {
            'complex': subcarrier_data,
            'amp': np.abs(subcarrier_data),
            'phase': np.angle(subcarrier_data) # 原始相位，不做unwrap以便看分布
        }
        print(f"   -> {label}: {len(subcarrier_data)} 帧")

    if not results:
        print("未提取到任何数据。")
        return

    # ================= 绘图部分 =================
    print("3. 正在生成单个子载波分析图...")
    plt.figure(figsize=(18, 5))
    
    # 颜色映射
    colors = plt.cm.get_cmap('tab10', len(results)) 
    
    # --- 1. 复平面上的星座图 (IQ Plot) ---
    plt.subplot(1, 3, 1)
    for i, (label, data) in enumerate(results.items()):
        c_data = data['complex']
        # 绘制散点
        plt.scatter(c_data.real, c_data.imag, label=label, s=10, alpha=0.6, edgecolors='none')
        # 绘制中心点
        center = np.mean(c_data)
        plt.scatter(center.real, center.imag, c='black', marker='x', s=50) # 中心标记
        
    plt.title(f'IQ Constellation (Subcarrier {TARGET_SUBCARRIER_INDEX})')
    plt.xlabel('In-Phase (Real)')
    plt.ylabel('Quadrature (Imag)')
    plt.legend(fontsize='small')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axis('equal') # 保持比例

    # --- 2. 幅度分布 (箱线图) ---
    plt.subplot(1, 3, 2)
    labels = list(results.keys())
    amp_data = [results[l]['amp'] for l in labels]
    
    plt.boxplot(amp_data, labels=labels, patch_artist=True, 
                boxprops=dict(facecolor='lightblue', alpha=0.6))
    
    plt.title(f'Amplitude Distribution (Subcarrier {TARGET_SUBCARRIER_INDEX})')
    plt.ylabel('Amplitude')
    plt.xticks(rotation=45, ha='right') # 标签倾斜
    plt.grid(True, linestyle='--', alpha=0.5)

    # --- 3. 时域稳定性 (前200帧片段) ---
    plt.subplot(1, 3, 3)
    for label, data in results.items():
        amp = data['amp']
        # 只画前200帧，避免太拥挤
        limit = min(len(amp), 200)
        plt.plot(range(limit), amp[:limit], label=label, linewidth=1, alpha=0.8)
        
    plt.title(f'Amplitude Trace (First {200} frames)')
    plt.xlabel('Frame Index')
    plt.ylabel('Amplitude')
    plt.legend(fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

# ================= 配置区域 =================
# 输入文件名 (请确保这是您之前清洗后保存的Excel文件)
INPUT_FILE = 'data/Antanne_3_original.xlsx'

# # 时间段定义(ESP32C6)
# TIME_RANGES = {
#     'No Target':             ('22:20:45', '22:20:55'), # 无目标
#     'Position 1 ': ('22:21:09', '22:21:19'), # Close to Antenna 4
#     'Position 2 ':('22:21:31', '22:21:39'), # Close to Antenna 5
#     'Position 3 ':('22:21:52', '22:22:02'), # Close to Antenna 6
#     'Position 4 (Center)':   ('22:22:15', '22:22:25')  # 正中间
# }
# 时间段定义(ESP32)
TIME_RANGES = {
    'No Target':             ('11:20:40', '11:20:50'), # 无目标
    'Position 1 ': ('11:11:50', '11:12:00'), # Close to Antenna 4
    'Position 2 ':('11:12:20', '11:12:30'), # Close to Antenna 5
    'Position 3 ':('11:15:37', '11:15:47'), # Close to Antenna 6
    'Position 4 (Center)':   ('11:16:05', '11:16:15')  # 正中间
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
        
        # 确保数据成对
        if len(data_list) % 2 != 0:
            return None
            
        for i in range(0, len(data_list), 2):
            imag = data_list[i]   # 第一个是虚部
            real = data_list[i+1] # 第二个是实部
            
            # 构建复数
            c = complex(real, imag)
            complex_data.append(c)
            
        return np.array(complex_data)
    except Exception as e:
        return None

def main():
    print(f"1. 正在读取文件: {INPUT_FILE} ...")
    try:
        # 直接读取 Excel
        df = pd.read_excel(INPUT_FILE)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {INPUT_FILE}。请确保文件名正确且在当前目录下。")
        return

    # 确保时间列是字符串格式以便比较
    df['time'] = df['time'].astype(str)

    # 结果存储容器
    results = {}

    print("2. 正在根据时间段筛选并解析数据...")
    
    # 遍历定义好的时间段进行处理
    for label, (start_time, end_time) in TIME_RANGES.items():
        # 1. 筛选时间
        mask = (df['time'] >= start_time) & (df['time'] <= end_time)
        subset = df[mask].copy()
        
        if len(subset) == 0:
            print(f"   警告: 时间段 {label} ({start_time}-{end_time}) 未找到数据！")
            continue
            
        # 2. 解析 CSI 数据列
        # 将每一行的 string 转换为 numpy array (复数)
        # 使用 apply 方法批量处理
        parsed_series = subset['csidata'].apply(parse_cleaned_csi)
        
        # 去除解析失败的行 (None)
        valid_csi = parsed_series.dropna()
        
        if len(valid_csi) == 0:
            print(f"   警告: 时间段 {label} 数据解析失败或为空。")
            continue
            
        # 3. 堆叠成矩阵 (行=帧数, 列=子载波索引)
        # 这一步会自动对齐，如果清洗后的子载波数量不一致可能会报错，
        # 这里假设清洗后每帧保留的有效子载波数量是一致的。
        try:
            csi_matrix = np.vstack(valid_csi.values)
        except ValueError:
            print(f"   错误: 时间段 {label} 内的数据子载波长度不一致，无法合并分析。")
            # 简单尝试修复：取众数长度
            lens = valid_csi.apply(len)
            mode_len = lens.mode()[0]
            valid_csi = valid_csi[lens == mode_len]
            csi_matrix = np.vstack(valid_csi.values)
            print(f"   已自动修正: 仅保留长度为 {mode_len} 的帧。")

        # 4. 计算特征
        # 幅度
        amplitude = np.abs(csi_matrix)
        # 相位 (解卷绕)
        phase = np.unwrap(np.angle(csi_matrix), axis=1)
        
        # 计算平均值 (沿时间轴/行方向平均)
        # 修正: 先取模再平均，避免相位随机旋转导致的相消干涉
        avg_amp = np.mean(amplitude, axis=0)
        avg_phase = np.mean(phase, axis=0)
        
        results[label] = {
            'amp': avg_amp,
            'phase': avg_phase,
            'count': len(valid_csi)
        }
        print(f"   -> {label}: 获取到 {len(valid_csi)} 帧有效数据")

    # ================= 绘图部分 =================
    if len(results) < 2:
        print("数据不足，无法生成对比图。")
        return

    print("3. 正在生成对比图...")
    import os
    
    # 输出目录
    output_dir = '../../Thesis-figures/chap04/ESP32/antanne_3_diffrent_position'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # --- 1. 原始幅度图 ---
    fig1 = plt.figure(figsize=(8, 6))
    for label, data in results.items():
        x_axis = range(len(data['amp']))
        plt.plot(x_axis, data['amp'], label=label, marker='.', markersize=4, alpha=0.8)
    
    plt.title('Raw Amplitude (Avg of Abs)')
    plt.xlabel('Subcarrier Index')
    plt.ylabel('Amplitude')
    plt.legend(fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    out_1 = os.path.join(output_dir, 'Raw_Amplitude_Avg.jpg')
    plt.savefig(out_1, dpi=600)
    print(f"图片已保存: {out_1}")
    plt.close(fig1)

    # --- 2. 差分幅度图 (幅度减除) ---
    fig2 = plt.figure(figsize=(8, 6))
    
    # 获取背景基准
    bg_key = 'No Target'
    if bg_key in results:
        bg_amp = results[bg_key]['amp']
        print(f"   [背景减除] 使用 '{bg_key}' 作为幅度基准。")
        
        for label, data in results.items():
            if label == bg_key: continue # 跳过背景自身
            
            # 计算幅度差: Target Amp - Background Amp
            # 正值表示信号增强，负值表示信号减弱
            diff_amp = data['amp'] - bg_amp
            
            x_axis = range(len(diff_amp))
            plt.plot(x_axis, diff_amp, label=f"{label} (Diff)", marker='.', markersize=4, alpha=0.8)
            
        plt.title('Amplitude Difference (Target - NoTarget)')
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    else:
        plt.text(0.5, 0.5, "No 'No Target' data for subtraction", ha='center')
        plt.title('Amplitude Difference (Skipped)')
    
    plt.xlabel('Subcarrier Index')
    plt.ylabel('Amp Difference')
    plt.legend(fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    out_2 = os.path.join(output_dir, 'Amplitude_Difference_Target-NoTarget.jpg')
    plt.savefig(out_2, dpi=600)
    print(f"图片已保存: {out_2}")
    plt.close(fig2)


    # --- 3. 相位图 ---
    fig3 = plt.figure(figsize=(8, 6))
    for label, data in results.items():
        x_axis = range(len(data['phase']))
        plt.plot(x_axis, data['phase'], label=label, marker='.', markersize=4, alpha=0.8)
    
    plt.title('CSI Phase Comparison (Unwrapped)')
    plt.xlabel('Subcarrier Index (Cleaned)')
    plt.ylabel('Phase (Radians)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    out_3 = os.path.join(output_dir, 'CSI_Phase_Comparison_Unwrapped.jpg')
    plt.savefig(out_3, dpi=600)
    print(f"图片已保存: {out_3}")
    plt.close(fig3)
    
    print(f"所有分析图片已保存至目录: {output_dir}")

if __name__ == '__main__':
    main()
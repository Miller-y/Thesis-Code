import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import ast
import os

# ================= 配置区域 =================
# 输入文件名
INPUT_FILE = 'data/csi_data_106_handled.xlsx'

# CSI采样率 (Hz)
FS = 100 

# 时间段定义 (只选一段进行分析)
# 格式: (开始时间, 结束时间)
TARGET_TIME_RANGE = ('22:21:09', '22:21:19')

# 要分析的子载波索引
# 如果不知道选哪个，通常中间的子载波效果较好
SUBCARRIER_INDEX = 32  

# 滤波器配置列表用于对比: (cutoff_freq, order, label)
# cutoff_freq: 截止频率 (Hz), 必须小于 FS/2
# order: 滤波器阶数
FILTER_CONFIGS = [
    (50, 4, 'Butterworth (Fc=50Hz, N=4)'),
    (40, 4, 'Butterworth (Fc=40Hz, N=4)'),
    (30, 4, 'Butterworth (Fc=30Hz, N=4)'),
    (20, 4, 'Butterworth (Fc=20Hz, N=4)'), 
]

# 可视化截取的帧数 (None表示该时间段内全部帧)
# VIS_FRAMES = 500
VIS_FRAMES = None 

# 图片保存路径
OUTPUT_IMG_PATH = '../../Thesis-figures/chap04/ESP32C6/filter_comparison_106.jpg'
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

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """
    应用巴特沃斯低通滤波器 (使用filtfilt实现零相位滤波)
    """
    nyq = 0.5 * fs  # 奈奎斯特频率
    if cutoff >= nyq:
        print(f"警告: 截止频率 {cutoff}Hz >= 奈奎斯特频率 {nyq}Hz，已忽略该滤波器。")
        return data
        
    normal_cutoff = cutoff / nyq
    # b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    # 使用sos (Second-Order Sections) 具有更好的数值稳定性
    sos = signal.butter(order, normal_cutoff, btype='low', output='sos')
    y = signal.sosfiltfilt(sos, data)
    return y

def main():
    print(f"1. 正在读取文件: {INPUT_FILE} ...")
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到文件 {INPUT_FILE}。")
        return
        
    try:
        df = pd.read_excel(INPUT_FILE)
    except Exception as e:
        print(f"读取Excel失败: {e}")
        return

    df['time'] = df['time'].astype(str)

    start_time, end_time = TARGET_TIME_RANGE
    print(f"2. 正在提取时间段 {start_time} - {end_time} 的数据...")

    # 筛选时间
    mask = (df['time'] >= start_time) & (df['time'] <= end_time)
    subset = df[mask].copy()
    
    if len(subset) == 0:
        print(f"警告: 时间段 {start_time}-{end_time} 未找到数据！")
        return
        
    # 解析CSI
    print("   正在解析CSI数据...")
    parsed_series = subset['csidata'].apply(parse_cleaned_csi)
    valid_csi = parsed_series.dropna()
    
    if len(valid_csi) == 0:
        print("未提取到有效的CSI数据行。")
        return
        
    try:
        # 堆叠成矩阵 [N_frames, N_subcarriers]
        csi_matrix = np.vstack(valid_csi.values)
    except ValueError:
        # 简单修复长度不一致
        lens = valid_csi.apply(len)
        mode_len = lens.mode()[0]
        valid_csi = valid_csi[lens == mode_len]
        csi_matrix = np.vstack(valid_csi.values)
        
    # 计算幅度
    amplitude_matrix = np.abs(csi_matrix)
    n_total_frames, n_subcarriers = amplitude_matrix.shape
    print(f"   -> 获取数据形状: {n_total_frames} 帧, {n_subcarriers} 子载波")
    
    # 截取帧数
    if VIS_FRAMES and VIS_FRAMES < n_total_frames:
        amplitude_matrix = amplitude_matrix[:VIS_FRAMES, :]
        print(f"   -> 已截取前 {VIS_FRAMES} 帧")

    # 提取单子载波数据
    target_idx = SUBCARRIER_INDEX
    if target_idx >= n_subcarriers:
        target_idx = n_subcarriers // 2
        print(f"提示: 指定索引 {SUBCARRIER_INDEX} 超出范围，自动选择中间子载波: {target_idx}")
        
    raw_data = amplitude_matrix[:, target_idx]
    
    # 生成时间轴
    times = np.arange(len(raw_data)) / FS

    # ================= 绘图部分 =================
    print("3. 正在应用滤波器并对比绘图...")
    
    # 初始化数据导出 和 分析结果列表
    export_df = pd.DataFrame()
    export_df['Time_s'] = times
    export_df['Raw_Signal'] = raw_data
    
    metrics_list = []

    plt.figure(figsize=(12, 8))
    
    # 1. 绘制原始数据
    plt.plot(times, raw_data, label='Raw Signal', color='lightgray', linewidth=2.5, alpha=0.7)
    
    # 2. 循环应用不同的滤波器配置
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] # 预定义一些颜色
    
    for i, (cutoff, order, label_text) in enumerate(FILTER_CONFIGS):
        color = colors[i % len(colors)]
        try:
            filtered_data = butter_lowpass_filter(raw_data, cutoff, FS, order=order)
            plt.plot(times, filtered_data, label=label_text, linewidth=1.5, color=color)
            
            # 收集数据用于导出
            export_df[label_text] = filtered_data
            
            # --- 计算量化指标 ---
            # 1. RMSE (均方根误差) - 衡量偏离程度，越小越接近原始信号
            rmse = np.sqrt(np.mean((raw_data - filtered_data)**2))
            
            # 2. Smoothness (平滑度) - 使用一阶差分的标准差来衡量，越小越平滑
            smoothness = np.std(np.diff(filtered_data))
            
            # 3. Correlation (相关系数) - 衡量波形相似度，越接近1越好
            corr = np.corrcoef(raw_data, filtered_data)[0, 1]
            
            metrics_list.append({
                "Filter_Label": label_text,
                "Cutoff_Hz": cutoff,
                "Order": order,
                "RMSE": rmse,
                "Smoothness": smoothness,
                "Correlation": corr
            })
            
        except Exception as e:
            print(f"   滤波器 {label_text} 处理出错: {e}")
    
    # --- 生成分析报告 ---
    metrics_df = pd.DataFrame(metrics_list)
    # 计算原始信号的平滑度作为基准
    raw_smoothness = np.std(np.diff(raw_data))
    
    report_lines = []
    report_lines.append("=== 滤波器性能分析报告 ===")
    report_lines.append(f"原始信号平滑度 (基准 - 越小越平滑): {raw_smoothness:.6f}")
    report_lines.append("-" * 100)
    # 调整列顺序方便查看
    cols = ['Filter_Label', 'Cutoff_Hz', 'Order', 'Smoothness', 'Correlation', 'RMSE']
    report_lines.append(metrics_df[cols].to_string(index=False, float_format="%.6f"))
    report_lines.append("-" * 100)
    
    # 简单的自动结论生成
    # 策略：在保持相关性 > 0.9 (不过度失真) 的前提下，寻找 Smoothness 最小 (最平滑) 的配置
    candidates = metrics_df[metrics_df['Correlation'] > 0.90]
    
    report_lines.append("\n[自动推荐结论]")
    if not candidates.empty:
        best_row = candidates.sort_values(by='Smoothness').iloc[0]
        report_lines.append(f"综合考虑波形保真度(相关性>0.9)和平滑效果：")
        report_lines.append(f"★ 推荐配置: {best_row['Filter_Label']}")
        report_lines.append(f"  理由: 相比原始信号，平滑度优化至 {best_row['Smoothness']:.6f} (原始为 {raw_smoothness:.6f})，且保持了 {best_row['Correlation']:.4f} 的高相关性。")
        
        # 次优推荐 (如果存在更平滑但相关性略低的)
        smoother_candidates = metrics_df[(metrics_df['Smoothness'] < best_row['Smoothness']) & (metrics_df['Correlation'] > 0.8)]
        if not smoother_candidates.empty:
             smoother_row = smoother_candidates.sort_values(by='Smoothness').iloc[0]
             report_lines.append(f"\n★ 如果需要更强的去噪效果 (允许轻微波形滞后/失真):")
             report_lines.append(f"  可选配置: {smoother_row['Filter_Label']}")
             report_lines.append(f"  理由: 平滑度进一步降至 {smoother_row['Smoothness']:.6f}，相关性为 {smoother_row['Correlation']:.4f}。")
    else:
        report_lines.append("未找到相关性 > 0.9 的强平滑配置，建议适当提高截止频率以避免过度失真，或者检查原始信号噪声是否过大。")

    report_text = "\n".join(report_lines)
    print(report_text)

    plt.title(f'CSI Amplitude Filter Analysis (Subcarrier #{target_idx})\nTime: {start_time}-{end_time}, FS={FS}Hz')
    plt.xlabel('Time (seconds)')
    plt.ylabel('CSI Amplitude')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_IMG_PATH), exist_ok=True)
    
    # 保存图片
    plt.savefig(OUTPUT_IMG_PATH, dpi=600)
    print(f"图片已保存至: {OUTPUT_IMG_PATH}")
    
    # 保存分析报告到txt
    report_path = OUTPUT_IMG_PATH.replace('.jpg', '_report.txt').replace('.png', '_report.txt')
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"分析报告已保存至: {report_path}")
    except Exception as e:
        print(f"保存报告失败: {e}")

    # 保存数据到CSV
    output_csv_path = OUTPUT_IMG_PATH.replace('.jpg', '.csv').replace('.png', '.csv')
    try:
        export_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"关键数据已保存至: {output_csv_path}")
    except Exception as e:
        print(f"保存CSV数据失败: {e}")

    plt.show()

if __name__ == '__main__':
    main()

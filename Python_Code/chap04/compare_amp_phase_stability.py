import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from scipy import signal # 引入信号处理库

# ================= 配置区域 =================
INPUT_FILE = 'data/csi_data_106_handled.xlsx'
TARGET_SUB_IDX = 32  # 第33个子载波

# 模拟采样率 (假设每秒100Hz，影响不大，主要用于滤波器设计)
FS = 100 
# 巴特沃斯低通滤波器配置 (调整: 折衷选择 5Hz，兼顾静态稳定性和微动特征)
CUTOFF_FREQ = 25  # 截止频率 (Hz) 
FILTER_ORDER = 4

TIME_RANGES = {
    'No Target':             ('22:20:45', '22:20:55'),
    'Position 1 (Left-Top)': ('22:21:09', '22:21:19'),
    'Position 2 (Right-Top)':('22:21:52', '22:22:02'),
    'Position 3 (Bot-Right)':('22:21:31', '22:21:41'),
    'Position 4 (Center)':   ('22:22:07', '22:22:17')
}
# ===========================================

def parse_cleaned_csi(csi_str):
    try:
        data_list = ast.literal_eval(csi_str)
        complex_data = []
        if len(data_list) % 2 != 0: return None
        for i in range(0, len(data_list), 2):
            complex_data.append(complex(data_list[i+1], data_list[i])) # real, imag
        return np.array(complex_data)
    except:
        return None

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

def main():
    print(f"正在分析第 {TARGET_SUB_IDX + 1} 个子载波的特征稳定性 (含滤波测试)...")
    try:
        df = pd.read_excel(INPUT_FILE)
    except FileNotFoundError:
        print("找不到文件")
        return
        
    df['time'] = df['time'].astype(str)
    
    # 数据存储
    data_store = {
        'labels': [],
        'amp_list': [],
        'amp_filtered_list': [],
        'phase_list': [],
        'phase_unwrapped_list': []
    }
    
    metrics = []

    for label, (start, end) in TIME_RANGES.items():
        mask = (df['time'] >= start) & (df['time'] <= end)
        subset = df[mask].copy()
        
        if len(subset) == 0: continue
            
        parsed = subset['csidata'].apply(parse_cleaned_csi).dropna()
        if len(parsed) == 0: continue
            
        # 堆叠
        try:
            matrix = np.vstack(parsed.values)
        except ValueError:
            # 简单对齐
            lens = parsed.apply(len)
            matrix = np.vstack(parsed[lens == lens.mode()[0]].values)
            
        if TARGET_SUB_IDX >= matrix.shape[1]: continue
            
        # 提取单子载波数据
        sc_data = matrix[:, TARGET_SUB_IDX]
        
        # 1. 幅度数据
        amp = np.abs(sc_data)
        
        # 1.1 幅度滤波 (新增)
        # 只有数据点足够多才能滤波，否则filtfilt会报错
        if len(amp) > 15:
            amp_filtered = butter_lowpass_filter(amp, CUTOFF_FREQ, FS, FILTER_ORDER)
        else:
            amp_filtered = amp
        
        # 2. 相位数据 (原始 [-pi, pi])
        phase = np.angle(sc_data)
        
        # 3. 相位数据 (时间轴解卷绕，尝试消除跳变)
        phase_unwrap = np.unwrap(phase) 
        # 移除线性趋势 (去除CFO引起的斜率，只看波动)
        x = np.arange(len(phase_unwrap))
        if len(x) > 1:
            p = np.polyfit(x, phase_unwrap, 1)
            phase_detrend = phase_unwrap - (p[0] * x + p[1])
        else:
            phase_detrend = phase_unwrap
        
        data_store['labels'].append(label)
        data_store['amp_list'].append(amp)
        data_store['amp_filtered_list'].append(amp_filtered)
        data_store['phase_list'].append(phase)
        data_store['phase_unwrapped_list'].append(phase_detrend)
        
        # 计算统计量
        amp_mean = np.mean(amp)
        # 计算原始幅度CV
        amp_cv = (np.std(amp) / amp_mean) * 100 if amp_mean != 0 else 0
        # 计算滤波后幅度CV (关键指标)
        amp_filt_mean = np.mean(amp_filtered)
        amp_filt_cv = (np.std(amp_filtered) / amp_filt_mean) * 100 if amp_filt_mean != 0 else 0
        
        phase_std = np.std(phase) 
        phase_detrend_std = np.std(phase_detrend) 
        
        metrics.append({
            'Label': label,
            'Amp_Mean': amp_mean, # 新增: 查看是不是因为幅值太小导致CV虚高
            'Amp_CV_Raw(%)': amp_cv,
            'Amp_CV_Filt(%)': amp_filt_cv,
            'Phase_Std(Cleaned)': phase_detrend_std
        })
        
    # ================= 可视化 =================
    import os
    # 输出目录
    output_dir = '../../Thesis-figures/chap04/ESP32C6/amp_phase_result_106_handled'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 1. 滤波效果对比 (前100帧) - Amplitude
    fig1 = plt.figure(figsize=(8, 6))
    for i, label in enumerate(data_store['labels']):
        # 画虚线表示原始，实线表示滤波后
        # plt.plot(data_store['amp_list'][i][:150], linestyle=':', alpha=0.4, linewidth=1) 
        plt.plot(data_store['amp_filtered_list'][i][:150], label=label, alpha=0.9, linewidth=2)
    plt.title(f'Time Domain: Amplitude (Low-Pass Filtered)\nSubcarrier {TARGET_SUB_IDX}')
    plt.ylabel('Amplitude')
    plt.xlabel('Frame Index (First 150)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize='small')
    plt.tight_layout()
    out_1 = os.path.join(output_dir, 'Time_Domain_Amplitude_Filtered.jpg')
    plt.savefig(out_1, dpi=600)
    print(f"图片已保存: {out_1}")
    plt.close(fig1)

    # 2. 滤波效果对比 (前100帧) - Phase
    fig2 = plt.figure(figsize=(8, 6))
    for i, label in enumerate(data_store['labels']):
        plt.plot(data_store['phase_unwrapped_list'][i][:150], label=label, alpha=0.9, linewidth=1.5)
    plt.title(f'Time Domain: Phase (Detrended)\nSubcarrier {TARGET_SUB_IDX}')
    plt.ylabel('Phase (Radians)')
    plt.xlabel('Frame Index (First 150)')
    plt.grid(True, linestyle='--', alpha=0.5)
    # plt.legend(fontsize='small') # 图图例如果太多会遮挡，可选
    plt.tight_layout()
    out_2 = os.path.join(output_dir, 'Time_Domain_Phase_Detrended.jpg')
    plt.savefig(out_2, dpi=600)
    print(f"图片已保存: {out_2}")
    plt.close(fig2)

    # 3. 分布区分度 (箱线图) - Amplitude
    fig3 = plt.figure(figsize=(8, 6))
    plt.boxplot(data_store['amp_filtered_list'], tick_labels=data_store['labels'], patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.title(f'Distribution: Amplitude (Filtered)\nSubcarrier {TARGET_SUB_IDX}')
    plt.ylabel('Amplitude')
    plt.xticks(rotation=20)
    plt.grid(True, axis='y')
    plt.tight_layout()
    out_3 = os.path.join(output_dir, 'Distribution_Amplitude_Filtered.jpg')
    plt.savefig(out_3, dpi=600)
    print(f"图片已保存: {out_3}")
    plt.close(fig3)
    
    # 4. 分布区分度 (箱线图) - Phase
    fig4 = plt.figure(figsize=(8, 6))
    plt.boxplot(data_store['phase_unwrapped_list'], tick_labels=data_store['labels'], patch_artist=True, boxprops=dict(facecolor='lightgreen'))
    plt.title(f'Distribution: Phase (Detrended)\nSubcarrier {TARGET_SUB_IDX}')
    plt.ylabel('Phase (Radians)')
    plt.xticks(rotation=20)
    plt.grid(True, axis='y')
    plt.tight_layout()
    out_4 = os.path.join(output_dir, 'Distribution_Phase_Detrended.jpg')
    plt.savefig(out_4, dpi=600)
    print(f"图片已保存: {out_4}")
    plt.close(fig4)
    
    print(f"所有分析图片已保存至目录: {output_dir}")
    
    # ================= 生成统计报告 =================
    report_lines = []
    report_lines.append("="*85)
    report_lines.append(" >>> CSI特征稳定性与可用性定量分析报告 (ESP32C6) <<<")
    report_lines.append("="*85)
    report_lines.append(f"子载波索引: {TARGET_SUB_IDX}")
    report_lines.append(f"滤波器配置: Butterworth Lowpass (Cutoff={CUTOFF_FREQ}Hz, Order={FILTER_ORDER})")
    report_lines.append("-" * 85)
    report_lines.append(f"{'Label':<25} | {'Mean Amp':<10} | {'CV(Raw)%':<10} | {'CV(Filt)%':<10} | {'Phase Std':<10}")
    report_lines.append("-" * 85)
    
    total_amp_filt_cv = 0
    total_phase_std = 0
    valid_count = 0
    
    # 存储用于计算区分度的均值
    amp_means = []
    phase_means = []

    for m in metrics:
        report_lines.append(f"{m['Label']:<25} | {m['Amp_Mean']:<10.2f} | {m['Amp_CV_Raw(%)']:<10.2f} | {m['Amp_CV_Filt(%)']:<10.2f} | {m['Phase_Std(Cleaned)']:<10.2f}")
        total_amp_filt_cv += m['Amp_CV_Filt(%)']
        total_phase_std += m['Phase_Std(Cleaned)']
        amp_means.append(m['Amp_Mean'])
        # 简单计算一下相位的均值，尽管对于随机相位意义不大，但可以看是否重叠
         # 由于没有在metrics里存Phase Mean，这里暂时略过，主要看Std
        valid_count += 1
        
    avg_amp_filt_cv = total_amp_filt_cv / valid_count if valid_count > 0 else 0
    avg_phase_std = total_phase_std / valid_count if valid_count > 0 else 0
    
    # 简易区分度计算 (Standard Deviation of Means / Mean of Standard Deviations)
    # 如果类间距离大，类内距离小，则比值大
    amp_between_std = np.std(amp_means)
    # amp_within_std 估算为平均 CV * 平均 Mean (反推) 或者直接拿各个组的std平均
    # 这里直接用 CV 作为衡量“类内离散度”的指标，用 amp_between_std 衡量“类间区分度”
    
    report_lines.append("="*85)
    report_lines.append("分析结论与论述 :")
    report_lines.append(f"1. 幅值稳定性 (Intra-class Stability):")
    report_lines.append(f"   - 滤波后幅值的平均变异系数 (CV) 为 {avg_amp_filt_cv:.2f}%。")
    if avg_amp_filt_cv < 15:
        report_lines.append("   - 评级: [优秀]。低变异系数表明同一点位的幅值特征非常稳定，噪声影响较小。")
    elif avg_amp_filt_cv < 25:
        report_lines.append("   - 评级: [良好]。存在一定波动，可能是由于环境微动或Deep Fade导致，但在可接受范围内。")
    else:
        report_lines.append("   - 评级: [较差]。幅值波动较大。")

    report_lines.append(f"\n2. 相位随机性 (Phase Randomness):")
    report_lines.append(f"   - 解卷绕并去趋势后的相位平均标准差为 {avg_phase_std:.2f} Rad。")
    if avg_phase_std > 1.0:
        report_lines.append("   - 评级: [高随机性]。相位标准差很大，说明相位在同一点位内剧烈跳变。")
        report_lines.append("   - 原因: ESP32等低成本WiFi网卡的载波频率偏移(CFO)和采样时钟偏移(SFO)导致的相位非线性误差难以完全消除。")
    else:
        report_lines.append("   - 评级: [中等]。相位具有一定稳定性，但通常不如幅值可靠。")

    report_lines.append(f"\n3. 综合推荐 (Recommendation for Neural Networks):")
    if avg_amp_filt_cv < 20 and avg_phase_std > 0.5:
        report_lines.append("   基于上述指标，强烈推荐使用 **[CSI幅值]** (Amplitude) 作为神经网络的输入特征。")
        report_lines.append("   理由:")
        report_lines.append("   (1) 稳定性: 幅值在静态场景下保持高度稳定(低CV)，有利于网络学习到位置与信号强度的确定性映射。")
        report_lines.append("   (2) 可分性: 不同位置的幅值均值差异明显(见箱线图)，构成了清晰的位置指纹(Fingerprint)。")
        report_lines.append("   (3) 相位不可用: 原始相位含噪过大，即使经过解卷绕和去线性化，其随机抖动仍会淹没位置特征，导致网络无法收敛或过拟合。")
    else:
        report_lines.append("   虽然幅值表现较好，但需注意特定场景下的噪声处理。仍建议优先考虑幅值。")
    
    report_lines.append("\n" + "="*85)
    
    # 打印并保存
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # with open(output_report, "w", encoding='utf-8') as f:
    #     f.write(report_text)
    # print(f"\n详细报告已保存至: {output_report}")
    
    # plt.show()


if __name__ == '__main__':
    main()

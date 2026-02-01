import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import os
import itertools

# ================= 配置区域 =================
# 输入文件名
INPUT_FILE = 'data/csi_data_104_handled.xlsx'

# 目标子载波索引 (从0开始计数)
# 第33个子载波 -> 索引 32
TARGET_SUBCARRIER_INDEX = 20 

# 时间段定义
TIME_RANGES = {
    'No Target':             ('22:20:45', '22:20:55'),
    'Position 1 (Left-Top)': ('22:21:09', '22:21:19'),
    'Position 2 (Right-Top)':('22:21:52', '22:22:02'),
    'Position 3 (Bot-Right)':('22:21:31', '22:21:41'),
    'Position 4 (Center)':   ('22:22:07', '22:22:17')
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

    # ================= 4. 数据统计与分析报告生成 =================
    print("\n4. 正在进行统计分析与结论生成...")
    
    report_lines = []
    report_lines.append("="*60)
    report_lines.append(f"CSI 幅值区分度分析报告 - 子载波 Index {TARGET_SUBCARRIER_INDEX}")
    report_lines.append("="*60)
    
    # 4.1 基础统计量
    stats = {}
    report_lines.append("\n[1. 各位置基础统计量]")
    report_lines.append(f"{'位置 (Label)':<25} | {'均值 (Mean)':<10} | {'标准差 (Std)':<10} | {'变异系数 (CV, %)':<15}")
    report_lines.append("-" * 75)
    
    for label, data in results.items():
        amp = data['amp']
        mu = np.mean(amp)
        sigma = np.std(amp)
        cv = (sigma / mu) * 100 if mu != 0 else 0
        
        # 记录中心点 (复数均值)与统计信息
        stats[label] = {
            'mu': mu, 
            'sigma': sigma, 
            'cv': cv, 
            'center': np.mean(data['complex'])
        }
        
        report_lines.append(f"{label:<25} | {mu:.4f}     | {sigma:.4f}     | {cv:.2f}%")

    # 4.2 区分度分析 (Pairwise Distinguishability)
    report_lines.append("\n[2. 位置间区分度分析 (基于复平面IQ中心距离)]")
    # 计算两两之间的距离以及分离度 (Separation Score)
    labels = list(results.keys())
    combinations = list(itertools.combinations(labels, 2))
    
    report_lines.append(f"{'对比组 (Pair)':<40} | {'中心距离 (Dist)':<15} | {'区分度 (Score)':<20} | {'评价'}")
    report_lines.append("-" * 95)
    
    distinct_count = 0
    total_pairs = len(combinations)
    
    for l1, l2 in combinations:
        c1 = stats[l1]['center']
        c2 = stats[l2]['center']
        # 使用幅值标准差近似表示该簇的“半径”或离散程度
        std1 = stats[l1]['sigma'] 
        std2 = stats[l2]['sigma']
        
        # 复平面欧氏距离
        dist = np.abs(c1 - c2)
        
        # 区分度评分: 距离 / (两者的噪声半径之和)
        # 类似于 Fisher Discriminant Ratio 的概念: 越高越好
        # Score > 1 意味着中心距离已经大于两者的离散半径之和
        separation_score = dist / (std1 + std2) if (std1 + std2) > 0 else 0
        
        eval_str = "区分度极差"
        if separation_score > 3:
            eval_str = "极好 (Excellent)"
            distinct_count += 1
        elif separation_score > 1.5:
            eval_str = "良好 (Good)"
            distinct_count += 1
        elif separation_score > 1.0:
            eval_str = "勉强 (Marginal)"
        
        report_lines.append(f"{l1} vs {l2:<12} | {dist:.4f}          | {separation_score:.4f}               | {eval_str}")

    # 4.3 自动结论
    report_lines.append("\n[3. 自动结论推断]")
    conclusion = []
    
    # (1) 稳定性评估
    avg_cv = np.mean([s['cv'] for s in stats.values()])
    if avg_cv < 5:
        conclusion.append(f"1. 信号稳定性: [极高]。平均变异系数 CV 为 {avg_cv:.2f}% (<5%)。数据在时域上非常稳定，表明环境干扰较小且硬件采集稳定，适合做静态指纹定位。")
    elif avg_cv < 15:
        conclusion.append(f"1. 信号稳定性: [良好]。平均变异系数 CV 为 {avg_cv:.2f}%。数据存在一定波动，但均值特征依然稳定，可用于识别。")
    else:
        conclusion.append(f"1. 信号稳定性: [较差]。平均变异系数 CV 为 {avg_cv:.2f}% (>15%)。数据波动较大，建议检查是否有动态环境干扰或增加低通滤波算法。")
        
    # (2) 区分度评估
    ratio = distinct_count / total_pairs if total_pairs > 0 else 0
    if ratio >= 0.8:
        conclusion.append(f"2. 位置区分能力: [强]。在总共 {total_pairs} 组两两对比中，有 {distinct_count} 组 ({ratio:.0%}) 达到了良好以上的区分度。")
        conclusion.append("   - 论述: 实验数据显示，当铝箔包裹的小球处于不同位置时，CSI在复平面上的聚类中心发生了显著偏移。区分度评分(Score)普遍较高，说明位置变化对信道多径效应有特异性影响。")
        conclusion.append("   - 结论: 该测试有力地证明了 ESP32C6 获取的 CSI 幅值信息具备优秀的空间分辨率，能够有效区分不同位置的铝箔小球，由于其反射特性，其对信号路径的改变是可探测的。")
    elif ratio >= 0.5:
        conclusion.append(f"2. 位置区分能力: [中等]。部分位置可以明显区分 (占比 {ratio:.0%})，但仍有部分位置特征重叠。")
        conclusion.append("   - 建议: 当前子载波可能处于部分位置的衰落点或不敏感区。建议尝试更换子载波，或者综合利用多个子载波的数据进行联合判决。")
    else:
        conclusion.append(f"2. 位置区分能力: [弱]。大部分位置特征重叠严重。")
        conclusion.append("   - 可能原因: 小球反射截面积过小导致多径变化淹没在噪声中，或者测试位置间距过小。建议增大目标物体尺寸或使用相位差信息。")
        
    report_text = "\n".join(report_lines + conclusion)
    
    # 打印到控制台
    print(report_text)
    
    # 保存报告到文件
    report_filename = f'analysis_report_sub{TARGET_SUBCARRIER_INDEX}.txt'
    # 确保保存到 data 目录 (如果要保存在 data 目录下) 或者当前目录
    # 这里直接保存在当前目录方便查看
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n[INFO] 详细分析报告已保存至: {os.path.abspath(report_filename)}")

    # ================= 绘图部分 =================
    print("3. 正在生成单个子载波分析图...")
    
    # 输出目录
    output_dir = '../../Thesis-figures/chap04/ESP32C6/single_result_104_handled'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 颜色映射
    colors = plt.cm.get_cmap('tab10', len(results)) 
    
    # --- 1. 复平面上的星座图 (IQ Plot) ---
    fig1 = plt.figure(figsize=(6, 5))
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
    plt.tight_layout()
    
    out_1 = os.path.join(output_dir, f'IQ_Constellation_sub{TARGET_SUBCARRIER_INDEX}.jpg')
    plt.savefig(out_1, dpi=600)
    print(f"   -> 图片已保存: {out_1}")
    plt.close(fig1)

    # --- 2. 幅度分布 (箱线图) ---
    fig2 = plt.figure(figsize=(8, 5))
    labels = list(results.keys())
    amp_data = [results[l]['amp'] for l in labels]
    
    plt.boxplot(amp_data, labels=labels, patch_artist=True, 
                boxprops=dict(facecolor='lightblue', alpha=0.6))
    
    plt.title(f'Amplitude Distribution (Subcarrier {TARGET_SUBCARRIER_INDEX})')
    plt.ylabel('Amplitude')
    plt.xticks(rotation=45, ha='right') # 标签倾斜
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    out_2 = os.path.join(output_dir, f'Amplitude_Distribution_sub{TARGET_SUBCARRIER_INDEX}.jpg')
    plt.savefig(out_2, dpi=600)
    print(f"   -> 图片已保存: {out_2}")
    plt.close(fig2)

    # --- 3. 时域稳定性 (前200帧片段) ---
    fig3 = plt.figure(figsize=(6, 5))
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
    out_3 = os.path.join(output_dir, f'Amplitude_Trace_sub{TARGET_SUBCARRIER_INDEX}.jpg')
    plt.savefig(out_3, dpi=600)
    print(f"   -> 图片已保存: {out_3}")
    plt.close(fig3)

if __name__ == '__main__':
    main()

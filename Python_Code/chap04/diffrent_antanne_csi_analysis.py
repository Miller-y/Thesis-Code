import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import os

# ================= 配置区域 =================
# 文件配置字典：{ '标签名': '文件路径' }
FILES_CONFIG = {
    'Device 1': 'data/csi_data_104_handled.xlsx',
    'Device 2': 'data/csi_data_105_handled.xlsx',
    'Device 3': 'data/csi_data_106_handled.xlsx'
}

# 统一分析的时间段 (请确保所有文件中都包含此时间段的数据)
# 格式: ('开始时间', '结束时间')
TARGET_TIME_WINDOW = ('22:21:09', '22:21:19') 

# ESP32:目标时间段: 11:11:50 - 11:12:00
# ===========================================

def generate_analysis_report(results, output_dir):
    """
    生成详细的分析报告并保存为 txt 文件
    """
    report_lines = []
    report_lines.append("="*60)
    report_lines.append("CSI 数据多设备位置对比分析报告")
    report_lines.append("="*60)
    
    # 1. 基础数据统计
    report_lines.append("\n[1. 关键指标统计 (Key Metrics)]")
    metrics_summary = []
    
    for label, data in results.items():
        # 计算全子载波的聚合指标
        overall_amp_mean = np.mean(data['avg_amp']) # 平均信号强度
        overall_amp_std = np.mean(data['std_amp'])  # 平均波动幅度
        overall_phase_std = np.mean(data['std_phase']) # 平均相位波动
        
        # 变异系数 (CV) = 标准差 / 平均值 (越低越稳定)
        amp_cv = overall_amp_std / overall_amp_mean if overall_amp_mean != 0 else float('inf')
        
        metrics_summary.append({
            'label': label,
            'amp_mean': overall_amp_mean,
            'amp_std': overall_amp_std,
            'amp_cv': amp_cv,
            'phase_std': overall_phase_std,
            'count': data['count']
        })
        
        report_lines.append(f"设备: {label}")
        report_lines.append(f"  - 有效帧数: {data['count']}")
        report_lines.append(f"  - 平均幅度 (Signal Strength): {overall_amp_mean:.4f}")
        report_lines.append(f"  - 幅度稳定性 (Stability - Std Dev): {overall_amp_std:.4f}")
        report_lines.append(f"  - 幅度变异系数 (CV, 越小越好): {amp_cv:.4f}")
        report_lines.append(f"  - 相位标准差 (Phase Stability): {overall_phase_std:.4f}")

    # 2. 对比排序
    report_lines.append("\n[2. 对比排名 (Ranking)]")
    
    # 按信号强度排序 (降序)
    sorted_by_amp = sorted(metrics_summary, key=lambda x: x['amp_mean'], reverse=True)
    best_amp = sorted_by_amp[0]
    
    # 按稳定性 (CV) 排序 (升序)
    sorted_by_cv = sorted(metrics_summary, key=lambda x: x['amp_cv'])
    best_stable = sorted_by_cv[0]

    report_lines.append("-> 信号强度 (Amplitude Strength):")
    for rank, item in enumerate(sorted_by_amp, 1):
        report_lines.append(f"   {rank}. {item['label']}: {item['amp_mean']:.2f}")

    report_lines.append("\n-> 信号稳定性 (Stability - lowest CV):")
    for rank, item in enumerate(sorted_by_cv, 1):
        report_lines.append(f"   {rank}. {item['label']}: CV = {item['amp_cv']:.4f}")

    # 3. 结论建议
    report_lines.append("\n[3. 自动分析结论 (Conclusion)]")
    conclusion = f"根据对选定时间段数据的分析：\n"
    conclusion += f"1. **信号强度最佳**：设备 '{best_amp['label']}' 检测到的平均CSI幅值最高 ({best_amp['amp_mean']:.2f})，表明该位置接收到的信号能量最强。\n"
    conclusion += f"2. **信号稳定性最佳**：设备 '{best_stable['label']}' 的数据波动最小 (CV={best_stable['amp_cv']:.4f})，数据质量最可靠。\n"
    
    if best_stable['label'] == best_amp['label']:
        conclusion += f"\n**综合推荐**：设备 '{best_amp['label']}' 在强度和稳定性上均表现最优，是非常理想的观测位置。"
    else:
        conclusion += f"\n**综合对比**：'{best_amp['label']}' 信号更强，但 '{best_stable['label']}' 更加稳定。\n"
        conclusion += f"   - 如果算法对信噪比(SNR)敏感，建议优先考虑 '{best_amp['label']}'。\n"
        conclusion += f"   - 如果算法对波形抖动敏感，建议优先考虑 '{best_stable['label']}'。"

    report_lines.append(conclusion)
    
    # 输出内容
    report_content = "\n".join(report_lines)
    print(report_content)
    
    # 保存文件
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"\n[完成] 分析报告已保存至: {report_path}")

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
    print(f"1. 开始多文件对比分析...")
    print(f"   目标时间段: {TARGET_TIME_WINDOW[0]} - {TARGET_TIME_WINDOW[1]}")

    results = {}
    
    # 遍历定义好的文件列表进行处理
    for label, file_path in FILES_CONFIG.items():
        print(f"\n[处理中] {label}: {file_path}")
        try:
            # 读取 Excel
            df = pd.read_excel(file_path)
            # 确保时间列是字符串格式
            df['time'] = df['time'].astype(str)
        except FileNotFoundError:
            print(f"   错误: 找不到文件 {file_path}")
            continue
        except Exception as e:
            print(f"   错误: 读取文件失败 - {e}")
            continue

        # 1. 筛选时间
        start_time, end_time = TARGET_TIME_WINDOW
        mask = (df['time'] >= start_time) & (df['time'] <= end_time)
        subset = df[mask].copy()
        
        if len(subset) == 0:
            print(f"   警告: 该文件在指定时间段内无数据！")
            continue
            
        # 2. 解析 CSI 数据列
        parsed_series = subset['csidata'].apply(parse_cleaned_csi)
        valid_csi = parsed_series.dropna()
        
        if len(valid_csi) == 0:
            print(f"   警告: 数据解析失败或为空。")
            continue
            
        # 3. 堆叠成矩阵 (行=帧数, 列=子载波索引)
        try:
            csi_matrix = np.vstack(valid_csi.values)
        except ValueError:
            print(f"   错误: 子载波长度不一致，尝试自动修正...")
            lens = valid_csi.apply(len)
            mode_len = lens.mode()[0]
            valid_csi = valid_csi[lens == mode_len]
            csi_matrix = np.vstack(valid_csi.values)

        # 4. 计算特征
        amplitude = np.abs(csi_matrix)
        phase = np.unwrap(np.angle(csi_matrix), axis=1)
        
        # --- 统计指标 ---
        # 1. 平均幅度
        avg_amp = np.mean(amplitude, axis=0)
        # 2. 幅度标准差 (稳定性指标)
        std_amp = np.std(amplitude, axis=0)
        # 3. 平均相位
        avg_phase = np.mean(phase, axis=0)
        # 4. 相位标准差 (相位稳定性)
        std_phase = np.std(phase, axis=0)
        
        results[label] = {
            'avg_amp': avg_amp,
            'std_amp': std_amp,
            'avg_phase': avg_phase,
            'std_phase': std_phase,
            'count': len(valid_csi)
        }
        print(f"   -> 成功提取 {len(valid_csi)} 帧数据")

    # ================= 绘图部分 =================
    if len(results) == 0:
        print("没有有效数据，无法生成图表。")
        return

    print("\n3. 正在生成对比图...")
    import os
    
    # 输出目录
    output_dir = '../../Thesis-figures/chap04/ESP32C6/three_devices_comparison'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # --- 图1: 平均幅度对比 ---
    fig1 = plt.figure(figsize=(8, 6))
    for label, data in results.items():
        plt.plot(data['avg_amp'], label=label, marker='.', markersize=4, alpha=0.8)
    
    plt.title('Average Amplitude Comparison (ESP32C6)')
    plt.xlabel('Subcarrier Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    out_1 = os.path.join(output_dir, 'Average_Amplitude_Comparison.jpg')
    plt.savefig(out_1, dpi=600)
    print(f"图片已保存: {out_1}")
    plt.close(fig1)

    # --- 图2: 幅度稳定性对比 (标准差) ---
    fig2 = plt.figure(figsize=(8, 6))
    for label, data in results.items():
        plt.plot(data['std_amp'], label=label, linestyle='-', linewidth=2, alpha=0.8)
            
    plt.title('Amplitude Stability (Std Dev)')
    plt.xlabel('Subcarrier Index')
    plt.ylabel('Std Deviation')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    out_2 = os.path.join(output_dir, 'Amplitude_Stability_StdDev.jpg')
    plt.savefig(out_2, dpi=600)
    print(f"图片已保存: {out_2}")
    plt.close(fig2)


    # --- 图3: 相位对比 ---
    fig3 = plt.figure(figsize=(8, 6))
    for label, data in results.items():
        plt.plot(data['avg_phase'], label=label, marker='.', markersize=4, alpha=0.8)
    
    plt.title('Average Phase Comparison (Unwrapped)')
    plt.xlabel('Subcarrier Index')
    plt.ylabel('Phase (Radians)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    out_3 = os.path.join(output_dir, 'Average_Phase_Comparison.jpg')
    plt.savefig(out_3, dpi=600)
    print(f"图片已保存: {out_3}")
    plt.close(fig3)
    
    # 生成分析报告
    generate_analysis_report(results, output_dir)
    
    print(f"所有分析图片和报告已保存至目录: {output_dir}")


if __name__ == '__main__':
    main()
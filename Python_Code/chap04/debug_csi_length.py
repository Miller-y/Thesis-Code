import pandas as pd
import ast
from collections import Counter

# ================= 配置区域 =================
INPUT_FILE = 'csi_data_extracted.xlsx'

# 我们只关注报错的那个时间段
DEBUG_TIME_RANGE = ('14:42:07', '14:42:17') # No Target
# ===========================================

def get_list_length(csi_str):
    """
    只计算列表长度，不进行复杂的复数转换，用于快速排查。
    """
    try:
        data_list = ast.literal_eval(csi_str)
        # 我们的数据是实部+虚部，所以除以2才是子载波数量
        return len(data_list) // 2, data_list
    except:
        return -1, None

def main():
    print(f"正在读取文件: {INPUT_FILE} ...")
    try:
        df = pd.read_excel(INPUT_FILE)
    except FileNotFoundError:
        print("找不到文件")
        return

    df['time'] = df['time'].astype(str)
    
    start_time, end_time = DEBUG_TIME_RANGE
    print(f"\n正在分析时间段: {start_time} ~ {end_time}")
    
    # 筛选数据
    mask = (df['time'] >= start_time) & (df['time'] <= end_time)
    subset = df[mask].copy()
    
    if len(subset) == 0:
        print("该时间段没有数据！")
        return

    print(f"总共有 {len(subset)} 条数据。正在检查每一条的长度...")

    # 存储长度统计
    length_counter = Counter()
    # 存储异常样本 (长度 -> list of (time, raw_data))
    abnormal_samples = {}

    # 遍历检查
    for index, row in subset.iterrows():
        subcarrier_count, raw_list = get_list_length(row['csidata'])
        
        length_counter[subcarrier_count] += 1
        
        # 如果长度不是众数（为了找出异常值，我们需要先跑完一遍或者假设众数是正常的）
        # 这里我们先暂存所有非52长度的数据（根据你之前的报错，52似乎是正常的）
        if subcarrier_count != 52:
            if subcarrier_count not in abnormal_samples:
                abnormal_samples[subcarrier_count] = []
            # 只存前3个样本用于查看，避免刷屏
            if len(abnormal_samples[subcarrier_count]) < 3:
                abnormal_samples[subcarrier_count].append((row['time'], raw_list))

    print("\n========== 分析结果 ==========")
    print("子载波数量分布 (长度: 出现次数):")
    for length, count in length_counter.most_common():
        status = " (主要/正常)" if length == 52 else " <--- 异常"
        print(f"长度 {length}: {count} 次{status}")

    print("\n========== 异常数据详情 ==========")
    if not abnormal_samples:
        print("未发现异常长度的数据。")
    else:
        for length, samples in abnormal_samples.items():
            print(f"\n[ 长度为 {length} 的异常数据样本 ]:")
            for time_str, data_val in samples:
                print(f"  时间: {time_str}")
                print(f"  数据总个数: {len(data_val)} (预期应该是 104 -> 52对)")
                print(f"  数据内容: {data_val}")
                print("-" * 50)
                
    print("\n建议：")
    print("1. 如果异常数据很少（比如只有几帧），脚本自动剔除是可以接受的。")
    print("2. 观察异常数据中是否包含未剔除干净的0，或者本身数据就缺失。")

if __name__ == '__main__':
    main()
import pandas as pd
import numpy as np
import scipy.io as sio
from datetime import datetime
import os

# ================= 配置区域 =================

# 1. 文件路径配置 (请确保这三个文件在当前目录下，或修改为绝对路径)
file_paths = [
    'data/csi_data_104_handled.xlsx',  # 对应 Antenna Index 0
    'data/csi_data_105_handled.xlsx',  # 对应 Antenna Index 1
    'data/csi_data_106_handled.xlsx'   # 对应 Antenna Index 2
]

# 2. 位置与时间窗口定义
# 格式: 'Position Name': ('Start Time', 'End Time')
time_windows = {
    'Position 1': ('22:21:10', '22:21:18'),
    'Position 2': ('22:21:53', '22:22:01'),
    'Position 3': ('22:21:32', '22:21:40'),
    'Position 4': ('22:22:16', '22:22:25')
}

# 3. 空间坐标定义 (重要：请根据实际测量的坐标修改这里！)
# 维度: 4行 x 3列 (X, Y, Z)
# 这里暂时用假设坐标代替，请务必修改为真实值
real_positions = np.array([
    [1.0, 0.0, 0.0],  # Position 1 坐标
    [2.0, 0.0, 0.0],  # Position 2 坐标
    [0.0, 1.0, 0.0],  # Position 3 坐标
    [1.5, 1.5, 0.0]   # Position 4 (Center) 坐标
])

# ================= 工具函数 =================

def parse_csi_string(csi_str):
    """
    解析CSI字符串并计算幅值
    输入: "[1, 2, 3, 4 ...]" (224个数字)
    规则: [虚部1, 实部1, 虚部2, 实部2 ...]
    输出: (112,) 的幅值数组
    """
    # 去除方括号并转为numpy数组 (比 ast.literal_eval 快)
    # 假设数据是用逗号分隔的
    raw_data = np.fromstring(csi_str.strip('[]'), sep=',', dtype=np.float32)
    
    if len(raw_data) != 224:
        raise ValueError(f"CSI数据长度异常，期望224，实际{len(raw_data)}")
    
    # 切片分离虚部和实部
    # 题目要求：第一个是虚部，第二个是实部
    imag_parts = raw_data[0::2] # 索引 0, 2, 4...
    real_parts = raw_data[1::2] # 索引 1, 3, 5...
    
    # 计算幅值: sqrt(real^2 + imag^2)
    amplitudes = np.sqrt(real_parts**2 + imag_parts**2)
    
    return amplitudes

def get_time_obj(time_val):
    """将Excel中的时间转换为标准的datetime对象以便比较"""
    # 如果读取出来已经是datetime对象（包含日期），提取时间部分
    # 如果是字符串，解析为时间
    if isinstance(time_val, str):
        return datetime.strptime(time_val, "%H:%M:%S").time()
    elif isinstance(time_val, datetime): # Pandas Timestamp
        return time_val.time()
    else:
        # 处理 python datetime.time 对象或其他情况
        return time_val

# ================= 主处理逻辑 =================

def main():
    # 初始化最终的数据容器
    # 维度: 4(Pos) × 3(Ant) × 112(Subcarrier) × 100(Frame)
    final_data = np.zeros((4, 3, 112, 100), dtype=np.float32)
    
    positions_keys = list(time_windows.keys())
    
    print("开始处理数据...")

    # 遍历 3 个天线文件
    for ant_idx, file_path in enumerate(file_paths):
        if not os.path.exists(file_path):
            print(f"错误: 找不到文件 {file_path}")
            return

        print(f"正在读取文件 ({ant_idx+1}/3): {file_path} ...")
        
        # 读取 Excel
        df = pd.read_excel(file_path)
        
        # 确保列名没有空格干扰
        df.columns = [c.strip() for c in df.columns]
        if 'time' not in df.columns or 'csidata' not in df.columns:
            print(f"错误: 文件 {file_path} 列名不匹配，需要 'time' 和 'csidata'")
            return

        # 预处理时间列：统一转换为 datetime.time 对象以便比较
        # 这里为了能够比较跨天或者不同日期，我们统一映射到一个固定的日期（比如今天）
        # 或者直接比较 time 对象。直接比较 time 对象最简单。
        df['temp_time_obj'] = df['time'].apply(lambda x: get_time_obj(x))
        
        # 遍历 4 个位置
        for pos_idx, pos_name in enumerate(positions_keys):
            start_str, end_str = time_windows[pos_name]
            start_time = datetime.strptime(start_str, "%H:%M:%S").time()
            end_time = datetime.strptime(end_str, "%H:%M:%S").time()
            
            # 筛选时间窗口内的数据
            # 注意：这里假设时间都在同一天内
            mask = (df['temp_time_obj'] >= start_time) & (df['temp_time_obj'] <= end_time)
            window_data = df.loc[mask].copy()
            
            # 检查数据量
            count = len(window_data)
            if count < 100:
                print(f"警告: {file_path} 在 {pos_name} ({start_str}-{end_str}) 只有 {count} 帧，不足 100 帧！")
                # 策略：如果不足，报错退出或者填充？通常应该报错。
                # 这里为了脚本健壮性，我们报错。
                raise ValueError(f"数据不足：{file_path} - {pos_name}")
            
            # 取前 100 帧
            selected_rows = window_data.iloc[:100]
            
            # 处理每一帧的 CSI 数据
            for frame_idx, (_, row) in enumerate(selected_rows.iterrows()):
                csi_str = row['csidata']
                try:
                    amp_array = parse_csi_string(csi_str) # 形状 (112,)
                    
                    # 填入大矩阵
                    # final_data[Pos, Ant, Subcarrier, Time]
                    final_data[pos_idx, ant_idx, :, frame_idx] = amp_array
                    
                except Exception as e:
                    print(f"解析错误: {file_path} 行 {frame_idx} - {e}")
                    return

            print(f"  -> 已提取 {pos_name}: {start_str} ~ {end_str} (Antenna {ant_idx+1})")

    # ================= 保存结果 =================
    
    output_filename = 'data/train_esp32c6.mat'
    print(f"\n正在保存数据到 {output_filename} ...")
    
    # 构造字典用于保存
    mat_content = {
        'data': final_data,          # 4 x 3 x 112 x 100
        'position': real_positions   # 4 x 3
    }
    
    sio.savemat(output_filename, mat_content)
    
    print("处理完成！")
    print(f"data维度: {final_data.shape}")
    print(f"position维度: {real_positions.shape}")

if __name__ == "__main__":
    main()
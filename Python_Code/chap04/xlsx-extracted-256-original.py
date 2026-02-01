import pandas as pd
import ast

# 1. 设置文件路径
input_file = 'data/Antanne_3.xlsx'  # 请修改为您实际的文件名
output_file = 'data/Antanne_3_original.xlsx'

# 需要剔除的子载波对索引 (0-127)
# 头部2对: 0-1
# 中间11对: 28~38
# 尾部7对: 93~99
REMOVE_INDICES = {0, 1, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 93, 94, 95, 96, 97, 98, 99}

def clean_csi_row(row):
    """
    解析CSI数据字符串，按固定位置剔除无效子载波。
    保留有效的 108 对数据。
    """
    csi_str = row['csidata_raw']
    try:
        # 将字符串列表转换为真实的 Python 列表
        # data_list = ast.literal_eval(csi_str) # 旧版: 逗号分隔
        
        # 新版: 空格分隔 [12 34 ...]
        # 去掉首尾中括号，然后按空格分割并转为int
        content = csi_str.strip().replace('[', '').replace(']', '')
        # split() 不带参数默认会处理所有空白字符（包括多个空格）
        data_list = [int(x) for x in content.split()]
        
        
        # 1. 长度检查：标准CSI数据应该有256个数字 (128对)
        if len(data_list) != 256:
            # 打印具体行号（假设原数据有默认索引）
            print(f"警告: 第 {row.name} 行数据异常，长度为 {len(data_list)} (预期256)，已跳过。")
            return None  # 返回 None 表示这一行无效
        cleaned_list = []

        # 2. 按对遍历 (共128对)
        for i in range(128):
            # 如果当前对的索引在剔除列表中，则跳过
            if i in REMOVE_INDICES:
                continue
            
            # 提取这一对的数据 (虚部, 实部)
            # data_list 是扁平的，第 i 对对应的索引是 2*i 和 2*i+1
            val1 = data_list[2*i]
            val2 = data_list[2*i+1]
            
            cleaned_list.extend([val1, val2])
                
        # 将结果转换回字符串格式 "[x, x, ...]"
        return str(cleaned_list).replace(" ", "")
        
    except Exception as e:
        print(f"出错: 第 {row.name} 行解析失败: {e}")
        return None


# 2. 读取文件 (增加了对 CSV/Excel 的自动识别，更加稳健)
print("正在读取文件...")
try:
    if input_file.endswith('.xlsx'):
        df = pd.read_excel(input_file, header=None)
    else:
        # 尝试读取 CSV
        try:
            df = pd.read_csv(input_file, header=None, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(input_file, header=None, encoding='gbk')
    print("文件读取成功，正在处理...")
except FileNotFoundError:
    print(f"错误：找不到文件 {input_file}")
    exit()

# 3. 提取 Time 和 CSI Data
print("正在提取并清洗数据...")
# 修改为：直接使用第3列（时间，索引2）和第6列（数据，索引5）
extracted_df = df.iloc[:, [2, 5]].copy()
extracted_df.columns = ['time', 'csidata_raw']
# 确保数据列转为字符串，防止读取为其他对象类型
extracted_df['csidata_raw'] = extracted_df['csidata_raw'].astype(str)

# 4. 应用清洗函数
# 使用 apply 方法将 clean_csi_data 函数应用到每一行
extracted_df['csidata'] = extracted_df.apply(clean_csi_row, axis=1)


# 5. 选择最终要保存的列
original_count = len(extracted_df)
final_df = extracted_df.dropna(subset=['csidata'])[['time', 'csidata']]
cleaned_count = len(final_df)
print(f"\n处理统计: 原有 {original_count} 行，清洗后 {cleaned_count} 行。剔除了 {original_count - cleaned_count} 行异常数据。")


# 6. 预览检查
print("\n数据预览 (前3行):")
print(final_df.head(3))

# 验证逻辑 (检查第一行长度变化)
try:
    original_list = ast.literal_eval(extracted_df['csidata_raw'].iloc[0])
    cleaned_list = ast.literal_eval(final_df['csidata'].iloc[0])
    original_len = len(original_list)
    cleaned_len = len(cleaned_list)

    print(f"\n逻辑验证 (第1行):")
    print(f"原始数据长度: {original_len} ({original_len//2} 对)")
    print(f"清洗后长度:   {cleaned_len} ({cleaned_len//2} 对)")
    print(f"减少了 {original_len - cleaned_len} 个数字 (即 {(original_len - cleaned_len)//2} 对)")
    
    if (original_len - cleaned_len)//2 == 20:
        print("验证通过：成功剔除了 20 对无效数据。")
    else:
        print("警告：剔除的数量与预期不符！")
except:
    print("验证步骤跳过（数据可能为空）。")

# 7. 保存
final_df.to_excel(output_file, index=False)
print(f"\n处理完成！清洗后的文件已保存为: {output_file}")
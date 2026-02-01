import pandas as pd
import ast

# 1. 设置文件路径
input_file = 'data/csi_data_105.xlsx'  # 请修改为您实际的文件名
output_file = 'data/csi_data_105_handled.xlsx'

# 需要剔除的子载波对索引 (0-127)
# 头部4对: 0-3
# 中间9对: 32(-9),61-67,96(29)
# 导频子载波6对：8,16,30,78,92,120保留(4,12,26,66,80,108)
# 尾部3对: 125-127
REMOVE_INDICES = {0, 1, 2, 3, 32, 61, 62, 63, 64, 65, 66, 67, 96, 125, 126, 127}

def clean_csi_row(row):
    """
    解析CSI数据字符串，按固定位置剔除无效子载波。
    保留有效的 106 对数据。
    """
    csi_str = row['csidata_raw']
    try:
        # 将字符串列表转换为真实的 Python 列表
        data_list = ast.literal_eval(csi_str)
        
        
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
# 使用正则提取：时间 和 原始CSI数组字符串
extracted_df = df[0].astype(str).str.extract(r'(\d{1,2}:\d{1,2}:\d{1,2}).*(\[.*\])')
extracted_df.columns = ['time', 'csidata_raw']

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
    
    if (original_len - cleaned_len)//2 == 16:
        print("验证通过：成功剔除了 16 对无效数据。")
    else:
        print("警告：剔除的数量与预期不符！")
except:
    print("验证步骤跳过（数据可能为空）。")

# 7. 保存
final_df.to_excel(output_file, index=False)
print(f"\n处理完成！清洗后的文件已保存为: {output_file}")
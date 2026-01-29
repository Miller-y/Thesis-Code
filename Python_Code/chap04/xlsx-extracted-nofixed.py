import pandas as pd
import ast

# 1. 设置文件路径
input_file = 'esp32s3-csi-data.xlsx'  # 请修改为您实际的文件名
output_file = 'csi_data_extracted.xlsx'



def clean_csi_data(csi_str):
    """
    解析CSI数据字符串，去除成对出现的(0,0)，保留其他数据。
    """
    try:
        # 将字符串列表转换为真实的 Python 列表
        # 例如: "[0, 0, 0, -7]" -> [0, 0, 0, -7]
        data_list = ast.literal_eval(csi_str)
        
        cleaned_list = []
        
        # 确保数据长度是偶数，因为我们要成对处理
        if len(data_list) % 2 != 0:
            # 如果不是偶数，可能数据有问题，这里选择原样返回或者报错
            # 为了稳健，我们暂时原样返回，并打印警告
            print(f"警告: 数据长度为奇数，无法成对处理: {csi_str[:20]}...")
            return csi_str

        # 以步长为 2 遍历列表 (0, 2, 4, ...)
        for i in range(0, len(data_list), 2):
            real = data_list[i]
            imag = data_list[i+1]
            
            # 核心逻辑：只有当两个数同时为 0 时才认为是无效数据
            if real == 0 and imag == 0:
                continue # 跳过这一对
            else:
                # 否则保留这一对 (包括 0,-7 这种情况)
                cleaned_list.extend([real, imag])
                
        # 将结果转换回字符串格式 "[x, x, ...]"
        # 这里的 replace 是为了去掉列表转字符串后自带的空格，保持紧凑格式（可选）
        return str(cleaned_list).replace(" ", "")
        
    except Exception as e:
        print(f"解析数据出错: {e}")
        return csi_str


# 2. 读取 CSV 文件
# header=None 表示不将第一行作为列名，因为第一行可能也是数据
# 如果您的文件第一行是表头，可以将 header=None 去掉
try:
    df = pd.read_excel(input_file, header=None)
    print("文件读取成功，正在处理...")
except FileNotFoundError:
    print(f"错误：找不到文件 {input_file}")
    exit()

# 3. 使用正则表达式提取数据
# 假设数据都在第一列（索引为 0）
# 正则解释：
# (\d{1,2}:\d{1,2}:\d{1,2})  -> 捕获组1：匹配时间格式 (如 14:40:27)
# .* -> 匹配中间任意字符
# (\[.*\])                  -> 捕获组2：匹配方括号及其中内容 (即 csi 数据)
# 3. 提取 Time 和 CSI Data
print("正在提取并清洗数据...")
# 提取 raw 数据
extracted_df = df[0].astype(str).str.extract(r'(\d{1,2}:\d{1,2}:\d{1,2}).*(\[.*\])')
extracted_df.columns = ['time', 'csidata_raw']

# 4. 应用清洗函数
# 使用 apply 方法将 clean_csi_data 函数应用到每一行
extracted_df['csidata'] = extracted_df['csidata_raw'].apply(clean_csi_data)

# 5. 选择最终要保存的列 (去掉原始的 raw 列，只保留清洗后的)
final_df = extracted_df[['time', 'csidata']]

# 6. 预览检查
print("\n数据预览 (前3行):")
print(final_df.head(3))

# 验证特定逻辑 (检查第一行长度变化)
original_len = len(ast.literal_eval(extracted_df['csidata_raw'].iloc[0]))
cleaned_len = len(ast.literal_eval(final_df['csidata'].iloc[0]))
print(f"\n逻辑验证 (第1行):")
print(f"原始数据长度: {original_len}")
print(f"清洗后长度:   {cleaned_len}")
print(f"减少了 {original_len - cleaned_len} 个数字 (即 {(original_len - cleaned_len)//2} 对 (0,0))")


# 7. 保存
final_df.to_excel(output_file, index=False)
print(f"\n处理完成！清洗后的文件已保存为: {output_file}")
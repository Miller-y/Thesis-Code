import pandas as pd
import ast

# 1. 设置文件路径
input_file = 'esp32s3-csi-data.xlsx'  # 请修改为您实际的文件名
output_file = 'csi_data_cleaned_fixed.xlsx'

# 需要剔除的子载波对索引 (0-63)
# 头部6对: 0-5
# 中间1对: 32
# 尾部5对: 59-63
REMOVE_INDICES = {0, 1, 2, 3, 4, 5, 32, 59, 60, 61, 62, 63}

def clean_csi_data(csi_str):
    """
    解析CSI数据字符串，按固定位置剔除无效子载波。
    保留有效的 52 对数据。
    """
    try:
        # 将字符串列表转换为真实的 Python 列表
        data_list = ast.literal_eval(csi_str)
        
        cleaned_list = []
        
        # 1. 长度检查：标准CSI数据应该有128个数字 (64对)
        if len(data_list) != 128:
            # 如果长度不对，无法按固定位置处理，返回空列表或者原样返回
            # 这里选择原样返回并打印警告，方便排查
            print(f"警告: 数据长度不为128 ({len(data_list)})，无法按固定位置清洗")
            return csi_str

        # 2. 按对遍历 (共64对)
        for i in range(64):
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
        print(f"解析数据出错: {e}")
        return csi_str


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
extracted_df['csidata'] = extracted_df['csidata_raw'].apply(clean_csi_data)

# 5. 选择最终要保存的列
final_df = extracted_df[['time', 'csidata']]

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
    
    if (original_len - cleaned_len)//2 == 12:
        print("验证通过：成功剔除了 12 对无效数据。")
    else:
        print("警告：剔除的数量与预期不符！")
except:
    print("验证步骤跳过（数据可能为空）。")

# 7. 保存
final_df.to_excel(output_file, index=False)
print(f"\n处理完成！清洗后的文件已保存为: {output_file}")
import pandas as pd

# 假设数据存储在 'data.csv' 文件中
file_path = 'experience_biking/experiment_results_biking.csv'

# 读取数据
df = pd.read_csv(file_path)

# 计算 IFCM轮廓系数 - FCM轮廓系数
df['IFCM - FCM'] = df['IFCM轮廓系数'] - df['FCM轮廓系数']

# 保留 'IFCM - FCM' 列的小数点后四位
df['IFCM - FCM'] = df['IFCM - FCM'].round(4)

# 输出结果
print(df)

# 保存处理后的数据到新的CSV文件
df.to_csv('processed_data.csv', index=False)

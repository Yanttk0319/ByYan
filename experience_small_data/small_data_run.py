import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from new_IFCM_β import run_experiment

# 读取数据集
rice = pd.read_csv("../dataset/rice.csv", sep='\t')
gap = pd.read_csv("../dataset/gap.csv", sep='\t')
rating = pd.read_csv("../dataset/Ratings.csv", sep='\t')
taiwan = pd.read_csv("../dataset/bankruptcy.csv", sep='\t')
churn = pd.read_csv("../dataset/churn.csv", sep='\t')
abalone = pd.read_csv("../dataset/ablone.csv", sep='\t')

df = rating
columns = list(df.columns)
features = columns[:len(columns)]
dataset = df[features]


# 数据缩放
def scale_to_minus_one_to_one(df):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df


dataset = scale_to_minus_one_to_one(dataset)

# 定义参数范围
c_values = [4, 6, 8]
alpha_values = [0.2, 0.4,0.6, 0.8, 1.0]

# CSV 文件名
csv_file = 'experiment_results_rating.csv'

# 检查文件是否存在，若不存在则创建并写入表头
if not os.path.isfile(csv_file):
    results_df = pd.DataFrame(
        columns=["c", "top_ranked_points_num", "alpha", "公平提升值", "原轮廓系数", "公平轮廓系数"])
    results_df.to_csv(csv_file, index=False)

# 循环遍历所有组合
for c in c_values:
    for alpha in alpha_values:
        try:
            # 运行实验
            result = run_experiment(c, alpha, dataset)
            # 将结果添加到数据框
            results_df = pd.DataFrame([result])
            # 追加结果到CSV文件
            results_df.to_csv(csv_file, mode='a', header=False, index=False)
            print(len(dataset))
        except Exception as e:
            print(f"Error occurred for c={c}, alpha={alpha}: {e}")

print("所有实验已完成，结果已保存到 'experiment_results_rating.csv")

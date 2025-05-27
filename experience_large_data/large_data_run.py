import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from new_IFCM_β import run_experiment

# 读取数据集
occur = pd.read_csv("../dataset/bean.csv", sep='\t')
dry = pd.read_csv("../dataset/estimation.csv", sep='\t')
rental = pd.read_csv("../dataset/rental.csv", sep='\t')
df = occur  #这里放数据集
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


# 自定义相似度计算函数
def calculate_similarity_by_distance(distance):
    if distance > 1:
        similarity = -np.square(distance - 1)
    else:
        similarity = np.square(1 - distance)
    return similarity


# 计算特征之间的距离矩阵
distance_matrix = squareform(pdist(dataset, metric='cityblock'))

# 计算相似度矩阵
similarity_matrix = np.zeros(distance_matrix.shape)
for i in range(distance_matrix.shape[0]):
    for j in range(distance_matrix.shape[1]):
        similarity_matrix[i, j] = calculate_similarity_by_distance(distance_matrix[i, j])

# 将相似度矩阵转换为图
G = nx.Graph()
for i in range(len(similarity_matrix)):
    for j in range(i + 1, len(similarity_matrix)):
        if similarity_matrix[i, j] > 0.82:  # 只添加正相似度的边
            G.add_edge(i, j, weight=similarity_matrix[i, j])

# 找出最大连通子图
largest_cc = max(nx.connected_components(G), key=len)
largest_subgraph = G.subgraph(largest_cc).copy()

# 获取最大连通子图对应的行索引
indices = list(largest_subgraph.nodes)

# 更新原始DataFrame为最大连通子图的DataFrame
dataset = dataset.iloc[indices].reset_index(drop=True)

# 确保 similarity_matrix 和 dataset 匹配
similarity_matrix = similarity_matrix[np.ix_(indices, indices)]

# 定义参数范围
c_values = [4, 6, 8]
alpha_values = [0.2, 0.3, 0.4, 0.6, 0.8, 1.0]

csv_file = 'experiment_results_occur.csv'

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

print("所有实验已完成，结果已保存到 'experiment_results_occur.csv'")

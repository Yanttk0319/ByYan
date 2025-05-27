import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from new_IFCM_β import run_experiment
# 读取数据集
occur = pd.read_csv("dataset/bean.csv", sep='\t')
dry = pd.read_csv("dataset/estimation.csv", sep='\t')
biking = pd.read_csv("dataset/rental.csv", sep='\t')
df = biking
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
        if similarity_matrix[i, j] > 0.84:  # 只添加正相似度的边
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
print(len(similarity_matrix))
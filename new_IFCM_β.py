import os
import numpy as np
import pandas as pd
import operator
import math

import time

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

# 读取数据集




def calculate_similarity_by_distance(distance):
    if distance > 1:
        similarity = -np.square(distance - 1)
    else:
        similarity = np.square(1 - distance)
    return similarity


# 计算曼哈顿距离
def manhattan_distance(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.sum(np.abs(point1 - point2))


# 创建相似度排名
def create_similarity(dataset):
    n_samples = dataset.shape[0]
    similarity_rankings = {}

    for i in range(n_samples):
        similarities = []
        for j in range(n_samples):
            if i != j:
                distance = manhattan_distance(dataset.iloc[i], dataset.iloc[j])
                distance = distance / len(dataset.columns)
                similarity = calculate_similarity_by_distance(distance)
                similarities.append((j, similarity))

        non_zero_similarities = [sim for sim in similarities if sim[1] != 0]
        similarity_rankings[i] = sorted(non_zero_similarities, key=lambda x: x[1], reverse=True)

    return similarity_rankings


def calculate_percentage_0_1(number):
    # 计算百分之0.1，并将结果转换为整数
    return int(number * 0.01)


def auto_beta(X, k_nn, sigmoid_k=4):
    # 计算k近邻距离矩阵
    D = pairwise_distances(X)
    knn_distances = np.sort(D, axis=1)[:, k_nn]
    r_auto = np.median(knn_distances)

    # 计算局部密度（r_auto邻域内的点数）
    rho = np.sum(D < r_auto, axis=1).astype(float)

    # 稳健归一化
    rho_min = np.percentile(rho, 5)
    rho_max = np.percentile(rho, 95)
    rho_norm = (rho - rho_min) / (rho_max - rho_min + 1e-8)
    rho_norm = np.clip(rho_norm, 0, 1)

    # 非线性映射 + 截断
    beta = 1 / (1 + np.exp(-sigmoid_k * (rho_norm - 0.5)))
    beta = np.clip(beta, 0.01, 0.99)

    return beta


# 根据相似度排名创建相似度矩阵
def create_similarity_matrix_from_rankings(similarity_rankings, beta):
    n_samples = len(similarity_rankings)
    similarity_matrix = np.zeros((n_samples, n_samples))

    # 遍历每个数据点的相似度排名
    for i, rankings in similarity_rankings.items():
        # 获取当前样本的beta值
        current_beta = beta[i]
        # 根据当前的beta值计算top_k的数量
        top_ranked_points_num = int(current_beta * n_samples)

        # 获取前top_ranked_points_num个相似的点
        top_rankings = rankings[:top_ranked_points_num]

        # 填充相似度矩阵
        for j, similarity in top_rankings:
            similarity_matrix[i][j] = similarity

    return similarity_matrix


# 保存相似度数据
def save_similarity_data(similarity_rankings, similarity_matrix, filepath):
    try:
        np.savez(filepath, similarity_rankings=similarity_rankings, similarity_matrix=similarity_matrix)
        print(f"Similarity data saved to {filepath}")
    except Exception as e:
        print(f"Error saving similarity data: {e}")


# 加载相似度数据
def load_similarity_data(filepath):
    try:
        data = np.load(filepath, allow_pickle=True)
        return data['similarity_rankings'].item(), data['similarity_matrix']
    except Exception as e:
        print(f"Error loading similarity data from {filepath}: {e}")
        return None, None


# 获取相似度数据
def get_similarity_data(dataset, beta, base_filepath):
    filename = f'similarity_data_{len(dataset)}.npz'
    filepath = os.path.join(base_filepath, filename)
    if os.path.exists(filepath):
        print("Loading similarity data from file.")
        similarity_rankings, similarity_matrix = load_similarity_data(filepath)
        if similarity_rankings is not None and similarity_matrix is not None:
            return similarity_rankings, similarity_matrix
        else:
            print("Failed to load similarity data, generating new data.")
    print("Generating new similarity data.")
    similarity_rankings = create_similarity(dataset)
    similarity_matrix = create_similarity_matrix_from_rankings(similarity_rankings, beta)
    save_similarity_data(similarity_rankings, similarity_matrix, filepath)
    return similarity_rankings, similarity_matrix


# 计算随机游走拉普拉斯矩阵
def compute_random_walk_laplacian(similarity_matrix):
    degree_matrix = np.diag(np.sum(similarity_matrix, axis=1))
    inv_degree_matrix = np.linalg.inv(degree_matrix)
    laplacian_matrix = np.eye(similarity_matrix.shape[0]) - np.dot(inv_degree_matrix, similarity_matrix)
    return laplacian_matrix


# 计算公平成员隶属矩阵
def fair_membership_matrix(membership_matrix, D_inv_S_matrix, alpha):
    n_samples, c = membership_matrix.shape
    I = np.eye(n_samples)
    A = (1 + alpha) * I + alpha * D_inv_S_matrix
    fair_membership = np.linalg.inv(A) @ membership_matrix
    return fair_membership


# 计算FCM轮廓系数
def calculate_fcm_silhouette_coefficient(dataset, membership_matrix, cluster_labels):
    num_samples = len(dataset)
    dataset = dataset.values
    silhouette_values = []

    for i in range(num_samples):
        cluster_i = cluster_labels[i]
        distance_in_cluster_i = 0
        total_membership_in_cluster_i = 0
        for j in range(num_samples):
            if cluster_labels[j] == cluster_i:
                distance_in_cluster_i += membership_matrix[i, cluster_i] * np.linalg.norm(dataset[i] - dataset[j])
                total_membership_in_cluster_i += membership_matrix[i, cluster_i]
        a_i = distance_in_cluster_i / total_membership_in_cluster_i if total_membership_in_cluster_i != 0 else 0

        b_i = np.inf
        for k in range(len(membership_matrix[0])):
            if k != cluster_i:
                distance_in_cluster_k = 0
                total_membership_in_cluster_k = 0
                for j in range(num_samples):
                    if cluster_labels[j] == k:
                        distance_in_cluster_k += membership_matrix[i, k] * np.linalg.norm(dataset[i] - dataset[j])
                        total_membership_in_cluster_k += membership_matrix[i, k]
                if total_membership_in_cluster_k != 0:
                    b_i = min(b_i, distance_in_cluster_k / total_membership_in_cluster_k)

        silhouette_coefficient_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) != 0 else 0
        silhouette_values.append(silhouette_coefficient_i)

    return np.mean(silhouette_values)


# 获取最相似点
def get_top_m_similar_points(v, similarity_rankings, top_ranked_points_num):
    similarity_ranking_v = similarity_rankings.get(v, [])
    if not similarity_ranking_v:
        return []
    top_ranked_points = [point[0] for point in similarity_ranking_v[:top_ranked_points_num]]
    return top_ranked_points


# 计算相似度
def calculate_similarity(v, membership_matrix, similar_points, similarity_matrix, degree_matrix):
    membership_vector_v = membership_matrix[v]
    similarities = []
    D_ii = degree_matrix[v, v]
    for point in similar_points:
        membership_vector_j = membership_matrix[point]
        distance = manhattan_distance(membership_vector_v, membership_vector_j)
        S_ij = similarity_matrix[v, point]
        similarity = calculate_similarity_by_distance(distance) * S_ij / D_ii
        similarities.append(similarity)
    return similarities


# 计算成员隶属相似度
def calculate_membership_similarity(membership_matrix1, membership_matrix2, sample_list, similarity_rankings,
                                    beta, similarity_matrix, degree_matrix):
    ifi, fair_ifi = 0, 0
    n_samples = len(similarity_matrix)

    # 遍历样本列表，计算每个样本的相似度
    for v in sample_list:
        # 获取当前样本的beta值，根据beta值计算top_ranked_points_num
        top_ranked_points_num = int(beta[v] * n_samples)

        # 获取样本v的top_similar_points
        top_similar_points = get_top_m_similar_points(v, similarity_rankings, top_ranked_points_num)

        # 计算样本v的相似度向量1和向量2
        similarity_vector1 = calculate_similarity(v, membership_matrix1, top_similar_points, similarity_matrix,
                                                  degree_matrix)
        similarity_vector2 = calculate_similarity(v, membership_matrix2, top_similar_points, similarity_matrix,
                                                  degree_matrix)

        # 累积相似度
        ifi += sum(similarity_vector1)
        fair_ifi += sum(similarity_vector2)

    # 计算平均ifi和fair_ifi
    fair_ifi = fair_ifi / len(sample_list)
    ifi = ifi / len(sample_list)

    # 计算fi
    fi = fair_ifi - ifi

    return ifi, fair_ifi, fi


# 初始化成员矩阵
def initialize_membership_matrix(c, num_samples):
    membership_mat = np.random.rand(num_samples, c)
    membership_mat = membership_mat / np.sum(membership_mat, axis=1, keepdims=True)
    return membership_mat


# 计算聚类中心
def calculate_cluster_center(membership_mat, dataset, c, m):
    n_samples = len(dataset)
    cluster_mem_val = zip(*membership_mat)
    cluster_centers = []
    cluster_mem_val_list = list(cluster_mem_val)
    for j in range(c):
        x = cluster_mem_val_list[j]
        x_raised = [e ** m for e in x]
        denominator = sum(x_raised)
        temp_num = []
        for i in range(n_samples):
            data_point = list(dataset.iloc[i])
            prod = [x_raised[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, zip(*temp_num))
        center = [z / denominator for z in numerator]
        cluster_centers.append(center)
    return cluster_centers


# 更新公平成员隶属值
def update_fair_membership_value(membership_mat, cluster_centers, dataset, c, alpha, similarity_matrix):
    n_samples = len(dataset)
    for i in range(n_samples):
        x = list(dataset.iloc[i])
        distances = [np.linalg.norm(list(map(operator.sub, x, cluster_centers[j]))) for j in range(c)]
        for j in range(c):
            den = sum([math.pow(float(distances[j] / distances[k]), 2) for k in range(c)])
            membership_mat[i][j] = float(1 / den)
    normalized_laplacian = compute_random_walk_laplacian(similarity_matrix)
    fair_membership = fair_membership_matrix(membership_mat, normalized_laplacian, alpha)
    return fair_membership


# 更新成员隶属值
def update_membership_value(membership_mat, cluster_centers, dataset, c):
    n_samples = len(dataset)
    for i in range(n_samples):
        x = list(dataset.iloc[i])
        distances = [np.linalg.norm(list(map(operator.sub, x, cluster_centers[j]))) for j in range(c)]
        for j in range(c):
            den = sum([math.pow(float(distances[j] / distances[k]), 2) for k in range(c)])
            membership_mat[i][j] = float(1 / den)
    return membership_mat


# 获取聚类标签
def get_clusters(membership_mat):
    n_samples = len(membership_mat)
    cluster_labels = []
    for i in range(n_samples):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
        cluster_labels.append(idx)
    return cluster_labels


# 个人公平模糊C均值聚类
def individual_fairness_fuzzy_c_means_clustering(c, epsilon, m, T, alpha, dataset, similarity_matrix):
    start = time.time()
    n_samples = len(dataset)
    membership = initialize_membership_matrix(c, n_samples)
    clusters = calculate_cluster_center(membership, dataset, c, m)
    fair_clusters = clusters.copy()
    fair_membership = membership.copy()
    t = 0
    while t <= T:
        old_membership_mat = fair_membership.copy()
        membership = update_membership_value(membership, clusters, dataset, c)
        clusters = calculate_cluster_center(membership, dataset, c, m)
        if t < 1:
            fair_membership = update_fair_membership_value(membership, fair_clusters, dataset, c, alpha,
                                                           similarity_matrix)
            fair_clusters = calculate_cluster_center(membership, dataset, c, m)
        else:
            fair_membership = update_fair_membership_value(fair_membership, fair_clusters, dataset, c, alpha,
                                                           similarity_matrix)
            fair_clusters = calculate_cluster_center(fair_membership, dataset, c, m)
        cluster_labels = get_clusters(membership)
        fair_cluster_labels = get_clusters(fair_membership)
        diff_norm = np.linalg.norm(
            [np.linalg.norm(np.array(old_membership_mat[i]) - np.array(fair_membership[i])) for i in
             range(len(old_membership_mat))])
        if diff_norm < epsilon:
            break
        t += 1
    print("用时：{0}".format(time.time() - start))
    return clusters, fair_clusters, membership, fair_membership, cluster_labels, fair_cluster_labels, dataset


# 运行实验
def run_experiment(c, alpha, dataset):
    epsilon = 1e-5
    T = 100
    m = 2.00
    k_nn =  calculate_percentage_0_1(len(dataset))
    beta = auto_beta(dataset, k_nn,4)
    similarity_rankings, similarity_matrix = get_similarity_data(dataset, beta, '.')
    clusters, fair_clusters, membership, fair_membership, labels, fair_labels, dataset = individual_fairness_fuzzy_c_means_clustering(
        c, epsilon, m, T, alpha, dataset, similarity_matrix)
    sample_list = list(range(len(dataset)))
    degree_matrix = np.diag(np.sum(similarity_matrix, axis=1))
    ifi, fair_ifi, fi = calculate_membership_similarity(membership, fair_membership, sample_list, similarity_rankings,
                                                        beta, similarity_matrix,
                                                        degree_matrix)
    PC = calculate_fcm_silhouette_coefficient(dataset, membership, labels)
    fair_PC = calculate_fcm_silhouette_coefficient(dataset, fair_membership, fair_labels)
    result = {
        "c": c,
        "alpha": alpha,
        "公平提升值": fi,
        "FCM的公平评价指标": ifi,
        "IFCM的公平评价指标": fair_ifi,
        "原轮廓系数": PC,
        "公平轮廓系数": fair_PC
    }
    return result



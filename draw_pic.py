import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import os

# 读取Excel数据
file_path = 'processed_data.csv'  # 请确保文件路径正确
data = pd.read_csv(file_path)

# 提取独特的簇中心c值和α值
unique_cluster_centers = sorted(data['c'].unique())  # 4, 6, 8
unique_alphas = sorted(data['α'].unique())  # 0.2, 0.4, 0.6, 0.8, 1.0

# 创建一个函数，用于从数据框中提取特定条件下的 IFM - FCM 值
def get_silhouette_values(df, clusters, alpha_values):
    silhouette_values = []
    for alpha in alpha_values:
        value = df[(df['c'] == clusters) & (df['α'] == alpha)]['IFCM - FCM'].values
        if len(value) > 0:
            silhouette_values.append(value[0])
        else:
            silhouette_values.append(0)  # 如果没有找到对应的值，填充0
    return silhouette_values

# 准备数据用于绘图
x = []
y = []
z = []
for cluster in unique_cluster_centers:
    fcm_values = get_silhouette_values(data, cluster, unique_alphas)
    for alpha, fcm_val in zip(unique_alphas, fcm_values):
        x.append(cluster)  # c 值
        y.append(alpha)    # α 值
        z.append(fcm_val)  # IFCM - FCM 值

# 创建插值网格
x_grid, y_grid = np.meshgrid(np.linspace(min(x), max(x), 200), np.linspace(min(y), max(y), 200))  # 增加插值密度
z_grid = griddata((x, y), z, (x_grid, y_grid), method='cubic')

# 绘制3D曲面图
fig = plt.figure(figsize=(18, 10))
ax = fig.add_subplot(111, projection='3d')

# 使用3D曲面图
surf = ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis', edgecolor='none', alpha=0.8, antialiased=True)

# 设置标题和轴标签
ax.set_title('Silhouette Difference of Estimation')

# 设置x轴标签为 c=4,6,8
ax.set_xticks(unique_cluster_centers)  # 设置c轴刻度为4,6,8
ax.set_xticklabels(unique_cluster_centers)

# 设置y轴标签为 α=0.2, 0.4, 0.6, 0.8, 1.0
y_labels = [f'α={alpha}' for alpha in unique_alphas]
ax.set_yticks(np.arange(len(unique_alphas)))
ax.set_yticklabels(y_labels)

# 设置z轴范围为 -0.1 到 0.15
ax.set_zlim(-0.1, 0.15)

# 调整视角以提高空间感
ax.view_init(elev=30, azim=60)  # 调整视角：elev为仰角，azim为方位角

# 调整光照，使得表面更加有层次感
ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # 去掉z轴线
ax.grid(False)  # 去掉网格线

# 设置颜色条
fig.colorbar(surf, shrink=0.5, aspect=5)

# 保存图片到指定文件夹
output_dir = r'C:\Users\Administrator\Desktop\picture'
os.makedirs(output_dir, exist_ok=True)  # 如果文件夹不存在则创建
output_path = os.path.join(output_dir, 'silhouette_difference.png')
plt.savefig(output_path, bbox_inches='tight')

plt.show()

# 绘制统一的颜色条衡量标准
fig, ax = plt.subplots(figsize=(1, 8))  # 调整宽度为1英寸

# 创建固定颜色条
cmap = plt.get_cmap('viridis')
norm = plt.Normalize(-0.1, 0.15)
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='vertical')

# 保存颜色条衡量标准图片
colorbar_path = os.path.join(output_dir, 'colorbar_standard.png')
plt.savefig(colorbar_path, bbox_inches='tight', pad_inches=0.1)

plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# 9个专属颜色，依次对应9个数据集
colors = [
    "#1f77b4",  # Abalone - 蓝
    "#ff7f0e",  # Bean - 橙
    "#2ca02c",  # Churn - 绿
    "#d62728",  # Estimation - 红
    "#9467bd",  # Gap - 紫
    "#8c564b",  # Bankruptcy - 棕
    "#e377c2",  # Ratings - 粉
    "#7f7f7f",  # Rice - 灰
    "#bcbd22"   # Rental - 黄绿
]

# 9个文件路径
file_paths = [
    'draw_dataset/ab.csv',
    'draw_dataset/be.csv',
    'draw_dataset/ch.csv',
    'draw_dataset/es.csv',
    'draw_dataset/ga.csv',
    'draw_dataset/ba.csv',
    'draw_dataset/ra.csv',
    'draw_dataset/ri.csv',
    'draw_dataset/re.csv'
]

# 9个数据集名称
dataset_names = [
    'Abalone', 'Bean', 'Churn', 'Estimation', 'Gap',
    'Bankruptcy', 'Ratings', 'Rice', 'Rental'
]

# 创建画布
plt.figure(figsize=(12, 8))

# 遍历每个数据集
for i, file_path in enumerate(file_paths):
    try:
        # 读取数据
        data = pd.read_csv(file_path, encoding='utf-8')
        # 按c=4筛选（可按c=8改）
        data_filtered = data[data['c'] == 8]
        # 去除缺失值
        data_filtered = data_filtered.dropna(subset=['α', 'IFCM - FCM'])
        # 按α排序
        data_filtered = data_filtered.sort_values(by='α')
        # 获取横纵坐标
        alpha = data_filtered['α']
        sc_dif = data_filtered['IFCM - FCM']
        # 画散点
        plt.scatter(alpha, sc_dif, label=dataset_names[i], color=colors[i], marker='o', s=30)
        # 画折线
        plt.plot(alpha, sc_dif, color=colors[i], linestyle='--', linewidth=1.5)
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found. Please check the path or file name.")
        continue

# 图表美化
plt.xlabel('α', fontsize=14)
plt.ylabel('SC_difference', fontsize=14)
plt.ylim(-0.5, 0.5)
plt.yticks([i * 0.1 for i in range(-5, 6)])  # -0.5到0.5步长0.1
plt.legend(title='Datasets', loc='upper left')
plt.grid(True, linestyle='--', alpha=0.8)
plt.tight_layout()

# 保存图片
save_path = r'C:\Users\Administrator\Desktop\daxiu\IFI_SC8.png'
plt.savefig(save_path, dpi=300)
plt.show()
print(f"The chart has been saved to: {save_path}")

import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置中文显示和字体配置
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# 数据集路径和标签
file_paths = [
    'draw_dataset/ab.csv',
    'draw_dataset/be.csv',
    'draw_dataset/ch.csv',
    'draw_dataset/es.csv',
    'draw_dataset/ga.csv'
]
labels = ['AB', 'BE', 'CH', 'ES', 'GA']  # 自定义数据集标签

# 创建画布
plt.figure(figsize=(10, 6), dpi=300)
ax = plt.gca()

# 颜色和样式配置
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 精心挑选的配色
markers = ['o', 's', '^', 'D', 'v']  # 不同标记形状
line_styles = ['-', '--', '-.', ':', '-']  # 不同线型

# 遍历所有数据集
for idx, (file_path, label) in enumerate(zip(file_paths, labels)):
    try:
        # 读取数据并筛选c=4
        df = pd.read_csv(file_path)
        df_c4 = df[df['c'] == 4].sort_values(by='α')

        # 提取需要的列
        x = df_c4['α']
        y = df_c4['IFCM_IFI']

        # 绘制折线图
        ax.plot(x, y,
                color=colors[idx],
                linestyle=line_styles[idx],
                marker=markers[idx],
                markersize=8,
                linewidth=2,
                markeredgecolor='w',  # 标记边缘白色
                markerfacecolor=colors[idx],
                label=label)

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

# 设置坐标轴
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.0)
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# 添加标签和标题
ax.set_xlabel('α', fontsize=12, labelpad=10)
ax.set_ylabel('IFCM_IFI', fontsize=12, labelpad=10)
ax.set_title('IFCM_IFI Comparison (c=4)', fontsize=14, pad=20)

# 添加图例和网格
ax.legend(loc='upper left', frameon=True, fontsize=10)
ax.grid(True, linestyle='--', alpha=0.6)

# 确保保存目录存在
save_dir = r'C:\Users\Administrator\Desktop\DAATA_SAVE\pic_save'
os.makedirs(save_dir, exist_ok=True)

# 保存图片
save_path = os.path.join(save_dir, 'IFCM_IFI_comparison.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
print(f'图表已保存至：{save_path}')

# 显示图表
plt.show()
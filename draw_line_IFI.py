import pandas as pd
import matplotlib.pyplot as plt

# Define file paths
file_paths = [
    'draw_dataset/ab.csv',  # Replace with actual data file path
    'draw_dataset/be.csv',  # Replace with actual data file path
    'draw_dataset/ch.csv',  # Replace with actual data file path
    'draw_dataset/es.csv',  # Replace with actual data file path
    'draw_dataset/ga.csv',
    'draw_dataset/ba.csv',
    'draw_dataset/ra.csv',
    'draw_dataset/ri.csv',
    'draw_dataset/re.csv', # Replace with actual data file path
]

# Create a figure
plt.figure(figsize=(12, 8))  # Larger figure size

# Color list, assigning different colors to each line (using lighter shades for lines)
colors = [
    "#1f77b4",  # 蓝
    "#ff7f0e",  # 橙
    "#2ca02c",  # 绿
    "#d62728",  # 红
    "#9467bd",  # 紫
    "#8c564b",  # 棕
    "#e377c2",  # 粉
    "#7f7f7f",  # 灰
    "#bcbd22"  # 黄绿
]

# Define dataset names
dataset_names = ['Abalone', 'Bean', 'Churn', 'Estimation', 'Gap', 'Bankruptcy', 'Ratings', 'Rice', 'Rental']

# Read each data file and plot the line chart
for i, file_path in enumerate(file_paths):
    try:
        # Read data file
        data = pd.read_csv(file_path, encoding='utf-8')  # or use other encodings

        # Filter rows where c=4
        data_filtered = data[data['c'] == 8]

        # Drop rows where 'α' or 'IFCM_IFI' is NaN
        data_filtered = data_filtered.dropna(subset=['α', 'IFCM_IFI'])

        # Sort data by α to ensure the points are in the correct order
        data_filtered = data_filtered.sort_values(by='α')

        # Get α and IFCM_IFI columns
        alpha = data_filtered['α']
        ifcm_ifi = data_filtered['IFCM_IFI']

        # Plot the points with smaller size and no transparency
        plt.scatter(alpha, ifcm_ifi, label=f'{dataset_names[i]}', color=colors[i], marker='o', s=30)  # No transparency

        # Now, connect the points using a dashed line for the same dataset
        plt.plot(alpha, ifcm_ifi, color=colors[i], linestyle='--', linewidth=1.5)  # Lighter color for the line

    except FileNotFoundError:
        print(f"Warning: File {file_path} not found. Please check the path or file name.")
        continue

# Set chart labels (no title)
plt.xlabel('α', fontsize=14)
plt.ylabel('IFI', fontsize=14)

# Set Y-axis range from 0 to 1
plt.ylim(0, 1)

# Set legend
plt.legend(title='Datasets', loc='upper left')

# Add gridlines
plt.grid(True, linestyle='--', alpha=0.8)

# Adjust layout to make it look neat
plt.tight_layout()

# Save the figure to the specified path
save_path = r'C:\Users\Administrator\Desktop\daxiu\IFI_C8'  # Set save path and file name
plt.savefig(save_path, dpi=300)  # Save as a high-quality image

# Show the plot
plt.show()

print(f"The chart has been saved to: {save_path}")

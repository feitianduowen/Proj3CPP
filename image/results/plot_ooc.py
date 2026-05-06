import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 读取 CSV 文件
csv_path = os.path.join(os.path.dirname(__file__), 'ooc.csv')
df = pd.read_csv(csv_path)

# 避免整数溢出，将 N 转换为 float，并计算 log10(时间)
df['N_float'] = df['N'].astype(float)
df['log_time_ns'] = np.log10(df['time_ns'])
# 计算 GFLOPS = 2.0 * N^3 / time_ns
df['GFLOPS'] = 2.0 * (df['N_float'] ** 3) / df['time_ns']

# 将 chunks 转换为字符串，以便作为分类变量（离散颜色）
df['chunks'] = df['chunks'].astype(str)

# 设置绘图风格
plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")

# 绘制分组柱状图（左侧 Y 轴：log10 时间）
hue_order = ['64', '32', '16', '8', '4', '2']
ax = sns.barplot(data=df, x='N', y='log_time_ns', hue='chunks',
                  hue_order=hue_order, palette='viridis')

# 标题和轴标签
plt.title('OOC Matrix Multiplication: lg(Time) vs N and Chunks', fontsize=14, fontweight='bold')
ax.set_xlabel('Matrix Size (N)', fontsize=12)
ax.set_ylabel('lg(Time in ns)', fontsize=12)

# 调整纵轴下界
min_log_val = df['log_time_ns'].min()
ax.set_ylim(bottom=min_log_val - 0.5)

# 创建右侧 Y 轴，用于绘制 GFLOPS 折线
ax2 = ax.twinx()

# 计算每个柱子的 X 坐标（精确到同一个 N 内部的各个 chunk）
unique_N = sorted(df['N'].unique())
n_map = {n: i for i, n in enumerate(unique_N)}
h_map = {h: i for i, h in enumerate(hue_order)}
num_hues = len(hue_order)
bar_width = 0.8 / num_hues

# 对每个 N，在同一 N 内部按 chunks 顺序连接 GFLOPS 点
for n in unique_N:
    sub_df = df[df['N'] == n].copy()
    sub_df['hue_idx'] = sub_df['chunks'].map(h_map)
    sub_df = sub_df.sort_values('hue_idx').dropna(subset=['hue_idx'])
    
    if not sub_df.empty:
        x_coords = n_map[n] + (sub_df['hue_idx'] - (num_hues - 1) / 2.0) * bar_width
        ax2.plot(x_coords, sub_df['GFLOPS'], marker='o', color='orange', linewidth=2)

ax2.set_ylabel('GFLOPS', fontsize=12)

# 调整图例（仅显示柱状图图例）
ax.legend(title='Chunks \n(N/blocksize)', bbox_to_anchor=(1.10, 1), loc='upper left')

# 保存及显示
plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), 'ooc_barplot.png')
plt.savefig(output_path, dpi=300)

print("绘图成功！图片已保存至: {}".format(output_path))
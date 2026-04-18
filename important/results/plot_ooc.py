import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 读取 CSV 文件
csv_path = os.path.join(os.path.dirname(__file__), 'ooc.csv')
df = pd.read_csv(csv_path)

# 为了避免整数溢出，将 N 转换为 float，并计算 lg(time_ns) 即以 10 为底的对数
df['N_float'] = df['N'].astype(float)
df['log_time_ns'] = np.log10(df['time_ns'])
df['normalized_time'] = df['time_ns'] / (df['N_float'] ** 3)

# 将 chunks 转换为字符串，以便作为分类变量（离散颜色）
df['chunks'] = df['chunks'].astype(str)

# 设置绘图风格
plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")

# 绘制分组柱状图
# 横坐标为 N，纵坐标为 log_time_ns，用 chunks 区分颜色，并按照 64, 32, 16, 8, 4, 2 的顺序排列
hue_order = ['64', '32', '16', '8', '4', '2']
ax = sns.barplot(data=df, x='N', y='log_time_ns', hue='chunks', hue_order=hue_order, palette='viridis')

# 设置图表标题和坐标轴标签
plt.title('OOC Matrix Multiplication: lg(Time) vs N and Chunks', fontsize=14, fontweight='bold')
plt.xlabel('Matrix Size (N)', fontsize=12)
plt.ylabel('lg(Time in ns)', fontsize=12)

# 避免纵坐标从 0 开始，将下界设为比最小值稍微低一点
min_log_val = df['log_time_ns'].min()
ax.set_ylim(bottom=min_log_val - 0.5)

# 创建并配置右侧的纵坐标（双 Y 轴）
ax2 = ax.twinx()

# 手动计算每个柱子的精确 X 坐标，以便只在同一个 N 内部对不同 chunks 进行连线
unique_N = sorted(df['N'].unique())
n_map = {n: i for i, n in enumerate(unique_N)}
h_map = {h: i for i, h in enumerate(hue_order)}
num_hues = len(hue_order)
bar_width = 0.8 / num_hues

# 对每一个规模 N 单独绘制一条橙色的折线（跨越它自己的不同 chunks）
for n in unique_N:
    sub_df = df[df['N'] == n].copy()
    sub_df['hue_idx'] = sub_df['chunks'].map(h_map)
    sub_df = sub_df.sort_values('hue_idx').dropna(subset=['hue_idx'])
    
    if not sub_df.empty:
        # 计算每个数据点在图上的绝对 X 坐标
        x_coords = sub_df['N'].map(n_map) + (sub_df['hue_idx'] - (num_hues - 1) / 2.0) * bar_width
        ax2.plot(x_coords, sub_df['normalized_time'], marker='o', color='orange', linewidth=2)

ax2.set_ylabel('time_ns / N^3', fontsize=12)

# 调整图例 (只保留左侧的柱状图图例，避免重复)
ax.legend(title='Chunks \n(N/blocksize)', bbox_to_anchor=(1.10, 1), loc='upper left')

# 自动调整布局并保存图片
plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), 'ooc_barplot.png')
plt.savefig(output_path, dpi=300)

print("绘图成功！图片已保存至: {}".format(output_path))

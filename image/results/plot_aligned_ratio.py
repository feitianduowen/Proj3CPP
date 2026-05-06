import matplotlib.pyplot as plt
import numpy as np

# 数据
sizes = [16, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
aligned_time = [100, 5300, 40400, 171300, 931500, 7988600, 23413100, 288155900, 5906959600]
unaligned_time = [100, 5700, 46500, 231900, 1234900, 8668800, 24472700, 259054400, 6467801600]

x = np.arange(len(sizes))
width = 0.35

# 计算 log10 比例 (这里使用以10为底的对数)
ratio = np.array(unaligned_time) / np.array(aligned_time)
log_ratio = np.log10(ratio)

fig, ax1 = plt.subplots(figsize=(10, 6))

# 左侧坐标轴：绘制柱状图
rects1 = ax1.bar(x - width/2, aligned_time, width, label='Aligned Time', color='#1f77b4')
rects2 = ax1.bar(x + width/2, unaligned_time, width, label='Unaligned Time', color='#ff7f0e')

ax1.set_ylabel('Time (ns) [Log Scale]')
ax1.set_xlabel('Matrix Size (N)')
ax1.set_title('Performance Comparison: Aligned vs Unaligned Memory Allocation')
ax1.set_xticks(x)
ax1.set_xticklabels(sizes)
ax1.set_yscale('log')
ax1.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)

# 右侧坐标轴：绘制log(ratio)折线图
ax2 = ax1.twinx()
line = ax2.plot(x, log_ratio, color='green', marker='o', linestyle='-', linewidth=2, label='log10(Unaligned/Aligned)')
ax2.set_ylabel('log10(Unaligned / Aligned)', color='green')
ax2.tick_params(axis='y', labelcolor='green')

# 在折线上加上数据点标签，防止混淆
for i, val in enumerate(log_ratio):
    ax2.annotate(f'{val:.3f}', (x[i], val), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='darkgreen')

# 合并两边的图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

fig.tight_layout()
output_path = 'd:/comp sci/CS209A cpp/lab/project/proj_3/important/results/aligned_barplot.png'
plt.savefig(output_path, dpi=300)
print("Plot successfully saved to {}".format(output_path))

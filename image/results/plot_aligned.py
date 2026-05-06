import matplotlib.pyplot as plt
import numpy as np

# 数据
sizes = [16, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
aligned_time = [100, 5300, 40400, 171300, 931500, 7988600, 23413100, 288155900, 5906959600]
unaligned_time = [100, 5700, 46500, 231900, 1234900, 8668800, 24472700, 259054400, 6467801600]
x = np.arange(len(sizes))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
# 绘制柱状图
rects1 = ax.bar(x - width/2, aligned_time, width, label='Aligned', color='#1f77b4')
rects2 = ax.bar(x + width/2, unaligned_time, width, label='Unaligned', color='#ff7f0e')

# 设置标签和对数坐标轴
ax.set_ylabel('Time (ns) [Log Scale]')
ax.set_xlabel('Matrix Size (N)')
ax.set_title('Performance Comparison: Aligned vs Unaligned Memory Allocation')
ax.set_xticks(x)
ax.set_xticklabels(sizes)
ax.set_yscale('log')
ax.legend()

# 加上网格线方便查看对数刻度
ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)

fig.tight_layout()
output_path = 'd:/comp sci/CS209A cpp/lab/project/proj_3/important/results/aligned_barplot.png'
plt.savefig(output_path, dpi=300)
print("Plot successfully saved to {}".format(output_path))

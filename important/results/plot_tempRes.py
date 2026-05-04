import matplotlib.pyplot as plt
import numpy as np
import os

# 从数据中提取
sizes = [16, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
time_plain = [4600, 67500, 775800, 15154900, 60927600, 2300996000, 33709007300, 471051090000, 6568479443800]
time_improved2 = [100, 4400, 45000, 176600, 638400, 3132800, 26385100, 249706800, 3328205900]
time_openblas = [200, 4500, 42600, 152600, 470700, 3114700, 15363900, 117742900, 933920600]

# 计算加速比
speedup_openblas = [time_plain[i] / time_openblas[i] for i in range(len(sizes))]
speedup_improved2 = [time_plain[i] / time_improved2[i] for i in range(len(sizes))]

fig, ax1 = plt.subplots(figsize=(10, 6))

x = np.arange(len(sizes))
width = 0.25

# 左坐标轴 (对数时间)
ax1.bar(x - width, time_plain, width, label='PLAIN Time', color='#ff7f0e')
ax1.bar(x, time_openblas, width, label='OPENBLAS Time', color='#2ca02c')
ax1.bar(x + width, time_improved2, width, label='IMPROVED2 Time', color='#1f77b4')

ax1.set_yscale('log')
ax1.set_xlabel('Matrix Size (N)')
ax1.set_ylabel('Time (ns) [Log Scale]')
ax1.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)

# 统一 x 轴的 ticks
ax1.set_xticks(x)
ax1.set_xticklabels(sizes)

# 右坐标轴 (加速比)
ax2 = ax1.twinx()
line1, = ax2.plot(x, speedup_openblas, marker='d', linestyle='-', linewidth=2,
                  label='Speedup (PLAIN / OPENBLAS)', color='#d62728')
line2, = ax2.plot(x, speedup_improved2, marker='v', linestyle='-', linewidth=2,
                  label='Speedup (PLAIN / IMPROVED2)', color='#9467bd')

ax2.set_ylabel('Speedup Ratio', color='purple')
ax2.tick_params(axis='y', labelcolor='purple')

# ---------- 在折线点上方标注倍速比 ----------
# 为了避免两条折线标注重叠，openblas 标注向右偏移，improved2 向左偏移
offset_x = 0.08   # 水平偏移量
for i, (xi, yi) in enumerate(zip(x, speedup_openblas)):
    ax2.text(xi + offset_x, yi, f'{yi:.1f}', ha='left', va='bottom',
             fontsize=8, color='#d62728')
for i, (xi, yi) in enumerate(zip(x, speedup_improved2)):
    ax2.text(xi - offset_x, yi, f'{yi:.1f}', ha='right', va='bottom',
             fontsize=8, color='#9467bd')
# -------------------------------------------

# 合并图例
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.title('Performance Comparison & Speedup (vs PLAIN)')
fig.tight_layout()

output_path = os.path.join(os.path.dirname(__file__), 'improved2.png')
plt.savefig(output_path, dpi=300)
print("Plot successfully saved to {}".format(output_path))
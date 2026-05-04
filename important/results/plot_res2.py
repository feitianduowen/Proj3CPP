import matplotlib.pyplot as plt
import numpy as np
import os
import csv

output_dir = r'd:\comp sci\CS209A cpp\lab\project\proj_3\important\results'
csv_file = os.path.join(output_dir, 'res2.csv')

sizes = []
plain = []
improved = []
openblas = []

# 读取 CSV 数据
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader) # skip header
    for row in reader:
        if not row or len(row) < 4:
            continue
        sizes.append(str(row[0]))
        plain.append(float(row[1]))
        improved.append(float(row[2]))
        openblas.append(float(row[3]))

plain = np.array(plain)
improved = np.array(improved)
openblas = np.array(openblas)

# 取对数
log_plain = np.log10(plain)
log_improved = np.log10(improved)
log_openblas = np.log10(openblas)

x = np.arange(len(sizes))
width = 0.25

fig, ax1 = plt.subplots(figsize=(10, 6))

rects1 = ax1.bar(x - width, log_plain, width, label='Plain', color='skyblue')
rects2 = ax1.bar(x, log_improved, width, label='Improved', color='lightgreen')
rects3 = ax1.bar(x + width, log_openblas, width, label='OpenBLAS', color='salmon')

ax1.set_ylabel('log10(Time in ns)')
ax1.set_xlabel('Matrix Size (N)')
ax1.set_title('Performance Comparison: Plain vs Improved vs OpenBLAS')
ax1.set_xticks(x)
ax1.set_xticklabels(sizes)
ax1.legend(loc='upper left')

# 添加数值标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax1.annotate('{:.1f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# 建立共享X轴的第二Y轴
ax2 = ax1.twinx()
ratio_imp = plain / improved
ratio_ob = plain / openblas

# 绘制折线图
ax2.plot(x, ratio_imp, color='green', marker='o', linestyle='-', linewidth=2, label='Speedup (Plain/Improved)')
ax2.plot(x, ratio_ob, color='red', marker='s', linestyle='--', linewidth=2, label='Speedup (Plain/OpenBLAS)')

ax2.set_ylabel('Speedup Ratio (Multiplier)', color='black')
# 调整副坐标图例的位置，避免与主坐标图例重叠
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0))

# 给折线图数据标注具体加速比倍数
for i in range(len(sizes)):
    # Improved 倍率标签向上偏移
    ax2.annotate('{:.1f}x'.format(ratio_imp[i]), (x[i], ratio_imp[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='green', weight='bold')
    # OpenBLAS 倍率标签向下偏移，防止重叠
    ax2.annotate('{:.1f}x'.format(ratio_ob[i]), (x[i], ratio_ob[i]), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9, color='red', weight='bold')

fig.tight_layout()
output_path = os.path.join(output_dir, 'res2_barplot.png')
plt.savefig(output_path, dpi=300)
print('Plot saved to {}'.format(output_path))
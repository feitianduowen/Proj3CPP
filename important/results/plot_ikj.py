import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure directory exists
output_dir = r'd:\comp sci\CS209A cpp\lab\project\proj_3\important\results'
os.makedirs(output_dir, exist_ok=True)

N = ['16', '50', '128', '256', '400', '800','1600','2000']
plain = [1100, 31100, 692900, 6328200, 21145900, 228186800,2268065300,13165999100]
ikj = [300, 7300, 106200, 828300, 2667200, 20861200,234924100,575011800]

log_plain = np.log10(plain)
log_ikj = np.log10(ikj)

x = np.arange(len(N))
width = 0.35
ratio = np.array(plain) / np.array(ikj)

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, log_plain, width, label='plain', color='skyblue')
rects2 = ax.bar(x + width/2, log_ikj, width, label='ikj', color='salmon')

ax.set_ylabel('log10(Time in ns)')
ax.set_xlabel('Matrix Size (N)')
ax.set_title('Plain vs IKJ Loop Optimization Performance')
ax.set_xticks(x)
ax.set_xticklabels(N)
ax.legend(loc='upper left')

# 建立共享X轴的第二Y轴
ax2 = ax.twinx()
ax2.plot(x, ratio, color='magenta', marker='o', linestyle='-', linewidth=2, label='Speedup (plain/ikj)')
ax2.set_ylabel('Speedup Ratio (plain / ikj)', color='magenta')
ax2.tick_params(axis='y', labelcolor='magenta')
ax2.legend(loc='upper right')

# 添加数值标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)

# 给折线图数据标注具体加速比倍数
for i, val in enumerate(ratio):
    ax2.annotate('{:.2f}x'.format(val), (x[i], val), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='magenta', weight='bold')

fig.tight_layout()
output_path = os.path.join(output_dir, 'ikj_barplot.png')
plt.savefig(output_path, dpi=300)
print('Plot saved to {}'.format(output_path))

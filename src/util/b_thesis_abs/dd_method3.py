from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os
import japanize_matplotlib

# プロットするデータ
current_points = [
    (1.89968694, 1.71819066, 1.25367675), #1
    (1.39968694, 1.31819066, 1.85367675), #2
    (0.25249593, 1.54140857, 0.00995917), #3
]

past_points = [
    (0.65087773, 1.22789779, -0.10376296), #1
    (0.50964872, -0.30561939, 0.83686038), #2
    (-0.04270765, 0.08497858, -0.04638656), #3
    (-0.59968694, 2.11819066, 0.05367675), #4
    (0.94988517, 1.00705003, 1.08114783) #5
]

# 保存先
output_path = '/mnt/nas0/g005/murasemaru/exp/other'
file_name = 'dd_method3.png'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# グラフの描画
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 過去データをプロット
for i, (x, y, z) in enumerate(past_points):
    ax.scatter(x, y, z, color='red', alpha=0.7, s=300)

# 現在データをプロット
for i, (x, y, z) in enumerate(current_points):
    ax.scatter(x, y, z, color='green', alpha=0.7, s=300)

# 軸範囲を設定（データ範囲に基づいて適切な範囲を選択）
ax.set_xlim(-1, 2)
ax.set_ylim(-1, 2)
ax.set_zlim(-1, 2)

# 軸目盛りを一定間隔で設定
ax.set_xticks(np.arange(-1, 2.5, 0.5))
ax.set_yticks(np.arange(-1, 2.5, 0.5))
ax.set_zticks(np.arange(-1, 2.5, 0.5))

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

# レイアウトを調整
plt.tight_layout()

# 保存（白い余白を最小化）
plt.savefig(os.path.join(output_path, file_name), dpi=800, format='png', bbox_inches='tight')
plt.show()

import math
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#rows, cols = 9, 11
patch_size = 87

def get_bad_patch(path, cube):
    bad_idx = []
    unsure_idx = []
    for file in os.listdir(path):
        if not file.startswith(cube):
            continue
        if file.endswith("_NG"):
            file_path = os.path.join(path, file)
            if not os.listdir(file_path):
                continue
            for patch in os.listdir(file_path):
                bad_idx.append(patch.split("_")[-1][:-4])
            continue
        if file.endswith("G"):
            file_path = os.path.join(path, file)
            if not os.listdir(file_path):
                continue
            for patch in os.listdir(file_path):
                unsure_idx.append(patch.split("_")[-1][:-4])

    return bad_idx, unsure_idx


'''
 Dict 5       | P9_ROI_02_C01_T_NG   |      73 |
+--------------+----------------------+---------+
| Dict 5       | P9_ROI_02_C06_T_NG   |      66 |
+--------------+----------------------+---------+
| Dict 5       | P9_ROI_02_C03_T_NG   |      62 |
+--------------+----------------------+---------+
| Dict 5       | P10_ROI_01_C05_T_NG  |      58 |
+--------------+----------------------+---------+
| Dict 5       | P10_ROI_01_C11_T_NG  |      50 |
+--------------+----------------------+---------+
| Dict 6       | P7_ROI_01_C01_NT_NG  |      97 |
+--------------+----------------------+---------+
| Dict 6       | P7_ROI_01_C06_NT_NG  |      97 |
+--------------+----------------------+---------+
| Dict 6       | P7_ROI_01_C10_NT_NG  |      94 |
+--------------+----------------------+---------+
| Dict 6       | P7_ROI_01_C02_NT_NG  |      91 |
+--------------+----------------------+---------+
| Dict 6       | P7_ROI_01_C07_NT_NG  |      87 |

P11_ROI_01_C03_T
P9_ROI_02_C06_T
'''



path_dir = "ntp_90_90_275/"
cube = "P13_ROI_01_C04_T" #"P9_ROI_02_C06_T"
bad_patches, unsure_patches = get_bad_patch(path_dir, cube)
print(len(bad_patches), len(unsure_patches))
img = plt.imread(os.path.join("PKG - HistologyHSI-GB/", f'{cube.split("_")[0]}/', f'ROI{cube.split("ROI")[1]}/', "rgb.png"))

fig, ax = plt.subplots(figsize=(10.04, 8.0))

print(img.shape)
height, width, _ = img.shape
cols = width//patch_size
rows = height//patch_size
cell_width = patch_size
cell_height = patch_size

ax.imshow(img)


for i in range(rows + 1):
    ax.hlines(i * cell_height, 0, cell_width*cols, color='blue', linewidth=0.5)
for j in range(cols + 1):
    ax.vlines(j * cell_width, 0, cell_height*rows, color='blue', linewidth=0.5)

for pos in bad_patches:
    pos = int(pos)
    y = pos//cols  #math.floor(pos / 11)
    x = pos - (y) * cols
    print("bad", pos, x, y)
    rect = patches.Rectangle((x * cell_width, y * cell_height), cell_width, cell_height, edgecolor='red', facecolor="red", alpha = 0.35)
    ax.add_patch(rect)

for pos in unsure_patches:
    pos = int(pos)
    y = pos//cols  #math.floor(pos / 11)
    x = pos - (y) * cols
    print("unsure", pos, x, y)
    rect = patches.Rectangle((x * cell_width, y * cell_height), cell_width, cell_height, edgecolor='yellow', facecolor="yellow", alpha = 0.35)
    ax.add_patch(rect)

plt.subplots_adjust(left=0.01, right=.99, top=0.999, bottom=0.0, wspace=0.0, hspace=0.0)
ax.set_xlim(0, width)
ax.set_ylim(height, 0)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig(f"Rejected Patches for Cube {cube}.pdf", dpi=600, transparent=True)

plt.show()

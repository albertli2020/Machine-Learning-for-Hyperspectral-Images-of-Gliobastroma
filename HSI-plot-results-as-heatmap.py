import math
import os
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from matplotlib.colors import Normalize
from PIL import Image


def read_csv(file_path):
    data = []
    with open(file_path, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

# Extract grid indices, ROI, and Cube from file_path
def parse_file_path(file_path):
    match = re.search(r"ntp_90_90_275/P(\d+)_ROI_0*(\d+)_C0*(\d+).*/patch_(\d+).npy", file_path)
    if match:
        dataset = int(match.group(1))  # Pn
        roi = int(match.group(2))     # ROI
        cube = int(match.group(3))    # Cube
        patch_id = int(match.group(4))  # Patch ID
        row_id = patch_id // 11       # Calculate row_id
        col_id = patch_id % 11        # Calculate col_id
        return dataset, roi, cube, row_id, col_id
    return None

def filter_data(data, dataset, roi, cid):
    filtered = []
    for row in data:
        file_path = row["Patch Path"]
        parsed = parse_file_path(file_path)
        if parsed and parsed[0] == dataset and parsed[1] == roi and parsed[2] == cid:
            filtered.append({
                "row_id": parsed[3],
                "col_id": parsed[4],
                "true_label": float(row["True Label"]),
                "predicted_probability": float(row["Probabilities"])
            })
    print("Number of output entries matched:", len(filtered))
    return filtered

path_dir = "ntp_90_90_275/"
# Specify the dataset (Pn) and ROI to analyzeP3_ROI_01_C08_T
dataset = 3  # For P1
roi = 1     # ROI3
expected_label = 1
cube_id = 8

if expected_label == 0:
    tnt_str = 'NT'
else:
    tnt_str = 'T'
cube = f"P{dataset}_ROI_{roi:02}_C{cube_id:02}_{tnt_str}"
csv_path = "test_output_2D_ROI_F9_3L.csv"  # Update this path as needed
patch_size = 87

img = plt.imread(os.path.join("PKG - HistologyHSI-GB/", f'{cube.split("_")[0]}/', f'ROI{cube.split("ROI")[1]}/', "rgb.png"))
fig, ax = plt.subplots(figsize=(8.0, 10.04))
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


data = read_csv(csv_path)


#P11_ROI_01_C11_T
# Filter data for the specified dataset and ROI
filtered_data = filter_data(data, dataset, roi, cube_id)

row_ids = [item["row_id"] for item in filtered_data]
col_ids = [item["col_id"] for item in filtered_data]

differences = [item["predicted_probability"] - item["true_label"] for item in filtered_data]
abs_differences = [abs(diff) for diff in differences]

grid_size = (9, 11)      
grid = np.full(grid_size, 1.5)

for row, col, diff in zip(row_ids, col_ids, abs_differences):
    grid[row, col] = diff

opacity = 0.35


norm = Normalize(vmin=min(abs_differences), vmax=max(abs_differences))
cmap = cm.get_cmap('RdBu_r')

for x, row in enumerate(grid):
    for y, diff in enumerate(row):
        if diff == 1.5:
            continue
        if(expected_label == 1):
            diff = 1 - diff
        color = cmap(norm(diff))  
        rect = patches.Rectangle(
            (y * cell_width, x * cell_height),
            cell_width,
            cell_height,
            edgecolor='none',  
            facecolor=color,
            alpha=0.65
        )
        ax.add_patch(rect)

    
ax.set_xlim(0, width)
ax.set_ylim(height, 0)
ax.set_xticks([])
ax.set_yticks([])
plt.title(f"Classification Results for Patches from Cube {cube}")

fig.savefig("plotted_results.png", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.0, 10.04))
img = Image.open("plotted_results.png")
ax.imshow(img, extent=(0, width, height, 0))

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=ax, shrink=0.5)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels(['Non-Tumor (Blue)', 'Rejected (Uncolored)', 'Tumor (Red)'])

ax.axis("off")
plt.show()

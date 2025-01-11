import os
from collections import defaultdict
import random

# Directory ntp_90_90_275
base_dir = 'ntp_90_90_275'

# Dictionary to organize patches by Pn_ROI_m
dataset = defaultdict(list)

# Walk through the directory and collect patches
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.startswith("patch_") and file.endswith(".npy"):
            dir_name = os.path.basename(root)
            parts = dir_name.split('_')
            if parts[0][1] == '6':
                continue  # skip P6
            Pn_ROI_m = parts[0] + '_ROI_' + parts[2]  # e.g., 'P1_ROI_02'
            dataset[Pn_ROI_m].append(os.path.join(root, file))

keys = list(dataset.keys())
random.shuffle(keys)
# Create a new dictionary with shuffled keys
shuffled_dataset = {key: dataset[key] for key in keys}

# Calculate the total number of patches
total_patches = sum(len(patches) for patches in dataset.values())
print("total patches: ", total_patches)

# Determine the number of patches for training
train_count = int(total_patches * 0.7)

# Initialize counters
current_train_count = 0
train_set = []
val_test_set = []

total_roi_count = 0
train_roi_count = 0
# Distribute patches based on the calculated train_count
for roi in shuffled_dataset:
    total_roi_count += 1
    patches = shuffled_dataset[roi]
    p_count = len(patches)
    if current_train_count + p_count <= train_count:
        train_set.extend(patches)
        current_train_count += p_count
        train_roi_count += 1
        print(roi, "selected for training:", p_count)
    elif current_train_count < train_count:
        if current_train_count + (p_count//2) <= train_count:
            if random.choice([True, False]):  # Random decision in tie cases
                train_set.extend(patches)
                current_train_count += p_count
                train_roi_count += 1
                print(roi, "selected for training: ", p_count)
            else:
                val_test_set.extend(patches)
                print(roi, "selected for val/test: ", p_count)
        else:
            val_test_set.extend(patches)
            print(roi, "selected for val/test: ", p_count)
    else:
        val_test_set.extend(patches)
        print(roi, "selected for val/test: ", p_count)

print(f"{current_train_count} ({current_train_count*100/total_patches:.2f}%) patches for training; {len(val_test_set)} patches for val/test")
print(f"{train_roi_count} out of {total_roi_count} ROIs selected for training")

# Example of saving to text files
with open('train_set.txt', 'w') as f:
    for item in train_set:
        f.write("%s\n" % item)

with open('val_test_set.txt', 'w') as f:
    for item in val_test_set:
        f.write("%s\n" % item)

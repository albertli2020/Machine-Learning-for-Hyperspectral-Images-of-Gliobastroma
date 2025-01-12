import os
from collections import defaultdict
import random

# Directory ntp_90_90_275
base_dir = 'ntp_90_90_275'

# Dictionary to organize patches by Pn_ROI_m
dataset_T = defaultdict(list)
dataset_NT = defaultdict(list)

# Walk through the directory and collect patches
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.startswith("patch_") and file.endswith(".npy"):
            dir_name = os.path.basename(root)
            parts = dir_name.split('_')
            if parts[0][1] == '6':
                continue  # skip P6
            if parts[-1] == 'T':
                Pn_ROI_m = parts[0] + '_ROI_' + parts[2]  # e.g., 'P1_ROI_02'
                dataset_T[Pn_ROI_m].append(os.path.join(root, file))
            elif parts[-1] == 'NT':                                
                Pn_ROI_m = parts[0] + '_ROI_' + parts[2]  # e.g., 'P1_ROI_02'
                dataset_NT[Pn_ROI_m].append(os.path.join(root, file))


keys = list(dataset_T.keys())
random.shuffle(keys)
# Create a new dictionary with shuffled keys
shuffled_dataset_T = {key: dataset_T[key] for key in keys}
keys = list(dataset_NT.keys())
random.shuffle(keys)
# Create a new dictionary with shuffled keys
shuffled_dataset_NT = {key: dataset_NT[key] for key in keys}

# Calculate the total number of patches
total_patches_T = sum(len(patches) for patches in dataset_T.values())
total_patches_NT = sum(len(patches) for patches in dataset_NT.values())
print("total Tumor patches: ", total_patches_T, " total Tumor patches: ", total_patches_NT)

# Initialize counters
train_set = []
val_test_set = []

total_roi_count = 0
val_test_roi_count = 0

# Distribute patches based on the calculated val_test_target_count
current_val_test_count = 0
sds = shuffled_dataset_T
# Determine the number of patches for validation and testing for the target fold
val_test_target_count = int(total_patches_T * 0.3)
print("--- Selecting Val/Test ROI set containing Tumor Patches")
for roi in sds:
    total_roi_count += 1
    patches = sds[roi]
    p_count = len(patches)
    if current_val_test_count + p_count <= val_test_target_count:
        val_test_set.extend(patches)
        current_val_test_count += p_count
        val_test_roi_count += 1
        print(roi, "selected for val_test:", p_count)
    elif current_val_test_count < val_test_target_count:
        if current_val_test_count + (p_count//2) <= val_test_target_count:
            if random.choice([True, False]):  # Random decision in tie cases
                val_test_set.extend(patches)
                current_val_test_count += p_count
                val_test_roi_count += 1
                print(roi, "selected for val_test: ", p_count)
            else:
                train_set.extend(patches)
                print(roi, "selected for training: ", p_count)
        else:
            train_set.extend(patches)
            print(roi, "selected for training: ", p_count)
    else:
        train_set.extend(patches)
        print(roi, "selected for training: ", p_count)

total_val_test_count = current_val_test_count
current_val_test_count = 0

# Distribute patches based on the calculated val_test_target_count
current_val_test_count = 0
sds = shuffled_dataset_NT
# Determine the number of patches for validation and testing for the target fo
val_test_target_count = int(total_patches_NT * 0.3)
print("--- Selecting Val/Test ROI set containing Non-Tumor Patches")
for roi in sds:
    total_roi_count += 1
    patches = sds[roi]
    p_count = len(patches)
    if current_val_test_count + p_count <= val_test_target_count:
        val_test_set.extend(patches)
        current_val_test_count += p_count
        val_test_roi_count += 1
        print(roi, "selected for val_test:", p_count)
    elif current_val_test_count < val_test_target_count:
        if current_val_test_count + (p_count//2) <= val_test_target_count:
            if random.choice([True, False]):  # Random decision in tie cases
                val_test_set.extend(patches)
                current_val_test_count += p_count
                val_test_roi_count += 1
                print(roi, "selected for val_test: ", p_count)
            else:
                train_set.extend(patches)
                print(roi, "selected for training: ", p_count)
        else:
            train_set.extend(patches)
            print(roi, "selected for training: ", p_count)
    else:
        train_set.extend(patches)
        print(roi, "selected for training: ", p_count)

total_val_test_count += current_val_test_count

print(f"{total_val_test_count} ({total_val_test_count*100/(total_patches_T+total_patches_NT):.2f}%) patches for val/test; {len(train_set)} patches for training")
print(f"{val_test_roi_count} out of {total_roi_count} ROIs selected for val/test")

fold_index = 1
# Example of saving to text files
with open(f"train_set_ROI_F{fold_index}.txt", 'w') as f:
    for item in train_set:
        f.write("%s\n" % item)

with open(f"val_test_set_ROI_F{fold_index}.txt", 'w') as f:
    for item in val_test_set:
        f.write("%s\n" % item)

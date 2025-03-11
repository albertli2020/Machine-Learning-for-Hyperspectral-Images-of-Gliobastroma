import os
from collections import defaultdict
import random
import copy

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
                if parts[-1] == 'T':
                    Pn_ROI_m = parts[0] + '_ROI_' + parts[2]  # e.g., 'P1_ROI_02'
                    dataset_T[Pn_ROI_m].append(os.path.join(root, file))
                elif parts[-1] == 'NT':                                
                    Pn_ROI_m = parts[0] + '_ROI_' + parts[2]  # e.g., 'P1_ROI_02'
                    dataset_NT[Pn_ROI_m].append(os.path.join(root, file))


keys = list(dataset_T.keys())
total_roi_count = len(keys)
random.shuffle(keys)
# Create a new dictionary with shuffled keys
shuffled_dataset_T = {key: dataset_T[key] for key in keys}
org_shuffled_dataset_T = copy.deepcopy(shuffled_dataset_T)
keys = list(dataset_NT.keys())
total_roi_count += len(keys)
random.shuffle(keys)
# Create a new dictionary with shuffled keys
shuffled_dataset_NT = {key: dataset_NT[key] for key in keys}
org_shuffled_dataset_NT = copy.deepcopy(shuffled_dataset_NT)

# Calculate the total number of patches
total_patches_T = sum(len(patches) for patches in dataset_T.values())
total_patches_NT = sum(len(patches) for patches in dataset_NT.values())
print("total Tumor patches: ", total_patches_T, " total Non-Tumor patches: ", total_patches_NT)

t_set = []
t_rois = []
save_partition_files = True
fold_index = 6
fold_indx_for_output_file = fold_index + 10
for roi in org_shuffled_dataset_T:
    patches = org_shuffled_dataset_T[roi]        
    t_set.extend(patches)
    t_rois.append(roi)
for roi in org_shuffled_dataset_NT:
    patches = org_shuffled_dataset_NT[roi]            
    t_set.extend(patches)
    t_rois.append(roi)

test_patch_count = len(t_set)
test_roi_count = len(t_rois)
print(f"For Partition/Fold # {fold_index}:")
print(f"{test_patch_count} ({test_patch_count*100/(total_patches_T+total_patches_NT):.2f}%) patches for test")
print(f"{test_roi_count} out of {total_roi_count} ROIs selected for test")
print(f"ROIs selected for test:", t_rois)
print(f"------------------------------")
if save_partition_files:
    with open(f"test_set_ROI_F{fold_indx_for_output_file}.txt", 'w') as f:
        for item in t_set:
            f.write("%s\n" % item)
    
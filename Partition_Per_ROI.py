import os
from collections import defaultdict
import random
import copy

# Directory ntp_90_90_275
base_dir = 'ntp_90_90_275'

def select_target_count_val_test(sds, val_test_target_count):
    val_test_set = []
    val_test_rois = []
    val_test_patch_count = 0
    for roi in sds:
        patches = sds[roi]
        p_count = len(patches)
        if val_test_patch_count + p_count <= val_test_target_count:
            val_test_set.extend(patches)
            val_test_patch_count += p_count
            val_test_rois += [roi]
            print(roi, "selected for val_test:", p_count)
        elif val_test_patch_count < val_test_target_count:
            if val_test_patch_count + (p_count//2) <= val_test_target_count:
                if random.choice([True, False]):  # Random decision in tie cases
                    val_test_set.extend(patches)
                    val_test_patch_count += p_count
                    val_test_rois += [roi]
                    print(roi, "selected for val_test: ", p_count)
    
    return val_test_rois, val_test_set

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


for fold_index in range(1, 6, 1):
    if fold_index == 5:
        percentage = 0.3
    else:
        percentage = 0.2
    print(f"--- Selecting Val/Test ROI set containing Tumor Patches for Fold{fold_index}")
    vt_rois_t, vt_set_t = select_target_count_val_test(shuffled_dataset_T, int(total_patches_T * percentage))
    print(f"--- Selecting Val/Test ROI set containing Non-Tumor Patches for Fold{fold_index}")
    vt_rois_nt, vt_set_nt = select_target_count_val_test(shuffled_dataset_NT, int(total_patches_NT * percentage))

    vt_rois = vt_rois_t + vt_rois_nt
    vt_set = vt_set_t + vt_set_nt

    for roi in vt_rois_t:
        del shuffled_dataset_T[roi]
    for roi in vt_rois_nt:
        del shuffled_dataset_NT[roi]

    val_test_patch_count = len(vt_set)
    val_test_roi_count = len(vt_rois)
    train_set = []
    for roi in org_shuffled_dataset_T:
        if roi in vt_rois_t:
            pass
        else:
            patches = org_shuffled_dataset_T[roi]
            train_set.extend(patches)

    for roi in org_shuffled_dataset_NT:
        if roi in vt_rois_nt:
            pass
        else:
            patches = org_shuffled_dataset_NT[roi]
            train_set.extend(patches)


    print(f"{val_test_patch_count} ({val_test_patch_count*100/(total_patches_T+total_patches_NT):.2f}%) patches for val/test; {len(train_set)} patches for training")
    print(f"{val_test_roi_count} out of {total_roi_count} ROIs selected for val/test")
    # Example of saving to text files
    with open(f"train_set_ROI_F{fold_index}.txt", 'w') as f:
        for item in train_set:
            f.write("%s\n" % item)

    with open(f"val_test_set_ROI_F{fold_index}.txt", 'w') as f:
        for item in vt_set:
            f.write("%s\n" % item)

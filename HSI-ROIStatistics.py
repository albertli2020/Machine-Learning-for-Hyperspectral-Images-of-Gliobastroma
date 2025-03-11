import time
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import json
CONFIG_FILE = "config.json"
SCRIPT_NAME = "Visualize Dataset Distribution of ROIs"  # The key in `config.json` for this script

num_bands = 275
patch_size = 90

test_roi_sets = [
    ['P1_ROI_02', 'P3_ROI_02', 'P7_ROI_02', 'P5_ROI_01', 'P9_ROI_01'],
    ['P1_ROI_03', 'P2_ROI_02', 'P5_ROI_02', 'P8_ROI_03', 'P11_ROI_01'],
    ['P1_ROI_04', 'P2_ROI_03', 'P5_ROI_04', 'P7_ROI_03', 'P12_ROI_01'],
    ['P5_ROI_03', 'P8_ROI_01', 'P1_ROI_01', 'P3_ROI_01', 'P13_ROI_01'],
    ['P7_ROI_01', 'P8_ROI_02', 'P2_ROI_01', 'P9_ROI_02', 'P10_ROI_01'],
    ['P6_ROI_01', 'P6_ROI_02', 'P6_ROI_03']
]

test_roi_set_truth_labels = [
    [0, 0, 0, 1, 1],
    [0, 0, 0, 1, 1],
    [0, 0, 0, 1, 1],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1],
    [1, 0, 0]
]

def determine_fold_index(patient_roi):
    found = False
    truth_label = 0
    fold_id = 0    
    for i in range(len(test_roi_sets)):
        if patient_roi in test_roi_sets[i]:
            fold_id = i
            found = True
    if found:
        test_set = test_roi_sets[fold_id]
        for i in range(len(test_set)):
            if patient_roi == test_set[i]:
                truth_label = test_roi_set_truth_labels[fold_id][i]
                break    
    return found, truth_label, fold_id

def countPatchesByROI(path):
    count_dict = defaultdict(lambda: defaultdict(int))  # 2D dictionary
    for entry in os.listdir(path):
        level_one_path = os.path.join(path, entry)
        if os.path.isdir(level_one_path):
            parts = entry.split('_')
            first_part = parts[0] + "_" + parts[1] + "_" + parts[2]
            last_part = parts[-1]
            if last_part == 'NG':
                last_part = parts[-2] + '_NG'
            elif last_part == '50G':
                last_part = parts[-2] + '_NG'
            elif last_part == '50NG':
                last_part = parts[-2] + '_NG'    
            c = sum(1 for entry in os.listdir(level_one_path))
            count_dict[first_part][last_part] += c
    return count_dict

def construct_and_plot_df(stats, by_str, num_cols=2, title_str=None):
    df = pd.DataFrame.from_dict(stats, orient="index").fillna(0).astype(int)

    # Given DataFrame (df) and new column names
    if num_cols == 4:
        new_column_names = ["Tumor", "Tumor-Rejected", "Non-Tumor", "Non-Tumor-Rejected"]  # New names for existing columns
    else:
        new_column_names = ["Tumor", "Non-Tumor"]  # New names for existing columns
    # Assign new column names
    df.columns = new_column_names
    print(df)  

    df.index.name = "ROI_Count"

    # Convert to long format for Seaborn
    df_long = df.reset_index().melt(id_vars="ROI_Count", var_name=by_str, value_name="Value")

    # Set style
    sns.set_theme(style="whitegrid")
    custom_palette = {"Tumor": "red", "Tumor-Rejected":"orange", "Non-Tumor": "green", "Non-Tumor-Rejected": "yellow"}

    # Create horizontal stacked bar chart
    fig = plt.figure(figsize=(8, 5))
    sns.barplot(x="Value", y="ROI_Count", hue=by_str, data=df_long, orient="h", palette=custom_palette)

    # Labels & Title
    plt.xlabel("ROI_Count")
    plt.ylabel(by_str)
    if title_str:
        plt.title(title_str, fontsize=12)
    plt.legend(title="", bbox_to_anchor=(1, 1))  # Move legend outside
    fig.canvas.manager.set_window_title(f"ROI Distribution among {by_str}s")
    plt.show()


def load_config():
    """ Load the configuration for this specific script """
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"Configuration file '{CONFIG_FILE}' not found.")

    with open(CONFIG_FILE, "r") as f:
        config_data = json.load(f)

    # Retrieve the script-specific configuration
    script_config = config_data.get(SCRIPT_NAME, {})

    if not script_config:
        raise ValueError(f"No configuration found for '{SCRIPT_NAME}' in {CONFIG_FILE}.")

    return script_config

if __name__ == "__main__":
    # Load script-specific parameters
    config = load_config()
    
    param = config.get(f"distributed_among", {"selected":"patients"})
    distributed_among = param['selected']

    patch_dir = f'ntp_{patch_size}_{patch_size}_{num_bands}'
    t_str = None
    if distributed_among == "patients":
        stats_roi = countPatchesByROI(patch_dir)
        by_str = "Patient"
        num_cols=2
        stats = defaultdict(lambda: defaultdict(int))  # 2D dictionary
        for roi in stats_roi:
            parts = roi.split('_')
            if stats_roi[roi]["T"] > 0:
                stats[parts[0]]["T"] += 1
            elif stats_roi[roi]["NT"] > 0:
                stats[parts[0]]["NT"] += 1
    elif distributed_among == "data folds (as test set)":
        stats_roi = countPatchesByROI(patch_dir)
        by_str = "Data Fold"
        num_cols=2
        stats = defaultdict(lambda: defaultdict(int))  # 2D dictionary
        for roi in stats_roi:
            found, truth_label, fold_id = determine_fold_index(roi)
            if found:
                if truth_label == 1:
                    stats[fold_id+1]["T"] += 1
                else:
                    stats[fold_id+1]["NT"] += 1
        t_str = "Note: Fold_6 is a special container for P6 ROI patches."
        

    construct_and_plot_df(stats, by_str, num_cols=num_cols, title_str=t_str)
import math
import os
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.metrics import roc_auc_score

from PIL import Image
import json
CONFIG_FILE = "config.json"
SCRIPT_NAME = "Plot Classification Results as Heatmap"  # The key in `config.json` for this script

def read_csv(file_path, nn_arch_fold_str, mff_str=None):
    data = []
    with open(file_path, mode="r") as file:
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        y_true = []
        y_score = []
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
            if row["True Label"] == "1.0":
                y_true.append(1.0)
                if row["Predicted Label"] == "1.0":
                    tp += 1
                else:
                    fn += 1
            else:
                y_true.append(0.0)
                if row["Predicted Label"] == "1.0":
                    fp += 1
                else:
                    tn += 1
            y_score.append(float(row["Probabilities"]))

        #print(tp, fn, tn, fp)
        x = (tp+fn)
        if x == 0:
            sens = 1.0
        else:
            sens = tp/x
        y = (tn+fp)   
        if y == 0:
            spec = 1.0
        else:
            spec = tn/y
        z = (tp+fp)
        if z == 0:
            prec = 1.0
        else:
            prec = tp/z

        if (x+y) == 0:
            accu = 1.0
        else:    
            accu = (tp+tn)/(x+y)
        
        f1_score = (2 * prec * sens) / (prec + sens) if (prec + sens) > 0.00000001 else 0.0
        auc = roc_auc_score(y_true, y_score)
        if mff_str:
            metrics_str = f"As reference, model {nn_arch_fold_str} for {mff_str} has these perf measures: auc={auc:.4f} accu={accu*100:.2f}% sens={sens*100:.2f}% spec={spec*100:.2f}% prec={prec*100:.2f}% f1={f1_score:.4f}"
        else:
            metrics_str = f"As reference, model {nn_arch_fold_str} has these perf measures: auc={auc:.4f} accu={accu*100:.2f}% sens={sens*100:.2f}% spec={spec*100:.2f}% prec={prec*100:.2f}% f1={f1_score:.4f}"
        print(metrics_str)
    return data

# Extract grid indices, ROI, and Cube from file_path
def parse_file_path(file_path, cube_str):
    p_str = f"ntp_90_90_275/{cube_str}/patch_(\d+).npy"
    match = re.search(p_str, file_path)
    if match:
        patch_id = int(match.group(1))  # Patch ID
        row_id = patch_id // 11       # Calculate row_id
        col_id = patch_id % 11        # Calculate col_id
        return row_id, col_id
    return None

def filter_data(data, cube_str):
    filtered = []
    for row in data:
        file_path = row["Patch Path"]
        parsed = parse_file_path(file_path, cube_str)
        if parsed:
            filtered.append({
                "row_id": parsed[0],
                "col_id": parsed[1],
                "true_label": float(row["True Label"]),
                "predicted_probability": float(row["Probabilities"])
            })
    return filtered

csv_paths_3l2d = ["test_output_2D_ROI_F11_3L",
             "test_output_2D_ROI_F12_3L",
             "test_output_2D_ROI_F13_3L",
             "test_output_2D_ROI_F14_3L", #F9
             "test_output_2D_ROI_F15_3L", #F10
             "test_output_2D_ROI_F16_3L"  #P6
             ]

csv_paths_3l2d_32b = ["test_output_2D_32B_ROI_F11_3L",
             "test_output_2D_32B_ROI_F12_3L",
             "test_output_2D_32B_ROI_F13_3L",
             "test_output_2D_32B_ROI_F14_3L", #F9
             "test_output_2D_32B_ROI_F15_3L", #F10
             "test_output_2D_32B_ROI_F16_3L"  #P6
             ]

csv_paths_s2l2d = ["test_output_2L2D_ROI_F11_2L",
             "test_output_2L2D_ROI_F12_2L",
             "test_output_2L2D_ROI_F13_2L",
             "test_output_2L2D_ROI_F14_2L", 
             "test_output_2L2D_ROI_F15_2L", 
             "test_output_2L2D_ROI_F16_2L"  #P6
             ]
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

val_roi_sets = [
    ['P1_ROI_01','P5_ROI_03'],
    ['P2_ROI_01','P8_ROI_02'],
    ['P5_ROI_01','P1_ROI_03'],
    ['P9_ROI_01','P2_ROI_02','P2_ROI_03'],
    ['P7_ROI_03','P1_ROI_03'],
    []
]

val_roi_set_truth_labels = [
    [1, 0],
    [1, 0],
    [1, 0],
    [1, 0, 0],
    [1, 0],
    []
]

def read_csv_and_create_heatmap_for_cube(data, cube, expected_label, num_rejected_patches_threshold=79, indeterminate_range=0.0):
    # Filter data for the specified dataset and ROI
    filtered_data = filter_data(data, cube)
    if len(filtered_data) == 0:
        return None, None

    patch_size = 87
    img = plt.imread(os.path.join("PKG - HistologyHSI-GB/", f'{cube.split("_")[0]}/', f'ROI{cube.split("ROI")[1]}/', "rgb.png"))
    fig, ax = plt.subplots(figsize=(4, 4))
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

    row_ids = [item["row_id"] for item in filtered_data]
    col_ids = [item["col_id"] for item in filtered_data]

    differences = [item["predicted_probability"] - item["true_label"] for item in filtered_data]
    abs_differences = [abs(diff) for diff in differences]

    grid_size = (9, 11)      
    grid = np.full(grid_size, 1.5)

    for row, col, diff in zip(row_ids, col_ids, abs_differences):
        grid[row, col] = diff

    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = cm.get_cmap('RdBu_r')

    count_rejected = 0
    count_results = np.zeros(4, dtype=int)
    for x, row in enumerate(grid):
        for y, diff in enumerate(row):
            if diff > 1.45:
                count_rejected += 1
                count_results[3] = count_rejected
                ax.plot([y * cell_width, (y+1)* cell_width], [x * cell_height, (x+1)*cell_height], color='red', linewidth=1)
                continue

            if diff > (0.5+indeterminate_range):
                count_results[1-expected_label] += 1
            elif diff >= (0.5-indeterminate_range):
                count_results[2] += 1
            else:
                count_results[expected_label] += 1

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

    if count_rejected > num_rejected_patches_threshold:
        plt.close(fig)
        return None, count_results
    
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    counts_str = f"NT:{count_results[0]}, T:{count_results[1]}, I:{count_results[2]}, R:{count_results[3]}"
    plt.title(cube.split("_")[3] + " {" + counts_str +"}", fontsize=15)

    # Save the figure into an image object (in memory)
    canvas = FigureCanvas(fig)
    canvas.draw()

    # Save the image to a NumPy array
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

    plt.close(fig)
    return img, count_results

def create_plots_in_grid(max_cols, cube_images, window_str, title_str, nn_arch_fold_str):
    # Create a single figure with a grid of subplots
    N = len(cube_images)
    if N < 1:
        return
    if N <= max_cols:
        cols = N
        rows = 1
    else:
        cols = max_cols  # Number of columns (adjust as needed)
        rows = (N + cols - 1) // cols  # Calculate the required rows based on N and cols
    
    fig, axs = plt.subplots(rows, cols, figsize=(max(cols * 3, 3), max(rows * 3, 3)))
    if N == 1:
        axs = np.array([[axs]])
    elif rows == 1:
        axs = np.array([axs])

    # Loop through the subplots and populate with images
    for i in range(N):
        ax = axs[i // cols, i % cols]
        img = cube_images[i]        
        # Display the image in the subplot
        ax.imshow(img)
        ax.axis('off')  # Turn off axis

    # Remove any empty subplots if N < rows * cols
    for i in range(N, rows * cols):
        axs[i // cols, i % cols].axis('off')
    
    # Reduce spacing
    plt.subplots_adjust(left=0.0, right=1.0, top=0.99, bottom=0.1, wspace=0.0, hspace=0.0)

    # Add a shared color bar across all subplots
    cax = fig.add_axes([0.1, 0.1, 0.8, 0.03])  # [x, y, w, h]
    cmap = cm.get_cmap('RdBu_r')
    norm = Normalize(vmin=0.0, vmax=1.0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cax, shrink=0.5, orientation='horizontal')
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Non-Tumor (Blue)', 'Rejected (Diag Line in Red) / Indeterminate', 'Tumor (Red)'])
    
    fig.suptitle(title_str, fontsize=12)
    fig.canvas.manager.set_window_title(f"Classification Results as Heatmap for {window_str} from Using the {nn_arch_fold_str} Model")
    # Show the plot
    plt.show()

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

def merge_cube_str(patient_roi, truth_label, cube_id):
    if truth_label == 0:
        tnt_str = 'NT'
    else:
        tnt_str = 'T'
    return f"{patient_roi}_C{(cube_id+1):02}_{tnt_str}"


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

max_num_cubes = 18
max_cols = 6
if __name__ == "__main__":
    # Load script-specific parameters
    config = load_config()
    
    #param = config.get("fold_id", {"selected":1})
    #fold_index = param['selected']
    #param = config.get(f"patient_roi_in_f{fold_index}", {"selected":"DNU"})
    param = config.get(f"patient_roi", {"selected":"DNU"})
    patient_roi = param['selected']
    param = config.get("cube", {"selected":255})    
    cube_id = param['selected']
    param = config.get("rejection_threshold", {"selected":95})    
    rejection_threshold = param['selected']
    param = config.get("range_of_indeterminate", {"selected":0.0}) 
    indeterminate_range = param['selected']

    param = config.get(f"nn_arch", {"selected":"3Layer_2D_275B"})
    nn_arch = param['selected']
    param = config.get(f"results_of_P6_using_model_trained_on_data_fold", {"selected":1})
    results_of_P6_using_model_trained_on_data_fold = param['selected']

    if nn_arch == "2Layer_2D_275B":
        csv_paths = csv_paths_s2l2d
    elif nn_arch == "3Layer_2D_275B":
        csv_paths = csv_paths_3l2d
    elif nn_arch == "3Layer_2D_32B":
        csv_paths = csv_paths_3l2d_32b

    print(f"{SCRIPT_NAME}:")
    #print(f"  - fold_id: {fold_index}")
    #print(f"  - patient_roi_in_f{fold_index}: {patient_roi}")
    print(f"  - for {patient_roi}")
    print(f"  - exlcuding cubes with more than {rejection_threshold} out of 99 patches rejected")
    print(f"  - probabilities between {(0.5-indeterminate_range):.2f} and {(0.5+indeterminate_range):.2f} result in indeterminate patch")
    if cube_id == 255:
        #print(f"  - Cube: ALL")
        pass
    elif cube_id < 1 or cube_id > max_num_cubes:
        print(f"  - Invalid Cube: {cube_id}")
        exit
    else:
        print(f"  - Cube: {cube_id}")

    found, truth_label, fold_index = determine_fold_index(patient_roi)
    if fold_index == 5:
        nn_arch_fold_str = nn_arch + f"_F{results_of_P6_using_model_trained_on_data_fold}"
        mff_str = "P6 ROIs"
        csv_path_str = csv_paths[fold_index]+f"_mff{results_of_P6_using_model_trained_on_data_fold+10}.csv"
    else:
        nn_arch_fold_str = nn_arch + f"_F{fold_index+1}"
        mff_str = None
        csv_path_str = csv_paths[fold_index]+".csv"
    #print(csv_path_str)   
    if found:
        cube_images = []
        total_counts = np.zeros(4, dtype=int)
        data = read_csv(csv_path_str, nn_arch_fold_str, mff_str)
        if cube_id == 255:
            cubes_to_check = range(max_num_cubes)
        else:
            cubes_to_check = [cube_id-1]
        for i in cubes_to_check:
            cube_str = merge_cube_str(patient_roi, truth_label, i)
            cube_image, counts= read_csv_and_create_heatmap_for_cube(data,
                        cube_str, truth_label, num_rejected_patches_threshold=rejection_threshold,
                        indeterminate_range = indeterminate_range)
            if cube_image is not None:
                cube_images.append(cube_image)
                total_counts = total_counts + counts
            else:
                if cube_id == 255:
                    pass
                else:
                    print(f"Cube {cube_str} not found in test_out csv file.")
        
        if total_counts[1] > total_counts[0]: # +  total_counts[2]:
            perc = total_counts[1] / (total_counts[1] + total_counts[0])
            roi_result = f"Tumor ({perc*100:.2f}%)"
        elif total_counts[0] > total_counts[1]: # +  total_counts[2]:
            perc = total_counts[0] / (total_counts[1] + total_counts[0])
            roi_result = f"Non-Tumor ({perc*100:.2f}%)"
        else:
            roi_result = "Indeterminate"
        total_counts_str = f"NT:{total_counts[0]}, T:{total_counts[1]}, I:{total_counts[2]}, R:{total_counts[3]}"
        title_str = "ROI-level Aggregated Assessment: " + roi_result + " {" + total_counts_str + "}"
        if truth_label == 0:
            tnt_str = '_NT'
        else:
            tnt_str = '_T'
        win_str = patient_roi + tnt_str
        print(f"{len(cube_images)} valid cubes plotted for {win_str}. {title_str}")        
        create_plots_in_grid(max_cols, cube_images, win_str, title_str, nn_arch_fold_str)
    else:
        print("Fold index for this roi and cube combination not found.")

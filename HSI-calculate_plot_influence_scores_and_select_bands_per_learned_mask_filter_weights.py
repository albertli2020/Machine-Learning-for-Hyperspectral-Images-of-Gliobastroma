import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import json
import os
CONFIG_FILE = "config.json"
SCRIPT_NAME = "Calc and Plot Influence Scores from Learned Band-wise Mask Weights"  # The key in `config.json` for this script

"""
Calculate a band's influence score using the Exponential Decay Function,
for smoother differentiation and more realistic modeling of diminishing influence. 

Formula:

Influence Factor = exp(-k * Ranked_Bin_Index)
 
where k is a decay rate (e.g., k=0.1 for slower decay, k=0.2 for faster decay),
and, Ranked_Bin_Index is determined by binning the ranked (mask weight from high to low) band indices.

An example is dividing 275 ranked band indices into 14 ranked 20-band bins:

Rank_of_band_per_mask_weight     Ranked_Bin_Index
0   - 19                          0
20  - 39                          1 
40  - 59                          2
60  - 79                          3
80  - 99                          4
...
240 - 259                         12 
260 - 274                         13 

Example (14 Ranked-Bins [275 bands into 14 20-band bins], k=0.375)

Ranked_Bin_Index	Influence Factor
0 	                1.00000
1                   0.68729
3 	                0.32465
4                   0.22313
...
12                  0.01111
13 	                0.00764

"""

def calculate_influence_scores_exponential_decay(mask_weight_ranking_list, num_bands=275, k=0.375):
    # Initialize influence scores array with zeros
    influence_scores = np.zeros(num_bands, dtype=float)
    bin_size = 20
    num_bins = num_bands//bin_size
    residue_bands = num_bands - num_bins*bin_size
    bin_indices = np.repeat(np.arange(num_bins), bin_size)
    res_indices = np.full(residue_bands, bin_indices[-1]+1)  
    exp_array = np.concatenate([bin_indices, res_indices])

    influence_factors = np.exp(-k *  exp_array)  # Apply exponential decay
    
    # Normalize so that the highest rank gets 1.0
    influence_factors /= influence_factors.max()

    # Process each ranking
    for mask_weight_ranking in mask_weight_ranking_list:
        for rank_index, band_index in enumerate(mask_weight_ranking):
            influence_scores[band_index] += influence_factors[rank_index]
    
    # Average the influence scores over the 5 (number of folds) rankings
    influence_scores /= len(mask_weight_ranking_list)
    
    return influence_scores

def load_csv_to_calc_and_plot(csv_file_list, embedded_2dcnn_arch):
    # Load the saved mask parameters from CSV (assuming it's saved in last row)
    ranked_bands_list = []
    num_folds = len(csv_file_list)
    num_input_bands = 275

    for csv_file in csv_file_list:
        # Read the CSV file
        df = pd.read_csv(csv_file, header=0)

        # Drop the first column (series labels)
        df = df.iloc[:, 1:]

        # Extract the last row (latest epoch mask parameters)
        mask_weight_values = df.iloc[-1].values  # Convert to NumPy array
        # Rank all the band indices based on absolute mask weight values (high to low)
        #ranked_indices = sorted(range(len(mask_weight_values)), key=lambda i: abs(mask_weight_values[i]), reverse=True)
        ranked_indices = sorted(range(len(mask_weight_values)), key=lambda i: mask_weight_values[i], reverse=True)
        ranked_bands_list.append(ranked_indices)

    def wavelength_to_bandindex(wl):
        return ((wl-400.482)/2.184).astype(int)

    def bandindex_to_wavelength(bi):
        return bi*2.184 + 400.482

    influence_scores = calculate_influence_scores_exponential_decay(ranked_bands_list, num_bands=num_input_bands)
    sorted_indices = sorted(range(len(influence_scores)), key=lambda i: influence_scores[i], reverse=True)
    num_bands_to_make_the_cut_3 = 112
    num_bands_to_make_the_cut_2 = 56
    num_bands_to_make_the_cut_1 = 32
    num_bands_to_make_the_cut_0 = 16

    # Take the top sorted indices
    top_sorted_indices = sorted_indices[:num_bands_to_make_the_cut_3]
    print(f"Top {num_bands_to_make_the_cut_3} indices in order of their influence scores:", top_sorted_indices)
    original_indices_cut_0 = [i for i in range(len(influence_scores)) if i in top_sorted_indices[:num_bands_to_make_the_cut_0]]
    print(f"Top {num_bands_to_make_the_cut_0} indices in their natural order:", original_indices_cut_0)
    original_indices_cut_1 = [i for i in range(len(influence_scores)) if i in top_sorted_indices[:num_bands_to_make_the_cut_1]]
    print(f"Top {num_bands_to_make_the_cut_1} indices in their natural order:", original_indices_cut_1)
    original_indices_cut_2 = [i for i in range(len(influence_scores)) if i in top_sorted_indices[:num_bands_to_make_the_cut_2]]
    print(f"Top {num_bands_to_make_the_cut_2} indices in their natural order:", original_indices_cut_2)
    original_indices_cut_3 = [i for i in range(len(influence_scores)) if i in top_sorted_indices[:num_bands_to_make_the_cut_3]]
    print(f"Top {num_bands_to_make_the_cut_3} indices in their natural order:", original_indices_cut_3)

    cut_off_data_to_plot_0 = (abs(influence_scores[sorted_indices[num_bands_to_make_the_cut_0-1]]) + abs(influence_scores[sorted_indices[num_bands_to_make_the_cut_0]])) / 2.0
    print(f"Min influence score of the set made the {num_bands_to_make_the_cut_0}-band cut: ", cut_off_data_to_plot_0)
    cut_off_data_to_plot_1 = (abs(influence_scores[sorted_indices[num_bands_to_make_the_cut_1-1]]) + abs(influence_scores[sorted_indices[num_bands_to_make_the_cut_1]])) / 2.0
    print(f"Min influence score of the set made the {num_bands_to_make_the_cut_1}-band cut: ", cut_off_data_to_plot_1)
    cut_off_data_to_plot_2 = (abs(influence_scores[sorted_indices[num_bands_to_make_the_cut_2-1]]) + abs(influence_scores[sorted_indices[num_bands_to_make_the_cut_2]])) / 2.0
    print(f"Min influence score of the set made the {num_bands_to_make_the_cut_2}-band cut: ", cut_off_data_to_plot_2)
    cut_off_data_to_plot_3 = (abs(influence_scores[sorted_indices[num_bands_to_make_the_cut_3-1]]) + abs(influence_scores[sorted_indices[num_bands_to_make_the_cut_3]])) / 2.0
    print(f"Min influence score of the set made the {num_bands_to_make_the_cut_3}-band cut: ", cut_off_data_to_plot_3)

    num_bands_to_plot = num_input_bands
    data_to_plot = influence_scores[:num_bands_to_plot]
    data_to_plot_desc = "Influence Score"

    # Plot the influence score for each input band
    fig, ax = plt.subplots(figsize=(15, 8))

    max_val = max(data_to_plot)
    min_val = min(data_to_plot)
    print(f"min and max {data_to_plot_desc}s are: ", min_val, max_val)
    y_min = min_val - (max_val - min_val) * 0.1  # Formula for lower bound

    x = bandindex_to_wavelength(np.arange(num_bands_to_plot))
    # Create a smooth curve using cubic spline interpolation
    x_smooth = np.linspace(x.min(), x.max(), num_bands_to_plot*16)  # More points for smoothness
    spline = make_interp_spline(x, data_to_plot, k=3)  # k=3 for cubic spline
    y_smooth = spline(x_smooth)

    plt.plot(x_smooth, y_smooth, label=f"{data_to_plot_desc} (Smoothed)", color='b', linewidth=1)
    plt.scatter(x, data_to_plot, marker='o', s=8, color='r', alpha=0.7, label=f"{data_to_plot_desc} (Original)")  # Optional: Show original points
    # Draw a horizontal line at y_cut
    plt.axhline(y=cut_off_data_to_plot_0, color='b', linestyle='--', linewidth=1, label=f"y = {cut_off_data_to_plot_0:.4f} ({num_bands_to_make_the_cut_0}-band cut)")
    plt.axhline(y=cut_off_data_to_plot_1, color='g', linestyle='--', linewidth=2, label=f"y = {cut_off_data_to_plot_1:.4f} ({num_bands_to_make_the_cut_1}-band cut)")
    plt.axhline(y=cut_off_data_to_plot_2, color='y', linestyle='--', linewidth=1, label=f"y = {cut_off_data_to_plot_2:.4f} ({num_bands_to_make_the_cut_2}-band cut)")
    plt.axhline(y=cut_off_data_to_plot_3, color='c', linestyle='--', linewidth=1, label=f"y = {cut_off_data_to_plot_3:.4f} ({num_bands_to_make_the_cut_3}-band cut)")

    plt.ylim(y_min, max_val) 

    # Add secondary x-axis
    ax2 = ax.secondary_xaxis('top', functions=(wavelength_to_bandindex, bandindex_to_wavelength))
    ax2.set_xlabel("Band Index", fontsize=15)
    # Move the label up
    ax2.xaxis.set_label_position('top')  # Ensure it's at the top
    ax2.xaxis.label.set_verticalalignment('bottom') #top')  # Align closer to top of ticks
    ax2.xaxis.label.set_x(0)  # Move label to left

    # Bar height and y-position just above the cut_off line
    #if (cut_off_data_to_plot_1-cut_off_data_to_plot_2) < 0.15:
    #    bar_height = (cut_off_data_to_plot_1-cut_off_data_to_plot_2)*0.25
    #else:
    #    bar_height = (1.0 - cut_off_data_to_plot_1)*0.25
    bar_height = (max_val - cut_off_data_to_plot_0)*0.25
    ax.bar(bandindex_to_wavelength(np.array(original_indices_cut_0)), bar_height*0.5, width=2, color='blue', alpha=0.25, align='center', bottom=cut_off_data_to_plot_0)
    top_y_position = cut_off_data_to_plot_1 #ax.get_ylim()[1] * cut_off_data_to_plot_1  # Position bars right above the cut-off-line-1
    ax.bar(bandindex_to_wavelength(np.array(original_indices_cut_1)), bar_height, width=2, color='green', alpha=0.25, align='center', bottom=top_y_position)
    top_y_position = cut_off_data_to_plot_2 #ax.get_ylim()[1] * cut_off_data_to_plot_2  # Position bars right above the cut-off-line-2
    bar_height = (cut_off_data_to_plot_1 - cut_off_data_to_plot_2)*0.25
    ax.bar(bandindex_to_wavelength(np.array(original_indices_cut_2)), bar_height, width=2, color='yellow', alpha=0.25, align='center', bottom=top_y_position)

    top_y_position = cut_off_data_to_plot_3 #ax.get_ylim()[1] * cut_off_data_to_plot_3  # Position bars right above the cut-off-line-2
    bar_height = (cut_off_data_to_plot_2 - cut_off_data_to_plot_3)*0.25
    ax.bar(bandindex_to_wavelength(np.array(original_indices_cut_3)), bar_height, width=2, color='c', alpha=0.25, align='center', bottom=top_y_position)

    # Formatting
    plt.xlabel("Band Wavelength (nm)", fontsize=15)
    ax.xaxis.label.set_x(0+0.045)#0.035)  # Move label to left
    plt.ylabel(data_to_plot_desc, fontsize=16)
    #plt.title(f"{data_to_plot_desc} Calculated from Learned Band-wise Mask Weights", fontsize=16)
    window_str = f"{data_to_plot_desc} Calculated from Band-wise Mask Weights Learned with Embedded {embedded_2dcnn_arch}-CNN"
    fig.canvas.manager.set_window_title(window_str)
    plt.legend(fontsize=14)
    plt.grid(True)                                                

    plt.tight_layout()
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
    
    param1 = config.get("embedded_2dcnn_arch", "3Layer_2D")
    embedded_2dcnn_arch = param1['selected']
    print(f"{SCRIPT_NAME}:")
    print(f"  - embedded_2dcnn_arch: {embedded_2dcnn_arch}")


    if embedded_2dcnn_arch == "3Layer_2D":    
        csv_file_list = ["MaskFilterParam_2D_ROI_F11_3L.csv",
                         "MaskFilterParam_2D_ROI_F12_3L.csv",
                         "MaskFilterParam_2D_ROI_F13_3L.csv",
                         "MaskFilterParam_2D_ROI_F14_3L.csv",
                         "MaskFilterParam_2D_ROI_F15_3L.csv"]
    else:
        csv_file_list = ["MaskFilterParam_2L2D_ROI_F11_2L.csv",
                         "MaskFilterParam_2L2D_ROI_F12_2L.csv",
                         "MaskFilterParam_2L2D_ROI_F13_2L.csv",
                         "MaskFilterParam_2L2D_ROI_F14_2L.csv",
                         "MaskFilterParam_2L2D_ROI_F15_2L.csv"]
    load_csv_to_calc_and_plot(csv_file_list, embedded_2dcnn_arch)


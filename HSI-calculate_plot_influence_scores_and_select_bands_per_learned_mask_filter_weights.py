import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

"""
Calculate a band's influence score using the Exponential Decay Function,
for smoother differentiation and more realistic modeling of diminishing influence. 

Formula:

Influence Factor = exp(-k * Ranked_Bin_Index)
 
where k is a decay rate (e.g., k=0.1 for slower decay, k=0.2 for faster decay),
and, Ranked_Bin_Index is determined by binning the ranked (mask weight from high to low) band indices.

An example is dividing 275 ranked band indices into 35 ranked 8-band bins:

Rank_of_band_per_mask_weight      Ranked_Bin_Index
0   - 7                           0
8   - 15                          1 
16  - 23                          2
24  - 31                          3
32  - 39                          4
...
264 - 271                         33 
272 - 274                         34 

Example (35 Ranked-Bins [275 bands into 35 8-band bins], k=0.25)

Ranked_Bin_Index	Influence Factor
0 	                1.00000
1                   0.77880
2                   0.60653
3 	                0.47237
4                   0.36788
...
33                  0.00026
34 	                0.00000

"""

def calculate_influence_scores_exponential_decay(mask_weight_ranking_list, num_bands=275, k=0.25):
    # Initialize influence scores array with zeros
    influence_scores = np.zeros(num_bands, dtype=float)
    bin_size = 8
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


# Load the saved mask parameters from CSV (assuming it's saved in last row)
csv_file_list = ["MaskFilterParam_2D_ROI_F11_3L.csv",
                 "MaskFilterParam_2D_ROI_F12_3L.csv",
                 "MaskFilterParam_2D_ROI_F13_3L.csv",
                 "MaskFilterParam_2D_ROI_F14_3L.csv",
                 "MaskFilterParam_2D_ROI_F15_3L.csv"]


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
    ranked_indices = sorted(range(len(mask_weight_values)), key=lambda i: abs(mask_weight_values[i]), reverse=True)
    #ranked_indices = sorted(range(len(mask_weight_values)), key=lambda i: mask_weight_values[i], reverse=True)
    ranked_bands_list.append(ranked_indices)

def wavelength_to_bandindex(wl):
    return ((wl-400.482)/2.284).astype(int)

def bandindex_to_wavelength(bi):
    return bi*2.284 + 400.482

influence_scores = calculate_influence_scores_exponential_decay(ranked_bands_list, num_bands=num_input_bands)
sorted_indices = sorted(range(len(influence_scores)), key=lambda i: influence_scores[i], reverse=True)
num_bands_to_make_the_cut_3 = 112
num_bands_to_make_the_cut_2 = 56
num_bands_to_make_the_cut_1 = 32

# Take the top sorted indices
top_sorted_indices = sorted_indices[:num_bands_to_make_the_cut_3]
print(f"Top {num_bands_to_make_the_cut_3} indices in order of their influence scores:", top_sorted_indices)
original_indices_cut_1 = [i for i in range(len(influence_scores)) if i in top_sorted_indices[:num_bands_to_make_the_cut_1]]
print(f"Top {num_bands_to_make_the_cut_1} indices in their natural order:", original_indices_cut_1)
original_indices_cut_2 = [i for i in range(len(influence_scores)) if i in top_sorted_indices[:num_bands_to_make_the_cut_2]]
print(f"Top {num_bands_to_make_the_cut_2} indices in their natural order:", original_indices_cut_2)
original_indices_cut_3 = [i for i in range(len(influence_scores)) if i in top_sorted_indices[:num_bands_to_make_the_cut_3]]
print(f"Top {num_bands_to_make_the_cut_3} indices in their natural order:", original_indices_cut_3)

cut_off_data_to_plot_1 = (abs(influence_scores[sorted_indices[num_bands_to_make_the_cut_1-1]]) + abs(influence_scores[sorted_indices[num_bands_to_make_the_cut_1]])) / 2.0
print("Min impact factor of the set made the 1st cut: ", cut_off_data_to_plot_1)
cut_off_data_to_plot_2 = (abs(influence_scores[sorted_indices[num_bands_to_make_the_cut_2-1]]) + abs(influence_scores[sorted_indices[num_bands_to_make_the_cut_2]])) / 2.0
print("Min impact factor of the set made the 2nd cut: ", cut_off_data_to_plot_2)
cut_off_data_to_plot_3 = (abs(influence_scores[sorted_indices[num_bands_to_make_the_cut_3-1]]) + abs(influence_scores[sorted_indices[num_bands_to_make_the_cut_3]])) / 2.0
print("Min impact factor of the set made the 3rd cut: ", cut_off_data_to_plot_3)

num_bands_to_plot = 263 #275 #263
data_to_plot = influence_scores[:num_bands_to_plot]
data_to_plot_desc = "Influence Score"

# Plot the influence score for each input band
_, ax = plt.subplots(figsize=(15, 8))

max_val = max(data_to_plot)
min_val = min(data_to_plot)
print(f"min and max {data_to_plot_desc} are: ", min_val, max_val)
y_min = min_val - (max_val - min_val) * 0.1  # Formula for lower bound

x = bandindex_to_wavelength(np.arange(num_bands_to_plot))
# Create a smooth curve using cubic spline interpolation
x_smooth = np.linspace(x.min(), x.max(), num_bands_to_plot*16)  # More points for smoothness
spline = make_interp_spline(x, data_to_plot, k=3)  # k=3 for cubic spline
y_smooth = spline(x_smooth)

plt.plot(x_smooth, y_smooth, label=f"{data_to_plot_desc} (Smoothed)", color='b', linewidth=1)
plt.scatter(x, data_to_plot, marker='o', s=8, color='r', alpha=0.7, label=f"{data_to_plot_desc} (Original)")  # Optional: Show original points
# Draw a horizontal line at y_cut
plt.axhline(y=cut_off_data_to_plot_1, color='g', linestyle='--', linewidth=2, label=f"y = {cut_off_data_to_plot_1:.4f} ({num_bands_to_make_the_cut_1} in 1st cut)")
plt.axhline(y=cut_off_data_to_plot_2, color='y', linestyle='--', linewidth=2, label=f"y = {cut_off_data_to_plot_2:.4f} ({num_bands_to_make_the_cut_2} in 2nd cut)")
plt.axhline(y=cut_off_data_to_plot_3, color='c', linestyle='--', linewidth=2, label=f"y = {cut_off_data_to_plot_3:.4f} ({num_bands_to_make_the_cut_3} in 3rd cut)")

plt.ylim(y_min, max_val) 

# Add secondary x-axis
ax2 = ax.secondary_xaxis('top', functions=(wavelength_to_bandindex, bandindex_to_wavelength))
ax2.set_xlabel("Band Index")
# Move the label up
ax2.xaxis.set_label_position('top')  # Ensure it's at the top
ax2.xaxis.label.set_verticalalignment('top')  # Align closer to top of ticks
#ax2.xaxis.label.set_horizontalalignment('right')
ax2.xaxis.label.set_x(0)  # Move label to left

# Bar height and y-position just above the cut_off line
bar_height = (1.0 - cut_off_data_to_plot_1)*0.25
#bar_y_position = 1.02  # Adjust as needed
top_y_position = ax.get_ylim()[1] * cut_off_data_to_plot_1  # Position bars near the top of the chart
# Add bars using `ax.bar()`
ax.bar(bandindex_to_wavelength(np.array(original_indices_cut_1)), bar_height, width=2, color='green', alpha=0.25, align='center', bottom=top_y_position)
top_y_position = ax.get_ylim()[1] * cut_off_data_to_plot_2  # Position bars near the top of the chart
# Add bars using `ax.bar()`
bar_height = (cut_off_data_to_plot_1 - cut_off_data_to_plot_2)*0.5
ax.bar(bandindex_to_wavelength(np.array(original_indices_cut_2)), bar_height, width=2, color='yellow', alpha=0.25, align='center', bottom=top_y_position)

# Formatting
plt.xlabel("Band Wavelength (nm)")
ax.xaxis.label.set_x(0+0.035)  # Move label to left
#ax.xaxis.label.set_horizontalalignment('right')
plt.ylabel(data_to_plot_desc)
plt.title(f"{data_to_plot_desc} Calculated from Learned Band-Wise Mask Weights", fontsize=16)
plt.legend()
plt.grid(True)                                                

plt.tight_layout()
plt.show()

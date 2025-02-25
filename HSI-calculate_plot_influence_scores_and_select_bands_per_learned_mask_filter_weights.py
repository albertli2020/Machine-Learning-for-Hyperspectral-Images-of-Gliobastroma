import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

def calculate_impact_factors(sorted_lists, num_bands=275):
    # Initialize impact factors array with zeros
    impact_factors = np.zeros(num_bands, dtype=float)
    
    impact_multiplier = 0.875
    # Define scoring tiers
    scores = [1.0, impact_multiplier, impact_multiplier*impact_multiplier, impact_multiplier*impact_multiplier*impact_multiplier*impact_multiplier]  # Corresponding to the top 30, 60, 120, etc.
    thresholds = [30, 60, 120, 180]  # Cutoff indices for each tier

    # Process each sorted list
    for sorted_list in sorted_lists:
        for idx, band_index in enumerate(sorted_list):
            if idx < thresholds[0]:  
                impact_factors[band_index] += scores[0]
            elif idx < thresholds[1]:  
                impact_factors[band_index] += scores[1]
            elif idx < thresholds[2]:  
                impact_factors[band_index] += scores[2]
            elif idx < thresholds[3]:  
                impact_factors[band_index] += scores[3]
            else:
                break 
    
    # Average the impact factor over the 5 lists
    impact_factors /= len(sorted_lists)
    
    return impact_factors


# Load the saved mask parameters from CSV (assuming it's saved in last row)
csv_file_list = ["MaskFilterParam_2D_ROI_F11_3L.csv",
                 "MaskFilterParam_2D_ROI_F12_3L.csv",
                 "MaskFilterParam_2D_ROI_F13_3L.csv",
                 "MaskFilterParam_2D_ROI_F14_3L.csv",
                 "MaskFilterParam_2D_ROI_F15_3L.csv"]

num_bands_to_make_the_cut = 150

sorted_bands_list = []
num_folds = len(csv_file_list)
num_input_channels = 275

for csv_file in csv_file_list:
    # Read the CSV file
    df = pd.read_csv(csv_file, header=0)

    # Drop the first column (series labels)
    df = df.iloc[:, 1:]

    # Extract the last row (latest epoch mask parameters)
    mask_values = df.iloc[-1].values  # Convert to NumPy array
    # Sort all indices based on absolute mask values (high to low)
    #sorted_indices = sorted(range(len(mask_values)), key=lambda i: abs(mask_values[i]), reverse=True)
    sorted_indices = sorted(range(len(mask_values)), key=lambda i: mask_values[i], reverse=True)
    sorted_bands_list.append(sorted_indices)

    # Display results    
    print(f"Top {num_bands_to_make_the_cut} indices high to low:", sorted_indices[:num_bands_to_make_the_cut])    
    
impact_factor = calculate_impact_factors(sorted_bands_list, num_bands=num_input_channels)
sorted_indices = sorted(range(len(impact_factor)), key=lambda i: impact_factor[i], reverse=True)
num_bands_to_make_the_cut_3 = 110
num_bands_to_make_the_cut_2 = 56
num_bands_to_make_the_cut_1 = 32

# Take the top sorted indices
print(f"Top {num_bands_to_make_the_cut_1} indices in original order of impact factor:", sorted_indices[:num_bands_to_make_the_cut_1])
cut_off_data_to_plot_1 = (abs(impact_factor[sorted_indices[num_bands_to_make_the_cut_1-1]]) + abs(impact_factor[sorted_indices[num_bands_to_make_the_cut_1]])) / 2.0
print("Min impact factor of the set made the 1st cut: ", cut_off_data_to_plot_1)
cut_off_data_to_plot_2 = (abs(impact_factor[sorted_indices[num_bands_to_make_the_cut_2-1]]) + abs(impact_factor[sorted_indices[num_bands_to_make_the_cut_2-1]])) / 2.0
print("Min impact factor of the set made the 2nd cut: ", cut_off_data_to_plot_2)
cut_off_data_to_plot_3 = (abs(impact_factor[sorted_indices[num_bands_to_make_the_cut_3-1]]) + abs(impact_factor[sorted_indices[num_bands_to_make_the_cut_3-1]])) / 2.0
print("Min impact factor of the set made the 3rd cut: ", cut_off_data_to_plot_3)

num_bands_to_plot = 263
data_to_plot = impact_factor[:num_bands_to_plot]
data_to_plot_desc = "Impact Factor"

# Plot the average weights for each input channel
plt.figure(figsize=(15, 8))

max_val = max(data_to_plot)
min_val = min(data_to_plot)
print(f"min and max {data_to_plot_desc} are: ", min_val, max_val)
y_min = min_val - (max_val - min_val) * 0.1  # Formula for lower bound

x = np.arange(num_bands_to_plot) * 2.284 + 400.482
# Create a smooth curve using cubic spline interpolation
x_smooth = np.linspace(x.min(), x.max(), num_bands_to_plot*8)  # More points for smoothness
spline = make_interp_spline(x, data_to_plot, k=2)  # k=3 for cubic spline
y_smooth = spline(x_smooth)

plt.plot(x_smooth, y_smooth, label=f"{data_to_plot_desc} (Smoothed)", color='b', linewidth=1)
plt.scatter(x, data_to_plot, marker='o', s=8, color='r', alpha=0.7, label=f"{data_to_plot_desc} (Original)")  # Optional: Show original points
# Draw a horizontal line at y_cut
plt.axhline(y=cut_off_data_to_plot_1, color='g', linestyle='--', linewidth=2, label=f"y = {cut_off_data_to_plot_1:.4f} ({num_bands_to_make_the_cut_1} in 1st cut)")
plt.axhline(y=cut_off_data_to_plot_2, color='y', linestyle='--', linewidth=2, label=f"y = {cut_off_data_to_plot_2:.4f} ({num_bands_to_make_the_cut_2} in 2nd cut)")
plt.axhline(y=cut_off_data_to_plot_3, color='c', linestyle='--', linewidth=2, label=f"y = {cut_off_data_to_plot_3:.4f} ({num_bands_to_make_the_cut_3} in 3rd cut)")

plt.ylim(y_min, max_val) 

# Formatting
plt.xlabel("Input Band (nm)")
plt.ylabel(data_to_plot_desc)
plt.title(f"Learned Band-Wise {data_to_plot_desc}")
plt.legend()
plt.grid(True)                                                

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Define categories (metrics) and corresponding values for each model
categories = np.array(["AUC", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1-Score"])

# Data in a NumPy-friendly structure (each row corresponds to a fold)
values_3Layer_2D_275B = np.array([
    [0.9595, 0.8469, 0.7475, 0.9405, 0.9220, 0.8256],  # Fold F1
    [0.9847, 0.9221, 0.9717, 0.8907, 0.8449, 0.9062],  # Fold F2
    [0.8841, 0.8127, 0.7249, 0.8920, 0.8582, 0.7859],  # Fold F3
    [0.9758, 0.9492, 0.9118, 0.9882, 0.9878, 0.9483],  # Fold F4
    [0.9223, 0.8971, 0.8742, 0.9284, 0.9434, 0.9075],  # Fold F5
])

values_3Layer_2D_32B = np.array([
    [0.8909, 0.7923, 0.7337, 0.8475, 0.8191, 0.7740],  # Fold F1
    [0.9609, 0.8351, 0.9651, 0.7529, 0.7116, 0.8192],  # Fold F2
    [0.9553, 0.8943, 0.8986, 0.8904, 0.8809, 0.8897],  # Fold F3
    [0.9842, 0.9531, 0.9347, 0.9722, 0.9724, 0.9532],  # Fold F4
    [0.9532, 0.9061, 0.9107, 0.8998, 0.9254, 0.9180],  # Fold F5
])

values_2Layer_2D_275B = np.array([
    [0.9454, 0.8349, 0.7107, 0.9519, 0.9229, 0.8068],  # Fold F1
    [0.9846, 0.9338, 0.8676, 0.9756, 0.9574, 0.9103],  # Fold F2
    [0.8820, 0.7557, 0.8955, 0.6296, 0.6857, 0.7767],  # Fold F3
    [0.9817, 0.9088, 0.9640, 0.8510, 0.8712, 0.9153],  # Fold F4
    [0.9112, 0.8303, 0.8820, 0.7597, 0.8337, 0.8571],  # Fold F5
])

# Fold names
fold_names = np.array(["F1", "F2", "F3", "F4", "F5"])

def plot_model_perf_w_radar_and_table(model_name_str):
    # Compute angles for radar chart
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    angles_closed = np.append(angles, angles[0]+2*np.pi)  # Close the loop
    #angles = np.concatenate((angles, [angles[0]]))  # Close the radar loop

    if model_name_str == "3Layer_2D_275B":
        values_orig = values_3Layer_2D_275B
    elif model_name_str == "3Layer_2D_32B":
        values_orig = values_3Layer_2D_32B
    elif model_name_str == "2Layer_2D_275B":
        values_orig = values_2Layer_2D_275B
    else:
        return

    values_closed = np.hstack((values_orig, values_orig[:, [0]]))
    # Append first value to each row to close the loop
    #values = np.column_stack((values_orig, values_orig[:, [0]]))

    fig, ax = plt.subplots(1, 2, figsize=(18, 6), subplot_kw={'projection': 'polar'})

    # Plot Radar Chart
    for i, fold in enumerate(fold_names):
        #ax[0].plot(angles, values[i], label=fold)
        #ax[0].fill(angles, values[i], alpha=0.2)
        # Remove duplicate closing point for interpolation
        # Interpolate for smooth curve
        fine_angles = np.linspace(angles_closed[0], angles_closed[-1], 100)
        interp_func = interp1d(angles_closed, values_closed[i, :], kind='quadratic', fill_value="extrapolate")  # Smooth curve
        fine_values = interp_func(fine_angles)

        # Plot smooth line without fill
        ax[0].plot(fine_angles, fine_values, label=fold, linewidth=2)

    ax[0].set_ylim(0.3, 1.0)
    ax[0].set_xticks(angles)
    ax[0].set_xticklabels(categories)
    ax[0].legend(loc="upper right", bbox_to_anchor=(1.2, 1))
    ax[0].grid(True, linestyle="dotted")

    # Plot Table
    ax[1].axis("tight")
    ax[1].axis("off")

    # Format the values to 4 decimal places
    formatted_values = np.vectorize(lambda x: f"{x:.4f}")(values_orig)
    table_data = np.column_stack((fold_names, formatted_values))  # Add fold names

    # Create Table
    table = ax[1].table(
        cellText=table_data,
        colLabels=["Fold"] + list(categories),  # Column headers
        cellLoc="center",
        loc="center"
    )

    # Increase Font Size
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 1.5)  # Scale table size

    #table_data = np.vstack((categories, values_orig)).T  # Combine categories and values
    #table_data = np.column_stack((categories, values_orig))  # Correct shape
    #table_data = np.vstack((["Metric"] + fold_names.tolist(), np.column_stack((categories, values_orig.T))))  # Transposed
    #ax[1].table(cellText=table_data, cellLoc="center", loc="center", colWidths=[0.25] * (len(fold_names) + 1), fontsize=14)

    fig.canvas.manager.set_window_title(f"Model Performance of {model_name_str}")
    plt.show()

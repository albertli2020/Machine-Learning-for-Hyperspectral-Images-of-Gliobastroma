import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from spectral import open_image
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import time
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import matplotlib.cm as cm

input_bands = 826
win_size = 3
output_bands = (input_bands//win_size) #275
r_band, g_band, b_band = 425//win_size, 192//win_size, 109//win_size
patch_stride = 87
patch_overlap = 3
patch_size = (patch_stride+patch_overlap) #90

def load_and_calibrate(file_path, dark_reference_fp, white_reference_fp, show_ref_plots=False):
    """Load the hyperspectral image and normalize+calibrate it with references."""
    # Open the HSI image using the spectral library
    img = open_image(file_path)
    data = img.load()  # Shape: (800, 1004, 826)
    data = data.astype(np.uint16)

    dr_img = open_image(dark_reference_fp)
    dr_data = dr_img.load()  # Shape: (1, 1004, 826)
    dr_data = dr_data.astype(np.uint16)
    wr_img = open_image(white_reference_fp)
    wr_data = wr_img.load()  # Shape: (1, 1004, 826)
    wr_data = wr_data.astype(np.uint16)
    if show_ref_plots:
        #print(data.shape, dr_data.shape, wr_data.shape)
        array1 = dr_data[0, :, :]
        array2 = wr_data[0, :, :]

        # Set up the figure and create two subplots
        fig = plt.figure(figsize=(18, 8))

        # Create color maps for rows
        colors1 = cm.viridis(np.linspace(0, 1, array1.shape[0]))
        colors2 = cm.plasma(np.linspace(0, 1, array2.shape[0]))

        # First subplot for array1
        ax1 = fig.add_subplot(121, projection='3d')
        x = np.arange(array1.shape[1])  # Column indices
        y = np.arange(array1.shape[0])  # Row indices

        # Plot each row of array1 with a different color
        for i in range(array1.shape[0]):
            z = array1[i, :]
            ax1.plot(x, np.full(array1.shape[1], y[i]), z, color=colors1[i])

        ax1.set_xlabel('Band Index')
        ax1.set_ylabel('Row Index')
        ax1.set_zlabel('Reference Radiance')
        ax1.set_title("Dark")

        # Second subplot for array2
        ax2 = fig.add_subplot(122, projection='3d')
        x = np.arange(array2.shape[1])  # Column indices

        # Plot each row of array2 with a different color
        for i in range(array2.shape[0]):
            z = array2[i, :]
            ax2.plot(x, np.full(array2.shape[1], y[i]), z, color=colors2[i])

        ax2.set_xlabel('Band Index')
        ax2.set_ylabel('Row Index')
        ax2.set_zlabel('Reference Radiance')
        ax2.set_title("White")

        plt.tight_layout()
        plt.show()

    # Normalize with dark and white references (each is of shape (1, 1004, 826))
    #print("Before normalization:", data.dtype, dr_data.dtype, wr_data.dtype)
    #data = (data - dr_data) / (wr_data - dr_data)
    transmittance_data = np.zeros((data.shape[0], data.shape[1], input_bands), dtype=np.float32)
    # Process each line in HSI
    for i in range(data.shape[0]):
        # Extract the current line of radiance data
        R = data[i, :, :]  # Shape: (1004, 826)

        # Apply the normalization formula for each pixel in the line
        T = (R - dr_data) / (wr_data - dr_data + 1e-6)  # Add small epsilon to avoid division by zero
        T = T.astype(np.float32)
        # Clip values to keep them within the range [0, 1]
        T = np.clip(T, 0, 1)
        # Store in the transmittance array
        transmittance_data[i, :, :] = T

    #print("After normalization:", transmittance_data.dtype)
    return transmittance_data

def apply_window_average(data):
    reduced_data = np.zeros((data.shape[0], data.shape[1], output_bands), dtype=np.float32)
    j = 0
    for i in range(0, input_bands-win_size+1, win_size):
        # Extract the current line of radiance data
        R = data[:, :, i] + data[:, :, i+1] + data[:, :, i+2]
        R = R / win_size                                        
        # Store in the reduced array        
        reduced_data[:, :, j] = R
        j += 1
    return reduced_data

def crop_patches(image_data):
    """Divide the data into overlapping patches and save each patch."""
    patches = []
    h, w, c = image_data.shape
    
    for i in range(0, h-patch_stride+1, patch_stride):
        for j in range(0, w-patch_stride+1, patch_stride):
            patch = image_data[i:i+patch_size, j:j+patch_size, :]
            patches.append(patch)
    return patches

def is_patch_blinded(patch, show_plot_mean_brightness=False, threshold=0.9):
    """
    Determines if more than half of a hyperspectral image patch is close to white brightness.
    
    Args:
        patch (np.ndarray): The patch with shape (height, width, bands).
        threshold (float): The brightness threshold to consider a pixel "close to white".
                           This is based on normalized values, where 1.0 represents white.
    
    Returns:
        float: [0.0, 1.0] ratio of too-bright (over threshold) pixels to the patch area.
    """
    #print("Checking brightness:", patch.dtype)
    # Calculate the mean brightness across the spectral bands for each pixel
    mean_brightness = np.mean(patch, axis=2)

    if show_plot_mean_brightness:
        # Display the mean image
        plt.imshow(mean_brightness, cmap='gray')
        plt.colorbar()
        plt.title("Mean Over Bands (Axis=2)")
        plt.show()

    # Count the number of pixels above the brightness threshold
    bright_pixels = np.sum(mean_brightness >= threshold)
    
    # Determine if more than half of the pixels are bright
    total_pixels = patch.shape[0] * patch.shape[1]
    blinded_ratio = bright_pixels / total_pixels
    #print(f'{bright_pixels} out of {total_pixels} pixels are close to white brightness')
    return blinded_ratio

def is_patch_blinded_per_RGB_saturation(patch_data, saturation_threshold = 0.25): #0.3 # Example threshold (tune empirically)
    # Step 1: Extract RGB bands
    rgb_image = patch_data[:, :, [r_band, g_band, b_band]]  # Shape: (90, 90, 3)

    # Step 2: Convert to HSV format
    # Ensure values are normalized to [0, 1] for matplotlib.colors
    #rgb_normalized = rgb_image / rgb_image.max()
    hsv_image = mcolors.rgb_to_hsv(rgb_image)

    # Step 3: Binarize based on Saturation threshold    
    binary_mask = hsv_image[:, :, 1] < saturation_threshold

    # Check if the patch has too many "empty" pixels
    blinded_ratio = binary_mask.sum() / binary_mask.size
    return blinded_ratio

def save_patches(patches, output_dir, output_subdir):
    # Save each patch as a .npy file
    output_subdir_ng = output_subdir + '_NG'
    for idx, patch in enumerate(patches):
        if idx == 0:
            continue
        #ratio = is_patch_blinded(patch, idx)
        ratio = is_patch_blinded_per_RGB_saturation(patch)
        #if  ratio > 0.495:
        if  ratio > 0.5:
            os.makedirs(f'{output_dir}/{output_subdir_ng}', exist_ok=True)
            r = int(ratio * 100)         
            patch_filename = f"{output_dir}/{output_subdir_ng}/patch_{r}_{idx}.npy"
        else:
            os.makedirs(f'{output_dir}/{output_subdir}', exist_ok=True)
            patch_filename = f"{output_dir}/{output_subdir}/patch_{idx}.npy"
        #print("Patch shape and datatype:", patch.shape, patch.dtype)
        np.save(patch_filename, patch)#, allow_pickle=False)
    print(f"Saved {len(patches)} to ", output_dir)


def process_and_save_patches(hsi_file, dark_ref_file, white_ref_file, output_dir, output_subdir):
    print(f"Processing {hsi_file}, into {output_subdir}")
    data = load_and_calibrate(hsi_file, dark_ref_file, white_ref_file)
    
    # Apply windowed averaging on GPU (MPS) for band reduction
    reduced_data = apply_window_average(data)
    
    # Create patches and transfer back to CPU for saving
    patches = crop_patches(reduced_data)
    save_patches(patches, output_dir, output_subdir)
    

def getImagePathsWithLabelsTwoLevels(path, max_num_images_to_get):
    """Lists all directories in the given path."""
    paths = []
    labels = []
    names = []
    for entry in os.listdir(path):
        if not entry.startswith("P9"):
            continue
        level_one_path = os.path.join(path, entry)
        if os.path.isdir(level_one_path):            
            for level_two_entry in os.listdir(level_one_path): 
                if "ROI_" in level_two_entry:
                    full_path = os.path.join(level_one_path, level_two_entry)
                    if os.path.isdir(full_path):
                        paths.append(full_path) # + "/raw.hdr")
                        names.append(entry+'_'+level_two_entry)
                        if level_two_entry.endswith("_T"):
                            labels.append(1)  
                        else:
                            labels.append(0)
                if len(labels) >= max_num_images_to_get:
                    break
        if len(labels) >= max_num_images_to_get:
            break

    return paths, names, labels

root_dir = "PKG - HistologyHSI-GB/"
image_paths, image_names, image_labels = getImagePathsWithLabelsTwoLevels(root_dir, 560)
output_dir = f'ntp_{patch_size}_{patch_size}_{output_bands}'
os.makedirs(output_dir, exist_ok=True)

for idx, image_path in enumerate(image_paths):
    # Example usage
    hsi_file = os.path.join(image_path, "raw.hdr")
    dark_ref_file = os.path.join(image_path, "darkReference.hdr")
    white_ref_file = os.path.join(image_path, "whiteReference.hdr")
    # Run in parallel to maximize I/O and computation with a thread pool
    #with ThreadPoolExecutor() as executor:
    #    executor.submit(process_and_save_patches, hsi_file, dark_ref_file, white_ref_file, output_dir, image_names[idx])
    process_and_save_patches(hsi_file, dark_ref_file, white_ref_file, output_dir, image_names[idx])
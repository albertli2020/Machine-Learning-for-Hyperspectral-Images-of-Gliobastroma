import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

#loaded_array = np.load("ntp_90_90_275/P1_ROI_01_C01_T_NG/patch_55_91.npy")
loaded_array = np.load("ntp_90_90_275/P1_ROI_01_C01_T/patch_0.npy")
print(f"Loaded numpy array of shape {loaded_array.shape}, and type {loaded_array.dtype}")

# Example patch shape (90, 90, 275) of float32
# Step 1: Extract RGB bands
r_band, g_band, b_band = 425//3, 192//3, 109//3
rgb_image = loaded_array[:, :, [r_band, g_band, b_band]]  # Shape: (90, 90, 3)

# Step 2: Convert to HSV format
# Ensure values are normalized to [0, 1] for matplotlib.colors
#rgb_normalized = rgb_image / rgb_image.max()
hsv_image = mcolors.rgb_to_hsv(rgb_image)

# Step 3: Binarize based on Hue threshold
saturation_threshold = 0.25
binary_mask = hsv_image[:, :, 1] < saturation_threshold

# Check if the patch has too many "empty" pixels
background_ratio = binary_mask.sum() / binary_mask.size
threshold_ratio = 0.5  # Example threshold ratio (50% of pixels)
is_background = background_ratio > threshold_ratio

# Output Results
print(f"Background ratio: {background_ratio:.2%}")
print(f"Patch classified as {'Background' if is_background else 'Valid Content'}")

# Overlay the binary mask onto the RGB image
overlay_image = rgb_image.copy()
overlay_color = np.array([1.0, 0.0, 0.0])  # Red color for mask overlay
overlay_image[binary_mask] = (overlay_image[binary_mask] * 0.5 + overlay_color * 0.5)

# Plot the combined image
plt.figure(figsize=(8, 8))
plt.imshow(overlay_image)
plt.title('HSV Image with Binary Mask Overlay')
plt.axis('off')
plt.show()































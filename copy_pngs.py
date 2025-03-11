import os
import shutil

# Define source and destination directories
D1 = "/Users/albert/PKG - HistologyHSI-GB/P2"  
D2 = "/Volumes/X31/albert_HistoDL/PKG - HistologyHSI-GB/P2"

# Iterate through all subdirectories and files in D1/P1
for root, dirs, files in os.walk(D1):
    for file in files:
        if file == "rgb.png":
            # Get the relative path of the subfolder
            relative_path = os.path.relpath(root, D1)
            
            # Construct the corresponding destination folder
            dest_folder = os.path.join(D2, relative_path)
            
            # Create the destination subfolder if it doesn't exist
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
            
            # Construct the full path of the source and destination files
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_folder, file)
            
            # Copy the rgb.png file to the destination
            shutil.copy2(src_file, dest_file)
            print(f"Copied {src_file} to {dest_file}")

print("All files have been copied successfully.")

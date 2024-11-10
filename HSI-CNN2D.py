import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from spectral import open_image  # To open ENVI files
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import time
import os
from collections import Counter

def getImagePathsWithLabelsTwoLevels(path):
    """Lists all directories in the given path."""

    paths = []
    labels = []
    for entry in os.listdir(path):
        level_one_path = os.path.join(path, entry)
        if os.path.isdir(level_one_path):            
            for level_two_entry in os.listdir(level_one_path): 
                if "ROI_" in level_two_entry:
                    full_path = os.path.join(level_one_path, level_two_entry)
                    if os.path.isdir(full_path):
                        paths.append(full_path + "/raw.hdr")
                        if level_two_entry.endswith("_T"):
                            labels.append(1)
                        else:
                            labels.append(0) 
    return paths, labels

def getImagePathsWithLabelsOneLevel(path):
    """Lists all directories in the given path."""

    paths = []
    labels = []
    for entry in os.listdir(path):
        if "ROI_" in entry:
            full_path = os.path.join(path, entry)
            if os.path.isdir(full_path):
                paths.append(full_path + "/raw.hdr")
                if entry.endswith("_T"):
                    labels.append(1)
                else:
                    labels.append(0) 
    return paths, labels

# Custom Dataset Class for Hyperspectral Images

class HyperspectralDataset(Dataset):

    def __init__(self, image_paths, labels, patch_size=87, n_components=50, gpu_device=None):

        self.image_paths = image_paths

        self.labels = labels

        self.patch_size = patch_size

        self.n_components = n_components

        self.gpu_device = gpu_device

        # Initialize PCA model for dimensionality reduction (band reduction)

        self.pca = PCA(n_components=n_components)
        

        # Fit PCA using a sample from the dataset

        self._fit_pca()

    

    def _fit_pca(self):

        # Collect patches to fit PCA (this can be a subset for efficiency)

        all_patches = []

        for path in self.image_paths[:10]:  # Use a subset to fit PCA

            hsi_image = open_image(path).open_memmap() #load()

            h, w, c = hsi_image.shape

            for _ in range(5):  # Collect multiple patches from each image

                x = np.random.randint(0, h - self.patch_size)

                y = np.random.randint(0, w - self.patch_size)

                patch = hsi_image[x:x+self.patch_size, y:y+self.patch_size, :].reshape(-1, c)  # (num_pixels, channels)

                all_patches.append(patch)

        

        # Stack patches and fit PCA on flattened data (shape: [num_samples, channels])

        all_patches = np.concatenate(all_patches, axis=0)  # Shape: (total_pixels, original_channels)

        self.pca.fit(all_patches)

    

    def __len__(self):

        return len(self.image_paths)

    

    def __getitem__(self, idx):

        # Load hyperspectral image and extract patch
        #print("Working on image path:", self.image_paths[idx])
        time1 = time.time()

        hsi_image = open_image(self.image_paths[idx]).open_memmap() #load()  # Shape: (H, W, C)

        label = self.labels[idx]

        

        # Select a random patch

        h, w, c = hsi_image.shape

        x = np.random.randint(0, h - self.patch_size)

        y = np.random.randint(0, w - self.patch_size)

        patch = hsi_image[x:x+self.patch_size, y:y+self.patch_size, :].reshape(-1, c)  # Flatten to (num_pixels, channels)

        

        # Apply PCA for band reduction
        time2 = time.time()
        patch = self.pca.transform(patch)  # Shape: (num_pixels, n_components)

        time3 = time.time()
        # Reshape back to (n_components, H, W) and normalize

        patch = patch.reshape(self.patch_size, self.patch_size, self.n_components)

        patch = (patch - np.mean(patch, axis=(0, 1))) / (np.std(patch, axis=(0, 1)) + 1e-5) # Normalize each band

        

        # Transpose to (C, H, W) for PyTorch and convert to tensor

        patch = torch.tensor(patch.transpose(2, 0, 1), dtype=torch.float32, device=self.gpu_device)

        label = torch.tensor(label, dtype=torch.long, device=self.gpu_device)

        time4 = time.time()
        print("Load, PCA_transform, and Tensorize each took:", f'{time2-time1:.8f}', f'{time3-time2:.8f}', f'{time4-time3:.8f}', " seconds.")

        return patch, label



# Updated CNN Model for Hyperspectral Image Classification

class HyperspectralCNN(nn.Module):

    def __init__(self, input_channels=50, patch_size=87, gpu_device=None):  # Reduced channels after PCA

        super(HyperspectralCNN, self).__init__()

        self.gpu_device = gpu_device

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, device=gpu_device)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, device=gpu_device)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, device=gpu_device)

        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU()

        
        self.fc1_size = patch_size//8
        # Flattened size after convolution and pooling

        self.fc1 = nn.Linear(256 * self.fc1_size * self.fc1_size, 128)

        self.fc2 = nn.Linear(128, 2)  # Binary classification (tumor vs non-tumor)

        

    def forward(self, x):
        x = self.relu(self.pool(self.conv1(x)))  # Output shape: (64, 43, 43)

        x = self.relu(self.pool(self.conv2(x)))  # Output shape: (128, 21, 21)

        x = self.relu(self.pool(self.conv3(x)))  # Output shape: (256, 10, 10)

        x = x.view(-1, 256 *  self.fc1_size *  self.fc1_size)            # Flatten for fully connected layers

        x = self.relu(self.fc1(x))

        x = self.fc2(x)                          # No softmax here; use CrossEntropyLoss

        return x

start_time = time.time()

attempt_gpu = False
train_validate_test_batch_size = 32 #16
# Load data and prepare dataset
#root_dir = "D:/HistologyHSI/PKG - HistologyHSI-GB/P1/"
root_dir = "PKG - HistologyHSI-GB/"
patch_size = 174 #87 # Example patch size
num_spectral_channels_to_use = 25

if attempt_gpu:
    if torch.backends.mps.is_available():
        gpu_device = torch.device("mps")
        #x = torch.ones(1, device=gpu_device)
        print ("Model and data will be moved to Metal Performance backend") #, x)
    elif torch.cuda.is_available():
        gpu_device = "cuda"
        print("Model and data will be moved to nVidia GPUs")
    else:
        gpu_device = None
        print("Model and data will stay on CPUs")
else:
    gpu_device = None

image_paths, labels = getImagePathsWithLabelsTwoLevels(root_dir)

# print (str(len(image_paths)) + " " + str(len(labels)))
#image_paths = [raw_file_name1, raw_file_name2, raw_file_name3, raw_file_name4, raw_file_name5, raw_file_name6]  # List of image file paths
#labels = [0, 0, 1, 1, 1, 1]  # Corresponding labels

# Split data for training and validation
# Step 1: Split the data into 60% train and 40% temporary (validation + test)
train_paths, temp_paths, train_labels, temp_labels = train_test_split(image_paths, labels, test_size=0.4) #.2

# Step 2: Split the temporary set into 50% validation and 50% test (i.e., 10% validation and 10% test)
val_paths, test_paths, val_labels, test_labels = train_test_split(temp_paths, temp_labels, test_size=0.5, random_state=42)

train_dataset = HyperspectralDataset(train_paths, train_labels, patch_size, n_components=num_spectral_channels_to_use, gpu_device=gpu_device)

val_dataset = HyperspectralDataset(val_paths, val_labels, patch_size, n_components=num_spectral_channels_to_use, gpu_device=gpu_device)

test_dataset = HyperspectralDataset(test_paths, test_labels, patch_size, n_components=num_spectral_channels_to_use, gpu_device=gpu_device)

train_loader = DataLoader(train_dataset, batch_size=train_validate_test_batch_size, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=train_validate_test_batch_size, shuffle=False)

test_loader = DataLoader(test_dataset, batch_size=train_validate_test_batch_size, shuffle=False)



# Initialize Model, Loss, Optimizer

model = HyperspectralCNN(input_channels=num_spectral_channels_to_use, patch_size=patch_size, gpu_device=gpu_device)  # Adjust channels based on your dataset

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

if gpu_device is not None:
    model.to(gpu_device)

end_time1 = time.time()
elapsed_time = end_time1 - start_time
print(f"Elapsed time for data loading into model: {elapsed_time} seconds")

# Training Loop

epochs = 10
last_epoch_time = end_time1
for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    batch_index = 0
    for inputs, labels in train_loader:
        batch_index += 1
        print("Training on epoch#:", epoch+1, ", batch#:", batch_index)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()



    # Validation

    model.eval()

    val_loss = 0.0

    correct = 0

    total = 0

    with torch.no_grad():

        for inputs, labels in val_loader:

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    this_epoch_time = time.time()
    tl = running_loss/len(train_loader)
    vl = val_loss/len(val_loader)
    va = correct / total
    print(f"Epoch [{epoch+1}/{epochs}], "

          f"Train Loss: {tl:.4f}, "

          f"Val Loss: {vl:.4f}, "

          f"Val Accuracy: {100 * va:.2f}%,",
          f"Time Taken: {(this_epoch_time-last_epoch_time):.4f}")
    
    last_epoch_time = this_epoch_time

    if (tl < 0.0001) and (vl < 0.0001) and (va > 0.999):
        break


print("Training Complete.")

end_time2 = time.time()
elapsed_time = end_time2 - end_time1
print(f"Elapsed time for training model: {elapsed_time} seconds")

end_time1 = end_time2

all_labels = []
all_preds = []
model.eval()
with torch.no_grad():  # Disable gradient computation for efficiency
    for inputs, labels in test_loader:
        outputs = model(inputs)    
        _, predicted = torch.max(outputs, 1)
        # Collect all predictions and labels
        all_labels.extend(labels.cpu().numpy())  # Move to CPU if necessary
        all_preds.extend(predicted.cpu().numpy())                

end_time2 = time.time()
print(f"Time taken to complete Testing: {(end_time2-end_time1):.4f} seconds")

print("True label distribution:", Counter(all_labels))
print("Predicted label distribution:", Counter(all_preds))

cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
else:
    print(cm.shape)
    raise ValueError("The confusion matrix does not have a 2x2 shape, which is required for binary classification.")

# Accuracy
accuracy = accuracy_score(all_labels, all_preds)
print("Accuracy:", accuracy)

# Sensitivity (Recall or True Positive Rate)
sensitivity = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
print("Sensitivity (Recall):", sensitivity)

# Specificity (requires manual calculation)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
print("Specificity:", specificity)

# Precision
precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
print("Precision:", precision)

# F1 Score
f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
print("F1 Score:", f1)

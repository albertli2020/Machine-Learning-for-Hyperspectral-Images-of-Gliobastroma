import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from spectral import open_image  # To open ENVI files
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import time
import os


def getImagePathsWithLabels(path):
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

    def __init__(self, image_paths, labels, patch_size=87, n_components=50):

        self.image_paths = image_paths

        self.labels = labels

        self.patch_size = patch_size

        self.n_components = n_components

        

        # Initialize PCA model for dimensionality reduction (band reduction)

        self.pca = PCA(n_components=n_components)

        

        # Fit PCA using a sample from the dataset

        self._fit_pca()

    

    def _fit_pca(self):

        # Collect patches to fit PCA (this can be a subset for efficiency)

        all_patches = []

        for path in self.image_paths[:10]:  # Use a subset to fit PCA

            hsi_image = open_image(path).load()

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

        hsi_image = open_image(self.image_paths[idx]).load()  # Shape: (H, W, C)

        label = self.labels[idx]

        

        # Select a random patch

        h, w, c = hsi_image.shape

        x = np.random.randint(0, h - self.patch_size)

        y = np.random.randint(0, w - self.patch_size)

        patch = hsi_image[x:x+self.patch_size, y:y+self.patch_size, :].reshape(-1, c)  # Flatten to (num_pixels, channels)

        

        # Apply PCA for band reduction

        patch = self.pca.transform(patch)  # Shape: (num_pixels, n_components)

        

        # Reshape back to (n_components, H, W) and normalize

        patch = patch.reshape(self.patch_size, self.patch_size, self.n_components)

        patch = (patch - np.mean(patch, axis=(0, 1))) / (np.std(patch, axis=(0, 1)) + 1e-5) # Normalize each band

        

        # Transpose to (C, H, W) for PyTorch and convert to tensor

        patch = torch.tensor(patch.transpose(2, 0, 1), dtype=torch.float32)

        label = torch.tensor(label, dtype=torch.long)

        

        return patch, label



# Updated CNN Model for Hyperspectral Image Classification

class HyperspectralCNN(nn.Module):

    def __init__(self, input_channels=50):  # Reduced channels after PCA

        super(HyperspectralCNN, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU()

        

        # Flattened size after convolution and pooling

        self.fc1 = nn.Linear(256 * 10 * 10, 128)

        self.fc2 = nn.Linear(128, 2)  # Binary classification (tumor vs non-tumor)

        

    def forward(self, x):

        x = self.relu(self.pool(self.conv1(x)))  # Output shape: (64, 43, 43)

        x = self.relu(self.pool(self.conv2(x)))  # Output shape: (128, 21, 21)

        x = self.relu(self.pool(self.conv3(x)))  # Output shape: (256, 10, 10)

        x = x.view(-1, 256 * 10 * 10)            # Flatten for fully connected layers

        x = self.relu(self.fc1(x))

        x = self.fc2(x)                          # No softmax here; use CrossEntropyLoss

        return x

start_time = time.time()

# Load data and prepare dataset
root_dir = "D:/HistologyHSI/PKG - HistologyHSI-GB/P1/"

image_paths, labels = getImagePathsWithLabels(root_dir)

# print (str(len(image_paths)) + " " + str(len(labels)))
#image_paths = [raw_file_name1, raw_file_name2, raw_file_name3, raw_file_name4, raw_file_name5, raw_file_name6]  # List of image file paths
#labels = [0, 0, 1, 1, 1, 1]  # Corresponding labels

patch_size = 87 # Example patch size

# Split data for training and validation

train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2)

train_dataset = HyperspectralDataset(train_paths, train_labels, patch_size)

val_dataset = HyperspectralDataset(val_paths, val_labels, patch_size)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)




# Initialize Model, Loss, Optimizer

model = HyperspectralCNN(input_channels=50)  # Adjust channels based on your dataset

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

end_time1 = time.time()
elapsed_time = end_time1 - start_time
print(f"Elapsed time for data loading into model: {elapsed_time} seconds")

# Training Loop

epochs = 10

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for inputs, labels in train_loader:

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

    print(f"Epoch [{epoch+1}/{epochs}], "

          f"Train Loss: {running_loss/len(train_loader):.4f}, "

          f"Val Loss: {val_loss/len(val_loader):.4f}, "

          f"Val Accuracy: {100 * correct / total:.2f}%")


print("Training Complete.")

end_time2 = time.time()
elapsed_time = end_time2 - end_time1
print(f"Elapsed time for training model: {elapsed_time} seconds")
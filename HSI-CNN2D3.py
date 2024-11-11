import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from spectral import open_image
from sklearn.preprocessing import MinMaxScaler
import time
import os

# Custom Dataset class to handle multiple hyperspectral images
class HSIDataset(Dataset):
    def __init__(self, hsi_paths, labels, patch_size=87, spectral_channels=826, threshold=0.01):
        """
        hsi_paths: list of paths to ENVI format hyperspectral images
        labels: list of labels (0 or 1) indicating tumor vs non-tumor for each image
        patch_size: size of the square patch to extract from each image
        spectral_channels: number of spectral channels in the hyperspectral images
        threshold: variance threshold for blank space rejection
        """
        assert len(hsi_paths) == len(labels), "Mismatch between images and labels"
        
        self.hsi_paths = hsi_paths
        self.labels = labels
        self.patch_size = patch_size
        self.spectral_channels = spectral_channels
        self.threshold = threshold

        # Preprocess all images: normalization and spectral reduction
        self.processed_images = []
        for path in hsi_paths:
            hsi_data = open_image(path).load()       # Load ENVI file
            hsi_data = self.normalize_hsi(hsi_data)  # Normalize and reduce spectral dimension
            self.processed_images.append(hsi_data)
    
    def normalize_hsi(self, hsi_data):
        # Normalize each spectral channel
        scaler = MinMaxScaler()
        hsi_data = scaler.fit_transform(hsi_data.reshape(-1, self.spectral_channels)).reshape(hsi_data.shape)

        # Reduce spectral channels using 1D convolution
        hsi_data = self.reduce_spectral_dim(hsi_data, reduced_channels=64)
        return hsi_data
    
    def reduce_spectral_dim(self, data, reduced_channels=64):
        # Reshape data to [height * width, spectral_channels] and add a batch dimension for Conv1d
        height, width, spectral_channels = data.shape
        data_reshaped = torch.Tensor(data).view(-1, spectral_channels).unsqueeze(0)  # Shape: [1, height*width, spectral_channels]
        
        # Define Conv1d layer for spectral reduction
        conv1d = nn.Conv1d(spectral_channels, reduced_channels, kernel_size=1)  # Using kernel_size=1 to avoid spatial reduction
        with torch.no_grad():
            reduced_data = conv1d(data_reshaped.permute(0, 2, 1)).squeeze(0).permute(1, 0)  # Shape: [height*width, reduced_channels]
        
        # Reshape back to [height, width, reduced_channels]
        return reduced_data.view(height, width, reduced_channels).numpy()

    
    def __len__(self):
        # Return the total number of patches across all images
        total_patches = 0
        for img in self.processed_images:
            total_patches += (img.shape[0] - self.patch_size + 1) * (img.shape[1] - self.patch_size + 1)
        return total_patches
    
    def __getitem__(self, idx):
        # Find which image and patch the idx refers to
        current_idx = idx
        for img_index, img in enumerate(self.processed_images):
            num_patches = (img.shape[0] - self.patch_size + 1) * (img.shape[1] - self.patch_size + 1)
            if current_idx < num_patches:
                break
            current_idx -= num_patches
        
        # Calculate patch start position within the selected image
        x = current_idx % (img.shape[0] - self.patch_size + 1)
        y = current_idx // (img.shape[0] - self.patch_size + 1)
        patch = img[x:x+self.patch_size, y:y+self.patch_size, :]
        label = self.labels[img_index]  # Label for the current image

        # Reject blank patches based on variance threshold
        if np.var(patch) < self.threshold:
            return self.__getitem__((idx + 1) % len(self))
        
        return torch.Tensor(patch).permute(2, 0, 1), torch.tensor(label, dtype=torch.long)

# Define a CNN model for patch classification
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=64, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # Adaptive pooling to ensure a consistent feature map size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))  # Adjust output spatial size if needed
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Match the size based on adaptive pooling output
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.adaptive_pool(x)  # Ensures consistent feature map size
        
        x = x.view(x.size(0), -1)  # Flatten dynamically based on current batch size
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Parameters and data paths
start_time = time.time()

root_dir_name = "D:/HistologyHSI/PKG - HistologyHSI-GB/P1/"
raw_file_name1 = root_dir_name + 'ROI_02_C01_NT/raw.hdr'
raw_file_name2 = root_dir_name + 'ROI_02_C02_NT/raw.hdr'
raw_file_name3 = root_dir_name + 'ROI_01_C01_T/raw.hdr'
raw_file_name4 = root_dir_name + 'ROI_01_C02_T/raw.hdr'
raw_file_name5 = root_dir_name + 'ROI_01_C03_T/raw.hdr'
raw_file_name6 = root_dir_name + 'ROI_01_C04_T/raw.hdr'

hsi_paths = [raw_file_name1, raw_file_name2, raw_file_name3, raw_file_name4, raw_file_name5, raw_file_name6]  # List of image file paths
labels = [0, 0, 1, 1, 1, 1]  # Corresponding labels

batch_size = 16
learning_rate = 0.001
num_epochs = 10
validation_split = 0.2

# Initialize dataset
dataset = HSIDataset(hsi_paths, labels)
dataset_size = len(dataset)
print ("dataset size = " + str(dataset_size))
val_size = int(validation_split * dataset_size)
train_size = dataset_size - val_size

# Split the dataset into training and validation sets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

end_time1 = time.time()
elapsed_time = end_time1 - start_time
print(f"Elapsed time for data loading into model: {elapsed_time} seconds")

# Initialize model, criterion, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training and validation loop
best_val_loss = float('inf')  # Track the best validation loss for model saving

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_train_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
    
    avg_train_loss = running_train_loss / len(train_loader)

    # Validation phase
    model.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_val_loss = running_val_loss / len(val_loader)
    val_accuracy = correct / total

    # Print epoch metrics
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")

    # Save the model if validation loss improves
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Model saved with val loss: {best_val_loss:.4f}")

print("Training complete.")

end_time2 = time.time()
elapsed_time = end_time2 - end_time1
print(f"Elapsed time for training model: {elapsed_time} seconds")

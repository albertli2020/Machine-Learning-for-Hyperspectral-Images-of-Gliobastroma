import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from spectral import open_image

# Specify wavelengths and select only those in the specified ranges
wavelengths = np.array([440.5, 465.96, 498.71, 509.62, 556.91, 575.1, 593.29, 615.12,
                        636.94, 666.05, 698.79, 731.53, 884.32, 902.51])

# Loading ENVI hyperspectral images
def load_hyperspectral_image(path):
    img = open_image(path)  # Open the ENVI file
    img = img.load()  # Load the entire image array
    return img

# Normalizing hyperspectral data
def normalize(data):
    data_min = np.min(data, axis=(0, 1), keepdims=True)
    data_max = np.max(data, axis=(0, 1), keepdims=True)
    normalized_data = (data - data_min) / (data_max - data_min)
    return normalized_data

# Reducing channels to only specified wavelengths
def select_channels(data, wavelengths, target_ranges):
    selected_indices = []
    for min_wl, max_wl in target_ranges:
        indices = np.where((wavelengths >= min_wl) & (wavelengths <= max_wl))[0]
        selected_indices.extend(indices)
        print (str(len(indices)) + " " + str(len(selected_indices)))
    reduced_data = data[:, :, selected_indices]
    return reduced_data

# Extracting patches of size 87x87 and filtering out blank ones
def extract_patches(image, patch_size=87, threshold=0.5):
    patches = []
    for i in range(0, image.shape[0] - patch_size + 1, patch_size):
        for j in range(0, image.shape[1] - patch_size + 1, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size, :]
            # Filter out blank patches
            if np.mean(patch) > threshold:
                patches.append(patch)
    return patches

# CNN model definition
class TumorClassifierCNN(nn.Module):
    def __init__(self, input_channels):
        super(TumorClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * (patch_size // 8) * (patch_size // 8), 128)
        self.fc2 = nn.Linear(128, 2)  # Binary classification: tumor vs. non-tumor
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * (patch_size // 8) * (patch_size // 8))
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Load and preprocess the dataset
path = "D:/HistologyHSI/PKG - HistologyHSI-GB/P1/ROI_01_C01_T/raw.hdr"
raw_image = load_hyperspectral_image(path)
target_ranges = [(440.5, 465.96), (498.71, 509.62), (556.91, 575.1), 
                 (593.29, 615.12), (636.94, 666.05), (698.79, 731.53), (884.32, 902.51)]

reduced_image = select_channels(raw_image, wavelengths, target_ranges)
normalized_image = normalize(reduced_image)
patches = extract_patches(normalized_image)

print ("the number of patches is " + str(len(patches)))
l = [1]*len(patches)
# Convert patches and labels to tensors
patch_size = 87
tensor_patches = torch.tensor(patches).permute(0, 3, 1, 2).float()  # Reshape to (N, C, H, W)
labels = torch.tensor(l)

# Dataset and DataLoader
dataset = data.TensorDataset(tensor_patches, labels)
train_loader = data.DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, loss function, and optimizer
input_channels = tensor_patches.shape[1]
model = TumorClassifierCNN(input_channels=input_channels)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}")
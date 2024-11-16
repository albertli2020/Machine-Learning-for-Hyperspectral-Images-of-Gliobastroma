import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from spectral import open_image
from spectral import envi
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import scipy
import os
import glob
import time
from collections import Counter

# Define target wavelength ranges
wavelengths = np.array([440.5, 465.96, 498.71, 509.62, 556.91, 575.1, 593.29, 615.12,
                        636.94, 666.05, 698.79, 731.53, 884.32, 902.51])
target_ranges = [(440.5, 465.96), (498.71, 509.62), (556.91, 575.1), 
                 (593.29, 615.12), (636.94, 666.05), (698.79, 731.53), (884.32, 902.51)]
patch_size = 87

def getImagePathsWithLabels(root_dir):
    labels = []
    image_paths = glob.glob(os.path.join(root_dir, "./ROI_*/"), recursive=False)
    for image_path in image_paths:
            # path = os.path.basename(image_path)
            # print(image_path)
            if "_NT" in image_path:
                labels.append(0)
            else:
                labels.append(1)
    return image_paths, labels


def readHyperspectralImage(file_name):
    """Function used to read an ENVI hyperspectral image, and return a numpy
    data structure containing the 3D hyperspectral cube

    Parameters
    ----------
    file_name : str
        The file location of the ENVI header related to the hyperspectral data.

    Returns
    -------
    hyperspectral_data: ndarray
        3-dimensional numpy array containing the hyperspectral cube
    wavelengths: ndarray
        1-dimensional numpy array containing the spectral bands corresponding to
        the hyperspectral cube
    """

    ENVI_structure = open_image(file_name)
    hyperspectral_image = ENVI_structure.open_memmap()
    # wavelengths = ENVI_structure.bands.centers
    return hyperspectral_image


def getCalibratedImage(image_path):
    raw_hyperspectral_image = readHyperspectralImage(image_path + "/raw.hdr")
    white_reference  = readHyperspectralImage(image_path + "/whiteReference.hdr")
    dark_reference = readHyperspectralImage(image_path + "/darkReference.hdr")
    return (raw_hyperspectral_image-dark_reference)/(white_reference-dark_reference)


def band_reduction(hyperspectral_image, wavelengths, n = 3):
    """Perform the band reduction to the hyperspectral cube

    Parameters
    ----------
    input_data : ndarray
        Array containing the raw hyperspectral cube
    wavelengths : ndarray
        1-dimensional numpy array containing the spectral bands corresponding to
        the hyperspectral cube
    n : Integer
        Width of the window used for the moving average window.
    Returns
    -------
    band_reduced_cube: ndarray
        3-dimensional numpy array containing the hyperspectral cube after the
        band reduction
    wavelength_reduced: ndarray
        1-dimensional numpy array containing the spectral bands corresponding to
        the hyperspectral cube after the band reduction
    """
    w = np.ones(n)/n
    moving_averaged_image = scipy.ndimage.convolve1d(hyperspectral_image, w, axis=2)
    band_reduced_image = moving_averaged_image[:,:,1:-1:n]
    wavelength_reduced = wavelenghts[1:-1:n]
    return band_reduced_image, wavelength_reduced

# Hyperspectral Dataset Class
class HyperspectralDataset(Dataset):
    def __init__(self, image_paths, labels, wavelengths, target_ranges, patch_size=87, blank_threshold=0.5, gpu_device=None):
        self.root_dir = root_dir
        self.gpu_device = gpu_device
        self.wavelengths = wavelengths
        self.target_ranges = target_ranges
        self.patch_size = patch_size
        self.blank_threshold = blank_threshold
        self.image_paths = image_paths
        self.labels = labels

    
    def load_hyperspectral_image(self, path):
        img = open_image(path).load()
        return img
    
    def select_channels(self, data):
        selected_indices = []
        for min_wl, max_wl in self.target_ranges:
            indices = np.where((self.wavelengths >= min_wl) & (self.wavelengths <= max_wl))[0]
            selected_indices.extend(indices)
        return data[:, :, selected_indices]
    
    def normalize(self, data):
        data_min = np.min(data, axis=(0, 1), keepdims=True)
        data_max = np.max(data, axis=(0, 1), keepdims=True)
        return (data - data_min) / (data_max - data_min)
    
    def extract_patches(self, image):
        patches = []
        for i in range(0, image.shape[0] - self.patch_size + 1, self.patch_size):
            for j in range(0, image.shape[1] - self.patch_size + 1, self.patch_size):
                patch = image[i:i+self.patch_size, j:j+self.patch_size, :]
                if np.mean(patch) > self.blank_threshold:
                    patches.append(patch)
        return patches
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        # raw_image = self.load_hyperspectral_image(image_path)
        # reduced_image = self.select_channels(raw_image)
        reduced_image = getCalibratedImage(image_path)
        normalized_image = self.normalize(reduced_image)
        patches = self.extract_patches(normalized_image)
        
        patches_np = np.array(patches)
        tensor_patches = torch.from_numpy(patches_np).permute(0, 3, 1, 2).float()
        # Convert patches and labels to tensors
        # tensor_patches = torch.tensor(patches).permute(0, 3, 1, 2).float()
        tensor_labels = torch.full((tensor_patches.shape[0],), label, dtype=torch.long, device=self.gpu_device)
        # print ("Tensor size: " + str(len(tensor_patches)) + " " + str(len(tensor_labels)))
        return tensor_patches, tensor_labels


# Define the CNN model (same as before)
class TumorClassifierCNN(nn.Module):
    def __init__(self, input_channels, gpu_device=None):
        super(TumorClassifierCNN, self).__init__()
        self.gpu_device = gpu_device
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, device=gpu_device)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, device=gpu_device)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, device=gpu_device)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * (patch_size // 8) * (patch_size // 8), 128)
        self.fc2 = nn.Linear(128, 2)  # Binary classification
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        # x = x.view(-1, 128 * (patch_size // 8) * (patch_size // 8))
        x = x.reshape(-1, 128 * (patch_size // 8) * (patch_size // 8))
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

start_time = time.time()

attempt_gpu = False

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

# root_dir = "D:/HistologyHSI/PKG - HistologyHSI-GB"
root_dir = "C:/projects/P2"
image_paths, labels = getImagePathsWithLabels(root_dir)
# Initialize the dataset and DataLoader
# dataset = HyperspectralDataset(image_paths, labels, wavelengths, target_ranges, patch_size, blank_threshold=0.5, gpu_device=gpu_device)
# train_loader = data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda x: (torch.cat([item[0] for item in x]), torch.cat([item[1] for item in x])))
input_channels = 826//3

# Split data for training and validation
# Step 1: Split the data into 60% train and 40% temporary (validation + test)
train_paths, temp_paths, train_labels, temp_labels = train_test_split(image_paths, labels, test_size=0.3) #.2

# Step 2: Split the temporary set into 50% validation and 50% test (i.e., 10% validation and 10% test)
val_paths, test_paths, val_labels, test_labels = train_test_split(temp_paths, temp_labels, test_size=0.7)

train_dataset = HyperspectralDataset(train_paths, train_labels, wavelengths, target_ranges, patch_size, gpu_device=gpu_device)

val_dataset = HyperspectralDataset(val_paths, val_labels, wavelengths, target_ranges, patch_size, gpu_device=gpu_device)

test_dataset = HyperspectralDataset(test_paths, test_labels, wavelengths, target_ranges, patch_size, gpu_device=gpu_device)

print("Image Dirstribution: " + str(len(train_paths)) + " " + str(len(val_paths)) + " " + str(len(test_paths)))

print("Dataset Dirstribution: " + str(len(train_dataset)) + " " + str(len(val_dataset)) + " " + str(len(test_dataset)))

print("True label distribution:", Counter(train_labels), Counter(val_labels), Counter(test_labels))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: (torch.cat([item[0] for item in x]), torch.cat([item[1] for item in x])))

val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=lambda x: (torch.cat([item[0] for item in x]), torch.cat([item[1] for item in x])))

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=lambda x: (torch.cat([item[0] for item in x]), torch.cat([item[1] for item in x])))
# Initialize model, loss function, and optimizer

model = TumorClassifierCNN(input_channels=input_channels, gpu_device=gpu_device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

if gpu_device is not None:
    model.to(gpu_device)

end_time1 = time.time()
elapsed_time = end_time1 - start_time
print(f"Elapsed time for data loading into model: {elapsed_time} seconds")

# Training loop
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
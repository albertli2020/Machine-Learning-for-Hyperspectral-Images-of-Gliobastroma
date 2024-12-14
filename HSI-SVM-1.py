import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import time
from processData import *

# Define Dataset
class HyperspectralDataset(Dataset):
    def __init__(self, image_files, labels):
        self.image_files = image_files
        # self.data = []
        self.labels = labels
        """
        # Load data and labels
        for label in [0, 1]:  # 0: non-tumor, 1: tumor
            class_dir = self.root_dir / str(label)
            for npy_path in class_dir.glob("*.npy"):  # Assuming patches are stored as .npy files
                self.data.append(npy_path)
                self.labels.append(label)
        """
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        npy_path = self.image_files[idx]
        label = self.labels[idx]
        patch_np = np.load(npy_path)  # Load the numpy array
        patch = patch_np[:87, :87, :]
        assert patch.shape == (87, 87, 275), f"Unexpected patch size: {patch.shape} at {npy_path}"
        patch = torch.tensor(patch, dtype=torch.float32)  # Convert to PyTorch tensor
        patch = patch.permute(2, 0, 1)  # Rearrange to (Bands, Height, Width) for PyTorch
        return patch, label

# Define VSM Model for 3D Hyperspectral Input
class VSMModel(nn.Module):
    def __init__(self):
        super(VSMModel, self).__init__()
        self.conv1 = nn.Conv2d(275, 64, kernel_size=3, padding=1)  # 275 bands as input channels
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 43 * 43, 256)  # Flattened feature map size
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)  # Binary classification (non-tumor, tumor)
    
    def forward(self, x):
        x = self.conv1(x)  # (275, 87, 87) -> (64, 87, 87)
        x = self.relu(x)
        x = self.pool(x)   # (64, 87, 87) -> (64, 43, 43)
        x = self.conv2(x)  # (64, 43, 43) -> (128, 43, 43)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten: (128, 43, 43) -> (128 * 43 * 43)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Set up data loading
def prepare_data(root_dir, batch_size=32, val_size=0.1, test_size=0.2, data_precentage=0.02):
    t_images, nt_images = generatePatchLists(root_dir)
    training_set, training_labels, val_set, val_labels, test_set, test_labels = HSIDataSetSplit(t_images, nt_images,val_size, test_size, data_precentage )
    training_dataset = HyperspectralDataset(training_set, training_labels)
    val_dataset = HyperspectralDataset(val_set, val_labels)
    test_dataset = HyperspectralDataset(test_set, test_labels)
    #train_indices, val_indices = train_test_split(range(len(dataset)), test_size=test_size, random_state=42)
    
    # train_loader = DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_indices))
    # val_loader = DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(val_indices))
    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# Training and Validation Function
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for patches, labels in train_loader:
            patches, labels = patches.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(patches)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for patches, labels in val_loader:
                patches, labels = patches.to(device), labels.to(device)
                outputs = model(patches)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        """
        print("Classification Report:")
        print(classification_report(labels, predicted))

        cm = confusion_matrix(labels, predicted)  # Assuming y_val contains true validation labels
        print("Confusion Matrix:")
        print(cm)
        """

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {100 * correct/total:.2f}%")

# Prediction Function
def predict(model, test_loader, device="cpu"):

    total_labels = []
    total_preds = []
    model.eval()
    with torch.no_grad():
        for patches, labels in test_loader:
            patches, labels = patches.to(device), labels.to(device)
            # print(len(patches))
            outputs = model(patches)
            _, predicted = torch.max(outputs, 1)
            total_labels.extend(labels)
            total_preds.extend(predicted)

    #print("test label distribution:", Counter(total_labels))
    #print("Predicted label distribution:", Counter(total_preds))
    print("Classification Report:")
    print(classification_report(total_labels, total_preds))


"""
def predict(model, patch, device="cpu"):
    model.eval()
    with torch.no_grad():
        patch = patch.to(device)
        output = model(patch.unsqueeze(0))  # Add batch dimension
        _, predicted = torch.max(output, 1)
        return predicted.item()
"""
# Main Execution
if __name__ == "__main__":
    root_dir = "C:/HSI-ML/ntp_90_90_275"  # Replace with the path to your dataset
    batch_size = 32
    val_size = 0.1
    test_size = 0.2
    data_precentage = 1
    epochs = 10
    learning_rate = 0.0001
    device = torch.device("cpu")  # Force CPU usage

    start_time = time.time()

    # Prepare data
    train_loader, val_loader, test_loader = prepare_data(root_dir, batch_size, val_size, test_size, data_precentage)

    # Initialize model
    model = VSMModel().to(device)

    # Train model
    train_model(model, train_loader, val_loader, epochs=epochs, lr=learning_rate, device=device)

    # Save model
    torch.save(model.state_dict(), "vsm_model.pth")

    # Load model for prediction
    # model.load_state_dict(torch.load("vsm_model.pth"))
    # model.to(device)
    predict(model, test_loader, device)
    # Example prediction (replace with an actual patch)
    # patch = np.load("path/to/example_patch.npy")  # Load an example patch
    # patch_tensor = torch.tensor(patch, dtype=torch.float32).permute(2, 0, 1).to(device)
    # label = predict(model, patch_tensor, device=device)
    # print(f"Predicted label: {label}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
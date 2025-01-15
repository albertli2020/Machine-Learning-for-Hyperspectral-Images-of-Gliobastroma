import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, log_loss, classification_report, roc_auc_score
from collections import Counter
import time
import os
import glob
import random

def labelHSIDataSet(data_files):

    labels = []

    for file in data_files:
        p = os.path.dirname(file)
        # print(p)
        if p.endswith("_T") or p.endswith("_T_50G"):  
            labels.append(1)
        else:
            labels.append(0)

    return labels

def find_nth_occurrence(string, char, n):
    index = -1
    for i in range(n):
        index = string.find(char, index + 1)
        if index == -1:
            return -1
    return index

def generateLists(path):

    patch_list = []
    t_roi_list = []
    nt_roi_list =[]
    t_roi_dir = {}
    nt_roi_dir = {}

    for entry in os.listdir(path):
        next_level_path = os.path.join(path, entry)
        if os.path.isdir(next_level_path):
            paths = glob.glob(os.path.join(next_level_path, "*.npy"), recursive=False)
            if not entry.startswith("P6") and entry.endswith("_T"):            
                patch_list.extend(paths)
                index = find_nth_occurrence(entry, "_", 3)
                roi = entry[:index]
                if roi in t_roi_dir:
                    t_roi_dir[roi].extend(paths)
                else:
                    t_roi_dir[roi] = paths
            elif not entry.startswith("P6") and entry.endswith("_NT"):            
                patch_list.extend(paths)
                index = find_nth_occurrence(entry, "_", 3)
                roi = entry[:index]
                if roi in nt_roi_dir:
                    nt_roi_dir[roi].extend(paths)
                else:
                    nt_roi_dir[roi] = paths

    t_roi_list = t_roi_dir.keys()
    nt_roi_list = nt_roi_dir.keys()        
    return patch_list, t_roi_list, nt_roi_list, t_roi_dir, nt_roi_dir


def splitDataset(patch_list, t_roi_dir, nt_roi_dir,  test_roi_list, val_roi_list):
    training_set = []
    val_set = []
    test_set = []

    for roi in test_roi_list:
        if roi in t_roi_dir:
            test_set.extend(t_roi_dir[roi])
        elif roi in nt_roi_dir:
            test_set.extend(nt_roi_dir[roi])
    
    for roi in val_roi_list:
        if roi in t_roi_dir:
            val_set.extend(t_roi_dir[roi])
        elif roi in nt_roi_dir:
            val_set.extend(nt_roi_dir[roi])

    training_set = list(set(patch_list) - set(test_set) - set(val_set))
    random.shuffle(training_set)

    return training_set, val_set, test_set

test_roi_sets = [
    ['P1_ROI_02', 'P3_ROI_02', 'P7_ROI_02', 'P5_ROI_01', 'P9_ROI_01'],
    ['P1_ROI_03', 'P2_ROI_02', 'P5_ROI_02', 'P8_ROI_03', 'P11_ROI_01'],
    ['P1_ROI_04', 'P2_ROI_03', 'P5_ROI_04', 'P7_ROI_03', 'P12_ROI_01'],
    ['P5_ROI_03', 'P8_ROI_01', 'P1_ROI_01', 'P3_ROI_01', 'P13_ROI_01'],
    ['P7_ROI_01', 'P8_ROI_02', 'P2_ROI_01', 'P9_ROI_02', 'P10_ROI_01']
]

val_roi_sets = [
['P1_ROI_01','P5_ROI_03'],
['P2_ROI_01','P8_ROI_02'],
['P5_ROI_01','P1_ROI_03'],
['P9_ROI_01','P2_ROI_02','P2_ROI_03'],
['P7_ROI_03','P1_ROI_03']
]


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
        # R band indice
        r_band = 425 // 3

        # Extract RGB bands
        patch_r = patch_np[:87, :87, r_band]
        assert patch_r.shape == (87, 87), f"Unexpected patch size: {patch_rgb.shape} at {npy_path}"

        # Convert to PyTorch tensor and add the band dimension
        patch = torch.tensor(patch_r, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 87, 87)
        # patch = patch.permute(2, 0, 1)  # Rearrange to (Bands, Height, Width) for PyTorch
        return patch, label

# Define VSM Model for 3D Hyperspectral Input
class VSMModel(nn.Module):
    def __init__(self):
        super(VSMModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)  # 3 bands as input channels
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


# Training and Validation Function
def train_model(model, train_loader, val_loader, epochs=10, lr=0.0001, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_accuracy = 0
    best_loss = float('inf') 
    
    for epoch in range(epochs):
        y_true_trn = []
        y_pred_trn = []
        model.train()
        train_loss = 0
        trn_correct = 0
        trn_total = 0
        for patches, labels in train_loader:
            patches, labels = patches.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(patches)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            trn_correct += (predicted == labels).sum().item()
            trn_total += labels.size(0)
            y_true_trn.extend(labels.cpu().numpy())
            y_pred_trn.extend(predicted.cpu().numpy())
        
        #trn_loss = log_loss(y_true_trn, y_pred_trn)
        #trn_accuracy = accuracy_score(y_true_trn, y_pred_trn)

        y_true_val = []
        y_pred_val = []
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for patches, labels in val_loader:
                patches, labels = patches.to(device), labels.to(device)
                outputs = model(patches)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(predicted.cpu().numpy())

        #val_loss = log_loss(y_true_val, y_pred_val)
        #val_accuracy = accuracy_score(y_true_val, y_pred_val)
        #print("Classification Report:")
        #print(classification_report(total_labels, total_preds))

        #cm = confusion_matrix(labels, predicted)  # Assuming y_val contains true validation labels
        #print("Confusion Matrix:")
        #print(cm)
        
        
        trn_loss = train_loss/len(train_loader)
        trn_accuracy = trn_correct/trn_total
        val_loss = val_loss/len(val_loader)
        val_accuracy = val_correct/val_total

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {trn_loss:.4f}, Train Accuracy: {100 * trn_accuracy:.2f}%,"
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {100 * val_accuracy:.2f}%")

        epoch_avg_accuracy = (trn_accuracy + val_accuracy)/2
        epoch_avg_loss = (trn_loss + val_loss)/2

        if (epoch_avg_loss <= best_loss) and (epoch_avg_accuracy >= best_accuracy):
            best_loss = epoch_avg_loss
            best_accuracy = epoch_avg_accuracy
            torch.save(model.state_dict(), "svm_model_roi_sb.pth")
            print(f"Trained model saved w/ avg loss: {best_loss:.4f}, avg accuracy: {100*best_accuracy:.4f}%")

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
            total_labels.extend(labels.cpu().numpy())
            total_preds.extend(predicted.cpu().numpy())

    print("test label distribution:", Counter(total_labels))
    print("Predicted label distribution:", Counter(total_preds))
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
    data_precentage = 1
    epochs = 10
    learning_rate = 0.0001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    start_time = time.time()

    patch_list, t_roi_list, nt_roi_list, t_roi_dir, nt_roi_dir = generateLists(root_dir)
    training_set, val_set, test_set = splitDataset(patch_list, t_roi_dir, nt_roi_dir, test_roi_sets[3], val_roi_sets[3])
    training_labels = labelHSIDataSet(training_set)
    val_labels = labelHSIDataSet(val_set)
    test_labels = labelHSIDataSet(test_set)

    print("Total number of training data:", len(training_set))
    print("Trainging label distribution:", Counter(training_labels))
    print("Total number of validation data:", len(val_set))
    print("Predicted label distribution:", Counter(val_labels))
    print("Total number of test data:", len(test_set))
    print("Predicted label distribution:", Counter(test_labels))


    training_dataset = HyperspectralDataset(training_set, training_labels)
    val_dataset = HyperspectralDataset(val_set, val_labels)
    test_dataset = HyperspectralDataset(test_set, test_labels)

    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # Initialize model
    model = VSMModel().to(device)

    # Train model
    train_model(model, train_loader, val_loader, epochs=epochs, lr=learning_rate, device=device)

    # Save model
    # torch.save(model.state_dict(), "vsm_model_roi.pth")

    # Load model for prediction
    model.load_state_dict(torch.load("svm_model_roi_sb.pth"))
    model.to(device)
    predict(model, test_loader, device)
    # Example prediction (replace with an actual patch)
    # patch = np.load("path/to/example_patch.npy")  # Load an example patch
    # patch_tensor = torch.tensor(patch, dtype=torch.float32).permute(2, 0, 1).to(device)
    # label = predict(model, patch_tensor, device=device)
    # print(f"Predicted label: {label}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
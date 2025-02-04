import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, log_loss, classification_report, roc_auc_score, hinge_loss
from collections import Counter
import matplotlib.pyplot as plt
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
            labels.append(1.0)
        else:
            labels.append(0.0)

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
    random.shuffle(test_set)
    random.shuffle(val_set)
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
        self.labels = labels
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        npy_path = self.image_files[idx]
        label = self.labels[idx]
        patch_np = np.load(npy_path)  # Load the numpy array
        patch = patch_np[:87, :87, :]
        assert patch.shape == (87, 87, 275), f"Unexpected patch size: {patch.shape} at {npy_path}"
        patch = torch.tensor(patch, dtype=torch.float32)  # Convert to PyTorch tensor
        # patch = patch.permute(2, 0, 1)  # Rearrange to (Bands, Height, Width) for PyTorch
        patch = patch.reshape(-1)
        #label = torch.FloatTensor(label).squeeze()
        return patch, label

class SVM(nn.Module):
    def __init__(self, input_dim):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

# Training and Validation Function

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()      

    epoch_loss = running_loss / len(dataloader.dataset)
    
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Main Execution
if __name__ == "__main__":
    root_dir = "C:/HSI-ML/ntp_90_90_275"  # Replace with the path to your dataset
    batch_size = 100
    #epochs = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    start_time = time.time()

    patch_list, t_roi_list, nt_roi_list, t_roi_dir, nt_roi_dir = generateLists(root_dir)

    for i in range(len(test_roi_sets)):
        fold = "fold" + str(i+1)
        model_file = fold + "_best_svm_roi_model.pth"

        training_set, val_set, test_set = splitDataset(patch_list, t_roi_dir, nt_roi_dir, test_roi_sets[i], val_roi_sets[i])
        training_labels = labelHSIDataSet(training_set)
        val_labels = labelHSIDataSet(val_set)
        test_labels = labelHSIDataSet(test_set)

        training_dataset = HyperspectralDataset(training_set, training_labels)
        val_dataset = HyperspectralDataset(val_set, val_labels)
        test_dataset = HyperspectralDataset(test_set, test_labels)

        train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


        # Initialize model
        input_dim = 87 * 87 * 275
        lr = 0.004
        model = SVM(input_dim).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training loop
        num_epochs = 10
        best_val_acc = 0.0
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0

        print("\n")
        print(f"{fold}: Training started with learning rate={lr}.")

        print("Total number of training data:", len(training_set))
        print("Trainging label distribution:", Counter(training_labels))
        print("Total number of validation data:", len(val_set))
        print("Predicted label distribution:", Counter(val_labels))
        print("Total number of test data:", len(test_set))
        print("Predicted label distribution:", Counter(test_labels))
        
        for epoch in range(num_epochs):

            start = time.time()
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
        
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
            f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

            end = time.time()
            elapsed = end - start
            print(f"Elapsed time: {elapsed:.2f} seconds")

            if val_loss < best_val_loss and val_acc > best_val_acc:
                best_val_loss = val_loss
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), model_file)
                print(f"New best model saved with validation loss: {best_val_loss:.4f} and validation accuracy: {best_val_acc:.4f}")
            else:
                patience_counter += 1
        
            # Early stopping
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        print(f"{fold}: Training completed.")

        # Testing
        model.load_state_dict(torch.load(model_file))
        model.eval()

        all_preds = []
        all_labels = []
        all_scores = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predicted = (outputs > 0.5).float()
                all_scores.extend(outputs.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Calculate metrics
        auc = roc_auc_score(all_labels, all_scores)
        accuracy = accuracy_score(all_labels, all_preds)
        sensitivity = recall_score(all_labels, all_preds)
        specificity = recall_score(all_labels, all_preds, pos_label=0)
        precision = precision_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)

        print(f"AUC: {auc:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Sensitivity: {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"Precision: {precision:.4f}")
        print("Confusion Matrix:")
        print(cm)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total Elapsed time: {elapsed_time:.2f} seconds")
        print("\n")

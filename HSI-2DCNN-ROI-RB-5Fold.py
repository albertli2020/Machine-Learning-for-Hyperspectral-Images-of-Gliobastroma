import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, log_loss, classification_report, roc_auc_score
from collections import Counter
import matplotlib.pyplot as plt
import time
import os
import glob
import random
import sys

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
    def __init__(self, image_files, labels, bands):
        self.image_files = image_files
        # self.data = []
        self.labels = labels
        self.bands = bands
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
        patch = patch_np[:87, :87, bands]
        assert patch.shape == (87, 87, len(bands)), f"Unexpected patch size: {patch.shape} at {npy_path}"
        patch = torch.tensor(patch, dtype=torch.float32)  # Convert to PyTorch tensor
        patch = patch.permute(2, 0, 1)  # Rearrange to (Bands, Height, Width) for PyTorch
        return patch, label

# Define VSM Model for 3D Hyperspectral Input
class CNNModel(nn.Module):
    def __init__(self, num_bands):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(num_bands, 64, kernel_size=3, padding=1)  # 275 bands as input channels
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 43 * 43, 256)  # Flattened feature map size
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)  # Binary classification (non-tumor, tumor)
    
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
def train_model(model, train_loader, val_loader, class_weight=1, fold="", epochs=10, lr=0.0001, device="cpu"):
    #criterion = nn.CrossEntropyLoss()
    pos_weight = torch.tensor([class_weight], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    #best_accuracy = 0
    #best_loss = float('inf') 
    best_auc = 0
    patience = 8
    patience_counter = 0
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    
    for epoch in range(epochs):
        y_true_trn = []
        y_pred_trn = []

        model.train()
        train_loss = 0
        #trn_correct = 0
        #trn_total = 0
        for patches, labels in train_loader:
            patches, labels = patches.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(patches).squeeze(1)
            # _, predicted = torch.max(outputs, 1)
            # predicted = (outputs > 0.5).float()
            probabilities = torch.sigmoid(outputs)  # Convert logits to probabilities
            predicted = (probabilities > 0.5).float()  
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*patches.size(0)
            #trn_correct += (predicted == labels).sum().item()
            #trn_total += labels.size(0)
            y_true_trn.extend(labels.cpu().numpy())
            y_pred_trn.extend(predicted.cpu().numpy())

        y_true_val = []
        y_pred_val = []
        y_prob_val = []
        model.eval()
        val_loss = 0
        #val_correct = 0
        #val_total = 0
        with torch.no_grad():
            for patches, labels in val_loader:
                patches, labels = patches.to(device), labels.to(device)
                outputs = model(patches).squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()*patches.size(0)
                probabilities = torch.sigmoid(outputs)  # Convert logits to probabilities
                predicted = (probabilities > 0.5).float()  
                #_, predicted = torch.max(outputs, 1)
                #val_correct += (predicted == labels).sum().item()
                #val_total += labels.size(0)
                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(predicted.cpu().numpy())
                y_prob_val.extend(probabilities.cpu().numpy())
                # y_prob_val.extend(probabilities[:, 1].detach().cpu().numpy())

        trn_loss = train_loss/len(train_loader.dataset)
        trn_accuracy = accuracy_score(y_true_trn, y_pred_trn)
        val_loss = val_loss/len(val_loader.dataset)
        val_accuracy = accuracy_score(y_true_val, y_pred_val)
        val_auc = roc_auc_score(y_true_val, y_prob_val)


        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {trn_loss:.4f}, Train Accuracy: {100 * trn_accuracy:.2f}%,"
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {100 * val_accuracy:.2f}%, Val AUC: {val_auc:.4f}")

        train_losses.append(trn_loss)
        train_accs.append(trn_accuracy)
        val_losses.append(val_loss)
        val_accs.append(val_accuracy)

        #epoch_avg_accuracy = (trn_accuracy + val_accuracy)/2
        #epoch_avg_loss = (trn_loss + val_loss)/2

        #if (epoch_avg_loss <= best_loss) and (val_accuracy >= best_accuracy):
        if val_auc > best_auc:
            #best_loss = epoch_avg_loss
            # best_accuracy = epoch_avg_accuracy
            #best_loss = val_loss
            #best_accuracy = val_accuracy
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), fold + "_best_2dcnn_model_roi.pth")
            # print(f"Trained model saved w/ avg loss: {best_loss:.4f}, validation accuracy: {100*best_accuracy:.4f}%")
            print(f"Trained model saved w/ best AUC: {best_auc:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print("Early stopping triggered")
            break


    # Plot training and validation curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fold + '_2dcnn_roi_training_curves.png')
    plt.close()


# Prediction Function
def predict(model, test_loader, fold="", device="cpu"):

    total_labels = []
    total_preds = []
    total_scores = []
    model.eval()
    with torch.no_grad():
        for patches, labels in test_loader:
            patches, labels = patches.to(device), labels.to(device)
            # print(len(patches))
            outputs = model(patches).squeeze(1)
            #probabilities = nn.functional.softmax(outputs, dim=1)
            #_, predicted = torch.max(outputs, 1)
            probabilities = torch.sigmoid(outputs)  # Convert logits to probabilities
            predicted = (probabilities > 0.5).float()  
            total_labels.extend(labels.cpu().numpy())
            total_preds.extend(predicted.cpu().numpy())
            #total_scores.extend(probabilities[:, 1].detach().cpu().numpy())
            total_scores.extend(probabilities.cpu().numpy())

        cm = confusion_matrix(total_labels, total_preds, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            print("True Neg, False Pos, False Neg, True Pos are: ", tn, fp, fn, tp)
        else:
            print(cm.shape)
            raise ValueError("The confusion matrix does not have a 2x2 shape, which is required for binary classification.")

        # Calculate AUC
        if len(set(total_labels)) < 2:
            auc = -0.0
        else:
            auc = roc_auc_score(total_labels, total_scores)

        # Accuracy
        accuracy = accuracy_score(total_labels, total_preds)
        # Sensitivity (Recall or True Positive Rate)
        sensitivity = recall_score(total_labels, total_preds, pos_label=1, zero_division=0)
        # Specificity (requires manual calculation)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        # Precision
        precision = precision_score(total_labels, total_preds, pos_label=1, zero_division=0)
        # F1 Score
        f1 = f1_score(total_labels, total_preds)
        print(f"AUC, Test accuracy, sensitivity (recall), specificity, Precision. F1_Score are: {auc:.4f}, {accuracy*100:.4f}%, {sensitivity*100:.4f}%, {specificity*100:.4f}%, {precision*100:.4f}%, {f1*100:.4f}%")

    print("test label distribution:", Counter(total_labels))
    print("Predicted label distribution:", Counter(total_preds))
    print("Classification Report:")
    print(classification_report(total_labels, total_preds))

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Non-tumor', 'Tumor'])
    plt.yticks(tick_marks, ['Non-tumor', 'Tumor'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    plt.tight_layout()
    plt.savefig( fold + '_2dcnn_roi_confusion_matrix.png')
    plt.close()

# Main Execution
if __name__ == "__main__":
    root_dir = "C:/HSI-ML/ntp_90_90_275"  # Replace with the path to your dataset
    batch_size = 32
    epochs = 20
    learning_rate = 0.0001

    bands = list(range(0,275))
    if sys.argv[1] == None or sys.argv[1] == "b275":
        bands = list(range(0,275))
    elif sys.argv[1] == "b150":
        bands = list(range(0,150))
    elif sys.argv[1] == "b120":
        bands = list(range(0,40)) + list(range(60,140))
    elif sys.argv[1] == "b100":
        bands = list(range(0,20)) + list(range(60,140))
    elif sys.argv[1] == "b80":
        bands = list(range(0,20)) + list(range(70,130))   
    elif sys.argv[1] == "b60":
        bands = list(range(70, 130))

    num_bands = len(bands)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    patch_list, t_roi_list, nt_roi_list, t_roi_dir, nt_roi_dir = generateLists(root_dir)

    for i in range(len(test_roi_sets)):
        fold = "fold" + str(i+1)
        if sys.argv[1]:
            fold = sys.argv[1] + "_" + fold
        start_time = time.time()

        training_set, val_set, test_set = splitDataset(patch_list, t_roi_dir, nt_roi_dir, test_roi_sets[i], val_roi_sets[i])
        training_labels = labelHSIDataSet(training_set)
        val_labels = labelHSIDataSet(val_set)
        test_labels = labelHSIDataSet(test_set)

        training_label_dist = Counter(training_labels)
        training_t = training_label_dist[1.0]
        class_weight_tumor = len(training_labels)/(2*training_t)

        print(f"\nFold #{i+1}:")
        print("\nTotal number of training data:", len(training_set))
        print("Trainging label distribution:", Counter(training_labels))
        print("Total number of validation data:", len(val_set))
        print("Predicted label distribution:", Counter(val_labels))
        print("Total number of test data:", len(test_set))
        print("Predicted label distribution:", Counter(test_labels))


        training_dataset = HyperspectralDataset(training_set, training_labels, bands)
        val_dataset = HyperspectralDataset(val_set, val_labels, bands)
        test_dataset = HyperspectralDataset(test_set, test_labels, bands)

        train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


        # Initialize model
        model = CNNModel(num_bands).to(device)

        # Train model
        train_model(model, train_loader, val_loader, class_weight=class_weight_tumor, fold=fold, epochs=epochs, lr=learning_rate, device=device)

        # Load model for prediction
        model.load_state_dict(torch.load(fold + "_best_2dcnn_model_roi.pth"))
        model.to(device)
        predict(model, test_loader, fold=fold, device=device)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Fold #{i+1} Elapsed time: {elapsed_time} seconds")
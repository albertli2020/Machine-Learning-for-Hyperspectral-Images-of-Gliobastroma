import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, log_loss, roc_auc_score, recall_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import glob
import random
from collections import Counter

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

def decision_function_to_proba(decision):
    def sigmoid(x):
        mask = x >= 0
        result = np.empty_like(x)
        result[mask] = 1 / (1 + np.exp(-x[mask]))
        result[~mask] = np.exp(x[~mask]) / (1 + np.exp(x[~mask]))
        return result
    return sigmoid(decision)

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

root_dir = "C:/HSI-ML/ntp_90_90_275"  # Replace with the path to your dataset
batch_size = 32

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

start_time = time.time()

patch_list, t_roi_list, nt_roi_list, t_roi_dir, nt_roi_dir = generateLists(root_dir)

for i in range(len(test_roi_sets)):
    fold = "fold" + str(i+1)

    training_set, val_set, test_set = splitDataset(patch_list, t_roi_dir, nt_roi_dir, test_roi_sets[i], val_roi_sets[i])
    training_labels = labelHSIDataSet(training_set)
    val_labels = labelHSIDataSet(val_set)
    test_labels = labelHSIDataSet(test_set)

    print(f"\nFold #{i+1} training starts")
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


    # Define parameter grid for SVM (SGDClassifier)
    param_grid = {
        'alpha': [0.01],  # Equivalent to 1/C in SVM
        'eta0': [0.1],  # Initial learning rate
    }

    best_val_auc = 0
    best_model = None
    best_scaler = None

    all_classes = np.unique(training_dataset.labels)

    for alpha in param_grid['alpha']:
        for eta0 in param_grid['eta0']:

            start = time.time()
            
            # Create SGDClassifier (SVM approximation)
            svm_model = SGDClassifier(loss='hinge', alpha=alpha, learning_rate='optimal', 
                                    eta0=eta0, power_t=0.5, random_state=42)
            
            # Initialize the scaler
            scaler = StandardScaler()

            # Train the model
            for batch_data, batch_labels in train_loader:
                X_batch = batch_data.numpy()
                y_batch = batch_labels.numpy().ravel()
                
                # Partial fit the scaler
                scaler.partial_fit(X_batch)
                
                # Scale the batch data
                X_batch_scaled = scaler.transform(X_batch)
                
                # Partial fit the SVM
                #svm_model.partial_fit(X_batch_scaled, y_batch, classes=np.unique(y_batch))
                svm_model.partial_fit(X_batch_scaled, y_batch, classes=all_classes)

            # Calculate training metrics
            train_pred, train_prob, y_true = [], [], []
            for batch_data, batch_labels in train_loader:
                X_batch = batch_data.numpy()
                y_batch = batch_labels.numpy().ravel()
                
                X_batch_scaled = scaler.transform(X_batch)
                
                batch_pred = svm_model.predict(X_batch_scaled)
                #batch_prob = svm_model.predict_proba(X_batch_scaled)
                batch_decision = svm_model.decision_function(X_batch_scaled)
                batch_prob = decision_function_to_proba(batch_decision)
                
                train_pred.extend(batch_pred)
                train_prob.extend(batch_prob)
                y_true.extend(y_batch)

            train_loss = log_loss(y_true, train_prob)
            train_accuracy = accuracy_score(y_true, train_pred)
            
            # Calculate validation metrics
            val_pred, val_prob, y_val = [], [], []
            for batch_data, batch_labels in val_loader:
                X_batch = batch_data.numpy()
                y_batch = batch_labels.numpy().ravel()
                
                X_batch_scaled = scaler.transform(X_batch)
                
                batch_pred = svm_model.predict(X_batch_scaled)
                #batch_prob = svm_model.predict_proba(X_batch_scaled)
                batch_decision = svm_model.decision_function(X_batch_scaled)
                batch_prob = decision_function_to_proba(batch_decision)
                
                val_pred.extend(batch_pred)
                val_prob.extend(batch_prob)
                y_val.extend(y_batch)

            val_loss = log_loss(y_val, val_prob)
            val_accuracy = accuracy_score(y_val, val_pred)
            #val_auc = roc_auc_score(y_val, [prob[1] for prob in val_prob])
            val_auc = roc_auc_score(y_val, val_prob)

            print(f"\nParameters: alpha={alpha}, eta0={eta0}")
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Training Accuracy: {train_accuracy:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            print(f"Validation AUC: {val_auc:.4f}")

            # Update best model if validation AUC improves
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model = svm_model
                best_params = {'alpha': alpha, 'eta0': eta0}
                best_scaler = scaler

    end = time.time()
    elapsed = end - start
    print(f"Elapsed time: {elapsed:.2f} seconds")       

    #print(f"\nBest Parameters: {best_params}")

    # Save the best model and scaler
    joblib.dump(best_model, fold + '_best_svm_roi_model.joblib')
    joblib.dump(best_scaler, fold + '_best_svm_roi_scaler.joblib')

    # Evaluate on test set
    test_pred, test_prob, y_test = [], [], []
    for batch_data, batch_labels in test_loader:
        X_batch = batch_data.numpy()
        y_batch = batch_labels.numpy().ravel()
        
        X_batch_scaled = best_scaler.transform(X_batch)
        
        batch_pred = best_model.predict(X_batch_scaled)
        #batch_prob = best_model.predict_proba(X_batch_scaled)
        batch_decision = svm_model.decision_function(X_batch_scaled)
        batch_prob = decision_function_to_proba(batch_decision)
        
        test_pred.extend(batch_pred)
        test_prob.extend(batch_prob)
        y_test.extend(y_batch)

    test_loss = log_loss(y_test, test_prob)
    test_accuracy = accuracy_score(y_test, test_pred)
    #test_auc = roc_auc_score(y_test, [prob[1] for prob in test_prob])
    test_auc = roc_auc_score(y_test, test_prob)
    test_precision = precision_score(y_test, test_pred, zero_division=0)
    test_recall = recall_score(y_test, test_pred, zero_division=0)
    test_f1 = f1_score(y_test, test_pred)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, test_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate specificity
    test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Sensivity(Recall): {test_recall:.4f}")
    print(f"Test Specificity: {test_specificity:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test F1-score: {test_f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Save the heatmap to a file
    plt.savefig(fold + '_svm_roi_confusion_matrix.png', dpi=300, bbox_inches='tight')

    # Optionally, close the plot if you don't want to display it
    plt.close()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Fold #{i+1} testing ends")
    print(f"Fold #{i+1} total Elapsed time: {elapsed_time:.2f} seconds")
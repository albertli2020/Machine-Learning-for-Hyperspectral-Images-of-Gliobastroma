import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, log_loss, classification_report, roc_auc_score
import os
import random
import glob
import time
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import csv

"""
This Python program uses incremental learning through layer-wise expansion.
Similar to Progressive GANs, layers are added gradually to a network during training.
The progressive complexity growing scheme inherits parameters from shallower networks to initialize deeper (more-layered) ones.
An up-to 8-layer 2D- or 4-layer-3D CNN is built progressively in a hierarchical fashion, with HSI patches of shape (87, 87, 275) * float32.
"""
"""
Do the following the activate and customize python environment, including a nightly development build of PyTorch:

conda activate /Users/albert/anaconda3/envs/pyTorch-MPS
# Do the following once to persist the customized PyTorch development build installation to support conv3d on MPS
pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
"""
data_root_dir = "ntp_90_90_275/"
sampled_data_percentage = 5 #100 #to reduce validation data size
input_spectral_bands = 826//3 #275
g_patch_size = 87

tvt_data_folds = [  [["P2", "P3", "P4", "P5", "P8", "P9", "P10", "P12", "P13"], ["P1"], ["P7", "P11"]],
                    [["P2", "P1", "P7", "P5", "P8", "P9", "P10", "P12", "P11"], ["P3"], ["P4", "P13"]],
                    [["P1", "P3", "P4", "P11", "P8", "P10", "P12", "P13"], ["P7"], ["P2", "P5", "P9"]],
                    [["P2", "P7", "P4", "P5", "P9", "P11", "P13"], ["P1"], ["P3", "P8", "P10", "P12"]],
                    ["train_set_ROI_F1.txt", "val_test_set_ROI_F1.txt"],
                    ["train_set_ROI_F2.txt", "val_test_set_ROI_F2.txt"],
                    ["train_set_ROI_F3.txt", "val_test_set_ROI_F3.txt"],
                    ["train_set_ROI_F4.txt", "val_test_set_ROI_F4.txt"],
                    ["train_set_ROI_F5.txt", "val_test_set_ROI_F5.txt"]
                ]
global_specifier_2D_F1 = {'nn_arch_name': '2D', 'data_fold_type': 'Patient', 'data_fold_name': 'F1', 'tvt_data_fold_idx': 0, 'batch_size':24, '1st_hl_size':256}
global_specifier_3D_F1 = {'nn_arch_name': '3D', 'data_fold_type': 'Patient', 'data_fold_name': 'F1', 'tvt_data_fold_idx': 0, 'batch_size':12, '1st_hl_size':16}
global_specifier_2D_ROI_F1 = {'nn_arch_name': '2D', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F1', 'tvt_data_fold_idx': 4+0, 'batch_size':24, '1st_hl_size':256}


global_specifier = global_specifier_2D_ROI_F1 #global_specifier_2D_F1 #global_specifier_3D_F1
g_batch_size = global_specifier['batch_size']
g_1st_hidden_layer_size = global_specifier['1st_hl_size']
attempt_gpu = True

# Define the steps of machine learning processing of HSI data patches
mlp_steps = [
    #{'desc':'1-L from scratch, TVT', 'nol_from':0, 'nol_new':1, 'lr':0.002, 'noe':8}, #18},
    {'desc':'1-L to 1-L, TVT', 'nol_from':1, 'nol_new':1, 'lr':0.0015, 'noe':6},
    {'desc':'1-L to 2-L, TVT', 'nol_from':1, 'nol_new':2, 'lr':0.001, 'noe':8}, #25},
    {'desc':'2-L to 3-L, TVT', 'nol_from':2, 'nol_new':3, 'lr':0.00025, 'noe':8},
    {'desc':'3-L to 4-L, TVT', 'nol_from':3, 'nol_new':4, 'lr':0.00005, 'noe':8},
    {'desc':'4-L to 5-L, TVT', 'nol_from':4, 'nol_new':5, 'lr':0.000025, 'noe':8}
    #{'desc':'5-L to 5-L, TVT', 'nol_from':5, 'nol_new':5, 'lr':0.00001, 'noe':8}
    #{'desc':'6-L to 6-L, TVT', 'nol_from':6, 'nol_new':6, 'lr':0.00002, 'noe':8}
    ]

mlp_testonly_1 = {'desc':'1-L Test-only', 'nol':1}
mlp_testonly_2 = {'desc':'2-L Test-only', 'nol':2}
mlp_testonly_3 = {'desc':'3-L Test-only', 'nol':3}
mlp_testonly_4 = {'desc':'4-L Test-only', 'nol':4}
mlp_testonly_5 = {'desc':'5-L Test-only', 'nol':5}
mlp_testonly_6 = {'desc':'6-L Test-only', 'nol':6}
mlp_testonly = mlp_testonly_1
if global_specifier['data_fold_type'] == 'Patient':
    testonly_patients = tvt_data_folds[global_specifier['tvt_data_fold_idx']][2] + tvt_data_folds[global_specifier['tvt_data_fold_idx']][1] + ["P6"]
#testonly_patients = ["P6"]
run_testonly = False #False # 

# File path for the log
mlp_log_file_path = f"mlp_pnn_{global_specifier['nn_arch_name']}_{global_specifier['data_fold_name']}_training_log_file.txt"

def read_dataset_file_list(file_path, duplicatePos = False):
    dataset_files = []
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            lines = file.readlines()
            #for line in lines:
            #    dataset_files.extend(line.strip())
            for line in lines:
                clean_line = line.rstrip('\n')
                dataset_files += [clean_line]
                if duplicatePos:
                    p = os.path.dirname(clean_line)                         
                    if p.endswith("_T") or p.endswith("_T_50G"):
                        dataset_files += [clean_line]
    
    return dataset_files
    
def read_log_file(file_path):
    """Read the log file and construct a dictionary."""
    log_data = {}
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()
                if line:  # Skip empty lines
                    nol, fn, last_saved, avg_loss, avg_accuracy = line.split(",")
                    log_data[int(nol)] = {
                        "nol": int(nol),
                        "fn": fn,
                        "last_saved": last_saved,
                        "avg_loss": float(avg_loss),
                        "avg_accuracy": float(avg_accuracy),
                    }
    return log_data

def update_log_file(file_path, data):
    """Write updated log data back to the file."""
    with open(file_path, "w") as file:
        for key, record in data.items():
            line = f"{record['nol']},{record['fn']},{record['last_saved']},{record['avg_loss']},{record['avg_accuracy']}\n"
            file.write(line)

def generatePatchListsByPatient(path, training_patients=[]):
    patient_patch_dict = {}
    for i in range(13):
        k = "P" + str(i+1)
        patient_patch_dict[k] = []
    
    for entry in os.listdir(path):
        next_level_path = os.path.join(path, entry)
        if os.path.isdir(next_level_path):
            paths = glob.glob(os.path.join(next_level_path, "*.npy"), recursive=False)                      
            i = entry.find("_")
            p = entry[:i]
            if entry.endswith("_NT"): # or entry.endswith("_T_50G"): or entry.endswith("_NT_50G"): 
                patient_patch_dict[p].extend(paths)
            elif entry.endswith("_T"):
                patient_patch_dict[p].extend(paths)
                if p in training_patients:
                    patient_patch_dict[p].extend(paths) # double T patches if in training patients list
    # print(patient_patch_dict)            
    return patient_patch_dict

def labelHSIDataSet(data_files):
    labels = []
    for file in data_files:
        #print(file)
        p = os.path.dirname(file)        
        #print(p)
        if p.endswith("_T") or p.endswith("_T_50G"):  
            labels.append(1)
        else:
            labels.append(0)
    return labels

def dataSetSample(precentage, set1, set2=[], attempt_to_balance=False):

    data_set = []

    if precentage < 1:
        count1 = int(len(set1) * precentage)
        count2 = int(len(set2) * precentage)
        if attempt_to_balance:
            avg_count = (count1 + count2)//2
            if avg_count > len(set1):
                avg_count = len(set1)
            if avg_count > len(set2):            
                avg_count = len(set2)
            count1 = avg_count
            count2 = avg_count
        data_set = random.sample(set1, k = count1) +  random.sample(set2, k = count2)
    else:
        data_set = set1 + set2
    
    random.shuffle(data_set)
    return data_set

def HSIDatasetSplitByPatientLists(patient_file_dict, training_patients, validation_patients, test_patients, data_precentage=1.0):
    validation_set = []
    test_set = []
    training_set = []

    for patient in validation_patients:
        validation_set.extend(patient_file_dict[patient])
    validation_set = dataSetSample(data_precentage, validation_set, attempt_to_balance=False)
    validation_labels = labelHSIDataSet(validation_set)

    for patient in test_patients:
        test_set.extend(patient_file_dict[patient])
    test_set = dataSetSample(1.0, test_set, attempt_to_balance=False)
    test_labels = labelHSIDataSet(test_set)

    for patient in training_patients:
        training_set.extend(patient_file_dict[patient])
    training_set = dataSetSample(1.0, training_set, attempt_to_balance=False)
    training_labels = labelHSIDataSet(training_set)

    return training_set, training_labels, validation_set, validation_labels, test_set, test_labels

# Hyperspectral Dataset Class
class HyperspectralDataset(Dataset):
    def __init__(self, image_paths, labels, randomize_pos_data=False, bands=range(0,275), patch_size=87, gpu_device=None):
        self.gpu_device = gpu_device
        self.patch_size = patch_size
        self.image_paths = image_paths
        self.bands = bands
        self.labels = labels
        self.rpd = randomize_pos_data
        # Total number of patches
        self.total_images = len(self.image_paths)
        self.indices = np.arange(self.total_images)
    
    def __len__(self):
        n = len(self.image_paths)
        n_batches = n//g_batch_size
        delta = n - (n_batches*g_batch_size)
        if delta > 1:
            n_batches += 1
        return n_batches
    
    def __getitem__(self, idx):
        #patch_np = patch_np[:self.patch_size, :self.patch_size, self.bands]
        start_idx = idx * g_batch_size
        end_idx = min(start_idx + g_batch_size, self.total_images)
        batch_indices = self.indices[start_idx:end_idx]
        
        y_batch = labelHSIDataSet(self.image_paths[start_idx:end_idx])
        # print(y_batch)

        # Dynamically load the image patches        
        X_batch = []
        if self.rpd:
            for i in batch_indices:
                image = np.load(self.image_paths[i])  # Assuming each file is a numpy array of the patch
                #image = image[:self.patch_size, :self.patch_size, :]
                X_batch.append(image)
                    
            for i in range(len(y_batch)):
                image = X_batch[i]
                max_offset = image.shape[0] - g_patch_size
                if y_batch[i] == 1:
                    rr = random.randint(0, max_offset)
                    rc = random.randint(0, max_offset)
                else:
                    rr = 0
                    rc = 0                
                X_batch[i] = image[rr:rr+self.patch_size, rc:rc+self.patch_size, :]
        else:
            for i in batch_indices:
                image = np.load(self.image_paths[i])  # Assuming each file is a numpy array of the patch
                image = image[:self.patch_size, :self.patch_size, :]
                X_batch.append(image)
        
        # Convert to numpy arrays
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
    
        if global_specifier['nn_arch_name'] == '2D':
            X_batch = X_batch.transpose(0, 3, 1, 2)
        tensor_patch = torch.tensor(X_batch, dtype=torch.float32).to(self.gpu_device)
        tensor_label = torch.tensor(y_batch, dtype=torch.float32).to(self.gpu_device)
        return tensor_patch, tensor_label

    def show_plot_patches_in_batch(self, idx):    
        rgb_images = []
        r_band, g_band, b_band = 425//3, 192//3, 109//3
        start_idx = idx * g_batch_size
        end_idx = min(start_idx + g_batch_size, self.total_images)
        batch_indices = self.indices[start_idx:end_idx]
        for idx in batch_indices:
            patch_image_path = self.image_paths[idx]
            label = self.labels[idx]
            print(f"Label={label}, path={patch_image_path}")
            patch_np = np.load(patch_image_path).astype(np.float32)
            #patch_np = patch_np[:self.patch_size, :self.patch_size, :]
            rgb_images.append(patch_np[:, :, [r_band, b_band, g_band]]) #[r_band, g_band, b_band]])

        # Create the figure and subplots
        fig, axes = plt.subplots(4, 4, figsize=(10, 12))
        
        # Plot each image
        for i, ax in enumerate(axes.flat):
            ax.imshow(rgb_images[i])
            ax.axis('off')  # Hide the axes
            t = self.image_paths[batch_indices[i]]
            parts = t.split('/')
            del parts[0]
            t = '/'.join(parts)
            ax.set_title(t, fontsize=8)
        
        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.show()

# Define the CNN model (same as before)
class TumorClassifierCNN(nn.Module):
    def init_2D(self, num_input_channels, num_layers, gpu_device):
        if self.conv1 is None:
            self.conv1 = nn.Conv2d(in_channels=num_input_channels, out_channels=g_1st_hidden_layer_size, kernel_size=3, stride=1, padding=0, device=gpu_device)  # (87, 87, 275) -> (85, 85, 256)
        if num_layers > 1 and self.conv2 is None:
            self.conv2 = nn.Conv2d(in_channels=g_1st_hidden_layer_size, out_channels=256, kernel_size=3, stride=1, padding=0, device=gpu_device)  # (85, 85, 256) -> (83, 83, 256)
        if num_layers > 2 and self.conv3 is None:
            self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0, device=gpu_device)  # (83, 83, 256) -> (81, 81, 512)
        if num_layers > 3 and self.conv4 is None:
            self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0, device=gpu_device)  # (81, 81, 512) -> (79, 79, 512)
        if num_layers > 4 and self.conv5 is None:
            self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=0, device=gpu_device) # -> (77, 77, 1024)
        if num_layers > 5 and self.conv6 is None:
            self.conv6 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=0, device=gpu_device) # (77, 77, 1024) -> (75, 75, 1024)
        if num_layers > 6 and self.conv7 is None:
            self.conv7 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=0, device=gpu_device) #-> (28, 28, 1024)
        if num_layers > 7 and self.conv8 is None:
            self.conv8 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=0, device=gpu_device) #-> (26, 26, 1024)            

    def init_3D(self, num_output_channels, num_layers, gpu_device):
        if self.conv1 is None:
            self.conv1 = nn.Conv3d(in_channels=1, out_channels=num_output_channels, kernel_size=(3, 3, 3), stride=1, padding=1, device=gpu_device)
        if num_layers > 1 and self.conv2 is None:        
            self.conv2 = nn.Conv3d(in_channels=num_output_channels, out_channels=num_output_channels, kernel_size=(3, 3, 3), stride=1, padding=1, device=gpu_device)
        if num_layers > 2 and self.conv3 is None:
            self.conv3 = nn.Conv3d(in_channels=num_output_channels, out_channels=num_output_channels, kernel_size=(3, 3, 3), stride=1, padding=1, device=gpu_device)
        if num_layers > 3 and self.conv4 is None:
            self.conv4 = nn.Conv3d(in_channels=num_output_channels, out_channels=num_output_channels, kernel_size=(3, 3, 3), stride=1, padding=1, device=gpu_device)

    def __init__(self, num_input_channels, gpu_device=None, num_layers=1, inherited_model=None):
        super(TumorClassifierCNN, self).__init__()
        self.gpu_device = gpu_device
        dense_layer_inputs = g_1st_hidden_layer_size
        self.conv1 = None
        self.conv2 = None
        self.conv3 = None
        self.conv4 = None
        self.conv5 = None
        self.conv6 = None
        self.conv7 = None
        self.conv8 = None
        self.dl = None
        self.unsqueezeInput = False
        # Define the Conv layers
        if inherited_model is not None:
            self.conv1 = inherited_model.conv1
        if num_layers > 1:
            dense_layer_inputs = 256
            if inherited_model is not None:
                self.conv2 = inherited_model.conv2
        if num_layers > 2:
            dense_layer_inputs = 512
            if inherited_model is not None:
                self.conv3 = inherited_model.conv3
        if num_layers > 3:
            dense_layer_inputs = 512
            if inherited_model is not None:
                self.conv4 = inherited_model.conv4

        if global_specifier['nn_arch_name'] == '2D':            
            if num_layers > 4:
                dense_layer_inputs = 1024
                if inherited_model is not None:
                    self.conv5 = inherited_model.conv5            
            if num_layers > 5:
                dense_layer_inputs = 1024
                if inherited_model is not None:
                    self.conv6 = inherited_model.conv6
            if num_layers > 6:
                dense_layer_inputs = 1024
                if inherited_model is not None:
                    self.conv7 = inherited_model.conv7            
            if num_layers > 7:
                dense_layer_inputs = 1024    
                if inherited_model is not None:
                    self.conv8 = inherited_model.conv8
            self.init_2D(num_input_channels, num_layers, gpu_device)
            # Global average pooling
            self.pool = nn.AdaptiveAvgPool2d((1, 1))  # (71, 71, 1024) -> (1, 1, 1024)
            if dense_layer_inputs > 256:
                self.dl = nn.Linear(dense_layer_inputs, 256)  # (1, 1024) -> (1, 256)            
            # Fully connected (dense) layer
            self.fc = nn.Linear(256, 1) #2)  # (1, 256) -> (1, 2)
            self.fc2 = None
        else:
            self.unsqueezeInput = True
            self.init_3D(g_1st_hidden_layer_size, num_layers, gpu_device)        
            # Global average pooling
            self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.fc2 = nn.Linear(g_1st_hidden_layer_size, 1)  # Binary classification
            self.fc = None #an earlier version of 3DCNN started to use fc2; so we have to stick with it...

        # Activation and Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        #print("At CNN instantiation time, 1st layer weight shape is:", self.conv1.weight.shape)
        
    def forward(self, x):
        if self.unsqueezeInput:
            x = x.unsqueeze(1)
        x = self.dropout(self.relu(self.conv1(x)))
        if self.conv2 is not None:
            x = self.dropout(self.relu(self.conv2(x)))
        if self.conv3 is not None:        
            x = self.dropout(self.relu(self.conv3(x)))
        if self.conv4 is not None:
            x = self.dropout(self.relu(self.conv4(x)))
        if self.conv5 is not None:        
            x = self.dropout(self.relu(self.conv5(x)))
        if self.conv6 is not None:
            x = self.dropout(self.relu(self.conv6(x)))
        if self.conv7 is not None:        
            x = self.dropout(self.relu(self.conv7(x)))
        if self.conv8 is not None:
            x = self.dropout(self.relu(self.conv8(x)))
        # Global average pooling
        x = self.pool(x)  # Shape: (batch_size, 1024, 1, 1)
        x = x.view(x.size(0), -1)    # Flatten to (batch_size, 1024)

        # Dense layer
        if self.dl is not None:
            x = self.dropout(self.relu(self.dl(x)))
        # Fully connected layer. No activation function as the loss function (e.g., BCEWithLogitsLoss) internally applies sigmoid for numerical stability.
        #x = self.dropout(self.relu(self.fc(x)))
        # Dropout is less commonly used in output layers.
        #x = self.dropout(self.fc(x))
        if self.fc is not None:
            x = self.fc(x)
        else:
            x = self.fc2(x)
        return x

class HyperspectralNetworkTrainer():
    def __init__(self, num_layers, input_bands=range(0,input_spectral_bands), class_weight_ratio=1.0, gpu_device=None, learning_rate=0.0001, num_layers_of_inherited_model=0):
        self.my_num_layer = num_layers
        self.gpu_device = gpu_device
        self.model_save_file = f"ByPatients_{global_specifier['nn_arch_name']}_{global_specifier['data_fold_name']}_{self.my_num_layer}LoP_best_model.pth"
        self.target_accuracy = 0.65
        # Initialize model
        self.input_bands = input_bands
        if num_layers_of_inherited_model == 0:
            pretrained_model = None
        else:
            if num_layers_of_inherited_model == -1:                
                # test-only; parameters will be loaded at test time.
                pretrained_model = TumorClassifierCNN(num_input_channels=len(input_bands), gpu_device=gpu_device, num_layers=self.my_num_layer)         
            else:
                pretrained_model = TumorClassifierCNN(num_input_channels=len(input_bands), gpu_device=gpu_device, num_layers=num_layers_of_inherited_model)
                pretrained_model.load_state_dict(torch.load(f"ByPatients_{global_specifier['nn_arch_name']}_{global_specifier['data_fold_name']}_{num_layers_of_inherited_model}LoP_best_model.pth"))            
        
        if num_layers_of_inherited_model == -1:
            print(f"Test only with model parameters to be loaded from {self.model_save_file}")
            self.model = pretrained_model
        elif num_layers_of_inherited_model == self.my_num_layer:
            # Attempt to fine-train the model of same numer of layers
            print(f"Fine-train with lower LR={learning_rate}, starting at the last saved best model parameters.")
            self.model = pretrained_model
        else:
            if num_layers_of_inherited_model == 0:
                print(f"Train with LR={learning_rate}, starting from scratch.")
            else:
                print(f"Train with LR={learning_rate}, starting by inheriting saved best model parameters from a shallower network.")
            self.model = TumorClassifierCNN(num_input_channels=len(input_bands), gpu_device=gpu_device, num_layers=self.my_num_layer, inherited_model=pretrained_model)

        # This model.to() step might not be needed as the CNN instantiation above was executed with gpu_device specified.
        if gpu_device is not None:
            self.model.to(gpu_device)

        #Initialize loss function and optimizer. These are not needed for test-only.
        if num_layers_of_inherited_model >= 0:
            print(f"Class weight Ratio = {class_weight_ratio:.4f}")
            class_weights = torch.tensor([class_weight_ratio], dtype=torch.float32, device=gpu_device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights) #For binary classification, this is perferred over nn.CrossEntropyLoss(weight=class_weights)
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)

    def train_with_validation(self, train_dataset, val_dataset, loss_th_to_stop=0.005, accrucy_th_to_stop=0.99, epochs=100):
        # Read the log file
        log_data = read_log_file(mlp_log_file_path)

        train_start_time = time.time()
        if self.my_num_layer in log_data:
            best_loss = log_data[self.my_num_layer]["avg_loss"]
            best_accuracy = log_data[self.my_num_layer]["avg_accuracy"]
        else:
            best_loss = float('inf')  # Track the best validation loss for model saving
            best_accuracy = 0.0 
        target_met = False

        epoch_start_time = train_start_time
        for epoch in range(epochs):
            # Training
            y_true_trn = []
            y_pred_trn = []
            batch_loss = 0.0
            self.model.train()
            for batch_idx in range(len(train_dataset)):
                X_batch, y_batch = train_dataset[batch_idx]  # X_batch shape: (batch_size, num_features)

                self.optimizer.zero_grad()
                outputs = self.model(X_batch).squeeze()
                loss = self.criterion(outputs, y_batch)
                batch_loss = loss.item()
                loss.backward()
                #loss.backward(retain_graph=True)
                self.optimizer.step()

                probabilities = torch.sigmoid(outputs)  # Convert logits to probabilities
                if probabilities.ndim == 0:
                    probabilities = probabilities.unsqueeze(0)
                y_pred_batch = probabilities.float().detach()
                y_true_trn.extend(y_batch.cpu().numpy())
                y_pred_trn.extend(y_pred_batch.cpu().numpy())                
                
            # Compute metrics for the training process
            trn_loss = log_loss(y_true_trn, y_pred_trn)
            y_pred_trn = (np.array(y_pred_trn) > 0.5).astype(int)
            trn_accuracy = accuracy_score(y_true_trn, y_pred_trn)

            # Validation
            y_true_val = []
            y_pred_val = []
            self.model.eval()
            with torch.no_grad():
                for batch_idx in range(len(val_dataset)):
                    # Load a batch
                    X_batch, y_batch = val_dataset[batch_idx]  # X_batch shape: (batch_size, num_features)                    
                    outputs = self.model(X_batch).squeeze()
                    probabilities = torch.sigmoid(outputs)  # Convert logits to probabilities
                    if probabilities.ndim == 0:
                        probabilities = probabilities.unsqueeze(0)
                    y_pred_batch = probabilities.float().detach()
                    y_true_val.extend(y_batch.cpu().numpy())
                    y_pred_val.extend(y_pred_batch.cpu().numpy())
            # Compute metrics for the validation process
            val_loss = log_loss(y_true_val, y_pred_val)
            y_pred_val = (np.array(y_pred_val) > 0.5).astype(int)
            val_accuracy = accuracy_score(y_true_val, y_pred_val)
        
            epoch_end_time = time.time()
            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"TL: {trn_loss:.4f}, "
                  f"TA: {trn_accuracy*100:.4f}%, last batch loss: {batch_loss:.4f}; "
                  f"VL: {val_loss:.4f}, "
                  f"VA: {100 * val_accuracy:.4f}%; ",                      
                  f"seconds taken: {(epoch_end_time-epoch_start_time):.2f}")
            epoch_start_time = epoch_end_time

            epoch_avg_accuracy = (trn_accuracy + val_accuracy)/2.0
            epoch_avg_loss = (trn_loss + val_loss)/2.0
            # Save the model if average loss and accuracy improves
            #if (epoch_avg_loss <= best_loss) and (epoch_avg_accuracy >= best_accuracy) and (epoch_avg_accuracy > self.target_accuracy):
            if (epoch_avg_accuracy >= best_accuracy) and (epoch_avg_accuracy > self.target_accuracy):
                best_loss = epoch_avg_loss
                best_accuracy = epoch_avg_accuracy
                target_met = True
                torch.save(self.model.state_dict(), self.model_save_file)
                print(f"Trained model saved to {self.model_save_file} w/ avg loss: {best_loss:.4f}, avg accuracy: {100*best_accuracy:.4f}%")
                log_data[self.my_num_layer] = {
                    "nol": self.my_num_layer,
                    "fn": self.model_save_file,
                    "last_saved": datetime.fromtimestamp(time.time()).strftime('%m-%d-%Y_%H:%M:%S_PT'),
                    "avg_loss": best_loss,
                    "avg_accuracy": best_accuracy,
                }
                update_log_file(mlp_log_file_path, log_data)
                if (trn_loss < loss_th_to_stop) and (val_loss < loss_th_to_stop) and (trn_accuracy > accrucy_th_to_stop) and (val_accuracy > accrucy_th_to_stop):
                    break

        train_end_time = time.time()
        print(f"Training Complete. Elapsed seconds for this training & validation cycle: {(train_end_time - train_start_time):.2f}")
        return target_met

    def test_model(self, dataset, load_model_from_best_saved=True, explicit_prev_saved_model_file=None):
        test_start_time = time.time()
        all_labels = []
        all_preds = []
        all_scores = []
        if load_model_from_best_saved:
            if explicit_prev_saved_model_file is None:
                self.model.load_state_dict(torch.load(self.model_save_file))
            else:
                self.model.load_state_dict(torch.load(explicit_prev_saved_model_file))

        self.model.eval()
        with torch.no_grad():  # Disable gradient computation for efficiency
            for batch_idx in range(len(dataset)):
                X_batch, y_batch = dataset[batch_idx]  # X_batch shape: (batch_size, num_features)
                outputs = self.model(X_batch).squeeze()
                probabilities = torch.sigmoid(outputs).detach() # Convert logits to probabilities
                if probabilities.ndim == 0:
                    probabilities = probabilities.unsqueeze(0)
                #print("Probabilities shape", probabilities.shape)
                y_pred_batch = (probabilities > 0.5).float()
                all_scores.extend(probabilities.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
                all_preds.extend(y_pred_batch.cpu().numpy())

        test_end_time = time.time()
        print(f"Seconds taken to complete Testing: {(test_end_time-test_start_time):.2f}")
        print("True label distribution:", Counter(all_labels))
        print("Predicted label distribution:", Counter(all_preds))

        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            print("True Neg, False Pos, False Neg, True Pos are: ", tn, fp, fn, tp)
        else:
            print(cm.shape)
            raise ValueError("The confusion matrix does not have a 2x2 shape, which is required for binary classification.")

        # Calculate AUC
        if len(set(all_labels)) < 2:
            auc = -0.0
        else:
            auc = roc_auc_score(all_labels, all_scores)

        # Accuracy
        accuracy = accuracy_score(all_labels, all_preds)
        # Sensitivity (Recall or True Positive Rate)
        sensitivity = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
        # Specificity (requires manual calculation)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        # Precision
        precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
        # F1
        f1 = f1_score(all_labels, all_preds, average='binary')  # For binary classification
        print(f"AUC, Test accuracy, sensitivity (recall), specificity, precision, F1 are: {auc:.4f}, {accuracy*100:.2f}%, " 
              f"{sensitivity*100:.2f}%, {specificity*100:.2f}%, {precision*100:.2f}%,  {f1*100:.2f}%")

        # Print detailed metrics
        print("Classification Report:")
        print(classification_report(all_labels, all_preds, zero_division=0))  # Use 0 for undefined metrics

        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))
        return all_labels, all_preds, all_scores

if attempt_gpu:
    if torch.backends.mps.is_available():
        g_gpu_device = torch.device("mps")
        #x = torch.ones(1, device=g_gpu_device)
        print ("Model and data will be moved to Metal Performance backend") #, x)
    elif torch.cuda.is_available():
        g_gpu_device = "cuda"
        print("Model and data will be moved to nVidia GPUs")
    else:
        g_gpu_device = None
        print("Model and data will stay on CPUs")
else:
    g_gpu_device = None

data_loading_start_time = time.time()
print(f"val data use percentage={sampled_data_percentage:.2f}%; patch_size={g_patch_size}; spectral_bands_to_use={input_spectral_bands}")
if run_testonly:
    if global_specifier['data_fold_type'] == 'Patient':
        trn_patients = []
        val_patients = []
        patient_file_dict = generatePatchListsByPatient(data_root_dir, trn_patients)
        joined_patients_str = '_'.join(testonly_patients)
    test_only_num_of_layers = mlp_testonly['nol']
    trainer = HyperspectralNetworkTrainer(test_only_num_of_layers, gpu_device=g_gpu_device, num_layers_of_inherited_model=-1)
    # Write to CSV file
    # CSV file path
    #output_csv_path = f"test_output_{global_specifier['nn_arch_name']}_F{global_specifier['tvt_data_fold_idx']+1}_{joined_patients_str}_{test_only_num_of_layers}L.csv"
    output_csv_path = f"test_output_{global_specifier['nn_arch_name']}_{global_specifier['data_fold_name']}_{test_only_num_of_layers}L.csv"
    print("Test output will be saved to: ", output_csv_path)
    with open(output_csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["Patch Path", "True Label", "Predicted Label", "Probabilities"])
        if global_specifier['data_fold_type'] == 'Patient':
            for tst_patient in testonly_patients:
                _, _, _, _, test_set, test_labels = HSIDatasetSplitByPatientLists(patient_file_dict,
                                                    trn_patients, val_patients, [tst_patient], data_precentage=sampled_data_percentage/100.0)
                print("Test Patient:", tst_patient, " with ", str(len(test_set)), " accepted patches.")
                test_dataset = HyperspectralDataset(test_set, test_labels, bands=range(0,input_spectral_bands), patch_size=g_patch_size, gpu_device=g_gpu_device)
                local_time = datetime.fromtimestamp(time.time()).strftime('%m-%d-%Y %H:%M:%S')
                print(f"{mlp_testonly['desc']} for {tst_patient} starts at Pacific Time: {local_time}")
                true_labels, predicted_labels, probabilities = trainer.test_model(test_dataset)                                
                # Write rows
                for path, true, pred, prob in zip(test_set, true_labels, predicted_labels, probabilities):
                    writer.writerow([path, true, pred, prob])
            print("----------------------")
            print(" ")
        else:
            test_set = read_dataset_file_list(tvt_data_folds[global_specifier['tvt_data_fold_idx']+1][1])
            test_labels = labelHSIDataSet(test_set)
            test_dataset = HyperspectralDataset(test_set, test_labels, bands=range(0,input_spectral_bands), patch_size=g_patch_size, gpu_device=g_gpu_device)
            local_time = datetime.fromtimestamp(time.time()).strftime('%m-%d-%Y %H:%M:%S')
            print(f"{mlp_testonly['desc']} for {global_specifier['data_fold_name']} starts at Pacific Time: {local_time}")
            true_labels, predicted_labels, probabilities = trainer.test_model(test_dataset)
             # Write rows
            for path, true, pred, prob in zip(test_set, true_labels, predicted_labels, probabilities):
                writer.writerow([path, true, pred, prob])                            

else:
    if global_specifier['data_fold_type'] == 'Patient':
        trn_patients = tvt_data_folds[global_specifier['tvt_data_fold_idx']][0] #["P2", "P3", "P4", "P5", "P8", "P9", "P10", "P12", "P13"]
        val_patients = tvt_data_folds[global_specifier['tvt_data_fold_idx']][1] #["P1"]
        tst_patients = tvt_data_folds[global_specifier['tvt_data_fold_idx']][2] #["P7", "P11"]
        patient_file_dict = generatePatchListsByPatient(data_root_dir, trn_patients)
        training_set, training_labels, validation_set, validation_labels, test_set, test_labels = HSIDatasetSplitByPatientLists(patient_file_dict,
                                                                                                    trn_patients, val_patients, tst_patients,
                                                                                                    data_precentage=sampled_data_percentage/100.0)
    else:
        training_set_read_file = tvt_data_folds[global_specifier['tvt_data_fold_idx']][0]
        #print(training_set_read_file)
        training_set = read_dataset_file_list(training_set_read_file, duplicatePos=True)
        random.shuffle(training_set)
        training_labels = labelHSIDataSet(training_set)
        val_test_dataset_file_list = read_dataset_file_list(tvt_data_folds[global_specifier['tvt_data_fold_idx']][1])
        random.shuffle(val_test_dataset_file_list)
        split_index = int(len(val_test_dataset_file_list) * 0.25)
        validation_set = val_test_dataset_file_list[:split_index]
        validation_labels = labelHSIDataSet(validation_set)
        test_set = val_test_dataset_file_list[split_index:]
        test_labels = labelHSIDataSet(test_set)

    '''
    print(training_set)
    print(training_labels)
    print(validation_set)
    print(validation_labels)
    print(test_set)
    print(test_labels)
    '''
    train_class_counts = Counter(training_labels)
    if global_specifier['data_fold_type'] == 'Patient':
        print("Training Patients:", trn_patients, "Validation Patients:", val_patients, "Test Patients:", tst_patients)
    print(f"[Training vs Validation vs Test] Patch Image Distribution: " + str(len(training_set)) + " " + str(len(validation_set)) + " " + str(len(test_set)))
    print(f"Training True label distribution [0: Non-Tumor; 1: Tumor]:", train_class_counts, Counter(validation_labels), Counter(test_labels))
    train_dataset = HyperspectralDataset(training_set, training_labels, randomize_pos_data=True, bands=range(0,input_spectral_bands), patch_size=g_patch_size, gpu_device=g_gpu_device)
    val_dataset = HyperspectralDataset(validation_set, validation_labels, bands=range(0,input_spectral_bands), patch_size=g_patch_size, gpu_device=g_gpu_device)
    test_dataset = HyperspectralDataset(test_set, test_labels, bands=range(0,input_spectral_bands), patch_size=g_patch_size, gpu_device=g_gpu_device)

    data_loading_end_time = time.time()
    print(f"Elapsed seconds for data loading: {(data_loading_end_time - data_loading_start_time):.2f}")

    for s in mlp_steps:
        # Convert current time to local time and format it
        local_time = datetime.fromtimestamp(time.time()).strftime('%m-%d-%Y %H:%M:%S')
        print(f"{s['desc']} starts at Pacific Time: {local_time}")
        model_num_layers = s['nol_new']
        from_model_nl = s['nol_from']
        training_lr = s['lr']
        num_epoches = s['noe']
        print(f"Num_layers={model_num_layers}; Training_LR={training_lr:.6f}; Batch_Size={g_batch_size}; Num_Epoches={num_epoches}")
        trainer = HyperspectralNetworkTrainer(model_num_layers, class_weight_ratio = train_class_counts[0] / train_class_counts[1],
                                        gpu_device=g_gpu_device, learning_rate=training_lr, num_layers_of_inherited_model=from_model_nl)
        is_trained_well = trainer.train_with_validation(train_dataset, val_dataset,
                                      loss_th_to_stop=0.025, accrucy_th_to_stop=0.98, epochs=num_epoches)
        if is_trained_well:
            pass #trainer.test_model(test_dataset)
        else:
            print("The most recent training and validation process didn't converge enough to run the test process.")

        print("----------------------")
        print(" ")

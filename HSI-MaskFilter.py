import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, log_loss, classification_report, roc_auc_score
import os
import time
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import csv

import HSI_Globals
from HSI_Dataset import HyperspectralDataset
from HSI_Core2DCNN import TumorClassifier2DCNN
from HSI_Workorder import HyperspectralWorkorderMLP

"""
This Python program bases on a pre-trained 2DCNN model and trains a band-wise mask filter (layer) of the input patch images.
The pre-trained 2DCNN model is "frozen" during the mask filter's training, while taking the output of the mask filter as its full-band (275) inputs. 
"""
"""
# Do the following to activate and customize python environment:
conda activate /Users/albert/anaconda3/envs/pyTorch-MPS
# Do the following once to persist the customized PyTorch development build installation to support conv3d on MPS
pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
"""

# Define the Mask Filter model
class MaskFilter(nn.Module):
    def __init__(self, num_input_channels, gpu_device):
        super(MaskFilter, self).__init__()
        
        # Initialize mask randomly between 0.9 and 1.1
        self.mask_param = nn.Parameter(
            torch.rand(num_input_channels, device=gpu_device) * 0.2 + 0.9,
            requires_grad=True
        )
    
    def forward(self, x):
        # Reshape inside forward()
        mask = self.mask_param.view(1, -1, 1, 1)
        
        # Apply mask
        x = x * mask  
        return x

class MaskFilterTrainer():
    def __init__(self, num_layers, input_bands, global_specifier, class_weight_ratio=1.0, gpu_device=None, learning_rate=0.0001, num_layers_of_inherited_model=0, min_accuracy=.40):
        self.my_num_layer = num_layers
        self.gpu_device = gpu_device
        self.mask_param_csv_file = f"MaskFilterParam_{global_specifier['nn_arch_name']}_{global_specifier['data_fold_name']}_{self.my_num_layer}L.csv"
        self.model_save_file = f"MaskFilter_{global_specifier['nn_arch_name']}_{global_specifier['data_fold_name']}_{self.my_num_layer}L_best_model.pth"
        self.pretrained_hsi_cnn_model_file = f"PNN_{global_specifier['nn_arch_name']}_{global_specifier['data_fold_name']}_{self.my_num_layer}L_best_model.pth"
        print(f"The current Mask_Filter_Trainer targets minimum_accuracy of {min_accuracy:.4f}")
        print(f"The underlying HSI_PNN model parameters are being loaded from {self.pretrained_hsi_cnn_model_file}")

        self.target_accuracy = min_accuracy
        # Initialize model
        self.input_bands = input_bands
        hls_1st = global_specifier['1st_hl_size']
        self.pretrained_hsi_nn_model = TumorClassifier2DCNN(num_input_channels=len(input_bands), hidden_layer_size_1st=hls_1st,
                                        gpu_device=gpu_device, num_layers=self.my_num_layer)
        self.pretrained_hsi_nn_model.load_state_dict(torch.load(self.pretrained_hsi_cnn_model_file))
        if gpu_device is not None:
           self.pretrained_hsi_nn_model.to(gpu_device)
        self.model = MaskFilter(len(input_bands), gpu_device)
        if num_layers_of_inherited_model > 0:
            inheriting_mask_param_file = f"MaskFilter_{global_specifier['nn_arch_name']}_{global_specifier['data_fold_name']}_{num_layers_of_inherited_model}L_best_model.pth"
            if os.path.exists(inheriting_mask_param_file):
                print(f"Inheriting mask fitler parameters from {inheriting_mask_param_file}")            
                self.model.load_state_dict(torch.load(inheriting_mask_param_file))
            else:
                print(f"Starting mask fitler parameters from random values between 0.9 and 1.1")            
        if gpu_device is not None:
            self.model.to(gpu_device)
        print(f"Class weight Ratio = {class_weight_ratio:.4f}")
        class_weights = torch.tensor([class_weight_ratio], dtype=torch.float32, device=gpu_device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights) #For binary classification, this is perferred over nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)

    def update_mask_param_file(self, epoch):
        # Ensure the file has headers if it does not exist
        if not os.path.exists(self.mask_param_csv_file):
            with open(self.mask_param_csv_file, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch"] + [f"B{i}" for i in range(self.model.mask_param.shape[0])])  # Header row
        mask_values = self.model.mask_param.detach().cpu().numpy()  # Convert to numpy for easy writing
        with open(self.mask_param_csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch] + mask_values.tolist())  # Append new row with epoch number

    def train_with_validation(self, train_dataset, val_dataset, mlp_log_file_path, loss_th_to_stop=0.0005, accrucy_th_to_stop=0.99, epochs=100):
        train_start_time = time.time()
        best_loss = float('inf')  # Track the best validation loss for model saving
        best_accuracy = 0.0 
        target_met = False

        epoch_start_time = train_start_time

        self.pretrained_hsi_nn_model.eval()
        for param in self.pretrained_hsi_nn_model.parameters():
            param.requires_grad = False  # Ensure no updates
        
        for epoch in range(epochs):
            # Training
            y_true_trn = []
            y_pred_trn = []
            batch_loss = 0.0
            self.model.train()
            for batch_idx in range(len(train_dataset)):
                X_batch, y_batch = train_dataset[batch_idx]  # X_batch shape: (batch_size, num_features)

                self.optimizer.zero_grad()
                mask_filtered_input = self.model(X_batch)
                outputs = self.pretrained_hsi_nn_model(mask_filtered_input).squeeze()

                lambda_sparsity = 0.01  # Regularization weight
                # Compute L1 loss on the learnable mask_param (not self.mask, since it's reshaped)
                mask_l1_loss = lambda_sparsity * torch.norm(self.model.mask_param, p=1)
                loss = self.criterion(outputs, y_batch) + mask_l1_loss                
                batch_loss = loss.item()
                loss.backward()

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
            trn_size = len(y_true_trn)

            # Validation
            y_true_val = []
            y_pred_val = []
            self.model.eval()
            with torch.no_grad():
                for batch_idx in range(len(val_dataset)):
                    # Load a batch
                    X_batch, y_batch = val_dataset[batch_idx]  # X_batch shape: (batch_size, num_features)
                    mask_filtered_input = self.model(X_batch)                    
                    outputs = self.pretrained_hsi_nn_model(mask_filtered_input).squeeze()
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
            val_size = len(y_true_val)
        
            epoch_end_time = time.time()
            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"TL: {trn_loss:.4f}, "
                  f"TA: {trn_accuracy*100:.4f}%, last batch loss: {batch_loss:.4f}; "
                  f"VL: {val_loss:.4f}, "
                  f"VA: {100 * val_accuracy:.4f}%; ",                      
                  f"seconds taken: {(epoch_end_time-epoch_start_time):.2f}")
            epoch_start_time = epoch_end_time

            epoch_avg_accuracy = (trn_accuracy * trn_size + val_accuracy * val_size)/(val_size+trn_size)
            epoch_avg_loss = (trn_loss*trn_size + val_loss * val_size)/(val_size+trn_size)
            # Save the model if average loss and accuracy improves
            #if (epoch_avg_loss <= best_loss) and (epoch_avg_accuracy >= best_accuracy) and (epoch_avg_accuracy > self.target_accuracy):
            if (epoch_avg_accuracy >= best_accuracy) and (epoch_avg_accuracy > self.target_accuracy):
                best_loss = epoch_avg_loss
                best_accuracy = epoch_avg_accuracy
                target_met = True
                torch.save(self.model.state_dict(), self.model_save_file)
                print(f"Trained model saved to {self.model_save_file} w/ avg loss: {best_loss:.4f}, avg accuracy: {100*best_accuracy:.4f}%")
                self.update_mask_param_file(epoch)
                if (trn_loss < loss_th_to_stop) and (val_loss < loss_th_to_stop) and (trn_accuracy > accrucy_th_to_stop) and (val_accuracy > accrucy_th_to_stop):
                    break

        train_end_time = time.time()
        print(f"Training Complete. Elapsed seconds for this training & validation cycle: {(train_end_time - train_start_time):.2f}")
        return target_met

mlp_steps_train_and_val = HSI_Globals.mlp_steps_train_and_val_for_mf
workorders = HSI_Globals.work_orders_train_mask_filter
mlp = HyperspectralWorkorderMLP(trainer_class=MaskFilterTrainer, attemp_gpu=True)
mlp.fill_orders(mlp_steps_train_and_val, None, workorders)


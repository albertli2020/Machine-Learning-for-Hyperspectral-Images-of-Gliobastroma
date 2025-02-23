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

import HSI_Globals
from HSI_Dataset import HyperspectralDataset
from HSI_Core2DCNN import TumorClassifier2DCNN
from HSI_Workorder import HyperspectralWorkorderMLP

"""
This Python program uses incremental learning through layer-wise expansion.
Similar to Progressive GANs, layers are added gradually to a network during training.
The progressive complexity growing scheme inherits parameters from shallower networks to initialize deeper (more-layered) ones.
An up-to 4-layer 2D-CNN is built progressively in a hierarchical fashion, with HSI patches of shape (87, 87, 275) * float32.
"""
"""
# Do the following to activate and customize python environment:
conda activate /Users/albert/anaconda3/envs/pyTorch-MPS
# Do the following once to persist the customized PyTorch development build installation to support conv3d on MPS
pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
"""
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

class HyperspectralNetworkTrainer():
    def __init__(self, num_layers, input_bands, global_specifier, class_weight_ratio=1.0, gpu_device=None, learning_rate=0.0001, num_layers_of_inherited_model=0, min_accuracy=0.4):
        self.my_num_layer = num_layers
        self.gpu_device = gpu_device
        self.model_save_file = f"PNN_{global_specifier['nn_arch_name']}_{global_specifier['data_fold_name']}_{self.my_num_layer}L_best_model.pth"
        print(f"The current HSI_PNN_Trainer targets minimum_accuracy of {min_accuracy:.4f}")
        self.target_accuracy = min_accuracy
        hls_1st = global_specifier['1st_hl_size']
        # Initialize model
        self.input_bands = input_bands
        if num_layers_of_inherited_model == 0:
            pretrained_model = None
        else:
            pretrained_model = TumorClassifier2DCNN(num_input_channels=len(input_bands), hidden_layer_size_1st=hls_1st,
                                        gpu_device=gpu_device, num_layers=num_layers_of_inherited_model)
            pretrained_model.load_state_dict(torch.load(f"PNN_{global_specifier['nn_arch_name']}_{global_specifier['data_fold_name']}_{num_layers_of_inherited_model}L_best_model.pth"))            
        
        if num_layers_of_inherited_model == self.my_num_layer:
            # Attempt to fine-train the model of same numer of layers
            print(f"Fine-train with lower LR={learning_rate}, starting at the last saved best model: {self.model_save_file}")
            self.model = pretrained_model
        else:
            if num_layers_of_inherited_model == 0:
                print(f"Train with LR={learning_rate}, starting from scratch.")
            else:
                print(f"Train with LR={learning_rate}, starting by inheriting saved best model parameters from a shallower network: {self.model_save_file}")
            self.model = TumorClassifier2DCNN(num_input_channels=len(input_bands), hidden_layer_size_1st=hls_1st,
                                    gpu_device=gpu_device, num_layers=self.my_num_layer, inherited_model=pretrained_model)

        # This model.to() step might not be needed as the CNN instantiation above was executed with gpu_device specified.
        if gpu_device is not None:
            self.model.to(gpu_device)

        #Initialize loss function and optimizer.
        print(f"Class weight Ratio of training dataset = {class_weight_ratio:.4f}")
        class_weights = torch.tensor([class_weight_ratio], dtype=torch.float32, device=gpu_device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights) #For binary classification, this is perferred over nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)

    def train_with_validation(self, train_dataset, val_dataset, log_file_path, loss_th_to_stop=0.005, accrucy_th_to_stop=0.99, epochs=100):
        # Read the log file
        log_data = read_log_file(log_file_path)

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
            trn_size = len(y_true_trn)

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

                log_data[self.my_num_layer] = {
                    "nol": self.my_num_layer,
                    "fn": self.model_save_file,
                    "last_saved": datetime.fromtimestamp(time.time()).strftime('%m-%d-%Y_%H:%M:%S_PT'),
                    "avg_loss": best_loss,
                    "avg_accuracy": best_accuracy,
                }
                update_log_file(log_file_path, log_data)
                if (trn_loss < loss_th_to_stop) and (val_loss < loss_th_to_stop) and (trn_accuracy > accrucy_th_to_stop) and (val_accuracy > accrucy_th_to_stop):
                    break

        train_end_time = time.time()
        print(f"Training Complete. Elapsed seconds for this training & validation cycle: {(train_end_time - train_start_time):.2f}")
        return target_met

class HyperspectralNetworkTester():
    def __init__(self, num_layers, input_bands, global_specifier, gpu_device=None):
        self.my_num_layer = num_layers
        self.gpu_device = gpu_device

        self.model_save_file = f"PNN_{global_specifier['nn_arch_name']}_{global_specifier['data_fold_name']}_{self.my_num_layer}L_best_model.pth"

        hls_1st = global_specifier['1st_hl_size']
        # Initialize model
        self.input_bands = input_bands
        
        pretrained_model = TumorClassifier2DCNN(num_input_channels=len(input_bands), hidden_layer_size_1st=hls_1st,
                                            gpu_device=gpu_device, num_layers=self.my_num_layer)
        print(f"Test only with model parameters to be loaded from {self.model_save_file}")
        # In test-only mode, parameters will be loaded at test time.
        #pretrained_model.load_state_dict(torch.load(f"PNN_{global_specifier['nn_arch_name']}_{global_specifier['data_fold_name']}_{self.my_num_layer}L_best_model.pth"))            
        
        self.model = pretrained_model
        # This model.to() step might not be needed as the CNN instantiation above was executed with gpu_device specified.
        if gpu_device is not None:
            self.model.to(gpu_device)


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

mlp_steps_train_and_val = HSI_Globals.mlp_steps_train_and_val_for_pnn_3L_only
mlp_steps_testonly = HSI_Globals.mlp_steps_testonly_for_pnn
#workorders = HSI_Globals.work_orders_275B_train_and_test
#workorders = HSI_Globals.work_orders_32B_test_only
workorders = HSI_Globals.work_orders_32RB_train_and_test

mlp = HyperspectralWorkorderMLP(trainer_class=HyperspectralNetworkTrainer, tester_class=HyperspectralNetworkTester, log_file_prefix='mlp_pnn', attemp_gpu=True)
mlp.fill_orders(mlp_steps_train_and_val, mlp_steps_testonly, workorders)

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
import matplotlib.pyplot as plt

"""
This Python program uses three stages to train an 8 layer 2D CNN on HSI patches of shape (87, 87, 275) * float32.
The first stage uses 2% of the "good" dataset, and 3 selected bands to coarse-train the NN for weights that get close to
85% accuracy on this smaller dataset.
The second stage then uses 6.25% of the "good" data, 128 bands to fine-train the NN with initial weights inherited from
the coarse-train stage.
The last stage then uses 100% of the "good" data, 275 bands to fine-tune the NN with initial weights inherited from
the fine-train stage.
"""
ct_data_percentage = 0.02 # 2%
fine_train_percentage = 0.0625 # 6.25%
current_stage = 2 # 0: coarse-train; 1: fine-train; 2: fine-tune
patch_size = 87
max_num_NT_patches_to_use = (10858 + 2298 + 2285)
max_num_T_patches_to_use = (10858 + 2298 + 2285)
# root_dir = "D:/HistologyHSI/PKG - HistologyHSI-GB"
#root_dir = "C:/projects/P2"
root_dir = "ntp_90_90_275/"
# Initialize the dataset and DataLoader
# dataset = HyperspectralDataset(image_paths, labels, patch_size, blank_threshold=0.5, gpu_device=gpu_device)
# train_loader = data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda x: (torch.cat([item[0] for item in x]), torch.cat([item[1] for item in x])))
input_channels = 826//3
g_batch_size = 16

def getImagePathsWithLabels(path):
    """Lists all directories in the given path."""
    paths = []
    labels = []
    total_num_nt = 0
    total_num_t = 0
    for entry in os.listdir(path):
        next_level_path = os.path.join(path, entry)
        if os.path.isdir(next_level_path):
            current_label = 0
            if entry.endswith("_T"):
                current_label = 1
            elif entry.endswith("_T_50G"):
                current_label = 1
            #elif entry.endswith("_NT_50G"):
            #    current_label = 0
            elif entry.endswith("_NT"):
                current_label = 0
            else:
                continue
            for file_entry in os.listdir(next_level_path):
                if current_label == 0:
                    if total_num_nt >= max_num_NT_patches_to_use:
                        break
                    total_num_nt += 1
                else:
                    if total_num_t >= max_num_T_patches_to_use:
                        break
                    total_num_t += 1
                paths.append(os.path.join(next_level_path, file_entry))
                labels.append(current_label)
    return paths, labels


# Hyperspectral Dataset Class
class HyperspectralDataset(Dataset):
    def __init__(self, image_paths, labels, bands=range(0,275), patch_size=87, gpu_device=None):
        self.root_dir = root_dir
        self.gpu_device = gpu_device
        self.patch_size = patch_size
        self.image_paths = image_paths
        self.bands = bands
        self.labels = labels
        self.plot_only = False
        self.batch_indices = []
    
    def __len__(self):
        n = len(self.image_paths)
        n = (n//g_batch_size)*g_batch_size
        return n
    
    def __getitem__(self, idx):
        if self.plot_only:
            self.batch_indices.append(idx)
            patch_np = np.zeros((2, 2, 2), dtype=np.float32)
            label = 0
        else:
            patch_image_path = self.image_paths[idx]
            label = self.labels[idx]
            patch_np = np.load(patch_image_path)
            patch_np = patch_np[:self.patch_size, :self.patch_size, self.bands]
            #r_band, g_band, b_band = 425//3, 192//3, 109//3
            #patch_np = patch_np[:self.patch_size, :self.patch_size, [r_band, g_band, b_band]] #:]
        tensor_patch = torch.tensor(patch_np.transpose(2, 0, 1), dtype=torch.float32, device=self.gpu_device)
        tensor_label = torch.tensor(label, dtype=torch.long, device=self.gpu_device)
        return tensor_patch, tensor_label
    
    def set_plot_only(self, set_on):
        self.plot_only = set_on
        self.batch_indices = []

    def show_plot_patches_in_batch(self):    
        rgb_images = []
        for idx in self.batch_indices:
            patch_image_path = self.image_paths[idx]
            label = self.labels[idx]
            print(f"Label={label}, path={patch_image_path}")
            patch_np = np.load(patch_image_path)
            #patch_np = patch_np[:self.patch_size, :self.patch_size, :]
            rgb_images.append(patch_np[:, :, [r_band, g_band, b_band]])

        # Create the figure and subplots
        fig, axes = plt.subplots(4, 4, figsize=(10, 12))
        
        # Plot each image
        for i, ax in enumerate(axes.flat):
            ax.imshow(rgb_images[i])
            ax.axis('off')  # Hide the axes
            t = self.image_paths[self.batch_indices[i]]
            parts = t.split('/')
            del parts[0]
            t = '/'.join(parts)
            ax.set_title(t)
        
        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.show()
        self.batch_indices = []

# Define the CNN model (same as before)
class TumorClassifierCNN(nn.Module):
    def __init__(self, input_channels, gpu_device=None):
        super(TumorClassifierCNN, self).__init__()
        self.gpu_device = gpu_device
        # Define the Conv2D layer
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=256, kernel_size=3, stride=1, padding=0, device=gpu_device)  # (87, 87, 275) -> (85, 85, 256)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0, device=gpu_device)  # (85, 85, 256) -> (83, 83, 256)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0, device=gpu_device)  # (83, 83, 256) -> (81, 81, 512)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0, device=gpu_device)  # (81, 81, 512) -> (79, 79, 512)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=0, device=gpu_device) # -> (77, 77, 1024)
        self.conv6 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=0, device=gpu_device) # (77, 77, 1024) -> (75, 75, 1024)
        #self.conv4 = nn.Conv2d(in_channels=275, out_channels=256, kernel_size=3, stride=2, padding=0, device=gpu_device)  # -> (44, 44, 256)
        #self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0, device=gpu_device)  # -> (42, 42, 512)
        #self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=0, device=gpu_device) # -> (40, 40, 1024)
        self.conv7 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=0, device=gpu_device) #-> (28, 28, 1024)
        self.conv8 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=0, device=gpu_device) #-> (26, 26, 1024)
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # (71, 71, 1024) -> (1, 1, 1024)
        self.dl = nn.Linear(1024, 256)  # (1, 1024) -> (1, 256)
        # Fully connected (dense) layer
        self.fc = nn.Linear(256, 2)  # (1, 256) -> (1, 2)

        # Activation and Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.05)
        print("At CNN instantiation time, 1st layer weight shape is:", self.conv1.weight.shape)
        
    def forward(self, x):
        x = self.dropout(self.relu(self.conv1(x)))
        x = self.dropout(self.relu(self.conv2(x)))
        x = self.dropout(self.relu(self.conv3(x)))
        x = self.dropout(self.relu(self.conv4(x)))
        x = self.dropout(self.relu(self.conv5(x)))
        x = self.dropout(self.relu(self.conv6(x)))
        x = self.dropout(self.relu(self.conv7(x)))
        x = self.dropout(self.relu(self.conv8(x)))
        # Global average pooling
        x = self.pool(x)  # Shape: (batch_size, 1024, 1, 1)
        x = x.view(x.size(0), -1)    # Flatten to (batch_size, 1024)

        # Dense layer
        x = self.dropout(self.dl(x))
        # Fully connected layer
        x = self.dropout(self.fc(x))

        # Sigmoid activation for probabilities
        #probabilities = torch.sigmoid(x)
        return x

attempt_gpu = True
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

data_loading_start_time = time.time()

image_paths, labels = getImagePathsWithLabels(root_dir)

r_band, g_band, b_band = 425//3, 192//3, 109//3
ct_train_bands = [b_band, g_band, r_band]
ft_train_bands = range(1, 256, 2)
fine_tune_bands = range(input_channels)


# Split data for coarse-training (CT) and fine-training/fine-tine (FT) stages
# Step 1: Split the data into 60% train and 40% temporary (validation + test)
ft_paths, ct_paths, ft_labels, ct_labels = train_test_split(image_paths, labels, test_size=ct_data_percentage)
ct_train_paths, ct_val_paths, ct_train_labels, ct_val_labels = train_test_split(ct_paths, ct_labels, test_size=0.2)
ct_train_dataset = HyperspectralDataset(ct_train_paths, ct_train_labels, bands=ct_train_bands, patch_size=patch_size, gpu_device=gpu_device)
ct_val_dataset = HyperspectralDataset(ct_val_paths, ct_val_labels, bands=ct_train_bands, patch_size=patch_size, gpu_device=gpu_device)
ct_train_loader = DataLoader(ct_train_dataset, batch_size=g_batch_size, shuffle=True)
ct_val_loader = DataLoader(ct_val_dataset, batch_size=g_batch_size, shuffle=False)
ct_train_class_counts = Counter(ct_train_labels)
print("Coarse Training Image Dirstribution: " + str(len(ct_train_paths)) + " " + str(len(ct_val_paths)))
print("Coarse Training Dataset Dirstribution: " + str(len(ct_train_dataset)) + " " + str(len(ct_val_dataset)) )
print("Coarse Training True label distribution:", ct_train_class_counts, Counter(ct_val_labels))

if current_stage == 2:
    fine_stage_bands = fine_tune_bands
else:
    fine_stage_bands = ft_train_bands
# Split data for FT stage training and validation
# Step 1: Split the data into 70% train and 30% temporary (validation + test)
train_paths, temp_paths, train_labels, temp_labels = train_test_split(ft_paths, ft_labels, test_size=0.3) #.2
# Step 2: Split the temporary set into 50% validation and 50% test (i.e., 10% validation and 10% test)
val_paths, test_paths, val_labels, test_labels = train_test_split(temp_paths, temp_labels, test_size=0.5)
ft_train_dataset = HyperspectralDataset(train_paths, train_labels, bands=fine_stage_bands, patch_size=patch_size, gpu_device=gpu_device)
ft_val_dataset = HyperspectralDataset(val_paths, val_labels, bands=fine_stage_bands, patch_size=patch_size, gpu_device=gpu_device)
ft_test_dataset = HyperspectralDataset(test_paths, test_labels, bands=fine_stage_bands, patch_size=patch_size, gpu_device=gpu_device)

ft_train_class_counts = Counter(train_labels)
print("Image Dirstribution: " + str(len(train_paths)) + " " + str(len(val_paths)) + " " + str(len(test_paths)))
print("Dataset Dirstribution: " + str(len(ft_train_dataset)) + " " + str(len(ft_val_dataset)) + " " + str(len(ft_test_dataset)))
print("True label distribution:", ft_train_class_counts, Counter(val_labels), Counter(test_labels))

ft_train_loader = DataLoader(ft_train_dataset, batch_size=g_batch_size, shuffle=True)
ft_val_loader = DataLoader(ft_val_dataset, batch_size=g_batch_size, shuffle=False)
ft_test_loader = DataLoader(ft_test_dataset, batch_size=g_batch_size, shuffle=False)
data_loading_end_time = time.time()
print(f"Elapsed time for data loading into model: {(data_loading_end_time - data_loading_start_time):.4f} seconds")

class HyperspectralNetworkTrainer():
    def __init__(self, input_bands, train_stage_name='Default', class_weight_ratio=1.0, gpu_device=None, learning_rate=0.0001):
        self.gpu_device = gpu_device
        self.train_stage_name = train_stage_name
        self.model_save_file = train_stage_name +'_best_model.pth'
        self.target_val_accuracy = 0.8
        # Initialize model, loss function, and optimizer
        self.input_bands = input_bands
        self.model = TumorClassifierCNN(input_channels=len(input_bands), gpu_device=gpu_device)
        if gpu_device is not None:
            self.model.to(gpu_device)
        print(f"Class weight Ratio = {class_weight_ratio:.4f}")
        class_weights = torch.tensor([1.0, class_weight_ratio], dtype=torch.float32, device=gpu_device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5) #0.002

    def expand_and_migrate(self, new_bands, train_stage_name='Default', class_weight_ratio=1.0, learning_rate=0.00015, saved_model_to_load=None):
        if saved_model_to_load is not None:
            # Load model for prediction
            self.model.load_state_dict(torch.load(saved_model_to_load))
            if self.gpu_device is not None:
                self.model.to(self.gpu_device)
        source_bands = self.input_bands
        self.input_bands = new_bands
        original_weights = self.model.conv1.weight
        print("New bands size=", len(new_bands))
        # Initialize expanded weights
        self.model.conv1 = nn.Conv2d(in_channels=len(new_bands), out_channels=256, kernel_size=3, stride=1, padding=0, device=gpu_device)  # (87, 87, 275) -> (85, 85, 256)        
        expanded_weights = self.model.conv1.weight

        ratio_bands = len(new_bands) // len(source_bands)
        '''
        # Create a map to count how many new_bands correspond to each source_band
        band_counts = {b: 0 for b in source_bands}
        
        # Calculate band counts
        
            for nb in new_bands:
                if nb < source_bands[0]:  # Case: nb is less than the first source band
                    band_counts[source_bands[0]] += 1
                elif nb > source_bands[-1]:  # Case: nb is greater than the last source band
                    band_counts[source_bands[-1]] += 1
                else:
                    for i in range(len(source_bands) - 1):
                        if source_bands[i] <= nb <= source_bands[i + 1]:
                            mid = (source_bands[i] + source_bands[i + 1]) // 2
                            selected_band = source_bands[i + 1] if nb > mid else source_bands[i]
                            band_counts[selected_band] += 1
                            break
        '''                
        with torch.no_grad():
            source_band_idx = 0            
            # Expand weights based on counts
            for nb_idx, nb in enumerate(new_bands):
                '''
                if nb < source_bands[0]:  # Case: nb is less than the first source band
                    selected_band = source_bands[0]
                elif nb > source_bands[-1]:  # Case: nb is greater than the last source band
                    selected_band = source_bands[-1]
                else:
                    for i in range(len(source_bands) - 1):
                        if source_bands[i] <= nb <= source_bands[i + 1]:
                            mid = (source_bands[i] + source_bands[i + 1]) // 2
                            selected_band = source_bands[i + 1] if nb > mid else source_bands[i]
                            break
                        
                divisor = band_counts[selected_band]  # Number of new bands sharing this source band
                source_band_idx = source_bands.index(selected_band)
                expanded_weights[:, nb_idx, :, :] = original_weights[:, source_band_idx, :, :] #/ divisor
                '''
                if source_band_idx >= len(source_bands):
                    expanded_weights[:, nb_idx, :, :] = 0.0
                    continue
                orig_source_band_idx = source_band_idx
                selected_band = source_bands[source_band_idx]
                delta = selected_band - nb
                if ratio_bands > 5: 
                    if delta >= 5: #7:
                        expanded_weights[:, nb_idx, :, :] = 0.0
                    #elif delta == 5 or delta == -5 or delta == 6 or delta == -6 :    
                    #    expanded_weights[:, nb_idx, :, :] = original_weights[:, source_band_idx, :, :] / 6.0
                    elif delta == 3 or delta == -3 or delta == 4 or delta == -4 :
                        expanded_weights[:, nb_idx, :, :] = original_weights[:, source_band_idx, :, :] / 4.0
                    elif delta == 2 or delta == -2:
                        expanded_weights[:, nb_idx, :, :] = original_weights[:, source_band_idx, :, :] / 2.0
                    elif delta == 0 or delta == 1 or delta == -1:
                        expanded_weights[:, nb_idx, :, :] = original_weights[:, source_band_idx, :, :]
                    else:
                        source_band_idx += 1
                        expanded_weights[:, nb_idx, :, :] = 0.0
                else:
                    if delta == 0:
                        expanded_weights[:, nb_idx, :, :] = original_weights[:, source_band_idx, :, :]
                        source_band_idx += 1
                    elif delta == 1:
                        expanded_weights[:, nb_idx, :, :] = original_weights[:, source_band_idx, :, :] / 8.0                    
                        

                print(f"nb={nb}, source_band_idx={orig_source_band_idx}, selected_band={selected_band}, e_w vs o_w={expanded_weights[0, nb_idx, 0, 0]} v {original_weights[0, orig_source_band_idx, 0, 0]}")

        print("Original 1st layer weight shape is:", original_weights.shape)
        print("New 1st layer weight shape is:", expanded_weights.shape)
        print(f"New class weight Ratio = {class_weight_ratio:.4f}")
        class_weights = torch.tensor([1.0, class_weight_ratio], dtype=torch.float32, device=gpu_device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5) #0.002
        self.train_stage_name = train_stage_name
        self.model_save_file = train_stage_name +'_best_model.pth'
        return True


    # Training loop
    #train_dataset.set_plot_only(True)
    #val_dataset.set_plot_only(True)
    #test_dataset.set_plot_only(True)
    def train_with_validation(self, train_loader, train_dataset, val_loader, val_dataset,
                            loss_th_to_stop=0.015, accrucy_th_to_stop=0.99, epochs=100, num_batches_to_show_time_elapsed=10):
        train_start_time = time.time()
        epoch_start_time = train_start_time
        best_val_loss = float('inf')  # Track the best validation loss for model saving
        target_met = False
        for epoch in range(epochs):    
            self.model.train()
            running_loss = 0.0
            batch_count = 0
            time3 = time.time()
            for inputs, labels in train_loader:
                if batch_count == 0:
                    time2 = time.time()            
                    #print(f'The last batch get-item took {time2-time3:.4f} seconds')

                if train_dataset.plot_only:
                    train_dataset.show_plot_patches_in_batch()
                    continue

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                #loss.backward()
                loss.backward(retain_graph=True)
                self.optimizer.step()
                loss_item = loss.item()
                running_loss += loss_item

                batch_count += 1
                if batch_count == num_batches_to_show_time_elapsed:
                    time3 = time.time()
                    batch_count = 0            
                    print(f'The last {num_batches_to_show_time_elapsed} batches of {self.train_stage_name} training took {time3-time2:.4f} seconds.')
                    print("Dense layer 1st 16 weights:", self.model.fc.weight.data.view(-1)[:16])  # Should not be None
                    #print("Dense layer weight gradient:", model.fc.weight.grad)  # Should not be None
                    print("Batch loss:", loss_item)

            # Validation
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            last_batch_labels = None
            last_batch_predicted = None
            with torch.no_grad():
                batch_count = 0
                time3 = time.time()
                for inputs, labels in val_loader:
                    if batch_count == 0:
                        time2 = time.time()            
                        #print(f'The last batch get-item took {time2-time3:.4f} seconds')
                    if val_dataset.plot_only:
                        val_dataset.show_plot_patches_in_batch()
                        continue
                    
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
                    #_, predicted = torch.max(outputs, 1)
                    predicted = torch.argmax(outputs, dim=1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    batch_count += 1
                    if batch_count == num_batches_to_show_time_elapsed:
                        time3 = time.time()
                        batch_count = 0
                        print(f'The last {num_batches_to_show_time_elapsed} batches of {self.train_stage_name} validation took {time3-time2:.4f} seconds.')
                        print("truth labels:", labels)
                        print("predicted:", predicted)
                    else:
                        last_batch_labels = labels
                        last_batch_predicted = predicted
                if batch_count>0:
                    print("truth labels:", last_batch_labels)
                    print("predicted:", last_batch_predicted)

            if val_dataset.plot_only:
                pass
            else:
                epoch_end_time = time.time()
                tl = running_loss/len(train_loader)
                vl = val_loss/len(val_loader)
                va = correct / total
                print(f"Epoch [{epoch+1}/{epochs}], "
                      f"Train Loss: {tl:.4f}, "
                      f"Val Loss: {vl:.4f}, "
                      f"Val Accuracy: {100 * va:.2f}%,",
                      f"Time Taken: {(epoch_end_time-epoch_start_time):.4f}")

                epoch_start_time = epoch_end_time
                    # Save the model if validation loss improves
                if (vl < best_val_loss) and (va > self.target_val_accuracy):
                    best_val_loss = vl
                    target_met = True
                    torch.save(self.model.state_dict(), self.model_save_file) #'best_model.pth')
                    print(f"{self.train_stage_name} training model saved with val loss: {best_val_loss:.4f}, and val accuracy: {va:.4f}")

                if (tl < loss_th_to_stop) and (vl < loss_th_to_stop) and (va > accrucy_th_to_stop):
                    break

        print("Training Complete.")
        train_end_time = time.time()
        print(f"Elapsed time for {self.train_stage_name} training model: {(train_end_time - train_start_time):.4f} seconds")
        return target_met

    def test_model(self, test_loader, test_dataset):
        test_start_time = time.time()
        all_labels = []
        all_preds = []
        self.model.eval()
        with torch.no_grad():  # Disable gradient computation for efficiency
            for inputs, labels in test_loader:
                if test_dataset.plot_only:
                    test_dataset.show_plot_patches_in_batch()
                    continue
                outputs = self.model(inputs)    
                #_, predicted = torch.max(outputs, 1)
                predicted = torch.argmax(outputs, dim=1)
                # Collect all predictions and labels
                all_labels.extend(labels.cpu().numpy())  # Move to CPU if necessary
                all_preds.extend(predicted.cpu().numpy())                

        test_end_time = time.time()
        print(f"Time taken to complete Testing: {(test_end_time-test_start_time):.4f} seconds")
        print("True label distribution:", Counter(all_labels))
        print("Predicted label distribution:", Counter(all_preds))

        if test_dataset.plot_only:
            pass
        else:        
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

go_through_coarse_training = False
go_through_fine_training = False
if current_stage==0 or current_stage==1:
    trainer = HyperspectralNetworkTrainer(ct_train_bands, train_stage_name='Coarse', class_weight_ratio = ct_train_class_counts[0] / ct_train_class_counts[1], gpu_device=gpu_device)
else:
    trainer = HyperspectralNetworkTrainer(ft_train_bands, train_stage_name='Fine-Train', class_weight_ratio = ft_train_class_counts[0] / ft_train_class_counts[1], gpu_device=gpu_device)    
if current_stage==0:
    is_trained_well = trainer.train_with_validation(ct_train_loader, ct_train_dataset, ct_val_loader, ct_val_dataset,
                                  loss_th_to_stop=0.02, accrucy_th_to_stop=0.9, epochs=100, num_batches_to_show_time_elapsed=20)
    #if is_trained_well:
    #    is_trained_well = trainer.expand_and_migrate(ft_train_bands, train_stage_name ='Fine', class_weight_ratio=ft_train_class_counts[0] / ft_train_class_counts[1])
else:
    if current_stage==1:
        is_trained_well = trainer.expand_and_migrate(ft_train_bands, train_stage_name ='Fine', class_weight_ratio=ft_train_class_counts[0] / ft_train_class_counts[1],
                                saved_model_to_load="Coarse_best_model.pth", learning_rate=0.00015)
    else:
        is_trained_well = trainer.expand_and_migrate(fine_tune_bands, train_stage_name ='Fine-Tune', class_weight_ratio=ft_train_class_counts[0] / ft_train_class_counts[1],
                        saved_model_to_load="Fine-Train_best_model.pth", learning_rate=0.0002)    

if is_trained_well:
    if current_stage==1 or current_stage==2:        
        is_trained_well = trainer.train_with_validation(ft_train_loader, ft_train_dataset, ft_val_loader, ft_val_dataset,                              
                              loss_th_to_stop=0.01, accrucy_th_to_stop=0.99, epochs=25, num_batches_to_show_time_elapsed=100)
    if is_trained_well:
        trainer.test_model(ft_test_loader, ft_test_dataset)
    else:
        print("Stopped after Coarse Train due to failure to meet accuracy target.")
else:
    print("Stopped after Coarse Train due to failure to meet accuracy target.")
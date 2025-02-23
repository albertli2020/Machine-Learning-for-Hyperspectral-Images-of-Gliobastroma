import numpy as np
import os
import random
import glob
import torch
from torch.utils.data import Dataset

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
    def __init__(self, image_file_list_read_from, batch_size, shuffle_files=False, split_for_val=None, duplicatePos=False, randomize_pos_data=False, bands=range(0,275), patch_size=87, gpu_device=None):
        self.gpu_device = gpu_device
        self.patch_size = patch_size

        #print(image_file_list_read_from)
        image_paths = read_dataset_file_list(image_file_list_read_from, duplicatePos=duplicatePos)
        if shuffle_files:
            random.shuffle(image_paths)
        if split_for_val is not None:
            split_index = int(len(image_paths) * split_for_val) #TBF
            image_paths = image_paths[:split_index]

        self.image_paths = image_paths        
        self.labels = labelHSIDataSet(image_paths)

        self.bands = bands
        self.batch_size = batch_size
        self.rpd = randomize_pos_data
        # Total number of patches
        self.total_images = len(self.image_paths)
        self.indices = np.arange(self.total_images)

    def __len__(self):
        n = len(self.image_paths)
        n_batches = n//self.batch_size
        delta = n - (n_batches*self.batch_size)
        if delta > 1:
            n_batches += 1
        return n_batches
    
    def __getitem__(self, idx):
        #patch_np = patch_np[:self.patch_size, :self.patch_size, self.bands]
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.total_images)
        batch_indices = self.indices[start_idx:end_idx]
        
        #y_batch = labelHSIDataSet(self.image_paths[start_idx:end_idx])
        y_batch = self.labels[start_idx:end_idx]

        # Dynamically load the image patches        
        X_batch = []
        if self.rpd:
            for i in batch_indices:
                image = np.load(self.image_paths[i])  # Assuming each file is a numpy array of the patch
                #image = image[:self.patch_size, :self.patch_size, :]
                X_batch.append(image)
                    
            for i in range(len(y_batch)):
                image = X_batch[i]
                max_offset = image.shape[0] - self.patch_size
                if y_batch[i] == 1:
                    rr = random.randint(0, max_offset)
                    rc = random.randint(0, max_offset)
                else:
                    rr = 0
                    rc = 0                
                X_batch[i] = image[rr:rr+self.patch_size, rc:rc+self.patch_size, self.bands]
        else:
            for i in batch_indices:
                image = np.load(self.image_paths[i])  # Assuming each file is a numpy array of the patch
                image = image[:self.patch_size, :self.patch_size, self.bands]
                X_batch.append(image)
        
        # Convert to numpy arrays
        X_batch = np.array(X_batch)
        #print(X_batch.shape)        
        y_batch = np.array(y_batch)        
    
        X_batch = X_batch.transpose(0, 3, 1, 2)
        tensor_patch = torch.tensor(X_batch, dtype=torch.float32).to(self.gpu_device)
        tensor_label = torch.tensor(y_batch, dtype=torch.float32).to(self.gpu_device)
        return tensor_patch, tensor_label
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import joblib
import os
import time
import glob
import random
from collections import Counter

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


class HSIImageDataset:
    def __init__(self, image_files, labels, batch_size, patch_size=(87, 87)):
        """
        Initializes the dataset.
        
        Args:
            image_dir (str): Directory where HSI image files are stored.
            label_file (str): Path to the file containing labels for the images.
            batch_size (int): Number of patches to load per batch.
            patch_size (tuple): Dimensions of each patch (default: (87, 87)).
        """
        # self.image_dir = image_dir
        # self.label_file = label_file
        self.batch_size = batch_size
        self.patch_size = patch_size

        # Load the list of file names and labels
        # t_images, nt_images = generatePatchLists(image_dir)
        self.image_files = image_files
        # print(self.image_files)
        # self.image_files = sorted(os.listdir(image_dir))
        # self.labels = np.load(label_file)  # Labels should match the image files
        self.labels = labels
        # assert len(self.image_files) == len(self.labels), "Mismatch in images and labels."
        
        # Total number of patches
        self.total_images = len(self.image_files)
        self.indices = np.arange(self.total_images)

    def __len__(self):
        """
        Returns the number of batches in the dataset.
        """
        return int(np.ceil(self.total_images / self.batch_size))

    def __getitem__(self, index):
        """
        Returns a batch of data dynamically loaded.
        
        Args:
            index (int): Batch index.
        
        Returns:
            X_batch (np.array): Batch of image patches.
            y_batch (np.array): Corresponding labels.
        """
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.total_images)
        batch_indices = self.indices[start_idx:end_idx]

        # Dynamically load the image patches
        X_batch = []
        # y_batch = []

        for idx in batch_indices:
            # img_path = os.path.join(self.image_dir, self.image_files[idx])
            image = np.load(self.image_files[idx])  # Assuming each file is a numpy array of the patch
            # X_batch.append(image[:self.patch_size, :self.patch_size, :])
            X_batch.append(image)
            # y_batch.append(self.labels[idx])
        
        # Convert to numpy arrays
        y_batch = labelHSIDataSet(self.image_files[start_idx:end_idx])
        # print(y_batch)
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)

        # Reshape X_batch to (batch_size, num_features)
        batch_size = X_batch.shape[0]
        num_features = np.prod(X_batch.shape[1:])  # Flatten all dimensions except batch size
        X_batch = X_batch.reshape(batch_size, num_features)

        return X_batch, y_batch

# Parameters
image_dir = 'C:/HSI-ML/ntp_90_90_275'  # Directory containing image files
#label_file = './labels.npy'  # Path to the labels file
batch_size = 500

start_time = time.time()

patch_list, t_roi_list, nt_roi_list, t_roi_dir, nt_roi_dir = generateLists(image_dir)
training_set, val_set, test_set = splitDataset(patch_list, t_roi_dir, nt_roi_dir, test_roi_sets[0], val_roi_sets[0])
training_set += val_set
training_labels = labelHSIDataSet(training_set)
test_labels = labelHSIDataSet(test_set)
print("Total number of training data:", len(training_set))
print("Trainging label distribution:", Counter(training_labels))
print("Total number of test data:", len(test_set))
print("Predicted label distribution:", Counter(test_labels))

# Create the dataset object
dataset = HSIImageDataset(training_set, training_labels, batch_size)

# Access a single batch
#for epoch in range(5):  # Loop through multiple epochs
  #  print(f"\nEpoch {epoch + 1}:")
  #  for batch_idx in range(len(dataset)):
  #      X_batch, y_batch = dataset[batch_idx]
  #      print(f"Batch {batch_idx + 1}: X shape = {X_batch.shape}, y shape = {y_batch.shape}")



# Initialize model and scaler
model = SGDClassifier(loss='log_loss', max_iter=1, tol=None, warm_start=True)
scaler = StandardScaler()

# Training Loop with Metrics
epochs = 5
metrics_per_epoch = []

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    y_true_epoch = []
    y_pred_epoch = []

    for batch_idx in range(len(dataset)):
        # Load a batch
        X_batch, y_batch = dataset[batch_idx]  # X_batch shape: (batch_size, num_features)

        # Scale the data
        if epoch == 0:
            scaler.partial_fit(X_batch)
        X_batch = scaler.transform(X_batch)

        # Incremental training
        model.partial_fit(X_batch, y_batch, classes=np.unique(dataset.labels))
        # model.partial_fit(X_batch, y_batch, classes=[0,1])
        
        # Predictions for this batch (optional, for metrics)
        y_pred_batch = model.predict(X_batch)
        y_true_epoch.extend(y_batch)
        y_pred_epoch.extend(y_pred_batch)
        
        #print(f"Trained on batch {batch_idx + 1}/{len(dataset)}")

    # Compute metrics for the epoch
    epoch_loss = log_loss(y_true_epoch, y_pred_epoch)
    epoch_accuracy = accuracy_score(y_true_epoch, y_pred_epoch)
    print(f"Epoch {epoch + 1} Training loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.4f}")

    # Store metrics
    metrics_per_epoch.append({
        'epoch': epoch + 1,
        'training loss': epoch_loss,
        'accuracy': epoch_accuracy,
    })

end_training_time = time.time()
elapsed_training_time = end_training_time - start_time
print(f"Training Elapsed time: {elapsed_training_time} seconds")

# Print final evaluation metrics on test set
print("\nFinal Evaluation on Test Set:")
test_dataset = HSIImageDataset(test_set, test_labels, batch_size)
# X_test, y_test = test_dataset[random.randint(0, len(test_dataset))]  # Replace with a dedicated test dataset
# X_test, y_test = test_dataset[0]
# X_test = scaler.transform(X_test)
# y_pred_test = model.predict(X_test)
# Transform the entire dataset
# X_test = scaler.transform([sample[0] for sample in test_dataset])
# y_test = [sample[1] for sample in test_dataset]

y_test = []
y_pred_test = []
for batch_idx in range(len(test_dataset)):
        # Load a batch
        X_batch, y_batch = test_dataset[batch_idx]  # X_batch shape: (batch_size, num_features)

        # Scale the data
        X_batch = scaler.transform(X_batch)

        y_pred_batch = model.predict(X_batch)
        y_test.extend(y_batch)
        y_pred_test.extend(y_pred_batch)

# Predict for the entire dataset
# y_pred_test = model.predict(X_test)

end_test_time = time.time()
elapsed_test_time = end_test_time - end_training_time
print(f"Training Elapsed time: {elapsed_test_time} seconds")

# Print detailed metrics
print("Classification Report:")
print(classification_report(y_test, y_pred_test))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_test))

joblib.dump(model, 'logistic_regression_hsi_dynamic.pkl')
print("\nModel saved as 'logistic_regression_hsi_dynamic.pkl'.")

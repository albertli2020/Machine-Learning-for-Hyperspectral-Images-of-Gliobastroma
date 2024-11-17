import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
#from torchvision.transforms import ToTensor
import numpy as np
from sklearn.model_selection import train_test_split

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.001
max_num_NT_patches_to_use = (10858 + 2298 + 2285)//40
max_num_T_patches_to_use = (10858 + 2298 + 2285)//40
root_dir = "ntp_90_90_275/"
input_channels = 826//3

#DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
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


# Custom Dataset
class HyperspectralDataset(Dataset):
    def __init__(self, image_paths, labels, patch_size=87, gpu_device=None):
        self.image_paths = image_paths
        self.labels = labels
        self.patch_size = patch_size
        self.gpu_device = gpu_device
    
    def __len__(self):
        n = len(self.image_paths)
        n = (n//BATCH_SIZE)*BATCH_SIZE
        return n
    
    def __getitem__(self, idx):
        patch_image_path = self.image_paths[idx]
        label = self.labels[idx]
        patch_np = np.load(patch_image_path)
        patch_np = patch_np[:self.patch_size, :self.patch_size, :]
        tensor_patch = torch.tensor(patch_np.transpose(2, 0, 1), dtype=torch.float32, device=self.gpu_device)
        tensor_label = torch.tensor(label, dtype=torch.long, device=self.gpu_device)
        return tensor_patch, tensor_label

# Define VSM model
class VSMModel(nn.Module):
    def __init__(self, input_dim):
        super(VSMModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        classification = self.classifier(latent)
        return reconstruction, classification

# Load Data
def load_data(image_dir, test_size=0.2, val_size=0.1):
    image_paths, labels = getImagePathsWithLabels(image_dir)
    dataset = HyperspectralDataset(image_paths, labels, gpu_device=gpu_device)
    train_data, test_data = train_test_split(dataset, test_size=test_size, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=42)
    return train_data, val_data, test_data

# Train function
def train_model(model, dataloader, optimizer, criterion_recon, criterion_class):
    model.train()
    running_loss = 0
    for inputs, labels in dataloader:
        #inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        reconstruction, classification = model(inputs)
        
        loss_recon = criterion_recon(reconstruction, inputs)
        loss_class = criterion_class(classification.squeeze(), labels.float())
        loss = loss_recon + loss_class
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

# Validation function
def validate_model(model, dataloader, criterion_recon, criterion_class):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            #inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            reconstruction, classification = model(inputs)
            loss_recon = criterion_recon(reconstruction, inputs)
            loss_class = criterion_class(classification.squeeze(), labels.float())
            loss = loss_recon + loss_class
            running_loss += loss.item()
    return running_loss / len(dataloader)

# Prediction function
def predict_model(model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            #inputs = inputs.to(DEVICE)
            _, classification = model(inputs)
            preds = (classification.squeeze() > 0.5).float().cpu().numpy()
            predictions.extend(preds)
    return predictions

# Main
if __name__ == '__main__':
    # Paths
    IMAGE_DIR = root_dir

    # Data Preparation
    train_data, val_data, test_data = load_data(IMAGE_DIR)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Model, Loss, Optimizer
    input_dim = 87 #input_channels  # Example: number of spectral bands
    model = VSMModel(input_dim).to(gpu_device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_recon = nn.MSELoss()
    criterion_class = nn.BCELoss()

    # Training Loop
    for epoch in range(EPOCHS):
        train_loss = train_model(model, train_loader, optimizer, criterion_recon, criterion_class)
        val_loss = validate_model(model, val_loader, criterion_recon, criterion_class)
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Prediction
    predictions = predict_model(model, test_loader)
    print("Predictions:", predictions)

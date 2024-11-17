import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import numpy as np
from sklearn.model_selection import train_test_split

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Custom Dataset
class HyperspectralDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.file_names = os.listdir(image_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.image_dir, self.file_names[idx])
        data = np.load(file_path)  # Assuming .npy files
        image = data['image']  # Replace 'image' with appropriate key
        label = data['label']  # Replace 'label' with appropriate key
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

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
    dataset = HyperspectralDataset(image_dir, transform=ToTensor())
    train_data, test_data = train_test_split(dataset, test_size=test_size, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=42)
    return train_data, val_data, test_data

# Train function
def train_model(model, dataloader, optimizer, criterion_recon, criterion_class):
    model.train()
    running_loss = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
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
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
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
            inputs = inputs.to(DEVICE)
            _, classification = model(inputs)
            preds = (classification.squeeze() > 0.5).float().cpu().numpy()
            predictions.extend(preds)
    return predictions

# Main
if __name__ == '__main__':
    # Paths
    IMAGE_DIR = 'path/to/hyperspectral/patches'

    # Data Preparation
    train_data, val_data, test_data = load_data(IMAGE_DIR)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Model, Loss, Optimizer
    input_dim = 200  # Example: number of spectral bands
    model = VSMModel(input_dim).to(DEVICE)
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

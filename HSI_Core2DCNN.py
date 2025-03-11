import torch
import torch.nn as nn

# Define Simpler 2-L 2DCNN Model for 3D Hyperspectral Input
class S2L2DCNNModel(nn.Module):
    def __init__(self, gpu_device=None, inherited_model=None):
        super(S2L2DCNNModel, self).__init__()
        if inherited_model is not None:
           self.conv1 = inherited_model.conv1
        else:
            self.conv1 = nn.Conv2d(275, 64, kernel_size=3, padding=1, device=gpu_device)  # 275 bands as input channels
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2
        if inherited_model is not None:
            self.conv2 = inherited_model.conv2
        else:
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, device=gpu_device)
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
    
# Define the Core 2DCNN model
class TumorClassifier2DCNN(nn.Module):
    def init_2D(self, num_input_channels, num_output_channels_1st, num_layers, gpu_device):
        if num_output_channels_1st <= 32:
            num_output_channels_2nd = 64
            num_output_channels_3rd = 128
        elif num_output_channels_1st <= 64:
            num_output_channels_2nd = 128
            num_output_channels_3rd = 256
        else:
            num_output_channels_2nd = 256
            num_output_channels_3rd = 512

        self.dl_inputs = num_output_channels_1st
        if self.conv1 is None:            
            self.conv1 = nn.Conv2d(in_channels=num_input_channels, out_channels=num_output_channels_1st, kernel_size=3, stride=1, padding=0, device=gpu_device)  # (87, 87, 275) -> (85, 85, 256)
            
        if num_layers > 1:
            self.dl_inputs = num_output_channels_2nd
            if self.conv2 is None:
                self.conv2 = nn.Conv2d(in_channels=num_output_channels_1st, out_channels=num_output_channels_2nd, kernel_size=3, stride=1, padding=0, device=gpu_device)  # (85, 85, 256) -> (83, 83, 256)
            
        if num_layers > 2:
            self.dl_inputs = num_output_channels_3rd
            if self.conv3 is None:
                self.conv3 = nn.Conv2d(in_channels=num_output_channels_2nd, out_channels=num_output_channels_3rd, kernel_size=3, stride=1, padding=0, device=gpu_device)  # (83, 83, 256) -> (81, 81, 512)
            
        if num_layers > 3:
            self.dl_inputs = 512
            if self.conv4 is None:
                self.conv4 = nn.Conv2d(in_channels=num_output_channels_3rd, out_channels=512, kernel_size=3, stride=1, padding=0, device=gpu_device)  # (81, 81, 512) -> (79, 79, 512)


    def __init__(self, num_input_channels, hidden_layer_size_1st, gpu_device=None, num_layers=1, inherited_model=None):
        super(TumorClassifier2DCNN, self).__init__()
        self.gpu_device = gpu_device       
               
        self.conv1 = None
        self.conv2 = None
        self.conv3 = None
        self.conv4 = None
        self.dl = None

        # Define the Conv layers
        if inherited_model is not None:
            self.conv1 = inherited_model.conv1
        if num_layers > 1:
            if inherited_model is not None:
                self.conv2 = inherited_model.conv2
        if num_layers > 2:
            if inherited_model is not None:
                self.conv3 = inherited_model.conv3
        if num_layers > 3:
            if inherited_model is not None:
                self.conv4 = inherited_model.conv4

        self.init_2D(num_input_channels, hidden_layer_size_1st, num_layers, gpu_device)
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # (79, 79, 512) -> (1, 1, 512)
        if self.dl_inputs > 256:
            self.dl = nn.Linear(self.dl_inputs, 256)  # (1, 512) -> (1, 256)

        if self.dl_inputs < 256:
            # Fully connected (dense) layer
            self.fc = nn.Linear(self.dl_inputs, 1) #2)  # (1, 256) -> (1, 1)
        else:
            # Fully connected (dense) layer
            self.fc = nn.Linear(256, 1) #2)  # (1, 256) -> (1, 1)

        # Activation and Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.dropout(self.relu(self.conv1(x)))
        if self.conv2 is not None:
            x = self.dropout(self.relu(self.conv2(x)))
        if self.conv3 is not None:        
            x = self.dropout(self.relu(self.conv3(x)))
        if self.conv4 is not None:
            x = self.dropout(self.relu(self.conv4(x)))
        # Global average pooling
        x = self.pool(x)  # Shape: (batch_size, 1024, 1, 1)
        x = x.view(x.size(0), -1)    # Flatten to (batch_size, 1024)

        # Dense layer
        if self.dl is not None:
            x = self.dropout(self.relu(self.dl(x)))
        
        x = self.fc(x)
        return x
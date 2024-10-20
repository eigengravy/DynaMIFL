import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 32 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 64 channels
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # 64 channels
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layer
        self.fc1 = nn.Linear(64 * 4 * 4, 64)  # Output size after convolutions and pooling
        self.fc2 = nn.Linear(64, num_classes)  # Final output layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the tensor
        x = x.view(-1, 64 * 4 * 4)  # Reshape for the fully connected layer
        
        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Final layer (no activation here; typically handled by loss function)
        
        return x
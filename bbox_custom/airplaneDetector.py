import torch
import torch.nn as nn
import torch.nn.functional as F

class AirplaneDetector(nn.Module):
    def __init__(self, width=128, height=128, num_classes=2):
        super(AirplaneDetector, self).__init__()
        self.width = width
        self.height = height
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((height // 8, width // 8))  # Adjusted to adapt to input size
        
        # Fully connected layers for classification and bounding box regression
        self.fc_cls = nn.Linear(64 * (height // 8) * (width // 8), num_classes)
        self.fc_bb = nn.Linear(64 * (height // 8) * (width // 8), 4)

    def forward(self, x):
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        
        # Adaptive pooling to adapt to variable input sizes
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classification output
        cls_output = self.fc_cls(x)
        
        # Bounding box regression output
        bb_output = self.fc_bb(x)
        
        return cls_output, bb_output

def sample():
    # Example usage with different input width and height
    width = 256
    height = 256
    model = AirplaneDetector(width=width, height=height)
    
    batch_size = 1  # For demonstration purposes, using a single sample
    channels = 3  # RGB channels
    random_input = torch.randn(batch_size, channels, height, width)
    print(model)
    cls_output, bb_output = model(random_input)
    
    # Print the result
    print("Classification output:", cls_output)
    print("Bounding box regression output:", bb_output)


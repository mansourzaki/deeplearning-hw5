# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        ResBlock with optional downsampling when stride > 1
        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            stride: stride for the first convolution and downsample path
        """
        super().__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut path
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()
    
    def forward(self, x):
        identity = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add shortcut
        out += self.shortcut(identity)
        out = F.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResBlocks
        self.layer1 = ResBlock(64, 64, 1)
        self.layer2 = ResBlock(64, 128, 2)
        self.layer3 = ResBlock(128, 256, 2)
        self.layer4 = ResBlock(256, 512, 2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final fully connected layer
        self.fc = nn.Linear(512, 2)
        
    def forward(self, x):
        # Initial convolution and pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        # ResBlocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling and flatten
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Final fully connected layer and sigmoid
        x = self.fc(x)
        x = torch.sigmoid(x)
        
        return x

    def __str__(self):
        return "ResNet" 
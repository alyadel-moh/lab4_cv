import torch
import torch.nn as nn

class DepthwiseSeparableBlock(nn.Module):
    """
    Depthwise Conv & Pointwise Conv
    """
    def __init__(self, in_channels, out_channels, stride):
        super(DepthwiseSeparableBlock, self).__init__()
        
        # Depthwise Convolution (groups=in_channels ensures 1 filter per channel)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                   stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Pointwise Convolution (1x1 conv to combine channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class MobileNet(nn.Module):
    """
    MobileNetV1 Architecture configured for 224x224 input.
    """
    def __init__(self, num_classes=4):
        super(MobileNet, self).__init__()
        
        # Initial 3x3 convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Architecture from original Paper
        # Format: (out_channels, stride)
        self.config = [
            (64, 1),
            (128, 2), (128, 1),
            (256, 2), (256, 1),
            (512, 2), (512, 1), (512, 1), (512, 1), (512, 1), (512, 1),
            (1024, 2), (1024, 1)
        ]
        
        self.layers = self._make_layers(in_channels=32)
        
        # Classification Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def _make_layers(self, in_channels):
        layers = []
        for out_channels, stride in self.config:
            layers.append(DepthwiseSeparableBlock(in_channels, out_channels, stride))
            in_channels = out_channels # Update input channels for the next block
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
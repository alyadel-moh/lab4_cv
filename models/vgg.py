import torch.nn as nn
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import NUM_CLASSES
from torchvision.models import vgg16_bn, VGG16_BN_Weights

class VGG16(nn.Module):
    def __init__(self,num_classes = NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(

            # Block 1 - 64 filters

            nn.Conv2d(3,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            # 112 x 112

            # Block 2 — 128 filters

            nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # 56 x 56

            # Block 3 — 256 filters

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),    
            
            # 28×28

            # Block 4 — 512 filters

            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # 14×14                                   
 
            # Block 5 — 512 filters

            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), 

            # 7×7
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))

        self.classifier = nn.Sequential(
            nn.Linear(512*7*7,4096),nn.ReLU(inplace=True),nn.Dropout(0.5),
            nn.Linear(4096,4096),nn.ReLU(inplace=True),nn.Dropout(0.5),
            nn.Linear(4096,num_classes)
        )

        pretrained = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
        self.features.load_state_dict(pretrained.features.state_dict())

        
        for param in self.features.parameters():
            param.requires_grad = False


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)
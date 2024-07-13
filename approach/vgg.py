import torch
import torch.nn as nn
import torch.nn.functional as F

# Class for VGG
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.dropout = nn.Dropout(0.2)
        
        self.classifier = nn.Sequential(
            nn.Linear(16384, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 7)
        )
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn1(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv5(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv6(x)))
        x = F.max_pool2d(x, 2)
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
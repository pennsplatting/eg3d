import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch_utils import persistence

@persistence.persistent_class
class EdgeExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.tensor(np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]]).reshape(1, 1, 3, 3), dtype=torch.float)
        self.conv = nn.Conv2d(1, 1, (3, 3), padding=1)
        self.conv.weight = nn.Parameter(self.w)
        self.transform = transforms.Grayscale(num_output_channels=1)
        
        # self.net = torch.nn.Sequential(
            
        # )
        
    def forward(self, img):
        return self.conv(self.transform(img))
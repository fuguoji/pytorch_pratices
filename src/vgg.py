import torch
import torch.nn as nn
from torchvision import models

class VGGModel(torch.nn.Module):
    def __init__(self):
        super(VGGModel, self).__init__()
        self.select = ['0', '5', '10', '19', '28']
        self.vgg = models.vgg11(pretrained=False).features

    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        
        return features


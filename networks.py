import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F

class CUB_Network(nn.Module):
    def __init__(self, embedding_size=512, device='cuda'):
        super().__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True).cuda()
        self.backbone.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.backbone.fc = nn.Identity()
        self.standardize = nn.LayerNorm(2048, elementwise_affine=False).to(device)
        self.remap = nn.Linear(2048, embedding_size, bias=True).to(device)
        
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=True):
            return F.normalize(self.remap(self.standardize(self.backbone(x))))
        
class SOP_Network(nn.Module):
    def __init__(self, embedding_size=512, device='cuda'):
        super().__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True).cuda()
        self.backbone.fc = nn.Identity()
        self.remap = nn.Linear(2048, embedding_size, bias=True).to(device)
        
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=True):
            return F.normalize(self.remap(self.backbone(x)))

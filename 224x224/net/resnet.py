import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn.init as init
from torchvision.models import resnet50

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM,self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)       
    
class Resnet50(nn.Module): ###
    def __init__(self,embedding_size, pretrained=True, is_norm=True, 
                 bn_freeze = True, layer_norm=False, gem=False):
        super(Resnet50, self).__init__()

        self.model = resnet50(pretrained)
        self.is_norm = is_norm
        self.embedding_size = embedding_size
        self.num_ftrs = self.model.fc.in_features
        
        self.gem = gem
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)
        self.model.gem = GeM()
        
        self.fp=True

        self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size)
        self._initialize_weights()

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)
                    
        self.layer_norm = layer_norm 
        if self.layer_norm:
            self.lnorm = nn.LayerNorm(self.embedding_size, elementwise_affine=False).cuda()

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=self.fp):
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

            if not self.gem:
                avg_x = self.model.gap(x)
                max_x = self.model.gmp(x)
                x = max_x + avg_x
            else:
                x = self.model.gem(x)

            xb = x.view(x.size(0), -1) ###
            x = self.model.embedding(xb)
    
            if self.layer_norm:
                x = self.lnorm(x)
            if self.is_norm or not self.training:
                x = self.l2_norm(x)
        
        return x.float()

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.embedding.weight, mode='fan_out')
        init.constant_(self.model.embedding.bias, 0)

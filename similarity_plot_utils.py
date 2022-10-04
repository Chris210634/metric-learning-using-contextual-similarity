import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms
import torchvision
from sampling import SOPDataset
from tqdm import tqdm

def get_hists(features, labels, d=512, n=50000):
    torch.set_num_threads(4)
    xs = torch.arange(1.2,1.6,0.001)
    def q(d, n):
        log_q = (n - 2) * torch.log(d + 1e-05) + (n-3)/2 * torch.log(1. - 0.25 * d.square() + 1e-5)
        return log_q.exp()
    torch.set_num_threads(6)
    bins = torch.arange(0,2,0.002).numpy()
    pos_sim = (features @ features.T)[((labels.unsqueeze(0).T == labels.unsqueeze(0)).float() - torch.eye(labels.shape[0])).nonzero(as_tuple=True)].view(-1)
    h_all = torch.histc((2. - 2. * (features @ features.T) + 1e-5).sqrt().view(-1), bins=1000, min=0.0, max=2.0)
    h_pos = torch.histc((2. - 2. * pos_sim + 1e-5).sqrt().view(-1), bins=1000, min=0.0, max=2.0)

    plt.plot(bins, h_all, label='all pairs')
    plt.plot(bins, h_pos, label='pos pairs')
    d = 512
    plt.plot(xs, q(xs, d)*40000000, label='uniform')
    plt.legend()
    plt.xlabel('distance')
    plt.title('Distribution of distances between pairs')
    plt.yscale('log')
    plt.grid()
    plt.show()
    
    return bins, h_all, h_pos, xs, q(xs, d)*40000000
    
class Network(nn.Module):
    def __init__(self, embedding_size=512, device='cuda'):
        super().__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True).cuda()
        self.backbone.fc = nn.Identity()
        self.remap = nn.Linear(2048, embedding_size, bias=True).to(device)
        
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=True):
            return F.normalize(self.remap(self.backbone(x)))
        
class SOP_Network(nn.Module):
    def __init__(self, embedding_size=512, device='cuda'):
        super().__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True).cuda()
        self.backbone.fc = nn.Identity()
        self.remap = nn.Linear(2048, embedding_size, bias=True).to(device)
        
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=True):
            return F.normalize(self.remap(self.backbone(x)))
        
def get_savepath(fn):
    with open(fn) as f:
        for line in f:
            if 'model will be saved in:' in line:
                return line.split(' ')[-1].strip()
            
def get_sim_mat(model, loader,n_batches, embedding_size=512, device='cuda'):
    torch.set_num_threads(4)
    features = []
    labels = []
    model.eval()
    i = 0
    shape = min(len(loader.dataset), n_batches * loader.batch_size)
    features = torch.zeros((shape,embedding_size))
    labels = torch.zeros(shape)
    with torch.no_grad():
        for img, target in tqdm(loader):
            feature = model(img.to(device)).cpu().detach()
            assert feature.shape[1] == features.shape[1]
            features[i:i+feature.shape[0]] = feature
            labels[i:i+feature.shape[0]] = target.view(-1)
            i += feature.shape[0]
            if i >= shape:
                break
    assert i == features.shape[0]
    return features, labels
            
def get_xform(augmentation):
    if augmentation == 'none':
        return transforms.Compose([
                    transforms.Resize((227,227)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    elif augmentation == 'bigtrain':
        return transforms.Compose([
                    transforms.Resize(288),
                    transforms.RandomResizedCrop(size=256, scale=[0.16, 1], ratio=[0.75, 1.33]),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    elif augmentation == 'bigtest':
        return transforms.Compose([
                    transforms.Resize((288,288)),
                    transforms.CenterCrop((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    else:
        raise NotImplemented
        
def get_features(model_path, loader, pt_folder='./', pretrained=False, n_batches=466):
    ''' input log path '''  
    model = torch.nn.DataParallel(Network())
    
    if not pretrained:
        model.module, loss_list, _= torch.load(pt_folder + model_path)
    
    model.module.backbone.eval()
    
    features, labels = get_sim_mat(model, loader, n_batches=n_batches)
    return features, labels
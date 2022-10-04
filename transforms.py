from torchvision import transforms

def iNat_get_xform(augmentation):
    ''' iNaturalist images are assumed to be pre-resized.'''
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if augmentation == 'none':
        return transforms.Compose([
                    transforms.Resize((227,227)),
                    transforms.ToTensor(),
                    normalize])
    elif augmentation == 'bigtrain':
        return transforms.Compose([
                    transforms.RandomResizedCrop(size=224, scale=[0.16, 1], ratio=[0.75, 1.33]), ###
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    normalize])
    
    elif augmentation == 'bigtest':
        return transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize])
    else:
        raise NotImplemented

def inshop_get_xform(augmentation):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if augmentation == 'none':
        return transforms.Compose([
                    transforms.Resize((227,227)),
                    transforms.ToTensor(),
                    normalize])
    elif augmentation == 'bigtrain':
        return transforms.Compose([
                    transforms.Resize(288),
                    transforms.RandomResizedCrop(size=256, scale=[0.16, 1], ratio=[0.75, 1.33]),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    normalize])
    
    elif augmentation == 'bigtest':
        return transforms.Compose([
                    transforms.Resize((288,288)),
                    transforms.CenterCrop((256, 256)),
                    transforms.ToTensor(),
                    normalize])
    else:
        raise NotImplemented
#     normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     if augmentation == 'none':
#         return transforms.Compose([
#                     transforms.Resize((227,227)),
#                     transforms.ToTensor(),
#                     normalize])
#     elif augmentation == 'bigtrain':
#         return transforms.Compose([
#                     transforms.RandomCrop(size=224), ###
#                     transforms.RandomHorizontalFlip(p=0.5),
#                     transforms.ToTensor(),
#                     normalize])
    
#     elif augmentation == 'bigtest':
#         return transforms.Compose([
#                     transforms.Resize(256),
#                     transforms.CenterCrop(224),
#                     transforms.ToTensor(),
#                     normalize])
#     else:
#         raise NotImplemented
        
# cars and CUB share same augmentation
def CUB_get_xform(augmentation):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if augmentation == 'none':
        return transforms.Compose([
                    transforms.Resize((227,227)),
                    transforms.ToTensor(),
                    normalize])
    elif augmentation == 'bigtrain':
        return transforms.Compose([
                    transforms.RandomResizedCrop(size=256, scale=[0.16, 1], ratio=[0.75, 1.33]),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    normalize])
    
    elif augmentation == 'bigtest':
        return transforms.Compose([
                    transforms.Resize(288),
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                    normalize])
    else:
        raise NotImplemented

def SOP_get_xform(augmentation):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if augmentation == 'none':
        return transforms.Compose([
                    transforms.Resize((227,227)),
                    transforms.ToTensor(),
                    normalize])
    elif augmentation == 'bigtrain':
        return transforms.Compose([
                    transforms.Resize(288),
                    transforms.RandomResizedCrop(size=256, scale=[0.16, 1], ratio=[0.75, 1.33]),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    normalize])
    
    elif augmentation == 'bigtest':
        return transforms.Compose([
                    transforms.Resize((288,288)),
                    transforms.CenterCrop((256, 256)),
                    transforms.ToTensor(),
                    normalize])
    else:
        raise NotImplemented
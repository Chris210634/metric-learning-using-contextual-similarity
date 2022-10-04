from __future__ import print_function
from torchvision import transforms
import torch

from PIL import Image
import os
import numpy as np

def parse_datalist(filename):
    '''return jpg_list, superlabel_list, class_list'''
    with open(filename) as f:
        fl = f.readlines()
    jpg_list = np.array([fli.strip().split(',')[0] for fli in fl])
    superlabel_list = np.array([int(fli.strip().split(',')[1]) for fli in fl])
    class_list = np.array([int(fli.strip().split(',')[2]) for fli in fl])
    return jpg_list, superlabel_list, class_list

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

root = 'train_mini/' # source images
test_txt = 'iNaturalist_test.txt'
train_txt = 'iNaturalist_train.txt'
imgs_train, _, _ = parse_datalist(train_txt)
imgs_test, _, _ = parse_datalist(test_txt)
imgs = np.concatenate((imgs_train, imgs_test))

from pathlib import Path
transform = transforms.Resize(288)
resized_root = "train_mini_resized/"

for index in range(len(imgs)):
    path = os.path.join(root, imgs[index])
    img = transform(pil_loader(path))
    new_path = resized_root + imgs[index]
    directory_path = '/'.join(new_path.split('/')[:-1])
    if not os.path.isdir(directory_path):
        Path(directory_path).mkdir(parents=False, exist_ok=False)
    img.save(new_path)
    
    if index % 1000 == 0:
        print('{} / {}, {}% done ...'.format(index, len(imgs), float(index / len(imgs))*100.))
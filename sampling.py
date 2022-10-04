# adapted from https://github.com/elias-ramzi/ROADMAP/tree/main/roadmap/samplers
import random
from collections import Counter
import numpy as np
import torch
import io
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from PIL import ImageFilter
import time
import os

###################################################################################################

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
def make_dataset_fromlist(image_list):
    ''' 
    Reads in the image list file e.g. cub200_train.txt.
    image_list dhould have a header: "image_id class_id super_class_id path"
    Then each line has four space delimited fields:
    "<image_id> <class_id> <super_class_id> <path>"
    The <image_id> and <super_class_id> fields are ignored.
    '''
    with open(image_list) as f:
        ll = f.readlines()
    if 'image_id' in ll[0]: # header
        ll.pop(0)
    image_index = np.array([x.split(' ')[3].strip() for x in ll])
    label_list = np.array([int(x.split(' ')[1].strip()) for x in ll])
    return image_index, label_list

def parse_datalist(filename): # iNat only
    '''
    Reads in the image list file e.g. iNaturalist_train.txt
    file must not have a header
    each line corresponds to an image. There are three comma delimited fields:
    <jpg_path>, <super label>, <label>
    return jpg_list, superlabel_list, class_list
    '''
    with open(filename) as f:
        fl = f.readlines()
    jpg_list = np.array([fli.strip().split(',')[0] for fli in fl])
    superlabel_list = np.array([int(fli.strip().split(',')[1]) for fli in fl])
    class_list = np.array([int(fli.strip().split(',')[2]) for fli in fl])
    return jpg_list, superlabel_list, class_list
    
class Imagelist(object):
    def __init__(self, image_list, root, transform=None):
        '''image_list is the txt file formatted like <jpg_path>, <super label>, <label>'''
        imgs, labels = make_dataset_fromlist(os.path.join(root, image_list))
        self.imgs = imgs
        self.labels = list(labels)
        self.super_labels = None
        self.transform = transform
        print(transform)
        self.loader = pil_loader
        self.root = root
        
    def __getitem__(self, index):
        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgs)
    
class Imagelist_iNat(object):
    def __init__(self, image_list, root, transform=None):
        '''image_list is the txt file formatted like <jpg_path>, <super label>, <label>'''
        imgs, superlabels, labels = parse_datalist(os.path.join(root, image_list))
        self.imgs = imgs
        self.labels = list(labels)
        self.super_labels = list(superlabels) # as required by HierarchicalSamplerBalanced
        self.transform = transform
        print(transform)
        self.loader = pil_loader
        self.root = root
        self.get_super_dict()
        
    def get_super_dict(self,):
        if hasattr(self, 'super_labels') and self.super_labels is not None:
            self.super_dict = {ct: {} for ct in set(self.super_labels)}
            for idx, cl, ct in zip(range(len(self.labels)), self.labels, self.super_labels):
                try:
                    self.super_dict[ct][cl].append(idx)
                except KeyError:
                    self.super_dict[ct][cl] = [idx]

    def __getitem__(self, index):
        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgs)
    
###################################################################################################

class BaseDataset(Dataset):

    def __init__(
        self,
        multi_crop=False,
        size_crops=[224, 96],
        nmb_crops=[2, 6],
        min_scale_crops=[0.14, 0.05],
        max_scale_crops=[1., 0.14],
        size_dataset=-1,
        return_label='none',
    ):
        super().__init__()

        if not multi_crop:
            self.get_fn = self.simple_get
        else:
            # adapted from
            # https://github.com/facebookresearch/swav/blob/master/src/multicropdataset.py
            self.get_fn = self.multiple_crop_get

            self.return_label = return_label
            assert self.return_label in ["none", "real", "hash"]

            color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
            mean = [0.485, 0.456, 0.406]
            std = [0.228, 0.224, 0.225]
            trans = []
            for i in range(len(size_crops)):
                randomresizedcrop = transforms.RandomResizedCrop(
                    size_crops[i],
                    scale=(min_scale_crops[i], max_scale_crops[i]),
                )
                trans.extend([transforms.Compose([
                    randomresizedcrop,
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.Compose(color_transform),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)])
                ] * nmb_crops[i])
            self.trans = trans

    def __len__(self,):
        return len(self.paths)

    @property
    def my_at_R(self,):
        if not hasattr(self, '_at_R'):
            self._at_R = max(Counter(self.labels).values())
        return self._at_R

    def get_instance_dict(self,):
        self.instance_dict = {cl: [] for cl in set(self.labels)}
        for idx, cl in enumerate(self.labels):
            self.instance_dict[cl].append(idx)

    def get_super_dict(self,):
        if hasattr(self, 'super_labels') and self.super_labels is not None:
            self.super_dict = {ct: {} for ct in set(self.super_labels)}
            for idx, cl, ct in zip(range(len(self.labels)), self.labels, self.super_labels):
                try:
                    self.super_dict[ct][cl].append(idx)
                except KeyError:
                    self.super_dict[ct][cl] = [idx]

    def simple_get(self, idx):
        pth = self.paths[idx]
       #  stream = io.BytesIO(self.jpeg_bytes[idx])
       #  img = Image.open(stream).convert('RGB')
        img = Image.open(pth).convert('RGB')
        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        label = torch.tensor([label])
        # out = {"image": img, "label": label, "path": pth}

#         if hasattr(self, 'super_labels') and self.super_labels is not None:
#             super_label = self.super_labels[idx]
#             super_label = torch.tensor([super_label])
#             out['super_label'] = super_label

        return img, label

    def multiple_crop_get(self, idx):
        pth = self.paths[idx]
        image = Image.open(pth).convert('RGB')
        multi_crops = list(map(lambda trans: trans(image), self.trans))

        if self.return_label == 'real':
            label = self.labels[idx]
            labels = [label] * len(multi_crops)
            return {"image": multi_crops, "label": labels, "path": pth}

        if self.return_label == 'hash':
            label = abs(hash(pth))
            labels = [label] * len(multi_crops)
            return {"image": multi_crops, "label": labels, "path": pth}

        return {"image": multi_crops, "path": pth}

    def __getitem__(self, idx):
        return self.get_fn(idx)

    def __repr__(self,):
        return f"{self.__class__.__name__}(mode={self.mode}, len={len(self)})"


class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

from os.path import join

import pandas as pd

class SOPDataset(BaseDataset):

    def __init__(self, data_dir, mode, transform=None, **kwargs):
        super().__init__(**kwargs)

        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        if mode == 'train':
            mode = ['train']
        elif mode == 'test':
            mode = ['test']
        elif mode == 'all':
            mode = ['train', 'test']
        else:
            raise ValueError(f"Mode unrecognized {mode}")

        self.paths = []
        self.labels = []
        self.super_labels = []
        for splt in mode:
            gt = pd.read_csv(join(self.data_dir, f'Ebay_{splt}.txt'), sep=' ')
            self.paths.extend(gt["path"].apply(lambda x: join(self.data_dir, x)).tolist())
            self.labels.extend((gt["class_id"] - 1).tolist())
            self.super_labels.extend((gt["super_class_id"] - 1).tolist())

        self.get_instance_dict()
        self.get_super_dict()
        
import itertools

import numpy as np
from torch.utils.data.sampler import BatchSampler

def safe_random_choice(input_data, size):
    replace = len(input_data) < size
    return np.random.choice(input_data, size=size, replace=replace)

class RandomSampler(BatchSampler):
    def __init__(self,
        dataset, samples_per_class,
        batch_size, num_batches, tries=10, superlabel=-1):
        self.dataset = dataset        
        self.tries = tries
        self.samples_per_class = samples_per_class
        self.batch_size = batch_size
        self.superlabel=superlabel
        self.num_batches = num_batches
        self.reshuffle()

    def __iter__(self,):
        # self.reshuffle()
        for batch in self.batches:
            yield batch

    def __len__(self,):
        return len(self.batches)

    def __repr__(self,):
        return (f"{self.__class__.__name__}(\n")
    
    def reshuffle(self):
        print("Shuffling data")
        t0 = time.time()
        batches = []
        
        labels = torch.tensor(self.dataset.labels)
        unique_labels = labels.unique().numpy()
        if self.superlabel >= 0:
            unique_labels = labels[(torch.tensor(self.dataset.super_labels) == self.superlabel).nonzero().view(-1)].unique().numpy()
        num_batches = len(unique_labels) // ( self.batch_size // self.samples_per_class)
        if num_batches > self.num_batches:
            num_batches = self.num_batches
        np.random.shuffle(unique_labels)
        print('There are {} unique labels'.format(len(unique_labels)))
        
        for _ in range(self.tries): ### Hack TODO
            i = 0
            for _ in range(num_batches):
                batch = []
                for _ in range(self.batch_size // self.samples_per_class):
                    batch.extend(safe_random_choice((labels == unique_labels[i]).nonzero().view(-1), size=self.samples_per_class))
                    i += 1
                batches.append(batch)
                if len(batches) >= self.num_batches:
                    break
            if len(batches) >= self.num_batches:
                break

        np.random.shuffle(batches)
        self.batches = batches
        t1 = time.time()
        print('reshuffle took {} seconds'.format(t1-t0))
    
# Inspired by
# https://github.com/kunhe/Deep-Metric-Learning-Baselines/blob/master/datasets.py
class HierarchicalSampler(BatchSampler):
    def __init__(
        self,
        dataset,
        batch_size,
        samples_per_class,
        batches_per_super_pair,
        nb_categories=2,
    ):
        """
        labels: 2D array, where rows correspond to elements, and columns correspond to the hierarchical labels
        batch_size: because this is a BatchSampler the batch size must be specified
        samples_per_class: number of instances to sample for a specific class. set to 0 if all element in a class
        batches_per_super_pairs: number of batches to create for a pair of categories (or super labels)
        inner_label: columns index corresponding to classes
        outer_label: columns index corresponding to the level of hierarchy for the pairs
        """
        self.batch_size = int(batch_size)
        self.batches_per_super_pair = int(batches_per_super_pair)
        self.samples_per_class = int(samples_per_class)
        self.nb_categories = int(nb_categories)

        # checks
        assert self.batch_size % self.nb_categories == 0, f"batch_size should be a multiple of {self.nb_categories}"
        self.sub_batch_len = self.batch_size // self.nb_categories
        if self.samples_per_class > 0:
            assert self.sub_batch_len % self.samples_per_class == 0, "batch_size not a multiple of samples_per_class"
        else:
            self.samples_per_class = None

        self.super_image_lists = dataset.super_dict.copy()
        self.super_pairs = list(itertools.combinations(set(dataset.super_labels), self.nb_categories))
        self.reshuffle()

    def __iter__(self,):
        # self.reshuffle()
        for batch in self.batches:
            yield batch

    def __len__(self,):
        return len(self.batches)

    def __repr__(self,):
        return (
            f"{self.__class__.__name__}(\n"
            f"    batch_size={self.batch_size},\n"
            f"    samples_per_class={self.samples_per_class},\n"
            f"    batches_per_super_pair={self.batches_per_super_pair},\n"
            f"    nb_categories={self.nb_categories}\n)"
        )

    def reshuffle(self):
        print("Shuffling data")
        t0 = time.time()
        batches = []
        for combinations in self.super_pairs:

            for b in range(self.batches_per_super_pair):

                batch = []
                for slb in combinations:

                    sub_batch = []
                    all_classes = list(self.super_image_lists[slb].keys())
                    np.random.shuffle(all_classes)
                    for cl in all_classes:
                        instances = self.super_image_lists[slb][cl]
                        samples_per_class = self.samples_per_class if self.samples_per_class else len(instances)
                        if len(sub_batch) + samples_per_class > self.sub_batch_len:
                            break
                        sub_batch.extend(safe_random_choice(instances, size=samples_per_class))

                    batch.extend(sub_batch)

                np.random.shuffle(batch)
                batches.append(batch)

        np.random.shuffle(batches)
        self.batches = batches
        t1 = time.time()
        print('reshuffle took {} seconds'.format(t1-t0))
        
class HierarchicalSamplerBalanced(BatchSampler):
    def __init__(
        self,
        dataset,
        batch_size,
        samples_per_class,
        batches_per_super_pair,
        nb_categories=2,
        batch_multiplier=66
    ):
        """
        labels: 2D array, where rows correspond to elements, and columns correspond to the hierarchical labels
        batch_size: because this is a BatchSampler the batch size must be specified
        samples_per_class: number of instances to sample for a specific class. set to 0 if all element in a class
        batches_per_super_pairs: number of batches to create for a pair of categories (or super labels)
        inner_label: columns index corresponding to classes
        outer_label: columns index corresponding to the level of hierarchy for the pairs
        """
        self.batch_size = int(batch_size)
        self.batches_per_super_pair = int(batches_per_super_pair)
        self.samples_per_class = int(samples_per_class)
        self.nb_categories = int(nb_categories)

        # checks
        assert self.batch_size % self.nb_categories == 0, f"batch_size should be a multiple of {self.nb_categories}"
        self.sub_batch_len = self.batch_size // self.nb_categories
        if self.samples_per_class > 0:
            assert self.sub_batch_len % self.samples_per_class == 0, "batch_size not a multiple of samples_per_class"
        else:
            self.samples_per_class = None

        self.super_image_lists = dataset.super_dict.copy()
        self.super_pairs = list(itertools.combinations(set(dataset.super_labels), self.nb_categories))
        self.num_half_batches = (torch.tensor(dataset.super_labels).unique(return_counts=True)[1] / len(dataset.super_labels) * batches_per_super_pair * batch_multiplier * 2).round().int()
        self.reshuffle()

    def __iter__(self,):
        # self.reshuffle()
        for batch in self.batches:
            yield batch

    def __len__(self,):
        return len(self.batches)

    def __repr__(self,):
        return (
            f"{self.__class__.__name__}(\n"
            f"    batch_size={self.batch_size},\n"
            f"    samples_per_class={self.samples_per_class},\n"
            f"    batches_per_super_pair={self.batches_per_super_pair},\n"
            f"    nb_categories={self.nb_categories}\n)"
        )

    def reshuffle(self):
        print("Shuffling data")
        t0 = time.time()
        batches = []
        
        samples_per_class = self.samples_per_class
        batch_size = self.batch_size
        size_of_half_batch = batch_size // 2
        num_classes_per_half_batch = size_of_half_batch // samples_per_class
        half_batches_master = [] # list of list of list, first index: super label, second index: half batch 
        superlabels = list(self.super_image_lists.keys())
        for superlabel in superlabels:
            num_half_batches_to_sample = self.num_half_batches[superlabel].item()
            all_classes = list(self.super_image_lists[superlabel].keys()) # all class labels that belong to this superlabel group
            num_classes_to_sample = int(num_half_batches_to_sample * num_classes_per_half_batch)
            classes_sampled = []
            while num_classes_to_sample > len(all_classes):
                np.random.shuffle(all_classes) # shuffle then get the first so many labels
                classes_sampled.extend(all_classes)
                num_classes_to_sample -= len(all_classes)
            if num_classes_to_sample > 0:
                np.random.shuffle(all_classes)
                classes_sampled.extend(all_classes[:num_classes_to_sample])
            # classes_sampled down contains a list of all the classes we want to sample
            # now sample the half batches
            half_batches = []
            sub_batch = []
            for cls in classes_sampled:
                instances = self.super_image_lists[superlabel][cls]
                sub_batch.extend(safe_random_choice(instances, size=samples_per_class))
                if len(sub_batch) >= size_of_half_batch:
                    half_batches.append(sub_batch)
                    sub_batch = []
            assert len(half_batches) == num_half_batches_to_sample
            half_batches_master.append(half_batches)

        # now round robin pair up half batches:
        batches = []
        superlabel1 = 0
        while sum([len(li) > 0 for li in half_batches_master]) > 1: # while there are at least 2 superlabels with available half batches
            for superlabel2 in superlabels:
                if superlabel2 == superlabel1:
                    continue
                if len(half_batches_master[superlabel1]) > 0 and len(half_batches_master[superlabel2]) > 0:
                    half_batch_1 = half_batches_master[superlabel1].pop()
                    half_batch_2 = half_batches_master[superlabel2].pop()
                    batches.append(half_batch_1 + half_batch_2)
                elif len(half_batches_master[superlabel1]) == 0:
                    break # no more half batches available from superlabel1, try next one
            superlabel1 += 1
            if superlabel1 >= len(half_batches_master):
                superlabel1 = 0

        assert len(half_batches_master) == len(superlabels)
        assert all([len(batch) == batch_size for batch in batches])

        np.random.shuffle(batches)
        self.batches = batches
        t1 = time.time()
        print('reshuffle took {} seconds'.format(t1-t0))
        
def kmeans(embeddings, k=200, device='cuda'):
    '''
    K-means ++ https://en.wikipedia.org/wiki/K-means%2B%2B and 
    https://en.wikipedia.org/wiki/K-means_clustering
    Assume normalized embeddings.
    '''
    cluster_centers = torch.zeros(k, embeddings.shape[1], device=device)
    initial = np.random.choice(embeddings.shape[0]) #Choose one center uniformly at random among the data points.
    cluster_centers[0, :] = embeddings[initial,:]

    for ki in range(1, k):
        # for each data point x not chosen yet, compute D(x), 
        # the distance between x and the nearest center that has already been chosen.
        D_x = (2. - 2. * (embeddings @ cluster_centers.T))[:,:ki] # N x k euclidean distance squared
        D_x, _ =  D_x.min(1)

        # Choose one new data point at random as a new center, 
        # using a weighted probability distribution where a point x is chosen 
        # with probability proportional to D(x)^2.
        D_x = D_x.clamp(min=0.)
        chosen_index = torch.distributions.categorical.Categorical(D_x / D_x.sum()).sample().item()
        cluster_centers[ki, :] = embeddings[chosen_index,:]

    track_list = []
    nearest_clusterid = torch.zeros(embeddings.shape[0], device=device)
    for it in range(100):
        # Assign each observation to the cluster with the nearest mean
        D_x = (2. - 2. * (embeddings @ cluster_centers.T)) # N x k euclidean distance squared
        _, nearest_clusterid_next =  D_x.min(1)

        if (nearest_clusterid != nearest_clusterid_next).sum() == 0:
            # The algorithm has converged when the assignments no longer change. 
            print('K-means converged at iteration {}, exiting ...'.format(it))
            break

        nearest_clusterid = nearest_clusterid_next

        # Recalculate means (centroids) for observations assigned to each cluster.
        mask = nearest_clusterid == torch.arange(k, device=device).unsqueeze(0).T # torch.Size([200, 8054]) bool
        # embeddings.shape is torch.Size([8054, 1024])

        # https://stackoverflow.com/questions/69314108/how-to-do-a-masked-mean-in-pytorch
        denom = torch.sum(mask, -1, keepdim=True)
        cluster_centers_next = torch.sum(embeddings * mask.unsqueeze(-1), dim=1) / denom
        cluster_centers_next = F.normalize(cluster_centers_next)
        track_list.append((cluster_centers_next - cluster_centers).sum().item())
        cluster_centers = cluster_centers_next
    return cluster_centers, nearest_clusterid
        
class HierarchicalSamplerKMeansBase(BatchSampler):
    # abstract, define sample_negative_pair(self, sim_mat)
    def __init__(
        self,
        dataset,
        batch_size,
        samples_per_class,
        batches_per_super_pair,
        features, labels, 
        sampling_power = 0., # number between 0 and -infty where 0 is uniform sampling and -infty is nearest neighbor sampling
        nb_categories=2,
        embedding_size=512
    ):
        """
        labels: 2D array, where rows correspond to elements, and columns correspond to the hierarchical labels
        batch_size: because this is a BatchSampler the batch size must be specified
        samples_per_class: number of instances to sample for a specific class. set to 0 if all element in a class
        batches_per_super_pairs: number of batches to create for a pair of categories (or super labels)
        inner_label: columns index corresponding to classes
        outer_label: columns index corresponding to the level of hierarchy for the pairs
        """
        self.batch_size = int(batch_size)
        self.batches_per_super_pair = int(batches_per_super_pair)
        self.samples_per_class = int(samples_per_class)
        self.nb_categories = int(nb_categories)

        # checks
        assert self.batch_size % self.nb_categories == 0, f"batch_size should be a multiple of {self.nb_categories}"
        self.sub_batch_len = self.batch_size // self.nb_categories
        if self.samples_per_class > 0:
            assert self.sub_batch_len % self.samples_per_class == 0, "batch_size not a multiple of samples_per_class"
        else:
            self.samples_per_class = None

        self.super_image_lists = dataset.super_dict.copy()
        self.super_pairs = list(itertools.combinations(set(dataset.super_labels), self.nb_categories))
        
        self.sampling_power = sampling_power
        self.embedding_size = embedding_size
        self.reshuffle(features, labels, dataset)

    def __iter__(self,):
        # self.reshuffle()
        for batch in self.batches:
            yield batch

    def __len__(self,):
        return len(self.batches)

    def __repr__(self,):
        return (
            f"{self.__class__.__name__}(\n"
            f"    batch_size={self.batch_size},\n"
            f"    samples_per_class={self.samples_per_class},\n"
            f"    batches_per_super_pair={self.batches_per_super_pair},\n"
            f"    nb_categories={self.nb_categories}\n)"
        )

    def reshuffle(self, features, labels, dataset):
        print("Shuffling data")
        t0 = time.time()
        ### set up centroids and centroid similarity matrix ###
        labels = torch.tensor(dataset.labels)
        unique_labels = labels.unique()
        centroids = []
        for label in unique_labels:
            centroids.append(features[(labels == label).nonzero().view(-1)].mean(0))
        centroids = torch.stack(centroids)
        centroids = F.normalize(centroids)
        cluster_centers, nearest_clusterid = kmeans(centroids.cuda(), k=12)
        nearest_clusterid = nearest_clusterid.cpu()
        ########################################################
        num_classes_per_batch = self.sub_batch_len // self.samples_per_class // 2
        batches = []
        for combinations in self.super_pairs:
            for b in range(self.batches_per_super_pair):
                batch = []
                for slb in combinations:
                    sub_batch = []
                    
                    all_classes = (nearest_clusterid == slb).nonzero().view(-1)
                    chosen_classes = all_classes[torch.randperm(len(all_classes))[:num_classes_per_batch]]
                    paired_classes = torch.tensor(self.sample_negative_pair(centroids[chosen_classes]))

                    for cl in torch.cat((chosen_classes, paired_classes)):
                        instances = (labels == cl).nonzero().view(-1)
                        sub_batch.extend(safe_random_choice(instances, size=self.samples_per_class))

                    batch.extend(sub_batch)
                np.random.shuffle(batch)
                batches.append(batch)

        np.random.shuffle(batches)
        self.batches = batches
        t1 = time.time()
        assert all([len(lt) == self.batch_size for lt in batches])
        print('reshuffle took {} seconds'.format(t1-t0))
        
class HierarchicalSamplerHardCentroidKmeans(HierarchicalSamplerKMeansBase):
    def sample_negative_pair(self, sim_mat):
        ''' sim_mat is cos similarity matrix. similarity with same sample is 8.0.
        return 1-D tensor same size as number of rows in the similarity matrix
        '''
        classes_closest_to_farthest = sim_mat.sort(descending=True)[1][:,:-1] # exclude the last row because it is itself
                    
        # define a categorical distribution over the number of samples and sample from it
        n_samples = classes_closest_to_farthest.shape[1]
        num_classes_per_bactch = classes_closest_to_farthest.shape[0]
        indices = torch.multinomial(torch.arange(1,n_samples+1).pow(self.sampling_power), num_classes_per_bactch, replacement=True)
        paired_classes = classes_closest_to_farthest[(torch.arange(num_classes_per_bactch),indices)]
        return paired_classes

class HierarchicalSamplerCentroidsBase(BatchSampler):
    # abstract, define sample_negative_pair(self, sim_mat)
    def __init__(
        self,
        dataset,
        batch_size,
        samples_per_class,
        batches_per_super_pair,
        features, labels, 
        sampling_power = 0., # number between 0 and -infty where 0 is uniform sampling and -infty is nearest neighbor sampling
        nb_categories=2,
        embedding_size=512
    ):
        """
        labels: 2D array, where rows correspond to elements, and columns correspond to the hierarchical labels
        batch_size: because this is a BatchSampler the batch size must be specified
        samples_per_class: number of instances to sample for a specific class. set to 0 if all element in a class
        batches_per_super_pairs: number of batches to create for a pair of categories (or super labels)
        inner_label: columns index corresponding to classes
        outer_label: columns index corresponding to the level of hierarchy for the pairs
        """
        self.batch_size = int(batch_size)
        self.batches_per_super_pair = int(batches_per_super_pair)
        self.samples_per_class = int(samples_per_class)
        self.nb_categories = int(nb_categories)

        # checks
        assert self.batch_size % self.nb_categories == 0, f"batch_size should be a multiple of {self.nb_categories}"
        self.sub_batch_len = self.batch_size // self.nb_categories
        if self.samples_per_class > 0:
            assert self.sub_batch_len % self.samples_per_class == 0, "batch_size not a multiple of samples_per_class"
        else:
            self.samples_per_class = None

        self.super_image_lists = dataset.super_dict.copy()
        self.super_pairs = list(itertools.combinations(set(dataset.super_labels), self.nb_categories))
        
        self.sampling_power = sampling_power
        self.embedding_size = embedding_size
        self.reshuffle(features, labels, dataset)

    def __iter__(self,):
        # self.reshuffle()
        for batch in self.batches:
            yield batch

    def __len__(self,):
        return len(self.batches)

    def __repr__(self,):
        return (
            f"{self.__class__.__name__}(\n"
            f"    batch_size={self.batch_size},\n"
            f"    samples_per_class={self.samples_per_class},\n"
            f"    batches_per_super_pair={self.batches_per_super_pair},\n"
            f"    nb_categories={self.nb_categories}\n)"
        )

    def reshuffle(self, features, labels, dataset):
        print("Shuffling data")
        t0 = time.time()
        ### set up centroids and centroid similarity matrix ###
        superlabels_unique = torch.tensor(dataset.super_labels).unique()
        centroids_dic = {}
        for superlabel in superlabels_unique:
            centroids = []
            for label in dataset.super_dict[superlabel.item()]:
                centroids.append(features[torch.tensor(dataset.super_dict[superlabel.item()][label])].mean(0))
            centroids = torch.stack(centroids)
            centroids = F.normalize(centroids)
            centroids_dic[superlabel.item()] = (centroids @ centroids.T) - 4. * torch.eye(centroids.shape[0])
        ########################################################
        num_classes_per_bactch = self.sub_batch_len // self.samples_per_class // 2
        batches = []
        for combinations in self.super_pairs:
            for b in range(self.batches_per_super_pair):
                batch = []
                for slb in combinations:
                    sub_batch = []
                    all_classes = list(self.super_image_lists[slb].keys())
                    chosen_classes = torch.randperm(len(all_classes))[:num_classes_per_bactch]
                    paired_classes = self.sample_negative_pair(centroids_dic[slb][chosen_classes])
                    
                    # setting sampling_power to -100. is equivalent to taking closest centroid. 
#                     assert (paired_classes != centroids_dic[slb][chosen_classes].max(1)[1]).sum() == 0
                    
                    for cl in torch.cat((chosen_classes, paired_classes)):
                        instances = self.super_image_lists[slb][all_classes[cl]]
                        sub_batch.extend(safe_random_choice(instances, size=self.samples_per_class))

                    batch.extend(sub_batch)
                np.random.shuffle(batch)
                batches.append(batch)

        np.random.shuffle(batches)
        self.batches = batches
        t1 = time.time()
        assert all([len(lt) == self.batch_size for lt in batches])
        print('reshuffle took {} seconds'.format(t1-t0))
         
class HierarchicalSamplerCentroidDistanceWeighted(HierarchicalSamplerCentroidsBase):
    def sample_negative_pair(self, sim_mat):
        mat = (2. - 2. * sim_mat + 1e-05).sqrt()
        d = self.embedding_size
        cutoff=0.5
        nonzero_loss_cutoff=1.4
        mat = torch.clamp(mat, min=cutoff)
        # See the first equation from Section 4 of the paper
        log_weights = (2.0 - d) * torch.log(mat) - ((d - 3) / 2) * torch.log(
            1.0 - 0.25 * (mat**2.0)
        )
        inf_or_nan = torch.isinf(log_weights) | torch.isnan(log_weights)
        # Subtract max(log(distance)) for stability.
        weights = torch.exp(log_weights - torch.max(log_weights[~inf_or_nan]))

        # Hack: mat < 2. gives negative pairs, cause distance > 2. is sample itself
        weights = weights * (mat <= 2.).float() * (mat < nonzero_loss_cutoff).float()
        weights[inf_or_nan] = 0.

        return torch.multinomial(weights + 1e-5, 1, replacement=True).view(-1)
    
class HierarchicalSamplerCentroidTopKUniform(HierarchicalSamplerCentroidsBase):
    def sample_negative_pair(self, sim_mat):
        weights = (sim_mat >= sim_mat.topk(10)[0][:,-1:]).float()
        return torch.multinomial(weights, 1, replacement=True).view(-1)
        
class HierarchicalSamplerHardCentroidConfusion(HierarchicalSamplerCentroidsBase):
    def sample_negative_pair(self, sim_mat):
        ''' sim_mat is cos similarity matrix. similarity with same sample is 8.0.
        return 1-D tensor same size as number of rows in the similarity matrix
        '''
        classes_closest_to_farthest = sim_mat.sort(descending=True)[1][:,:-1] # exclude the last row because it is itself
                    
        # define a categorical distribution over the number of samples and sample from it
        n_samples = classes_closest_to_farthest.shape[1]
        num_classes_per_bactch = classes_closest_to_farthest.shape[0]
        indices = torch.multinomial(torch.arange(1,n_samples+1).pow(self.sampling_power), num_classes_per_bactch, replacement=True)
        paired_classes = classes_closest_to_farthest[(torch.arange(num_classes_per_bactch),indices)]
        return paired_classes
    
class HierarchicalSamplerCentroidUniformHistogram(HierarchicalSamplerCentroidsBase):
    def uniform_sample(self, centroid_sim_vec):
        ''' centroid_sim_vec is 1D tensor.'''
        def random_choice(v):
            return v[torch.randperm(len(v))[0]]

        vv = (2. - 2.*centroid_sim_vec + 1e-05).sqrt()
        hh = torch.histc(vv, bins=100, max=2.0, min=0.0)
        hhh = hh.nonzero().view(-1)
        selected_bin = random_choice(hhh)
        assert hh[selected_bin] > 0
        lower = selected_bin * 0.02
        upper = lower + 0.02
        bin_indices = torch.logical_and((vv >= lower), (vv <= upper)).nonzero().view(-1)
        paired_index = random_choice(bin_indices)
        return paired_index
    
    def sample_negative_pair(self, sim_mat):
        ''' sim_mat is cos similarity matrix. similarity with same sample is 8.0.
        return 1-D tensor same size as number of rows in the similarity matrix
        '''
        paired_classes = []
        for row_i in range(sim_mat.shape[0]):
            # need for loop cause histc is not a vector operation
            paired_classes.append(self.uniform_sample(sim_mat[row_i]))

        return torch.tensor(paired_classes)
        
class HierarchicalSamplerMaxLoss(BatchSampler):
    def __init__(
        self,
        dataset,
        batch_size,
        samples_per_class,
        batches_per_super_pair,
        sim_mat, truth, loss_fn, 
        max_factor = 2,
        nb_categories=2,
    ):
        """
        labels: 2D array, where rows correspond to elements, and columns correspond to the hierarchical labels
        batch_size: because this is a BatchSampler the batch size must be specified
        samples_per_class: number of instances to sample for a specific class. set to 0 if all element in a class
        batches_per_super_pairs: number of batches to create for a pair of categories (or super labels)
        inner_label: columns index corresponding to classes
        outer_label: columns index corresponding to the level of hierarchy for the pairs
        """
        self.batch_size = int(batch_size)
        self.batches_per_super_pair = int(batches_per_super_pair)
        self.samples_per_class = int(samples_per_class)
        self.nb_categories = int(nb_categories)

        # checks
        assert self.batch_size % self.nb_categories == 0, f"batch_size should be a multiple of {self.nb_categories}"
        self.sub_batch_len = self.batch_size // self.nb_categories
        if self.samples_per_class > 0:
            assert self.sub_batch_len % self.samples_per_class == 0, "batch_size not a multiple of samples_per_class"
        else:
            self.samples_per_class = None

        self.super_image_lists = dataset.super_dict.copy()
        self.super_pairs = list(itertools.combinations(set(dataset.super_labels), self.nb_categories))
        self.reshuffle(sim_mat, truth, loss_fn, max_factor)

    def __iter__(self,):
        # self.reshuffle()
        for batch in self.batches:
            yield batch

    def __len__(self,):
        return len(self.batches)

    def __repr__(self,):
        return (
            f"{self.__class__.__name__}(\n"
            f"    batch_size={self.batch_size},\n"
            f"    samples_per_class={self.samples_per_class},\n"
            f"    batches_per_super_pair={self.batches_per_super_pair},\n"
            f"    nb_categories={self.nb_categories}\n)"
        )
    
    def find_max_loss(self, sim_mat, truth, sub_batches, n_select, loss_fn):
        ''' select n_select top loss out of sub_batches and return batches.'''
        losses = []
        # with torch.cuda.amp.autocast(enabled=True):
        for batch in sub_batches:
            scores = sim_mat[torch.tensor(batch)][:,torch.tensor(batch)]
            w = truth[torch.tensor(batch)][:,torch.tensor(batch)].float()
            loss = loss_fn(scores, w) ###
            losses.append(loss.item())
        _, indices = torch.tensor(losses).topk(n_select)
        batches = []
        for ind in indices:
            batches.append(sub_batches[ind])
        return batches

    def reshuffle(self, sim_mat, truth, loss_fn, max_factor):
        print("Shuffling data")
        t0 = time.time()
        batches = []
        for combinations in self.super_pairs:
            sub_batches = []
            for b in range(self.batches_per_super_pair * max_factor): # 10 * 2 = 20

                batch = []
                for slb in combinations:

                    sub_batch = []
                    all_classes = list(self.super_image_lists[slb].keys())
                    np.random.shuffle(all_classes)
                    for cl in all_classes:
                        instances = self.super_image_lists[slb][cl]
                        samples_per_class = self.samples_per_class if self.samples_per_class else len(instances)
                        if len(sub_batch) + samples_per_class > self.sub_batch_len:
                            break
                        sub_batch.extend(safe_random_choice(instances, size=samples_per_class))

                    batch.extend(sub_batch)

                np.random.shuffle(batch)
                sub_batches.append(batch)
                
            batches.extend(self.find_max_loss(sim_mat, truth, sub_batches, 
                                              n_select=self.batches_per_super_pair, 
                                              loss_fn=loss_fn))

        np.random.shuffle(batches)
        self.batches = batches
        t1 = time.time()
        print('reshuffle took {} seconds'.format(t1-t0))
        
class HierarchicalSamplerPositiveMargin(BatchSampler):
    def __init__(
        self,
        dataset,
        batch_size,
        samples_per_class,
        batches_per_super_pair,
        sim_mat, truth,
        nb_categories=2,
        pos_margin=0.9,
    ):
        self.batch_size = int(batch_size)
        self.batches_per_super_pair = int(batches_per_super_pair)
        self.samples_per_class = int(samples_per_class)
        self.nb_categories = int(nb_categories)

        # checks
        assert self.batch_size % self.nb_categories == 0, f"batch_size should be a multiple of {self.nb_categories}"
        self.sub_batch_len = self.batch_size // self.nb_categories
        if self.samples_per_class > 0:
            assert self.sub_batch_len % self.samples_per_class == 0, "batch_size not a multiple of samples_per_class"
        else:
            self.samples_per_class = None

        self.super_image_lists = dataset.super_dict.copy()
        self.super_pairs = list(itertools.combinations(set(dataset.super_labels), self.nb_categories))
        
        ########################################
        self.pos_margin = pos_margin
        self.reshuffle(sim_mat, truth)
    
    def __iter__(self,):
        for batch in self.batches:
            yield batch
            
    def __len__(self,):
        return len(self.batches)

    def __repr__(self,):
        return (
            f"{self.__class__.__name__}(\n"
            f"    batch_size={self.batch_size},\n"
            f"    samples_per_class={self.samples_per_class},\n"
            f"    batches_per_super_pair={self.batches_per_super_pair},\n"
            f"    nb_categories={self.nb_categories}\n)"
        )
        
    def reshuffle(self, sim_mat, truth):
        print("Shuffling data")
#         torch.set_num_threads(4)
        batches = []
        for combinations in self.super_pairs:

            for b in range(self.batches_per_super_pair):

                batch = []
                for slb in combinations:

                    sub_batch = []
                    all_classes = list(self.super_image_lists[slb].keys())
                    np.random.shuffle(all_classes)
                    err = True
                    for cl in all_classes:
                        instances = self.super_image_lists[slb][cl]
                        chosen_index = random.choice(instances)
                        elgible_positive_pairs = torch.logical_and(sim_mat[chosen_index] < self.pos_margin,
                                                                   truth[chosen_index]).nonzero().view(-1).numpy()
                        if len(elgible_positive_pairs) == 0:
                            continue
                        samples_per_class = self.samples_per_class if self.samples_per_class else len(instances)
                        if len(sub_batch) + samples_per_class > self.sub_batch_len:
                            err = False
                            break
                        sub_batch.append(chosen_index)
                        sub_batch.extend(safe_random_choice(elgible_positive_pairs, size=samples_per_class-1))
                    if err:
                        print('ran out of classes to sampler ERROR')
                        raise NotImplemented
                    batch.extend(sub_batch)

                np.random.shuffle(batch)
                batches.append(batch)
                
        assert all([len(lt) == self.batch_size for lt in batches])
        np.random.shuffle(batches)
        self.batches = batches
        
        
class HierarchicalSamplerNegativeMargin(BatchSampler):
    def __init__(
        self,
        dataset,
        batch_size,
        samples_per_class,
        batches_per_super_pair,
        sim_mat, truth,
        nb_categories=2,
        pos_margin=0.9,
        neg_margin=1.0
    ):
        self.batch_size = int(batch_size)
        self.batches_per_super_pair = int(batches_per_super_pair)
        self.samples_per_class = int(samples_per_class)
        self.nb_categories = int(nb_categories)

        # checks
        assert self.batch_size % self.nb_categories == 0, f"batch_size should be a multiple of {self.nb_categories}"
        self.sub_batch_len = self.batch_size // self.nb_categories
        if self.samples_per_class > 0:
            assert self.sub_batch_len % self.samples_per_class == 0, "batch_size not a multiple of samples_per_class"
        else:
            self.samples_per_class = None

        self.super_image_lists = dataset.super_dict.copy()
        self.super_pairs = list(itertools.combinations(set(dataset.super_labels), self.nb_categories))
        
        ########################################
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        print('setting pos_margin to {} and neg_margin to {}'.format(pos_margin, neg_margin))
        self.reshuffle(sim_mat, truth, dataset)
    
    def __iter__(self,):
        for batch in self.batches:
            yield batch
            
    def __len__(self,):
        return len(self.batches)

    def __repr__(self,):
        return (
            f"{self.__class__.__name__}(\n"
            f"    batch_size={self.batch_size},\n"
            f"    samples_per_class={self.samples_per_class},\n"
            f"    batches_per_super_pair={self.batches_per_super_pair},\n"
            f"    nb_categories={self.nb_categories}\n)"
        )
        
    def reshuffle(self, sim_mat, truth, dataset):
        print("Shuffling data")
        batches = []
        labels = torch.tensor(dataset.labels)
        for combinations in self.super_pairs:

            for b in range(self.batches_per_super_pair):

                batch = []
                for slb in combinations:

                    sub_batch = []
                    all_classes = list(self.super_image_lists[slb].keys())
                    np.random.shuffle(all_classes)
                    
                    samples_per_class = self.samples_per_class
                    super_class_mask = (torch.tensor(dataset.super_labels) == slb)

                    # start with a random class cl
                    cl = random.choice(all_classes)
                    # print('picked class {}'.format(cl))
                    
                    remaining_cl_mask = torch.clone(super_class_mask).float()
                    all_classes.remove(cl)
                    remaining_cl_mask = remaining_cl_mask - (labels == cl).float()
                    instances = self.super_image_lists[slb][cl]
                    # print('instances: ', instances)
                    
                    chosen_index = random.choice(instances)
                    # print('chosen_index: ', chosen_index)
                            
                    while len(all_classes) > 0 and len(sub_batch) + samples_per_class <= self.sub_batch_len:                           
                        elgible_positive_pairs = torch.logical_and(sim_mat[chosen_index] < self.pos_margin,
                                                                   truth[chosen_index]).nonzero().view(-1).numpy()
                        # print('elgible_positive_pairs: ', elgible_positive_pairs)
                        
                        if len(elgible_positive_pairs) == 0:
                            cl = random.choice(all_classes)
                            # print('picked class {}'.format(cl))
                            
                            all_classes.remove(cl)
                            remaining_cl_mask = remaining_cl_mask - (labels == cl).float()
                            instances = self.super_image_lists[slb][cl]
                            # print('instances: ', instances)
                            
                            chosen_index = random.choice(instances)
                            # print('chosen_index: ', chosen_index)
                            continue
                            
                        samples_per_class = self.samples_per_class if self.samples_per_class else len(instances)
                        sub_batch.append(chosen_index)
                        positive_indices = safe_random_choice(elgible_positive_pairs, size=samples_per_class-1)
                        # print('positive_indices: ', positive_indices)
                        
                        sub_batch.extend(positive_indices)
                        
                        # pick the closest sample to the margin
                        # margin = 1.0 means pick closest negative sample
                        # margin = -1.0 means pick farthest negative sample
                        closest_value, closest_index = (-(sim_mat[chosen_index] - self.neg_margin).square() + remaining_cl_mask.float()*10.).max(0)
                        # print('closest negative value: {}, index: {}'.format(closest_value-10., closest_index))
                        
                        chosen_index = closest_index.item()
                        # print('chosen_index: ', chosen_index)
                        
                        cl = labels[chosen_index].item()
                        all_classes.remove(cl)
                        remaining_cl_mask = remaining_cl_mask - (labels == cl).float()
                        instances = self.super_image_lists[slb][cl]

                    batch.extend(sub_batch)

                np.random.shuffle(batch)
                batches.append(batch)
                
        assert all([len(lt) == self.batch_size for lt in batches])
        np.random.shuffle(batches)
        self.batches = batches
        
        
# class HierarchicalSamplerHard(BatchSampler):
#     def __init__(
#         self,
#         dataset,
#         batch_size,
#         samples_per_class,
#         batches_per_super_pair,
#         sim_mat, truth, margin=0.5,
#         nb_categories=2,
#     ):
#         """
#         labels: 2D array, where rows correspond to elements, and columns correspond to the hierarchical labels
#         batch_size: because this is a BatchSampler the batch size must be specified
#         samples_per_class: number of instances to sample for a specific class. set to 0 if all element in a class
#         batches_per_super_pairs: number of batches to create for a pair of categories (or super labels)
#         inner_label: columns index corresponding to classes
#         outer_label: columns index corresponding to the level of hierarchy for the pairs
#         """
#         self.batch_size = int(batch_size)
#         self.batches_per_super_pair = int(batches_per_super_pair)
#         self.samples_per_class = int(samples_per_class)
#         self.nb_categories = int(nb_categories)

#         # checks
#         assert self.batch_size % self.nb_categories == 0, f"batch_size should be a multiple of {self.nb_categories}"
#         self.sub_batch_len = self.batch_size // self.nb_categories
#         if self.samples_per_class > 0:
#             assert self.sub_batch_len % self.samples_per_class == 0, "batch_size not a multiple of samples_per_class"
#         else:
#             self.samples_per_class = None

#         self.super_image_lists = dataset.super_dict.copy()
#         self.super_pairs = list(itertools.combinations(set(dataset.super_labels), self.nb_categories))
        
#         ########################################
#         self.margin = margin
#         self.reshuffle(sim_mat, truth, dataset)

#     def __iter__(self,):
#         for batch in self.batches:
#             yield batch

#     def __len__(self,):
#         return len(self.batches)

#     def __repr__(self,):
#         return (
#             f"{self.__class__.__name__}(\n"
#             f"    batch_size={self.batch_size},\n"
#             f"    samples_per_class={self.samples_per_class},\n"
#             f"    batches_per_super_pair={self.batches_per_super_pair},\n"
#             f"    nb_categories={self.nb_categories}\n)"
#         )

#     def reshuffle(self, sim_mat, truth, dataset):
#         print("Shuffling data")
#         batches = []
#         n_samples = self.sub_batch_len // self.samples_per_class
#         _, closest_neighbors = ((100.-((2. - 2. * sim_mat) - self.margin).square()) * truth.float()).topk(self.samples_per_class-1)
        
#         for combinations in self.super_pairs:
#             for b in range(self.batches_per_super_pair):
#                 batch = []
#                 for slb in combinations:
#                     sub_batch = []
                    
#                     # list containing indices of all samples with superlabel == slb
#                     all_superclass_indices = list((torch.tensor(dataset.super_labels) == slb).nonzero().view(-1))
#                     np.random.shuffle(all_superclass_indices)
#                     chosen_anchors = torch.tensor(all_superclass_indices[:n_samples])
#                     # pick self.samples_per_class-1 "closest neighbors" to chosen_anchors
#                     closest_neighbors[chosen_anchors]

#                     batch.extend(sub_batch)

#                 np.random.shuffle(batch)
#                 batches.append(batch)

#         np.random.shuffle(batches)
#         self.batches = batches
        
        
        
        
 
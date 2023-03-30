import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler, BatchSampler
from tqdm import *
import itertools, time, pickle

class BalancedSampler(Sampler):
    def __init__(self, data_source, batch_size, images_per_class=3):
        self.data_source = data_source
        self.ys = np.array(data_source.ys)
        self.num_groups = batch_size // images_per_class
        self.batch_size = batch_size
        self.num_instances = images_per_class
        self.num_samples = len(self.ys)
        self.num_classes = len(set(self.ys))
        self.counter = 0

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        num_batches = len(self.data_source) // self.batch_size
        ret = []
        
        assert all(list(torch.tensor(self.ys).unique() == torch.arange(self.num_classes)))
        remaining_samples = []
        for class_id in range(self.num_classes):
            remaining_samples.append(list(np.nonzero(self.ys == class_id)[0]))
        
        while num_batches > 0:

            sampled_classes = np.random.choice(self.num_classes, 96, replace=False)
        
            for i in range(len(sampled_classes)):
                
                if len(remaining_samples[sampled_classes[i]]) < self.num_instances:
                    remaining_samples[sampled_classes[i]] = list(np.nonzero(self.ys == sampled_classes[i])[0])
                
                class_sel = np.random.choice(remaining_samples[sampled_classes[i]], size=self.num_instances, replace=False)
                ret.extend(np.random.permutation(class_sel))
                
                for j in class_sel:
                    remaining_samples[sampled_classes[i]].remove(j)
                    
            num_batches -= 96 // self.num_groups
        return iter(ret)
    
def safe_random_choice(input_data, size):
    replace = len(input_data) < size
    return np.random.choice(input_data, size=size, replace=replace)
        
class HierarchicalSamplerBalanced(BatchSampler):
    def __init__(
        self,
        dataset,
        batch_size,
        samples_per_class=4,
        batches_per_super_pair=10,
        nb_categories=2,
        batch_multiplier=66
    ):

        self.batch_size = int(batch_size)
        self.batches_per_super_pair = int(batches_per_super_pair)
        self.samples_per_class = int(samples_per_class)
        self.nb_categories = int(nb_categories)

        # checks
        assert self.batch_size % self.nb_categories == 0, f"batch_size should be a multiple of {self.nb_categories}"
        self.sub_batch_len = self.batch_size // self.nb_categories
        assert self.samples_per_class > 0
        assert self.sub_batch_len % self.samples_per_class == 0, "batch_size not a multiple of samples_per_class"

        self.super_image_lists = self.get_super_dict(dataset.super_labels, dataset.ys)
        # self.super_pairs = list(itertools.combinations(set(dataset.super_labels), self.nb_categories))
        self.num_half_batches = (torch.tensor(dataset.super_labels).unique(return_counts=True)[1] / len(dataset.super_labels) * batches_per_super_pair * batch_multiplier * self.nb_categories).round().int()
        self.reshuffle()

    def __iter__(self,):
        self.reshuffle()
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
    
    def get_super_dict(self, super_labels, labels):
        super_dict = {ct: {} for ct in set(super_labels)}
        for idx, cl, ct in zip(range(len(labels)), labels, super_labels):
            try:
                super_dict[ct][cl].append(idx)
            except KeyError:
                super_dict[ct][cl] = [idx]
        return super_dict

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
        self.batches = batches[:84480//self.batch_size]
        t1 = time.time()
        print('reshuffle took {} seconds'.format(t1-t0))
     
class NearestNeighborSampler(Sampler):
    def __init__(self, data_source, batch_size, images_per_class=3, k=8):
        self.data_source = data_source
        self.ys = np.array(data_source.ys)
        self.num_groups = batch_size // images_per_class
        self.batch_size = batch_size
        self.num_instances = images_per_class
        self.num_samples = len(self.ys)
        self.num_classes = len(set(self.ys))
        self.k = k
        
    def get_topk_neighbors(self, trn_normal_dataset, dl_tmp, model, sz_embedding):
        print('Calculating topk training dataset neighbors ...')
        features = torch.zeros(len(trn_normal_dataset), sz_embedding)
        labels = torch.zeros(len(trn_normal_dataset))
        model.eval()
        ptr = 0
        with torch.no_grad():
            for x, y in dl_tmp:
                f = model(x.squeeze().cuda())
                features[ptr:ptr+f.shape[0],:] = f.cpu()
                labels[ptr:ptr+f.shape[0]] = y.cpu()
                ptr += f.shape[0]
        assert ptr == features.shape[0]
        sim_mat = F.normalize(features) @ F.normalize(features).T
        assert (labels - torch.tensor(self.data_source.ys)).abs().sum() < 1
        self.topk_neighbors = sim_mat.topk(self.k).indices # N x k

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        num_batches = len(self.data_source) // self.batch_size
        ret = []
        while num_batches > 0:
            sampled_samples = np.random.choice(len(self.ys), self.num_groups, replace=False)
            for chosen_sample in sampled_samples:
                class_sel = np.random.choice(self.topk_neighbors[chosen_sample], size=self.num_instances, replace=False)
                ret.extend(np.random.permutation(class_sel))
            num_batches -= 1
        return iter(ret)
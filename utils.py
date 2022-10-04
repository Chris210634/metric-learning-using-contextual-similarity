# from IPython.display import display, HTML
# display(HTML("<style>.container { width:100% !important; }</style>"))

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

def get_savepath(fn):
    with open(fn) as f:
        for line in f:
            if 'model will be saved in:' in line:
                return line.split(' ')[-1].strip()

def to_dict(line):
    ll = line[line.find('Namespace('):].strip().strip(')')[len('Namespace('):].split(',')
    d = {}
    for li in ll:
        assert '=' in li
        assert len(li.split('=')) == 2
        d[li.split('=')[0].strip()] = li.split('=')[1].strip()
    return d
            
def get_acc_list(fn, get_namespace=False):
    acc_list = []
    namespace = ''
    with open(fn) as f:
        for line in f:
            if "R @ 1:" in line:
                acc_list.append(float(line.strip().split(' ')[-1]))
            if 'Namespace(' in line:
                namespace = to_dict(line)
                
    if get_namespace:
        return acc_list, namespace
    return acc_list

def smooth(l):
    m = 500
    return [sum(l[i:i+m])/m for i in range(len(l)-m)]

#### for plotting ######
# plt.figure(figsize=(15, 10), dpi=80)
# logs_dir = '.'
# fl = os.listdir(logs_dir)
# for fn in fl:
#     if fn[-2:] == '.o':
#         acc_list, namespace = get_acc_list(logs_dir + '/' + fn, get_namespace=True)
#         plt.plot(acc_list, label=fn)
    
# plt.grid()
# plt.legend()
# plt.show()
########################

def get_features_and_labels_from_model_that_normalizes(G, test_loader, embedding_size=512, device='cuda'):
    torch.set_num_threads(4)
    features = []
    labels = []
    G.eval()
    i = 0
    features = torch.zeros((len(test_loader.dataset),embedding_size))
    labels = torch.zeros(len(test_loader.dataset))
    with torch.no_grad():
        for img, target in tqdm(test_loader):
            feature = G(img.to(device)).cpu().detach()
            assert feature.shape[1] == features.shape[1]
            features[i:i+feature.shape[0]] = feature
            labels[i:i+feature.shape[0]] = target.view(-1)
            i += feature.shape[0]

    assert i == features.shape[0]
    return features, labels

def _get_features_and_labels(G, test_loader=None, query_loader=None, gallery_loader=None, embedding_size=512, device='cuda'):
    if not test_loader is None:
        assert query_loader is None and gallery_loader is None
        features, labels = get_features_and_labels_from_model_that_normalizes(G, test_loader, 
                                                                          embedding_size=embedding_size, 
                                                                          device=device)
        print('features.shape: {}, labels.shape: {}'.format(features.shape, labels.shape))
        query_features = features
        gallery_features = features
        query_labels = labels
        gallery_labels = labels
    else:
        query_features, query_labels = get_features_and_labels_from_model_that_normalizes(G, query_loader, 
                                                                          embedding_size=embedding_size, 
                                                                          device=device)
        gallery_features, gallery_labels = get_features_and_labels_from_model_that_normalizes(G, gallery_loader, 
                                                                          embedding_size=embedding_size, 
                                                                          device=device)
        print('query_features.shape: {}, query_labels.shape: {}'.format(query_features.shape, query_labels.shape))
        print('gallery_features.shape: {}, gallery_labels.shape: {}'.format(gallery_features.shape, gallery_labels.shape))
    return query_features, gallery_features, query_labels, gallery_labels

def _get_AP_on_sub_matrix(sm, query_labels, query_index_start, query_index_end, 
                          gallery_labels, check_for_sample_repeat=True, device='cuda'):
    values, indices = sm.sort(descending=True)
    if check_for_sample_repeat:
        assert (values[:,-1].view(-1) != -2).sum() == 0.0 
        # the last column should all be -2.0 because this is the sample itself, and we set it to -2.0
    indices = indices[:,:-1] # don't include the last comlumn which is itself

    #AP_i = ( \sum_{k=1}^n Prec(k) x Relevance(k) ) / (# relevant documents)
    relevance = (query_labels[query_index_start:query_index_end].unsqueeze(0).T == gallery_labels[indices]).float()
    positive_rank = torch.cumsum(relevance, 1)
    overall_rank = (torch.arange(relevance.shape[1]).to(device) + 1).float().repeat(relevance.shape[0],1)
    precision =  positive_rank / overall_rank # precision is the # of relevant documents ranked less than or equal to k over the rank of k
    AP_sub = (precision * relevance).sum(1) / relevance.sum(1)
    first_R_mask = (overall_rank <= relevance.sum(1, keepdim=True)).float()
    AP_at_R_sub = (precision * relevance * first_R_mask).sum(1) / relevance.sum(1)
    return AP_sub, AP_at_R_sub

def test_APs(G, test_loader=None, query_loader=None, gallery_loader=None, embedding_size=512, device='cuda'):
    '''Get mAP and mAP@R'''
    query_features, gallery_features, query_labels, gallery_labels = _get_features_and_labels(G, 
                                                                                              test_loader=test_loader, 
                                                                                              query_loader=query_loader, 
                                                                                              gallery_loader=gallery_loader, 
                                                                                              embedding_size=embedding_size, 
                                                                                              device=device)
    query_labels = query_labels.to(device)
    gallery_labels = gallery_labels.to(device)
    
    # The similarity matrix is too big to fit in memory at once
    # we need to split into 24 pieces and evaluate independently
    # if you encounter memory issues, make n_splits bigger
    n_splits = query_labels.shape[0] // 512
    inc = query_labels.shape[0] // n_splits
    predictions = []
    f_T = gallery_features.T.cuda()
    
    AP_list = []
    AP_at_R_list = []
    
    sample_itself_place_holder = -2.
    for i in tqdm(range(n_splits)):
        sm = query_features[inc*i:inc*(i+1)].cuda() @ f_T
        for j in range(sm.shape[0]):
            if not test_loader is None:
                # when the query and gallery sets are the same, we want to make sure we don't retrieve the sample itself
                sm[j,inc*i+j] = sample_itself_place_holder

        AP_sub, AP_at_R_sub = _get_AP_on_sub_matrix(sm, 
                                                    query_labels=query_labels,
                                                    query_index_start=inc*i, 
                                                    query_index_end=inc*(i+1), 
                                                    gallery_labels=gallery_labels, 
                                                    check_for_sample_repeat=(not test_loader is None), 
                                                    device=device)
        del sm
        AP_list.append(AP_sub)
        AP_at_R_list.append(AP_at_R_sub)
        
    if query_labels.shape[0] % n_splits > 0:
        sm = query_features[inc*n_splits:].cuda() @ f_T
        for j in range(sm.shape[0]):
            if not test_loader is None:
                # when the query and gallery sets are the same, we want to make sure we don't retrieve the sample itself
                sm[j,inc*n_splits+j] = sample_itself_place_holder
            
        AP_sub, AP_at_R_sub = _get_AP_on_sub_matrix(sm, 
                                                    query_labels=query_labels,
                                                    query_index_start=inc*n_splits, 
                                                    query_index_end=len(query_labels), 
                                                    gallery_labels=gallery_labels, 
                                                    check_for_sample_repeat=(not test_loader is None), 
                                                    device=device)
        del sm
        AP_list.append(AP_sub)
        AP_at_R_list.append(AP_at_R_sub)
        
    assert len(torch.cat(AP_list)) == query_labels.shape[0]
    assert len(torch.cat(AP_at_R_list)) == query_labels.shape[0]
    
    return torch.cat(AP_list).mean().item()*100., torch.cat(AP_at_R_list).mean().item()*100.
    
        
def test_ks(G, test_loader=None, query_loader=None, gallery_loader=None, ks=[1, 10, 100], embedding_size=512, device='cuda'):
    '''Get recall at Ks.'''
    query_features, gallery_features, query_labels, gallery_labels = _get_features_and_labels(G, 
                                                                                              test_loader=test_loader, 
                                                                                              query_loader=query_loader, 
                                                                                              gallery_loader=gallery_loader, 
                                                                                              embedding_size=embedding_size, 
                                                                                              device=device)
        
    # The similarity matrix is too big to fit in memory at once
    # we need to split into 24 pieces and evaluate independently
    # if you encounter memory issues, make n_splits bigger
    n_splits = 24
    inc = query_labels.shape[0] // n_splits
    predictions = []
    f_T = gallery_features.T.cuda()

    topk_ik_list = []
    for k in ks:
        topk_ik_list.append([])

    for i in tqdm(range(n_splits)):
        sm = query_features[inc*i:inc*(i+1)].cuda() @ f_T
        for j in range(sm.shape[0]):
            if not test_loader is None:
                # when the query and gallery sets are the same, we want to make sure we don't retrieve the sample itself
                sm[j,inc*i+j] = -1.

        for i in range(len(ks)):
            k = ks[i]
            vk, ik = sm.topk(k)
            topk_ik_list[i].append(ik.cpu())

        _, indices = sm.max(1)
        predictions.append(indices.cpu())
        del indices, sm

    if query_labels.shape[0] % n_splits > 0:
        sm = query_features[inc*n_splits:].cuda() @ f_T
        for j in range(sm.shape[0]):
            if not test_loader is None:
                # when the query and gallery sets are the same, we want to make sure we don't retrieve the sample itself
                sm[j,inc*n_splits+j] = -1.
            
        for i in range(len(ks)):
            k = ks[i]
            vk, ik = sm.topk(k)
            topk_ik_list[i].append(ik.cpu())
            
        _, indices = sm.max(1)
        predictions.append(indices.cpu())
        del indices, sm

    del f_T
    predictions = torch.cat(predictions)
    print(predictions.shape)
    for i in range(predictions.shape[0]):
        if not test_loader is None:
            # when the query and gallery sets are the same, we want to make sure we don't retrieve the sample itself
            assert predictions[i] != i

    # predictions are indices into the gallery
    print('Test R @ 1: ', (query_labels == gallery_labels[predictions]).sum().item() / query_labels.shape[0])

    return_list = []
    for i in range(len(ks)):
        k = ks[i]
        ik = torch.cat(topk_ik_list[i])
        assert ik.shape[0] == query_labels.shape[0]
        count = 0
        for h in range(query_labels.shape[0]): # for each query
            label = query_labels[h]
            labels_n = gallery_labels[ik[h]]
            if label in labels_n:
                count += 1
        print('R @ {}: {}'.format(k, float(count) / float(query_labels.shape[0]) * 100.))
        return_list.append(float(count) / float(query_labels.shape[0]) * 100.)
    return return_list


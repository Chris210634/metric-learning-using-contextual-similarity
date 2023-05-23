import numpy as np
import torch
import losses
from tqdm import tqdm
import torch.nn.functional as F

############ AP test code ############
# model.eval()
# n = len(dl_ev.dataset)
# d = 512
# fs = torch.zeros((n,d), device='cuda')
# ys = torch.zeros(n, device='cuda')
# i = 0
# with torch.no_grad():
#     for x, y in dl_ev:
#         b = x.shape[0]
#         assert b == len(y)
#         fs[i:i+b, :] = model(x.cuda())
#         ys[i:i+b] = y
#         i += b
# assert i == n
# fs = F.normalize(fs)
# cos_sim = fs @ fs.T
# for i in range(n):
#     cos_sim[i,i] = -1.1
    
# w = (ys == ys.unsqueeze(0).T).float()
# sort_v, sort_i = (-cos_sim).sort(1)
# pp = (ys[sort_i] == ys.unsqueeze(0).T).float()
# pp = pp[:, :-1]
# kk = ((pp.cumsum(dim=1).cuda() / torch.arange(1, n).repeat(n,1).float().cuda()) * pp).sum(1) / pp.sum(1)
# mAP = kk.mean().item()*100.
####################################

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output

def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    s = 0
    for t,y in zip(T,Y):
        if t in torch.Tensor(y).long()[:k]:
            s += 1
    return s / (1. * len(T))


def predict_batchwise(model, dataloader):
    device = "cuda"
    model_is_training = model.training
    model.eval()
    
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    J = model(J.cuda())

                for j in J:
                    A[i].append(j)
    model.train()
    model.train(model_is_training) # revert to previous training state
    
    return [torch.stack(A[i]) for i in range(len(A))]

def proxy_init_calc(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()
    X, T, *_ = predict_batchwise(model, dataloader)

    proxy_mean = torch.stack([X[T==class_idx].mean(0) for class_idx in range(nb_classes)])

    return proxy_mean

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

def evaluate_cos(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)

    # get predictions by assigning nearest 8 neighbors with cosine
    K = 32
    Y = []
    xs = []
    
    cos_sim = F.linear(X, X)
    Y = T[cos_sim.cpu().topk(1 + K)[1][:,1:]]
    Y = Y.float().cpu()
    
    recall = []
    for k in [1, 2, 4, 8, 16, 32]:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))
        
    sample_itself_place_holder = -2.
    assert cos_sim.shape[0] == len(T)
    for i in range(cos_sim.shape[0]):
        cos_sim[i,i] = sample_itself_place_holder
    AP_sub, AP_at_R_sub = _get_AP_on_sub_matrix(cos_sim.cuda(), query_labels=T.cuda(),query_index_start=0, 
                                                query_index_end=len(T), gallery_labels=T.cuda(), 
                                                check_for_sample_repeat=True, device='cuda')
    print("mAP : {}".format(AP_sub.mean().item()*100.))
    print("mAP@R : {}".format(AP_at_R_sub.mean().item()*100.))

    return recall

def evaluate_cos_Inshop(model, query_dataloader, gallery_dataloader):
    nb_classes = query_dataloader.dataset.nb_classes()
    
    # calculate embeddings with model and get targets
    query_X, query_T = predict_batchwise(model, query_dataloader)
    gallery_X, gallery_T = predict_batchwise(model, gallery_dataloader)
    
    query_X = l2_norm(query_X)
    gallery_X = l2_norm(gallery_X)
    
    # get predictions by assigning nearest 8 neighbors with cosine
    K = 50
    Y = []
    xs = []
    
    cos_sim = F.linear(query_X, gallery_X)

    def recall_k(cos_sim, query_T, gallery_T, k):
        m = len(cos_sim)
        match_counter = 0

        for i in range(m):
            pos_sim = cos_sim[i][gallery_T == query_T[i]]
            neg_sim = cos_sim[i][gallery_T != query_T[i]]

            thresh = torch.max(pos_sim).item()

            if torch.sum(neg_sim > thresh) < k:
                match_counter += 1
            
        return match_counter / m
    
    # calculate recall @ 1, 2, 4, 8
    recall = []
    for k in [1, 10, 20, 30, 40, 50]:
        r_at_k = recall_k(cos_sim, query_T, gallery_T, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))
                
    return recall

def evaluate_cos_SOP(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()
    
    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)
    
    # get predictions by assigning nearest 8 neighbors with cosine
    K = 1000
    Y = []
    xs = []
    for x in X:
        if len(xs)<10000:
            xs.append(x)
        else:
            xs.append(x)            
            xs = torch.stack(xs,dim=0)
            cos_sim = F.linear(xs,X)
            y = T[cos_sim.cpu().topk(1 + K)[1][:,1:]]
            Y.append(y.float().cpu())
            xs = []
            
    # Last Loop
    xs = torch.stack(xs,dim=0)
    cos_sim = F.linear(xs,X)
    y = T[cos_sim.cpu().topk(1 + K)[1][:,1:]]
    Y.append(y.float().cpu())
    Y = torch.cat(Y, dim=0)

    # calculate recall @ 1, 2, 4, 8
    recall = []
    for k in [1, 10, 100, 1000]:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))
    return recall


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
#         print('features.shape: {}, labels.shape: {}'.format(features.shape, labels.shape))
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
#         print('query_features.shape: {}, query_labels.shape: {}'.format(query_features.shape, query_labels.shape))
#         print('gallery_features.shape: {}, gallery_labels.shape: {}'.format(gallery_features.shape, gallery_labels.shape))
    return query_features, gallery_features, query_labels, gallery_labels

def test_all(G, test_loader=None, query_loader=None, gallery_loader=None, ks=[1, 10, 100], embedding_size=512, device='cuda'):
    '''Get recall at Ks.'''
    
    model_is_training = G.training
    G.eval()
    
    query_features, gallery_features, query_labels, gallery_labels = _get_features_and_labels(G, 
                                                                                              test_loader=test_loader, 
                                                                                              query_loader=query_loader, 
                                                                                              gallery_loader=gallery_loader, 
                                                                                              embedding_size=embedding_size, 
                                                                                              device=device)
        
    # The similarity matrix is too big to fit in memory at once
    # we need to split into 24 pieces and evaluate independently
    # if you encounter memory issues, make n_splits bigger
    n_splits = query_labels.shape[0] // 512
    inc = query_labels.shape[0] // n_splits
    predictions = []
    f_T = gallery_features.T.cuda()

    topk_ik_list = []
    for k in ks:
        topk_ik_list.append([])
        
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
                                                    query_labels=query_labels.cuda(),
                                                    query_index_start=inc*i, 
                                                    query_index_end=inc*(i+1), 
                                                    gallery_labels=gallery_labels.cuda(), 
                                                    check_for_sample_repeat=(not test_loader is None), 
                                                    device=device)
        
        AP_list.append(AP_sub)
        AP_at_R_list.append(AP_at_R_sub)
        
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
                sm[j,inc*n_splits+j] = sample_itself_place_holder
            
        
        
        AP_sub, AP_at_R_sub = _get_AP_on_sub_matrix(sm, 
                                                    query_labels=query_labels.cuda(),
                                                    query_index_start=inc*n_splits, 
                                                    query_index_end=len(query_labels), 
                                                    gallery_labels=gallery_labels.cuda(), 
                                                    check_for_sample_repeat=(not test_loader is None), 
                                                    device=device)
        
        AP_list.append(AP_sub)
        AP_at_R_list.append(AP_at_R_sub)
        
        for i in range(len(ks)):
            k = ks[i]
            vk, ik = sm.topk(k)
            topk_ik_list[i].append(ik.cpu())
            
        _, indices = sm.max(1)
        predictions.append(indices.cpu())
        
        del indices, sm

    del f_T
    predictions = torch.cat(predictions)
#     print(predictions.shape)
    for i in range(predictions.shape[0]):
        if not test_loader is None:
            # when the query and gallery sets are the same, we want to make sure we don't retrieve the sample itself
            assert predictions[i] != i

    # predictions are indices into the gallery
    # print('Test R @ 1: ', (query_labels == gallery_labels[predictions]).sum().item() / query_labels.shape[0])

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
        print('R@{} : {}'.format(k, float(count) / float(query_labels.shape[0]) * 100.))
        return_list.append(float(count) / float(query_labels.shape[0]) * 100.)
        
    assert len(torch.cat(AP_list)) == query_labels.shape[0]
    assert len(torch.cat(AP_at_R_list)) == query_labels.shape[0]
    
    mAP = torch.cat(AP_list).mean().item()*100.
    mAPR = torch.cat(AP_at_R_list).mean().item()*100.
    
    print("mAP : {}".format(mAP))
    print("mAP@R : {}".format(mAPR))
    
    return_list.extend([mAP, mAPR])
    
    G.train()
    G.train(model_is_training) # revert to previous training state
    
    return return_list
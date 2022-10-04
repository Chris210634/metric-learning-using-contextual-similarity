# Ablation 0: normal / default                  $\lambda \mathcal{L}_{context} + (1 - \lambda) \mathcal{L}_{contrast}$
# Ablation 1: Nk_mask only # with eps=1e-5      $\lambda \mathcal{L}_{1} + (1 - \lambda) \mathcal{L}_{contrast}$
# Ablation 2: Nk_mask with sigmoid # temp=0.01  $\lambda \mathcal{L}_{1,\sigma} + (1 - \lambda) \mathcal{L}_{contrast}$
# Ablation 3: No query expansion (W_tilde)      $\lambda \mathcal{L}_{2} + (1 - \lambda) \mathcal{L}_{contrast}$
# Ablation 4: No query expansion (W_tilde but no negative intersection) $\lambda \mathcal{L}_{2,M_+} + (1 - \lambda) \mathcal{L}_{contrast}$
# Ablation 5: Min instead of multiplication for logical-and
# Ablation 6: sigmoid everywhere                $\lambda \mathcal{L}_{context, \sigma} + (1 - \lambda) \mathcal{L}_{contrast}$
# Ablation 7: no negative intersection          $\lambda \mathcal{L}_{context, M_+} + (1 - \lambda) \mathcal{L}_{contrast}$
# Ablation 8: skip intersection (step 2)
# Ablation 9: Detach Rk2 normalization          Detach $|\mathcal{R}_{k/2 + \epsilon}(i)|$
# Ablation 10: Don't detach anything
# --ablation to set ablation test
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import torch
import torchvision
from smooth_rank_ap import *
from tqdm import tqdm
from pytorch_metric_learning import distances, losses, miners, reducers, testers

##### My imports #####
from losses import margin_contrastive, GreaterThan, contrastive_jaccard
from sampling import *
from transforms import *
from networks import *
######################

def greater_than(x, y, temp=0.01):
    return torch.sigmoid((x-y)/temp)

def bintersection(a, b):
    ''' batch intersection between matrices a and b. Equivalent to a @ b.'''
    d = a.shape[0]
#     c = torch.zeros_like(a)
#     for i in range(d):
#         c[i] = torch.minimum(a , b[:,i]).sum(1)
#     return c.T
    return torch.minimum(a.repeat(d,1,1) , torch.einsum('ijk -> kij', b.repeat(d,1,1))).sum(-1).T

def get_jaccard_similarity(w, scores, k, eps):
    # features has been normalized before this function
    # D is ||u-v||^2 = 2 - 2<u,v>
    D = 2. - 2. * scores  # Squared Euclidedan, assume features are L2 normalized
    D = D.clamp(min=0.)   # for stability
    vk, ik = (-D).topk(k) # get k closest neighbors

    ### Ablations that only do step 1
    if args.ablation == 1:
        Nk_mask = GreaterThan.apply(-D + 1e-5, vk[:,-1:].detach())
        return Nk_mask
    elif args.ablation == 2:
        Nk_mask = greater_than(-D + 1e-5, vk[:,-1:].detach())
        return Nk_mask
    
    if args.ablation == 6:
        Nk_mask = greater_than(-D + eps, vk[:,-1:].detach())
    else:
        Nk_mask = GreaterThan.apply(-D + eps, vk[:,-1:].detach())
        
    if args.ablation == 5:
        intersection = bintersection(Nk_mask , Nk_mask.T) / Nk_mask.sum(1, keepdim=True).detach()
    elif args.ablation == 10:
        intersection = (Nk_mask @ Nk_mask.T) / Nk_mask.sum(1, keepdim=True)
    else:
        intersection = (Nk_mask @ Nk_mask.T) / Nk_mask.sum(1, keepdim=True).detach()
    Nk_mask_not = 1. - Nk_mask
    
    if args.ablation == 5:
        intersection_not = bintersection(Nk_mask_not , Nk_mask_not.T) / Nk_mask_not.sum(1, keepdim=True).detach()
    elif args.ablation == 10:
        intersection_not = (Nk_mask_not @ Nk_mask_not.T) / Nk_mask_not.sum(1, keepdim=True)
    else:
        intersection_not = (Nk_mask_not @ Nk_mask_not.T) / Nk_mask_not.sum(1, keepdim=True).detach()
    
    # before query expansion
    if args.ablation == 4 or args.ablation == 7:
        W_tilde = intersection * Nk_mask
    elif args.ablation == 5:
        W_tilde = torch.minimum(0.5 * (intersection + intersection_not) , Nk_mask)
    elif args.ablation == 8:
        W_tilde = Nk_mask
    else:
        W_tilde = 0.5 * (intersection + intersection_not) * Nk_mask
    
    if args.ablation == 3 or args.ablation == 4:
        return W_tilde
    
    # Query Expansion
    # Nk_over_2_mask = (-D >= vk[:,(k // 2)-1:(k // 2)]).float() # True if in k Euclidean neighborhood
    if args.ablation == 6:
        Nk_over_2_mask = greater_than(-D + eps, vk[:,(k // 2)-1:(k // 2)].detach())
    else:
        Nk_over_2_mask = GreaterThanR.apply(-D + eps, vk[:,(k // 2)-1:(k // 2)].detach())
        
    if args.ablation == 5:
        Rk_over_2_mask = torch.minimum(Nk_over_2_mask , Nk_over_2_mask.T)
    else:
        Rk_over_2_mask = Nk_over_2_mask * Nk_over_2_mask.T
        
    if args.ablation == 5:
        jaccard_similarity = bintersection(Rk_over_2_mask , W_tilde) / Rk_over_2_mask.sum(1, keepdim=True)
    elif args.ablation == 9:
        jaccard_similarity = (Rk_over_2_mask @ W_tilde) / Rk_over_2_mask.sum(1, keepdim=True).detach()
    else:
        jaccard_similarity = (Rk_over_2_mask @ W_tilde) / Rk_over_2_mask.sum(1, keepdim=True) #.detach()
        
    # Symmeterization
    return 0.5 * (jaccard_similarity + jaccard_similarity.T)

device = torch.device("cuda")

def test(G, test_loader):
    '''Please make sure G outputs normalized features.'''
    torch.set_num_threads(4)
    features = []
    labels = []
    G.eval()
    i = 0
    features = torch.zeros((len(test_loader.dataset),args.embedding_size))
    labels = torch.zeros(len(test_loader.dataset))
    with torch.no_grad():
        for img, target in test_loader:
            feature = G(img.to(device)).cpu().detach()
            assert feature.shape[1] == features.shape[1]
            features[i:i+feature.shape[0]] = feature
            labels[i:i+feature.shape[0]] = target.view(-1)
            i += feature.shape[0]

    assert i == features.shape[0]
    
    # The similarity matrix is too big to fit in memory at once
    # we need to split into 24 pieces and evaluate independently
    # if you encounter memory issues, make n_splits bigger
    n_splits = 24
    inc = labels.shape[0] // n_splits
    predictions = []
    f_T = features.T.cuda()
    for i in tqdm(range(n_splits)):
        sm = features[inc*i:inc*(i+1)].cuda() @ f_T
        for j in range(sm.shape[0]):
            sm[j,inc*i+j] = -1.
        _, indices = sm.max(1)
        predictions.append(indices.cpu())
        del indices, sm

    # left over piece
    if labels.shape[0] % n_splits > 0:
        sm = features[inc*n_splits:].cuda() @ f_T
        for j in range(sm.shape[0]):
            sm[j,inc*n_splits+j] = -1.
        _, indices = sm.max(1)
        predictions.append(indices.cpu())
        del indices, sm

    del f_T
    predictions = torch.cat(predictions)
    print(predictions.shape)
    for i in range(predictions.shape[0]):
        assert predictions[i] != i
        
    print('Test R @ 1: ', (labels == labels[predictions]).sum().item() / labels.shape[0])
    return (labels == labels[predictions]).sum().item() / labels.shape[0]

#################################################################################### 
parser = argparse.ArgumentParser(description='')
parser.add_argument('--trainxform', type=str, default='bigtrain', help='')
parser.add_argument('--testxform', type=str, default='bigtest', help='')
parser.add_argument('--root', type=str, default='', help='where data is located')
parser.add_argument('--lam', type=float, default=0.4, help='')
parser.add_argument('--gamma', type=float, default=0.1, help='')
parser.add_argument('--loss', type=str, default='hybrid', help='')
parser.add_argument('--embedding_size', type=int, default=512, help='')
parser.add_argument('--batch_size', type=int, default=128, help='')
parser.add_argument('--n_epochs', type=int, default=80, help='')
parser.add_argument('--test_freq', type=int, default=5, help='')
parser.add_argument('--lr', type=float, default=0.00001, help='')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='')
parser.add_argument('--eps', type=float, default=0.05, help='')
parser.add_argument('--start_epoch', type=int, default=0, help='')
parser.add_argument('--checkpoint', type=str, default='', help='')
parser.add_argument('--dataset', type=str, default='', help='')
parser.add_argument('--neg_margins', type=float, default=0.6, help='')
parser.add_argument('--pos_margins', type=float, default=0.75, help='')
parser.add_argument('--reg_sim', type=float, default=0.25, help='')
parser.add_argument('--loss_lr', type=float, default=0.001, help='for classification methods only')
parser.add_argument('--ablation', type=int, default=0, help='')
args = parser.parse_args()
print(args)
#################################################################################### 

roadmap_criterion = SupAP()
smoothAP_criterion = SmoothAP()
miner = miners.DistanceWeightedMiner()
triplet_criterion = losses.TripletMarginLoss(margin=0.05)
ms_criterion = losses.MultiSimilarityLoss()
fastap_criterion = losses.FastAPLoss()
ntxent_criterion = losses.NTXentLoss()

root = args.root
root = root.strip()
if root[-1] != '/':
    root += '/'
    
if args.dataset == 'SOP':
    testxform = SOP_get_xform(augmentation=args.testxform)
    trainxform = SOP_get_xform(augmentation=args.trainxform)
    trainset = SOPDataset(root, 'train', transform=trainxform)
    testset = SOPDataset(root, 'test', transform=testxform)
elif args.dataset == 'CUB':
    train_txt = 'cub200_train.txt'
    test_txt = 'cub200_test.txt'
    testxform = CUB_get_xform(augmentation=args.testxform)
    trainxform = CUB_get_xform(augmentation=args.trainxform)
    trainset = Imagelist(image_list=train_txt, root=root, transform=trainxform)
    testset = Imagelist(image_list=test_txt, root=root, transform=testxform)
elif args.dataset == 'Cars':
    train_txt = 'cars196_train.txt'
    test_txt = 'cars196_test.txt'
    testxform = CUB_get_xform(augmentation=args.testxform)
    trainxform = CUB_get_xform(augmentation=args.trainxform)
    trainset = Imagelist(image_list=train_txt, root=root, transform=trainxform)
    testset = Imagelist(image_list=test_txt, root=root, transform=testxform)
elif args.dataset == 'iNat':
    assert args.batch_size == 256
    test_txt = 'iNaturalist_test.txt'
    train_txt = 'iNaturalist_train.txt'
    testxform = iNat_get_xform(augmentation=args.testxform)
    trainxform = iNat_get_xform(augmentation=args.trainxform)
    trainset = Imagelist_iNat(image_list=train_txt, root=root, transform=trainxform)
    testset = Imagelist_iNat(image_list=test_txt, root=root, transform=testxform)
else:
    assert False # Need to input a valid dataset.

if args.dataset == 'SOP' or args.dataset == 'iNat':
    model = torch.nn.DataParallel(SOP_Network(embedding_size=args.embedding_size))
else:
    model = torch.nn.DataParallel(CUB_Network(embedding_size=args.embedding_size))
    
scaler = torch.cuda.amp.GradScaler()
if args.start_epoch > 0:
    assert args.checkpoint != ''
if args.checkpoint != '':
    print('loading checkpoint {} and starting on epoch {}'.format(args.checkpoint, args.start_epoch))
    model.module, _, scaler = torch.load(args.checkpoint)

loss_list = []
model_save_path = '{}_{}_lam_{}_gamma_{}_{}.pt'.format(args.dataset, args.loss, args.lam, args.gamma, random.random()*10000000)
print('model will be saved in: ', model_save_path)

G_lr = args.lr / 2.
W_lr = args.lr

optimizer1 = optim.Adam(model.module.backbone.parameters(), lr=G_lr, weight_decay=args.weight_decay)
optimizer2 = optim.Adam(model.module.remap.parameters(), lr=W_lr, weight_decay=args.weight_decay)

scheduler1 = optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[15,30,45], gamma=0.3)
scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer2, milestones=[15,30,45], gamma=0.3)

loss_optimizer = None
if args.loss == 'proxynca' or args.loss == 'normsoftmax':
    unique_labels = torch.tensor(trainset.labels).unique()
    num_classes = len(unique_labels)
    if args.loss == 'proxynca':
        criterion = losses.ProxyNCALoss(num_classes, args.embedding_size, softmax_scale=10.).to(device)
    else:
        criterion = losses.NormalizedSoftmaxLoss(num_classes, args.embedding_size, temperature=0.05).to(device)
    loss_optimizer = torch.optim.Adam(criterion.parameters(), lr=args.loss_lr, weight_decay=0.0001)

for _ in range(args.start_epoch):
    scheduler1.step()
    scheduler2.step()
#################################################################################### 
best_acc = 0.

for epoch in range(args.start_epoch, args.n_epochs):
    print('epoch {} G_lr {} W_lr {}'.format(epoch, optimizer1.param_groups[0]['lr'], optimizer2.param_groups[0]['lr']))
    
    model.module.backbone.eval()
    
    if args.dataset == 'SOP':
        trainsampler = HierarchicalSamplerBalanced(
                    dataset = trainset,
                    batch_size=args.batch_size,
                    samples_per_class=4,
                    batches_per_super_pair=10,
                    nb_categories=2
                )
    elif args.dataset == 'iNat':
        trainsampler = HierarchicalSamplerBalancediNat(
                    dataset = trainset,
                    batch_size=args.batch_size,
                    samples_per_class=4,
                    batches_per_super_pair=10,
                    nb_categories=2
                )
    else:
        trainsampler = RandomSampler(
                dataset = trainset,
                batch_size=args.batch_size,
                samples_per_class=4,
                num_batches = 42,
                tries = 5 if args.dataset == 'CUB' else 10
        )
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_sampler=trainsampler, num_workers=16, pin_memory=True)
    
    for x, y in tqdm(trainloader):
        with torch.cuda.amp.autocast(enabled=True):
            f = model(x.to(device))
            
        with torch.cuda.amp.autocast(enabled=False):
            y = y.reshape(-1).to(device)
            
            if args.loss == 'proxynca' or args.loss == 'normsoftmax':
                # for classification methods, the value of the label matters.
                # In particular, the labels must be numbered [0 , num_classes)
                # This is what this piece of code does
                y = (unique_labels.to(device) == y.unsqueeze(0).T).nonzero().T[1]
                
            w = (y.float().unsqueeze(0).T == y.float().unsqueeze(0)).float().to(device)
            scores = f.float() @ f.float().T

            if args.loss == 'roadmap':
                loss_supap = roadmap_criterion(scores, w)
                loss_con = margin_contrastive(scores, w, pos_margins=0.9, neg_margins=0.6)
                loss = 0.5 * loss_supap + 0.5 * loss_con
            elif args.loss == 'triplet':
                miner_output = miner(f, y)
                loss = triplet_criterion(f, y, miner_output)
            elif args.loss == 'multisimilarity':
                loss = ms_criterion(f, y)
            elif args.loss == 'smoothap':
                loss = smoothAP_criterion(scores, w)
            elif args.loss == 'fastap':
                loss = fastap_criterion(f, y)
            elif args.loss == 'contrastive':
                loss = margin_contrastive(scores, w, pos_margins=0.9, neg_margins=0.6)
            elif args.loss == 'ntxent':
                loss = ntxent_criterion(f, y)
            elif args.loss == 'proxynca' or args.loss == 'normsoftmax':
                loss = criterion(f, y)
            else:
                assert args.loss == 'hybrid' # our loss
                loss_context, reg_loss = contrastive_jaccard(scores, w, args.reg_sim, args.eps)
                loss_contrast = margin_contrastive(scores, w, pos_margins=args.pos_margins, neg_margins=args.neg_margins)
                loss =  args.lam * loss_context + args.gamma * reg_loss + (1. - args.lam - args.gamma) * loss_contrast
            
            loss_list.append(loss.item())
            
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer1)
        scaler.step(optimizer2)
        if not loss_optimizer is None:
            scaler.step(loss_optimizer)
        scaler.update()
        
    del trainloader, f, y, w
    scheduler1.step()
    scheduler2.step()
    
    if (epoch + 1) % args.test_freq == 0:
        print('Testing there are {} samples in testset'.format(len(testset)))
        testloader =  torch.utils.data.DataLoader(testset, batch_size=128, num_workers=16, shuffle=False, drop_last=False, pin_memory=True)
        test_acc = test(model, testloader)
        if test_acc > best_acc:
            print('beat test acc, saving model ... ')
            torch.save((model.module, loss_list, scaler), model_save_path)
            best_acc = test_acc
        del testloader
        
print('Done. Best acc was: ', best_acc)

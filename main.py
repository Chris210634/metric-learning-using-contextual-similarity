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
from losses import *
from sampling import *
from transforms import *
from networks import *
from utils import test_ks, get_features_and_labels_from_model_that_normalizes
######################

device = torch.device("cuda")

def test(G, test_loader=None, query_loader=None, gallery_loader=None):
    '''Please make sure G outputs normalized features.'''
    if not test_loader is None:
        assert query_loader is None and gallery_loader is None
        features, labels = get_features_and_labels_from_model_that_normalizes(G, test_loader, 
                                                                          embedding_size=args.embedding_size, 
                                                                          device=device)
        print('features.shape: {}, labels.shape: {}'.format(features.shape, labels.shape))
        query_features = features
        gallery_features = features
        query_labels = labels
        gallery_labels = labels
    else:
        query_features, query_labels = get_features_and_labels_from_model_that_normalizes(G, query_loader, 
                                                                          embedding_size=args.embedding_size, 
                                                                          device=device)
        gallery_features, gallery_labels = get_features_and_labels_from_model_that_normalizes(G, gallery_loader, 
                                                                          embedding_size=args.embedding_size, 
                                                                          device=device)
        print('query_features.shape: {}, query_labels.shape: {}'.format(query_features.shape, query_labels.shape))
        print('gallery_features.shape: {}, gallery_labels.shape: {}'.format(gallery_features.shape, gallery_labels.shape))
    
    # The similarity matrix is too big to fit in memory at once
    # we need to split into 24 pieces and evaluate independently
    # if you encounter memory issues, make n_splits bigger
    def helper(i, inc, f_T, test_loader):
        sm = query_features[inc*i:inc*(i+1)].cuda() @ f_T
        for j in range(sm.shape[0]):
            if not test_loader is None:
                # when the query and gallery sets are the same, we want to make sure we don't retrieve the sample itself
                sm[j,inc*i+j] = -1.
        _, indices = sm.max(1)
        predictions.append(indices.cpu())
        
    n_splits = 24
    inc = query_labels.shape[0] // n_splits
    predictions = []
    f_T = gallery_features.T.cuda()
    for i in tqdm(range(n_splits)):
        helper(i, inc, f_T, test_loader)

    # left over piece
    if query_labels.shape[0] % n_splits > 0:
        sm = query_features[inc*n_splits:].cuda() @ f_T
        for j in range(sm.shape[0]):
            if not test_loader is None:
                # when the query and gallery sets are the same, we want to make sure we don't retrieve the sample itself
                sm[j,inc*n_splits+j] = -1.
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
    return (query_labels == gallery_labels[predictions]).sum().item() / query_labels.shape[0]

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
elif args.dataset == 'inshop':
    train_txt = 'inshop_train.txt'
#     train_txt = 'inshop_train_super.txt'
    gallery_txt = 'inshop_gallery.txt'
    query_txt = 'inshop_query.txt'
    testxform = inshop_get_xform(augmentation=args.testxform)
    trainxform = inshop_get_xform(augmentation=args.trainxform)
    trainset = Imagelist(image_list=train_txt, root=root, transform=trainxform)
#     trainset = Imagelist_iNat(image_list=train_txt, root=root, transform=trainxform) # with superlabels
    testset = None
    queryset = Imagelist(image_list=query_txt, root=root, transform=testxform)
    galleryset = Imagelist(image_list=gallery_txt, root=root, transform=testxform)
else:
    assert False # Need to input a valid dataset.

if args.dataset == 'SOP' or args.dataset == 'iNat' or args.dataset == 'inshop':
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
    
    if args.dataset in ['SOP', 'iNat']:
#     if args.dataset in ['SOP', 'iNat', 'inshop']: # has superlabels
        trainsampler = HierarchicalSamplerBalanced(
                    dataset = trainset,
                    batch_size=args.batch_size,
                    samples_per_class=4,
                    batches_per_super_pair = 5 if args.dataset == 'inshop' else 10,
                    nb_categories=2,
                    batch_multiplier = 66 if args.dataset == 'SOP' else 70
                )
    else:
        trainsampler = RandomSampler(
                dataset = trainset,
                batch_size=args.batch_size,
                samples_per_class=4,
                num_batches = 600 if args.dataset == 'inshop' else 42,
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
        if not testset is None:
            print('Testing there are {} samples in testset'.format(len(testset)))
            testloader =  torch.utils.data.DataLoader(testset, batch_size=128, 
                                                      num_workers=16, shuffle=False, drop_last=False, pin_memory=True)
            test_acc = test(model, testloader)
            del testloader
        else:
            # use query and gallery set for inshop
            assert args.dataset == 'inshop'
            queryloader = torch.utils.data.DataLoader(queryset, batch_size=128, 
                                        num_workers=16, shuffle=False, drop_last=False, pin_memory=True)
            galleryloader = torch.utils.data.DataLoader(galleryset, batch_size=128, 
                                        num_workers=16, shuffle=False, drop_last=False, pin_memory=True)
            test_acc = test(model, test_loader=None, query_loader=queryloader, gallery_loader=galleryloader)
            del queryloader, galleryloader
        if test_acc > best_acc:
            print('beat test acc, saving model ... ')
            torch.save((model.module, loss_list, scaler), model_save_path)
            best_acc = test_acc
        
print('Done. Best acc was: ', best_acc)
print('Running final evaluation ... ')
model.module, loss_list, scaler = torch.load(model_save_path)
if args.dataset in ['CUB', 'Cars']:
    ks = [1, 2, 4, 8]
elif args.dataset == 'SOP':
    ks = [1, 10, 100]
elif args.dataset == 'iNat':
    ks = [1, 4, 16, 32]
elif args.dataset == 'inshop':
    ks = [1, 10, 20, 30, 40, 50]
else:
    assert False
    
if not testset is None:
    testloader =  torch.utils.data.DataLoader(testset, batch_size=128, num_workers=16, shuffle=False, drop_last=False, pin_memory=True)
    queryloader = None
    galleryloader = None
else:
    testloader = None
    queryloader = torch.utils.data.DataLoader(queryset, batch_size=128, 
                                        num_workers=16, shuffle=False, drop_last=False, pin_memory=True)
    galleryloader = torch.utils.data.DataLoader(galleryset, batch_size=128, 
                                num_workers=16, shuffle=False, drop_last=False, pin_memory=True)
test_ks(model, test_loader=testloader, query_loader=queryloader, gallery_loader=galleryloader, ks=ks, embedding_size=args.embedding_size)
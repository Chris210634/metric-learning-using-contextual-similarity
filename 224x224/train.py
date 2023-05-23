import torch, math, time, argparse, os
import random, dataset, utils, losses, net
import numpy as np
from net.resnet import *
from dataset import sampler
from torch.utils.data.sampler import BatchSampler
from tqdm import *

parser = argparse.ArgumentParser(description='')
# export directory, training and val datasets, test datasets
parser.add_argument('--dataset', default='cub')
parser.add_argument('--embedding-size', default = 512, type = int, dest = 'sz_embedding',
    help = 'Size of embedding that is appended to backbone model.'
)
parser.add_argument('--train_embedding', default = 2048, type = int)
parser.add_argument('--batch-size', default = 128, type = int, dest = 'sz_batch',
    help = 'Number of samples per batch.'
)
parser.add_argument('--epochs', default = 80, type = int, dest = 'nb_epochs',
    help = 'Number of training epochs.'
)
parser.add_argument('--gpu-id', default = 0, type = int,
    help = 'ID of GPU that is used for training.'
)
parser.add_argument('--workers', default = 8, type = int,
    dest = 'nb_workers',
    help = 'Number of workers for dataloader.'
)
parser.add_argument('--loss', default = 'Hybrid', help = 'Criterion for training')
parser.add_argument('--optimizer', default = 'adamw', help = 'Optimizer setting')
parser.add_argument('--bottleneck', default = 'linear')
parser.add_argument('--lr', default = 1e-4, type =float,help = 'Learning rate setting')
parser.add_argument('--weight-decay', default = 1e-4, type =float, help = 'Weight decay setting')
parser.add_argument('--lr-decay-step', default = 5, type =int, help = 'Learning decay step setting')
parser.add_argument('--lr-decay-gamma', default = 0.5, type =float, help = 'Learning decay gamma setting')
parser.add_argument('--gem', default = 1, type =int)
parser.add_argument('--projector_lr_multi', default = 2.0, type =float)
parser.add_argument('--embedding_lr_multi', default = 1.0, type =float)
parser.add_argument('--data_root', default = '', help = 'where is data?')

# HYBRID LOSS
parser.add_argument('--gamma', default = 0.1, type =float)
parser.add_argument('--lam', default = 0.5, type =float)
parser.add_argument('--pos_margin', default = 0.75, type =float)
parser.add_argument('--neg_margin', default = 0.6, type =float)
parser.add_argument('--regsim', default = 0.3, type =float)
parser.add_argument('--seed', default = 0, type =int)

# PROXY ANCHOR
parser.add_argument('--alpha', default = 32, type = float, help = 'Scaling Parameter setting')
parser.add_argument('--mrg', default = 0.1, type = float, help = 'Margin parameter setting')

# TRAINING
parser.add_argument('--IPC',  default = 4, type = int, help = 'Balanced sampling, images per class')
parser.add_argument('--warm', default = 0, type = int, help = 'Warmup training epochs')
parser.add_argument('--hierarchical', default = 0, type = int)
parser.add_argument('--bn-freeze', default = 1, type = int, help = 'Batch normalization parameter freeze')
parser.add_argument('--testfreq', default = 10, type = int )
parser.add_argument('--eps', default = 0.05, type = float )
parser.add_argument('--xform_scale', default = 0.08, type = float )
args = parser.parse_args()

################### RANDOM SEED #####################
save_seed = int(random.random()*1000000000)
save_path = '{}_{}_{}.pt'.format(args.loss, args.dataset, save_seed)
print('Save path: ', save_path)

seed = int(random.random()*1000000000)
if args.seed > 0:
    seed = args.seed

print('Using random seed : ', seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # set random seed for all gpus
######################################################

data_root = args.data_root

# Dataset Loader and Sampler
train_transform = dataset.utils.make_transform(
                is_train = True, 
                scale = args.xform_scale
            )

trn_dataset = dataset.load(
        name = args.dataset,
        root = data_root,
        mode = 'train',
        transform = train_transform)

print(train_transform)

# Sampling
if args.hierarchical:
    assert args.IPC
    balanced_sampler = sampler.HierarchicalSamplerBalanced(trn_dataset, batch_size=args.sz_batch, samples_per_class=args.IPC)
    print('Hierarchical Sampling')
    print(balanced_sampler)
    dl_tr = torch.utils.data.DataLoader(
        trn_dataset,
        num_workers = args.nb_workers,
        pin_memory = True,
        batch_sampler = balanced_sampler
    )
elif args.IPC:
    balanced_sampler = sampler.BalancedSampler(trn_dataset, batch_size=args.sz_batch, images_per_class=args.IPC)
    batch_sampler = BatchSampler(balanced_sampler, batch_size = args.sz_batch, drop_last = True)
    dl_tr = torch.utils.data.DataLoader(
        trn_dataset,
        num_workers = args.nb_workers,
        pin_memory = True,
        batch_sampler = batch_sampler
    )
    print('Balanced Sampling')
else:
    dl_tr = torch.utils.data.DataLoader(
        trn_dataset,
        batch_size = args.sz_batch,
        shuffle = True,
        num_workers = args.nb_workers,
        drop_last = True,
        pin_memory = True
    )
    print('Random Sampling')
    
# Setup evaluation datasets
eval_transform = dataset.utils.make_transform(
                is_train = False, 
                scale = args.xform_scale
            )

ev_dataset = dataset.load(
        name = args.dataset,
        root = data_root,
        mode = 'eval',
        transform = eval_transform)

dl_ev = torch.utils.data.DataLoader(
    ev_dataset,
    batch_size = 128, ###
    shuffle = False,
    num_workers = args.nb_workers,
    pin_memory = True
)

# Training Dataset with evaluation transform to track training accuracy
trn_normal_dataset = dataset.load(
        name = args.dataset,
        root = data_root,
        mode = 'train',
        transform = eval_transform)

dl_tmp = torch.utils.data.DataLoader(
    trn_normal_dataset,
    batch_size = 128,
    shuffle = False,
    num_workers = args.nb_workers,
    drop_last = False,
    pin_memory = True
)

print('eval_transform:')
print(eval_transform)

nb_classes = trn_dataset.nb_classes()

# Backbone Model
model = Resnet50(embedding_size=args.sz_embedding, pretrained=True, 
                 bn_freeze = args.bn_freeze, layer_norm=False, gem=bool(args.gem))
scaler = torch.cuda.amp.GradScaler()
model = model.cuda()
    
###########################################################################
class Bottleneck(nn.Module):
    def __init__(self, in_size, out_size):
        super(Bottleneck, self).__init__()
        assert args.bottleneck == 'linear'
        self.model = nn.Linear(in_size, out_size, bias=False)

    def forward(self, x):
        y = self.model(x)
        return F.normalize(y)
    
if args.bottleneck == 'identity':
    bottleneck = torch.nn.Identity()
else:
    bottleneck = Bottleneck(in_size=args.sz_embedding, out_size=args.train_embedding).cuda()
print('Bottleneck: ')
print(bottleneck)
###########################################################################
    
# DML Losses
def set_loss(loss):
    if loss == 'Proxy_Anchor':
        print(' Using Proxy Anchor ')
        emb = args.sz_embedding if args.bottleneck == 'identity' else args.train_embedding
        criterion = losses.Proxy_Anchor(nb_classes = nb_classes, sz_embed = emb, mrg = args.mrg, alpha = args.alpha).cuda()
    elif loss == 'Proxy_NCA':
        print(' Using Proxy NCA ')
        emb = args.sz_embedding if args.bottleneck == 'identity' else args.train_embedding
        criterion = losses.Proxy_NCA(nb_classes = nb_classes, sz_embed = emb).cuda()
    elif loss == 'NormalizedSoftmax':
        print(' Using Normalized Softmax ')
        criterion = losses.NormalizedSoftmax(nb_classes = nb_classes, sz_embed = args.train_embedding).cuda()
    elif loss == 'CosFace':
        print(' Using CosFace ')
        criterion = losses.CosFace(nb_classes = nb_classes, sz_embed = args.train_embedding).cuda()
    elif loss == 'MS': ###
        print(' Using MultiSimilarity ')
        criterion = losses.MultiSimilarityLoss().cuda()
    elif loss == 'MS+miner': ###
        print(' Using MultiSimilarity + miner')
        criterion = losses.MultiSimilarityLossWithMiner().cuda()
    elif loss == 'Contrastive': ###
        print(' Using Contrastive ')
        criterion = losses.ContrastiveLoss().cuda()
    elif loss == 'Roadmap': ###
        print(' Using Roadmap ')
        criterion = losses.RoadmapLoss(lam=0.5).cuda()
    elif loss == 'Hybrid': ###
        print(' Using Hybrid (Ours) ')
        criterion = losses.HybridLoss(pos_margin=args.pos_margin, 
                                      neg_margin=args.neg_margin, 
                                      lam=args.lam, k=args.IPC, eps=args.eps).cuda()
    elif loss == 'Triplet': ###
        print(' Using Triplet ')
        criterion = losses.TripletLoss().cuda()
    return criterion
        
criterion = set_loss(args.loss)

# Train Parameters
param_groups = [
    {'params': list(set(model.parameters()).difference(set(model.model.embedding.parameters()))) },
    {'params': model.model.embedding.parameters(), 'lr':float(args.lr) * args.embedding_lr_multi},
]

if args.loss in ['Proxy_Anchor', 'NormalizedSoftmax', 'CosFace']:
    print('setting criterion learning rate to : ', float(args.lr) * 100)
    print(' --> len(list(criterion.parameters())): ', len(list(criterion.parameters())))
    param_groups.append({'params': criterion.parameters(), 'lr':float(args.lr) * 100})
elif args.loss == 'Proxy_NCA':
    param_groups.append({'params': criterion.parameters(), 'lr':float(args.lr) * 100})

param_groups.append({'params': bottleneck.parameters(), 
                     'lr':float(args.lr) * args.projector_lr_multi
                    })

# Optimizer Setting
assert args.optimizer == 'adamw'
opt = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay = args.weight_decay)
    
print('******************************************************************')
print(args)
print('******************************************************************')

scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_decay_step, gamma = args.lr_decay_gamma)

print("Training parameters: {}".format(vars(args)))
print("Training for {} epochs.".format(args.nb_epochs))
losses_list = []
best_recall=[0]
best_epoch = 0

print(opt)

for epoch in range(0, args.nb_epochs):
    model.train()
    bottleneck.train()
    bn_freeze = args.bn_freeze
    if bn_freeze:
        modules = model.model.modules() 
        for m in modules: 
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    losses_per_epoch = []
    
    # Warmup: Train only new params, helps stabilize learning.
    if args.warm > 0:
        unfreeze_model_param = list(model.model.embedding.parameters()) + list(criterion.parameters())
        if epoch == 0:
            ## Turn on warmup
            for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                param.requires_grad = False
            for param in list(set(bottleneck.parameters())):
                assert param.requires_grad == True
            for param in list(set(model.model.embedding.parameters())):
                assert param.requires_grad == True
        if epoch == args.warm:
            ## Turn off warmup
            for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                param.requires_grad = True

    pbar = tqdm(enumerate(dl_tr))
    
    for batch_idx, batch in pbar:
        x, y = batch
        m = bottleneck(model(x.squeeze().cuda()))
        
        ####################################################################################
        criterion_loss = criterion(m, y.squeeze().cuda())
        loss = criterion_loss + args.gamma * losses.get_reg_loss(m, y.squeeze().cuda(), args.regsim)
        ####################################################################################
        
        opt.zero_grad()
        if not scaler is None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        pbar.set_description(
            'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx + 1, len(dl_tr),
                100. * batch_idx / len(dl_tr),
                loss.item()))
        
    scheduler.step()
    del m, loss, y, x
    
    if (epoch + 1) % args.testfreq == 0 or epoch == args.nb_epochs - 1:
        with torch.no_grad():
            print("**Evaluating on test data...**")
            if args.dataset != 'SOP':
                Recalls = utils.evaluate_cos(model, dl_ev)
            else:
                Recalls = utils.evaluate_cos_SOP(model, dl_ev)
            print("**Evaluating on train data...**")
            if args.dataset != 'SOP':
                Recalls = utils.evaluate_cos(model, dl_tmp)
            else:
                Recalls = utils.evaluate_cos_SOP(model, dl_tmp)
                
#         if best_recall[0] < Recalls[0]:
#             best_recall = Recalls
#             best_epoch = epoch
#             print("**Saving model ...**")
#             torch.save((model.state_dict(), bottleneck.state_dict()), save_path)
            
            
            
            
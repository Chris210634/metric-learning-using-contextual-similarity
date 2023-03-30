import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from pytorch_metric_learning import miners, losses
from smooth_rank_ap import SupAP, SmoothAP, HeavisideAP
from softbin_ap import SoftBinAP

def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        
    def forward(self, X, T):
        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term     
        
        return loss

# We use PyTorch Metric Learning library for the following codes.
# Please refer to "https://github.com/KevinMusgrave/pytorch-metric-learning" for details.
class NormalizedSoftmax(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed):
        super(NormalizedSoftmax, self).__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.loss_func = losses.NormalizedSoftmaxLoss(num_classes = self.nb_classes, embedding_size = self.sz_embed).cuda()

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
    
class CosFace(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed):
        super(CosFace, self).__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.loss_func = losses.CosFaceLoss(num_classes = self.nb_classes, embedding_size = self.sz_embed).cuda()

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
    
class Proxy_NCA(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale=9):
        super(Proxy_NCA, self).__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.scale = scale
        self.loss_func = losses.ProxyNCALoss(num_classes = self.nb_classes, embedding_size = self.sz_embed, softmax_scale = self.scale).cuda()

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
    
class MultiSimilarityLoss(torch.nn.Module): # also try with miner
    def __init__(self, ):
        super(MultiSimilarityLoss, self).__init__()
        self.loss_func = losses.MultiSimilarityLoss()
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
    
class MultiSimilarityLossWithMiner(torch.nn.Module):
    def __init__(self, **kwargs):
        super(MultiSimilarityLossWithMiner, self).__init__()
        self.miner = miners.MultiSimilarityMiner()
        self.loss_func = losses.MultiSimilarityLoss()
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        return  self.loss_func(embeddings, labels, hard_pairs)
    
def get_jaccard_similarity(w, scores, k, eps):
    # features have been normalized before this function
    # D is ||u-v||^2 = 2 - 2<u,v>
    D = 2. - 2. * scores  # Squared Euclidedan, assume features are L2 normalized
    D = D.clamp(min=0.)   # for stability
    vk, ik = (-D).topk(k) # get k closest neighbors

    Nk_mask = GreaterThan.apply(-D + eps, vk[:,-1:].detach())
    intersection = (Nk_mask @ Nk_mask.T) / Nk_mask.sum(1, keepdim=True).detach()
    Nk_mask_not = 1. - Nk_mask
    intersection_not = (Nk_mask_not @ Nk_mask_not.T) / Nk_mask_not.sum(1, keepdim=True).detach()
    W_tilde = 0.5 * (intersection + intersection_not) * Nk_mask

    Nk_over_2_mask = GreaterThan.apply(-D + eps, vk[:,(k // 2)-1:(k // 2)].detach())
    Rk_over_2_mask = Nk_over_2_mask * Nk_over_2_mask.T
    
    jaccard_similarity = (Rk_over_2_mask @ W_tilde) / Rk_over_2_mask.sum(1, keepdim=True)

    return 0.5 * (jaccard_similarity + jaccard_similarity.T)

class GreaterThan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return (x >= y).float()
    @staticmethod
    def backward(ctx, grad_output):
        x,y = ctx.saved_tensors
        g = 10.
        return grad_output * g, - grad_output * g
    
class SimilarityGradientMask(torch.autograd.Function):
    reverse = False
    @staticmethod
    def forward(ctx, scores, w):
        ctx.save_for_backward(w)
        return scores
    @staticmethod
    def backward(ctx, grad_output):
        w, = ctx.saved_tensors
        # scores is similarity matrix
        # if w == 1.0, score should increase, gradient should be negative
        # if w == 0.0, score should decrease, gradient should be positive
        return (1.0 - w) * grad_output.clamp(min=0.0) + w * grad_output.clamp(max=0.0), torch.zeros_like(w)

def get_reg_loss(embeddings, labels, reg_sim):
    w = (labels.unsqueeze(0).T == labels.unsqueeze(0)).float().cuda()
    scores = F.normalize(embeddings) @ F.normalize(embeddings).T
    reg_loss = (scores.mean() - reg_sim).square()
    return reg_loss
    
def contrastive_jaccard(scores, w, k, eps):
    js = get_jaccard_similarity(w, scores, k=k, eps=eps)
    I_neg = torch.ones_like(js) - torch.eye(js.shape[0], device=js.device)

    l_pos = ((1. - js).square() * w * I_neg)  # positive pairs 
    l_neg = (js.square() * (1. - w) * I_neg)  # negative pairs
    
    loss_context = ((l_pos.sum() + l_neg.sum()) / js.shape[0] / js.shape[0])
    return loss_context
    
def margin_contrastive(scores, w, pos_margins=0.9, neg_margins=0.6):
    '''pos_margins and neg_margins must be column vectors or scalars
    '''
    L_pos = F.relu(pos_margins - scores) * w
    L_neg = F.relu(scores - neg_margins) * (1. - w)
    
    if (L_pos > 0.).sum() < 1e-5:
        l_p = torch.tensor(0.)
    else:
        l_p = L_pos.sum() / (L_pos > 0.).sum()
    if (L_neg > 0.).sum() < 1e-5:
        l_n = torch.tensor(0.)
    else:
        l_n = L_neg.sum() / (L_neg > 0.).sum()
        
    return l_p + l_n

class SoftbinLoss(nn.Module):
    def __init__(self, pos_margin=0.9, neg_margin=0.6, lam=0.5, **kwargs):
        super(SoftbinLoss, self).__init__()
        self.criterion = SoftBinAP()
        print(self.criterion)
        self.lam = lam
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        
    def forward(self, embeddings, labels): 
        w = (labels.unsqueeze(0).T == labels.unsqueeze(0)).float().cuda()
        scores = F.normalize(embeddings) @ F.normalize(embeddings).T
        loss = self.lam * margin_contrastive(scores, w, pos_margins=self.pos_margin, neg_margins=self.neg_margin) + (1. - self.lam) * self.criterion(scores, w)
        return loss

class SmoothapLoss(nn.Module):
    def __init__(self, pos_margin=0.9, neg_margin=0.6, lam=0.5, **kwargs):
        super(SmoothapLoss, self).__init__()
        self.criterion = SmoothAP()
        print(self.criterion)
        self.lam = lam
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        
    def forward(self, embeddings, labels): 
        w = (labels.unsqueeze(0).T == labels.unsqueeze(0)).float().cuda()
        scores = F.normalize(embeddings) @ F.normalize(embeddings).T
        loss = self.lam * margin_contrastive(scores, w, pos_margins=self.pos_margin, neg_margins=self.neg_margin) + (1. - self.lam) * self.criterion(scores, w)
        return loss

class RoadmapLoss(nn.Module):
    def __init__(self, pos_margin=0.9, neg_margin=0.6, lam=0.5, **kwargs):
        super(RoadmapLoss, self).__init__()
        self.roadmap_criterion = SupAP()
        print(self.roadmap_criterion)
        self.lam = lam
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        
    def forward(self, embeddings, labels): 
        w = (labels.unsqueeze(0).T == labels.unsqueeze(0)).float().cuda()
        scores = F.normalize(embeddings) @ F.normalize(embeddings).T
        loss = self.lam * margin_contrastive(scores, w, pos_margins=self.pos_margin, neg_margins=self.neg_margin) + (1. - self.lam) * self.roadmap_criterion(scores, w)
        return loss

class HybridLoss(nn.Module):
    def __init__(self, pos_margin=0.75, neg_margin=0.6, lam=0.4, k=4, eps=0.05, **kwargs):
        super(HybridLoss, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.lam = lam
        self.k = k
        self.eps = eps
        
    def forward(self, embeddings, labels): 
        w = (labels.unsqueeze(0).T == labels.unsqueeze(0)).float().cuda()
        scores = F.normalize(embeddings) @ F.normalize(embeddings).T
        context_loss = contrastive_jaccard(scores, w, k=self.k, eps=self.eps) ###
        lambda_loss = margin_contrastive(scores, w, pos_margins=self.pos_margin, neg_margins=self.neg_margin)
        loss = self.lam * lambda_loss + (1. - self.lam) * context_loss
        
        scores.retain_grad()
        return loss, scores, w
    
def get_ap_logging(scores, w):
    crit = HeavisideAP()
    return crit(scores, w)
    
class ContrastiveLoss(nn.Module):
    def __init__(self, **kwargs):
        super(ContrastiveLoss, self).__init__()
        
    def forward(self, embeddings, labels):
        w = (labels.unsqueeze(0).T == labels.unsqueeze(0)).float().cuda()
        scores = F.normalize(embeddings) @ F.normalize(embeddings).T
        return margin_contrastive(scores, w, pos_margins=0.9, neg_margins=0.6)
    
class TripletLoss(nn.Module):
    def __init__(self, margin=0.05, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        # self.miner = miners.TripletMarginMiner(margin, type_of_triplets = 'semihard')
        self.miner = miners.DistanceWeightedMiner()
        self.loss_func = losses.TripletMarginLoss(margin = self.margin)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss
    
class NPairLoss(nn.Module):
    def __init__(self, l2_reg=0):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg
        self.loss_func = losses.NPairsLoss(l2_reg_weight=self.l2_reg, normalize_embeddings = False)
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
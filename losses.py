import torch.nn.functional as F
import torch

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
    
def contrastive_jaccard(scores, w, reg_sim, eps):
    reg_loss = (scores.mean() - reg_sim).square()

    js = get_jaccard_similarity(w, scores, k=4, eps=eps)
    I_neg = torch.ones_like(js) - torch.eye(js.shape[0], device=js.device)

    l_pos = ((1. - js).square() * w * I_neg)  # positive pairs 
    l_neg = (js.square() * (1. - w) * I_neg)  # negative pairs
    
    loss_context = ((l_pos.sum() + l_neg.sum()) / js.shape[0] / js.shape[0])

    return loss_context, reg_loss

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
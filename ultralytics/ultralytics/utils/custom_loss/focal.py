import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossWithSigmoid(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='none'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        pred_sigmoid = pred.sigmoid()
        
        target = target.type_as(pred)
        
        # 如果target=1,则pt=pred_sigmoid；否则为1-pred_sigmoid
        pt = target * pred_sigmoid + (1 - target) * (1 - pred_sigmoid)
        
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * (1.0-pt).pow(self.gamma)
        
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
        
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


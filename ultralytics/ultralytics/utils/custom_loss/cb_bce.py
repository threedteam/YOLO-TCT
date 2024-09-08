import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch

def compute_class_balance_weights(samples_per_cls, beta=0.9999):
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    
    # weights = weights / np.sum(weights) * len(samples_per_cls)
    weights = weights / np.sum(weights)  # Normalize weights
    
    weights = torch.tensor(weights, dtype=torch.float32)
    return weights


class CBBCELoss(nn.Module):
    def __init__(self, samples_per_cls, beta=0.9999, reduction='none'):
        super(CBBCELoss, self).__init__()
        self.weights = compute_class_balance_weights(samples_per_cls, beta)
        self.reduction = reduction
        
    def forward(self, pred_score, target, labels_one_hot):
        weights = self.weights.to(pred_score.device)

        sample_weights = weights.unsqueeze(0) * labels_one_hot
        sample_weights = sample_weights.sum(1).unsqueeze(1)
        
        bce_loss = F.binary_cross_entropy_with_logits(pred_score, target, reduction='none')
        
        cb_bce_loss = sample_weights * bce_loss
        
        if self.reduction == 'mean':
            cb_bce_loss = cb_bce_loss.mean()
        elif self.reduction == 'sum':
            cb_bce_loss = cb_bce_loss.sum()
        elif self.reduction == 'log':
            cb_bce_loss = torch.log1p(cb_bce_loss) # log1p(x) = log(1 + x)
        return cb_bce_loss
    
# Example usage
if __name__ == "__main__":
    samples_per_cls = [50, 100, 300]
    cb_bce_loss_fn = CBBCELoss(samples_per_cls)

    # Example logits and labels
    pred_score = torch.randn(10, len(samples_per_cls))
    labels = torch.randint(0, len(samples_per_cls), (10,))
    
    loss = cb_bce_loss_fn(pred_score, labels)
    print(loss)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CBFocalQualityLoss(nn.Module):
    def __init__(self, samples_per_cls, beta=0.9999, qf_beta=2.0):
        super(CBFocalQualityLoss, self).__init__()
        self.qf_beta = qf_beta
        
        # Calculate the effective number of samples for each class
        self.effective_num = 1.0 - np.power(beta, samples_per_cls)
        
        # Compute the class-balanced weights
        self.weights = (1.0 - beta) / np.array(self.effective_num)
        
        # Normalize the weights so their sum equals the number of classes
        self.weights = self.weights / np.sum(self.weights) * len(samples_per_cls)

        # Convert weights to a PyTorch tensor
        self.weights = torch.tensor(self.weights, dtype=torch.float32)

    def forward(self, pred_score, gt_score, gt_target_pos_mask, labels_one_hot):
        # Move weights to the same device as pred_score
        weights = self.weights.to(pred_score.device)

        # Compute per-sample weights
        sample_weights = weights.unsqueeze(0).unsqueeze(0) * labels_one_hot
        sample_weights = sample_weights.sum(2).unsqueeze(2)
        
        # negatives are supervised by 0 quality score
        pred_sigmoid = pred_score.sigmoid()
        scale_factor = pred_sigmoid
        
        zerolabel = scale_factor.new_zeros(pred_score.shape)
        with torch.cuda.amp.autocast(enabled=False):
            loss = F.binary_cross_entropy_with_logits(pred_score, zerolabel, reduction='none') * scale_factor.pow(self.qf_beta)
        
        scale_factor = gt_score[gt_target_pos_mask] - pred_sigmoid[gt_target_pos_mask]
        with torch.cuda.amp.autocast(enabled=False):
            loss[gt_target_pos_mask] = F.binary_cross_entropy_with_logits(pred_score[gt_target_pos_mask], gt_score[gt_target_pos_mask], reduction='none') * scale_factor.abs().pow(self.qf_beta)
        
        # Apply class-balanced weights
        cb_loss = sample_weights * loss
        
        return cb_loss

if __name__ == "__main__":
    # Example usage
    samples_per_cls = [50, 100, 300]
    no_of_classes = len(samples_per_cls)
    cb_qf_loss = CBFocalQualityLoss(samples_per_cls, no_of_classes)

    # Example logits, gt_score, gt_target_pos_mask, and labels
    pred_score = torch.randn(10, no_of_classes)
    gt_score = torch.rand(10, no_of_classes)
    gt_target_pos_mask = torch.rand(10, no_of_classes) > 0.5
    labels = torch.randint(0, no_of_classes, (10,))
    loss = cb_qf_loss(pred_score, gt_score, gt_target_pos_mask, labels)
    print(loss)

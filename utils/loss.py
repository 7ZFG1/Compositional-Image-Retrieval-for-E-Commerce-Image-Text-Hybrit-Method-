import torch
import torch.nn as nn
import torch.nn.functional as F
    
class InfoNCELoss(nn.Module):
    """
    Encourages the model to pull together the correct (query, target) 
    pairs while pushing apart all other combinations.
    """
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, query_features, target_features):
        query = F.normalize(query_features, dim=-1)
        target = F.normalize(target_features, dim=-1)
        
        # Cosinuse similarity matrix [B, B]
        logits = (query @ target.T) / self.temperature
        
        # Correct matches on the diagonal (0, 1, 2... B-1)
        labels = torch.arange(logits.shape[0], device=logits.device)
        
        # Calculate loss for both directions (query->target and target->query)
        loss_q = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        
        return (loss_q + loss_t) / 2.0

class MSLoss(nn.Module):
    """
    Multi-Similarity Loss.
    It penalizes hard negatives and positives more aggressively, making it effective for fine-grained distinctions
    """
    def __init__(self, alpha=2.0, beta=50.0, base=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base

    def forward(self, query_features, target_features):
        query = F.normalize(query_features, dim=-1)
        target = F.normalize(target_features, dim=-1)
        
        sim = query @ target.T
        batch_size = sim.shape[0]
        
        # Diagonal matrix mask
        mask = torch.eye(batch_size, device=sim.device).bool()
        
        # Positive similarities [B]
        pos_sim = sim[mask]
        
        # Negative similarities [B, B-1]
        neg_sim = sim[~mask].view(batch_size, -1)
        
        # MS Loss Formula
        pos_loss = torch.log(1 + torch.exp(-self.alpha * (pos_sim - self.base))).mean() / self.alpha
        neg_exp = torch.exp(self.beta * (neg_sim - self.base))
        neg_loss = torch.log(1 + neg_exp.sum(dim=-1)).mean() / self.beta
        
        return pos_loss + neg_loss
    
class TripletLoss(nn.Module):
    """
    Classic Triplet Loss.
    Encourages the model to pull together the correct (query, target) 
    pairs while pushing apart incorrect (query, negative) pairs by a margin.
    """
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2)

    def forward(self, query_features, target_features):
        # In this simplified version, we treat all other samples in the batch as negatives
        batch_size = query_features.shape[0]
        
        # Create negative features by shuffling the target features within the batch
        neg_indices = torch.randperm(batch_size)
        negative_features = target_features[neg_indices]
        
        return self.triplet_loss(query_features, target_features, negative_features)
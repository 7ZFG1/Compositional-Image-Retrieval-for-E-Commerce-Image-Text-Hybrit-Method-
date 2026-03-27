import torch
import torch.nn as nn
import torch.nn.functional as F

class FashionCombiner(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, image_features, text_features):
        combined = torch.cat([image_features, text_features], dim=-1)
        delta = self.fusion(combined)
        query_embedding = image_features + delta
        return F.normalize(self.layer_norm(query_embedding), p=2, dim=-1)
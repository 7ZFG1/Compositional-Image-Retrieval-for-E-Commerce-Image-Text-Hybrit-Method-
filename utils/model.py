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
    

class FashionCombiner_v2(nn.Module):
    """
    CLIP-based image + text combiner.

    Improvements:

     1- Cross-Attention Fusion — learns by attending to which regions of the image the text should influence.
     2- Learned Gating — dynamically balances the weights of image and text deltas for each sample.
     3- Projection Heads — enables transition from CLIP’s frozen embedding space to the combiner’s own space.

    Args:

     * embed_dim (int): CLIP embedding dimension (default: 512).
     * num_heads (int): Number of cross-attention heads (default: 8).
     * dropout (float): Dropout rate (default: 0.2).
    """
 
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.embed_dim = embed_dim
 
        # ── 3. Projection Heads ───────────────────────────────────────────
        # Projection from CLIP's frozen space into a separate representation space.
        # Each modality learns its own linear transformation.
        self.img_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.txt_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
 
        # ── 1. Cross-Attention Fusion ─────────────────────────────────────
        # Text (query) attends over image (key/value).
        # Learns the question: "Which parts of the image does this text affect?"
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(embed_dim)
 
        # ── Fusion MLP ────────────────────────────────────────────────────
        # Produces a delta from cross-attention output + original image.
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),
            nn.GELU(),                          # Activation compatible with CLIP
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
 
        # ── 2. Gating Mechanism ───────────────────────────────────────────
        # Learns per-sample decision: "how much image delta vs text delta"
        # using a sigmoid in the range [0,1].
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
 
        # ── Output normalization ──────────────────────────────────────────
        self.layer_norm = nn.LayerNorm(embed_dim)
 
    # ─────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        image_features: torch.Tensor,   # [B, embed_dim]  — CLIP image embedding
        text_features: torch.Tensor,    # [B, embed_dim]  — CLIP text embedding
    ) -> torch.Tensor:                  # [B, embed_dim]  — normalized query
        """
        Args:
            image_features: CLIP image encoder output, shape (B, D).
            text_features:  CLIP text encoder output, shape (B, D).
 
        Returns:
            Normalized query embedding, shape (B, D).
        """
 
        # ── 3. Projection ────────────────────────────────────────────────
        img = self.img_proj(image_features)   # [B, D]
        txt = self.txt_proj(text_features)    # [B, D]
 
        # ── 1. Cross-Attention ────────────────────────────────────────────
        # MultiheadAttention with batch_first=True → [B, seq, D]
        img_seq = img.unsqueeze(1)            # [B, 1, D]
        txt_seq = txt.unsqueeze(1)            # [B, 1, D]
 
        # Text queries the image: query=txt, key/value=img
        attended, _ = self.cross_attn(
            query=txt_seq,
            key=img_seq,
            value=img_seq,
        )                                     # [B, 1, D]
        attended = attended.squeeze(1)        # [B, D]
 
        # Residual + normalization (Post-LN style)
        attended = self.cross_attn_norm(attended + txt)   # [B, D]
 
        # ── Fusion MLP → text-guided delta ─────────────────────────────
        txt_delta = self.fusion(
            torch.cat([img, attended], dim=-1)
        )                                                  # [B, D]
 
        # ── 2. Gating ─────────────────────────────────────────────────────
        # img_delta: pure image contribution (identity-like)
        img_delta = img                                    # [B, D]
 
        gate = self.gate(
            torch.cat([image_features, text_features], dim=-1)
        )                                                  # [B, D] ∈ (0,1)
 
        # Weighted delta: if gate is high → image dominates, else text dominates
        delta = gate * img_delta + (1.0 - gate) * txt_delta   # [B, D]
 
        # ── Residual connection + normalization ────────────────────────────────
        query_embedding = image_features + delta
        return F.normalize(self.layer_norm(query_embedding), p=2, dim=-1)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, D = 4, 512
    model = FashionCombiner_v2(embed_dim=D, num_heads=8)
    model.eval()
 
    img_feat = torch.randn(B, D)
    txt_feat = torch.randn(B, D)
 
    with torch.no_grad():
        out = model(img_feat, txt_feat)
 
    assert out.shape == (B, D), f"Unexpected shape: {out.shape}"
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(B), atol=1e-5), "L2 normalization error"
 
    print(f"✓ Output shape : {out.shape}")
    print(f"✓ L2 norms     : {norms}")
    print("All checks passed.")
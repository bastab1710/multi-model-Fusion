"""
model_advanced.py
Cross-Attention & Dynamic-Gated Multimodal Model for Image-Text Matching (CADF-ITM).

Differentiating innovations over standard Late Fusion:
1. Spatial Patch Preservation: Retains 7x7=49 visual tokens from ResNet50 Layer4 rather than discarding spatial detail.
2. Bidirectional Cross-Modal Cross-Attention (CMA): Allows words to attend directly to visual regions (e.g. 'snow' vs 'grass').
3. Dynamic Modality Reliability Gating (DMF): Learns sample-adaptive weights to prevent modality collapse.
4. Dual-Head Joint Decision: Combines cross-attended deep fusion with metric contrastive cosine alignment.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Automatically set Torch cache to workspace local directory
if "TORCH_HOME" not in os.environ:
    workspace_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".torch_cache")
    os.environ["TORCH_HOME"] = workspace_cache


class PatchVisualEncoder(nn.Module):
    """
    Extracts 7x7 spatial patch tokens (49 visual tokens) + 1 global token from frozen ResNet50.
    """
    def __init__(self, embed_dim=128):
        super(PatchVisualEncoder, self).__init__()
        weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)

        # Freeze ResNet backbone
        for param in resnet.parameters():
            param.requires_grad = False

        # Extract up to layer4
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # 1x1 conv to project 2048 -> embed_dim
        self.proj_patches = nn.Conv2d(2048, embed_dim, kernel_size=1)
        self.proj_global = nn.Linear(2048, embed_dim)

    def forward(self, x):
        # x shape: (B, 3, 224, 224)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat_map = self.layer4(x)  # (B, 2048, 7, 7)

        # Spatial patch tokens
        patch_tokens = self.proj_patches(feat_map)  # (B, embed_dim, 7, 7)
        B, C, H, W = patch_tokens.shape
        patch_tokens = patch_tokens.flatten(2).transpose(1, 2)  # (B, 49, embed_dim)

        # Global vector
        global_feat = self.avgpool(feat_map).flatten(1)  # (B, 2048)
        global_token = self.proj_global(global_feat).unsqueeze(1)  # (B, 1, embed_dim)

        # Combined visual tokens: (B, 50, embed_dim)
        vis_tokens = torch.cat([global_token, patch_tokens], dim=1)
        return vis_tokens, global_token.squeeze(1)


class WordTextEncoder(nn.Module):
    """
    Learns token embeddings for words, producing a sequence of word tokens + global pooled token.
    """
    def __init__(self, vocab_size=5000, embed_dim=128, padding_idx=0):
        super(WordTextEncoder, self).__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, text_tokens):
        # text_tokens: (B, max_len)
        embed = self.embedding(text_tokens)  # (B, max_len, embed_dim)
        embed = self.proj(embed)

        mask = (text_tokens != self.padding_idx).unsqueeze(-1).float()
        masked = embed * mask
        global_text = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # (B, embed_dim)
        return embed, global_text


class CrossAttentionFusionBlock(nn.Module):
    """
    Bidirectional Cross-Modal Attention between Visual Patches and Text Words.
    """
    def __init__(self, embed_dim=128, num_heads=4, dropout=0.2):
        super(CrossAttentionFusionBlock, self).__init__()
        self.text_to_img_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.img_to_text_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm_t = nn.LayerNorm(embed_dim)
        self.norm_v = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, vis_tokens, text_tokens):
        # Text queries image patches
        t2i, _ = self.text_to_img_attn(query=text_tokens, key=vis_tokens, value=vis_tokens)
        t_out = self.norm_t(text_tokens + t2i)  # (B, max_len, embed_dim)

        # Image queries text words
        i2t, _ = self.img_to_text_attn(query=vis_tokens, key=text_tokens, value=text_tokens)
        v_out = self.norm_v(vis_tokens + i2t)  # (B, 50, embed_dim)

        # Pool cross-attended representations
        t_pool = t_out.mean(dim=1)  # (B, embed_dim)
        v_pool = v_out.mean(dim=1)  # (B, embed_dim)

        fused = torch.cat([v_pool, t_pool], dim=1)  # (B, 256)
        fused = self.ffn(fused)  # (B, 128)
        return fused


class DynamicGatedITMModel(nn.Module):
    """
    Cross-Attention & Dynamic-Gated Multimodal Model for ITM (CADF-ITM).
    """
    def __init__(self, vocab_size=5000, embed_dim=128):
        super(DynamicGatedITMModel, self).__init__()
        self.vis_encoder = PatchVisualEncoder(embed_dim=embed_dim)
        self.text_encoder = WordTextEncoder(vocab_size=vocab_size, embed_dim=embed_dim)
        self.cross_attn = CrossAttentionFusionBlock(embed_dim=embed_dim)

        # Dynamic Reliability Gating Subnetwork (DMF)
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

        # Deep Judge Decision Head
        self.judge = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, images, text_tokens):
        vis_tokens, g_vis = self.vis_encoder(images)
        txt_tokens, g_txt = self.text_encoder(text_tokens)

        # 1. Cross-Modal Attention features
        attn_features = self.cross_attn(vis_tokens, txt_tokens)  # (B, 128)

        # 2. Dynamic Gating weights
        gate_weights = self.gate_net(torch.cat([g_vis, g_txt], dim=1))  # (B, 2)
        alpha_v = gate_weights[:, 0:1]
        alpha_t = gate_weights[:, 1:2]

        # 3. Gated Global Representation
        gated_global = alpha_v * g_vis + alpha_t * g_txt  # (B, 128)

        # 4. Joint Multimodal Decision (Cross-Attended + Gated Global)
        joint_repr = torch.cat([attn_features, gated_global], dim=1)  # (B, 256)
        match_score = self.judge(joint_repr)  # (B, 1)

        # 5. Cosine Metric Similarity (for contrastive regularizer)
        cos_sim = F.cosine_similarity(g_vis, g_txt, dim=1).unsqueeze(1)  # in [-1, 1]

        return match_score, cos_sim, gate_weights

    def get_parameter_summary(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}


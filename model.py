"""
model.py
PyTorch implementation of the Late-Fusion Multimodal Model for Image-Text Matching (ITM).
Based on the architecture specified in Assam Science and Technology University's research project:
  1. Image branch (vision): Frozen ResNet50 backbone (2048-d) -> Linear projection (128-d) + ReLU + Dropout(0.2)
  2. Text branch (language): Embedding (128-d) + Global Average Pooling -> Linear projection (128-d) + ReLU + Dropout(0.2)
  3. Late Fusion (the judge): Concatenation (256-d) -> Dense(128) -> Dense(64) -> Dense(1) + Sigmoid
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models

# Automatically set Torch cache to workspace local directory if not specified
if "TORCH_HOME" not in os.environ:
    workspace_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".torch_cache")
    os.environ["TORCH_HOME"] = workspace_cache


class ImageBranch(nn.Module):
    """
    Image feature extractor using pre-trained ResNet50 with frozen weights.
    Outputs a 128-dimensional projected visual embedding.
    """
    def __init__(self, projection_dim=128, dropout_rate=0.2):
        super(ImageBranch, self).__init__()
        # Load pre-trained ResNet50
        weights = models.ResNet50_Weights.DEFAULT
        self.backbone = models.resnet50(weights=weights)

        # Freeze all backbone layers
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Extract 2048-dim pooled feature vector by replacing fc layer
        in_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Identity()

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(in_features, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        features = self.backbone(x)  # shape: (batch_size, 2048)
        embeddings = self.projection(features)  # shape: (batch_size, projection_dim)
        return embeddings


class TextBranch(nn.Module):
    """
    Text feature extractor using learned Token Embeddings and Global Average Pooling.
    Outputs a 128-dimensional projected textual embedding.
    """
    def __init__(self, vocab_size=5000, embedding_dim=128, projection_dim=128, dropout_rate=0.2, padding_idx=0):
        super(TextBranch, self).__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, text_tokens):
        # text_tokens shape: (batch_size, max_seq_len)
        embedded = self.embedding(text_tokens)  # (batch_size, seq_len, embed_dim)

        # Mask out padding tokens for Global Average Pooling
        mask = (text_tokens != self.padding_idx).unsqueeze(-1).float()  # (batch_size, seq_len, 1)
        masked_embed = embedded * mask
        sum_embed = masked_embed.sum(dim=1)  # (batch_size, embed_dim)
        lengths = mask.sum(dim=1).clamp(min=1.0)  # (batch_size, 1)
        gap = sum_embed / lengths  # (batch_size, embed_dim)

        embeddings = self.projection(gap)  # (batch_size, projection_dim)
        return embeddings


class LateFusionJudge(nn.Module):
    """
    Late-Fusion Decision Network ('The Judge').
    Takes concatenated visual and textual vectors (256-d) and predicts match score in [0, 1].
    """
    def __init__(self, in_features=256, dropout_rate=0.3):
        super(LateFusionJudge, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, combined_features):
        return self.classifier(combined_features)


class LateFusionMultimodalModel(nn.Module):
    """
    Full End-to-End Late-Fusion Multimodal Model for Image-Text Matching (ITM).
    Combines:
      - Image Branch (ResNet50 frozen -> 128-d)
      - Text Branch (Embedding + GAP -> 128-d)
      - Late-Fusion Judge (Concatenation 256-d -> Sigmoid Score)
    """
    def __init__(self, vocab_size=5000, embedding_dim=128, projection_dim=128):
        super(LateFusionMultimodalModel, self).__init__()
        self.image_branch = ImageBranch(projection_dim=projection_dim)
        self.text_branch = TextBranch(vocab_size=vocab_size, embedding_dim=embedding_dim, projection_dim=projection_dim)
        self.judge = LateFusionJudge(in_features=projection_dim * 2)

    def forward(self, images, text_tokens):
        img_feats = self.image_branch(images)       # (batch, 128)
        text_feats = self.text_branch(text_tokens)  # (batch, 128)
        fused = torch.cat([img_feats, text_feats], dim=1)  # (batch, 256)
        match_score = self.judge(fused)             # (batch, 1)
        return match_score

    def get_parameter_summary(self):
        """Returns counts of trainable vs frozen parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total = trainable + frozen
        return {"trainable": trainable, "frozen": frozen, "total": total}


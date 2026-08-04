# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
from torch import nn
from typing import Any, Optional, Tuple, Type
import math

class PromptEncoder(nn.Module):
    def __init__(self, embed_dim: int, image_embedding_size: Tuple[int, int, int],) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size 
        # Dense positional encoding for image grid (used as image_pe by the decoder)
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        # Dense "no-prompt" embedding (broadcast over the image embedding grid)
        self.no_prompt_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0) # dense positional encoding [1, 768, H′, W′, D′]

    def forward(self, text_embedding: Optional[torch.Tensor],) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.no_prompt_embed.weight.device
        dtype  = self.no_prompt_embed.weight.dtype
        
        bs = 1 if text_embedding is None else int(text_embedding.shape[0])
        
        H, W, D = map(int, self.image_embedding_size)
        dense_embeddings = self.no_prompt_embed.weight.reshape(1, -1, 1, 1, 1).to(device=device, dtype=dtype)
        dense_embeddings = dense_embeddings.expand(bs, -1, H, W, D)  # [B,C,H',W',D']
        
        return dense_embeddings # returns a dense no-prompt grid [B, 768, H′, W′, D′] added to features


class PositionEmbeddingRandom(nn.Module):
    """
    3D positional encoding using random spatial frequencies.
    Returns a dense PE of shape [C=2*num_pos_feats, H, W, D].
    """
    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if not scale or scale <= 0.0:
            scale = 1.0
        torch.manual_seed(0)
        self.register_buffer("positional_encoding_gaussian_matrix", scale * torch.randn((3, num_pos_feats)))

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        # coords in [0,1], shape [..., 3]
        x = 2.0 * coords - 1.0                         # [-1,1]
        x = (x @ self.positional_encoding_gaussian_matrix) * (2.0 * math.pi)  # [..., F]
        return torch.cat([x.sin(), x.cos()], dim=-1)   # [..., 2F]

    def forward(self, size: Tuple[int, int, int]) -> torch.Tensor:
        """Generate PE for a grid of size (H, W, D)."""
        h, w, d = size
        device = self.positional_encoding_gaussian_matrix.device
        dtype  = self.positional_encoding_gaussian_matrix.dtype

        # normalized centers in [0,1]
        y = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h
        x = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w
        z = (torch.arange(d, device=device, dtype=dtype) + 0.5) / d
        yy, xx, zz = torch.meshgrid(y, x, z, indexing="ij")   # [H,W,D] each

        pe = self._pe_encoding(torch.stack([xx, yy, zz], dim=-1))  # [H,W,D,2F]
        return pe.permute(3, 0, 1, 2)  # [C=2F, H, W, D]

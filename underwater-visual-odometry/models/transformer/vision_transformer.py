"""
Vision Transformer for feature extraction from underwater images
Based on ViT architecture but adapted for underwater visual odometry
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch embedding using convolution
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [batch_size, channels, height, width]
        Returns:
            Patch embeddings [batch_size, num_patches, embed_dim]
        """
        batch_size = x.shape[0]
        
        # Create patches and embed
        x = self.projection(x)  # [batch_size, embed_dim, H/P, W/P]
        x = x.flatten(2)        # [batch_size, embed_dim, num_patches]
        x = x.transpose(1, 2)   # [batch_size, num_patches, embed_dim]
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [batch_size, seq_len, embed_dim]
        Returns:
            Attended features [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ v  # [batch_size, num_heads, seq_len, head_dim]
        out = out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        
        # Final projection
        out = self.proj(out)
        
        return out


class MLP(nn.Module):
    """Multi-layer perceptron with GELU activation"""
    
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        mlp_ratio: float = 4.0, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [batch_size, seq_len, embed_dim]
        Returns:
            Output features [batch_size, seq_len, embed_dim]
        """
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for underwater image feature extraction
    
    Adapted from the original ViT architecture for visual odometry tasks
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        representation_size: Optional[int] = None
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size, 
            in_channels=in_channels,
            embed_dim=d_model
        )
        
        num_patches = self.patch_embed.num_patches
        
        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model) * 0.02)
        
        # Class token for global representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=d_model,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Optional representation layer
        if representation_size:
            self.pre_logits = nn.Linear(d_model, representation_size)
            self.pre_logits_act = nn.Tanh()
        else:
            self.pre_logits = None
            
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Vision Transformer
        
        Args:
            x: Input images [batch_size, channels, height, width]
        Returns:
            Global image features [batch_size, embed_dim]
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [batch_size, num_patches, embed_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, embed_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [batch_size, num_patches + 1, embed_dim]
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Layer normalization
        x = self.norm(x)
        
        # Extract class token (global representation)
        cls_token_final = x[:, 0]  # [batch_size, embed_dim]
        
        # Optional pre-logits layer
        if self.pre_logits is not None:
            cls_token_final = self.pre_logits(cls_token_final)
            cls_token_final = self.pre_logits_act(cls_token_final)
            
        return cls_token_final
    
    def get_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get patch-level features (for visualization or analysis)
        
        Args:
            x: Input images [batch_size, channels, height, width]
        Returns:
            Patch features [batch_size, num_patches, embed_dim]
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Layer normalization
        x = self.norm(x)
        
        # Return patch features (excluding class token)
        return x[:, 1:]  # [batch_size, num_patches, embed_dim]


# Factory functions for different model sizes
def vit_tiny_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ViT-Tiny model"""
    model = VisionTransformer(
        patch_size=16, d_model=192, num_layers=12, num_heads=3, **kwargs
    )
    return model


def vit_small_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ViT-Small model"""
    model = VisionTransformer(
        patch_size=16, d_model=384, num_layers=12, num_heads=6, **kwargs
    )
    return model


def vit_base_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ViT-Base model"""
    model = VisionTransformer(
        patch_size=16, d_model=768, num_layers=12, num_heads=12, **kwargs
    )
    return model


def vit_large_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ViT-Large model"""
    model = VisionTransformer(
        patch_size=16, d_model=1024, num_layers=24, num_heads=16, **kwargs
    )
    return model
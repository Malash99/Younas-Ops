"""
Multi-modal sensor fusion module for UW-TransVO

Fuses visual features with IMU and pressure sensor data using cross-modal attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class IMUEncoder(nn.Module):
    """Encode IMU data (accelerometer + gyroscope)"""
    
    def __init__(self, input_dim: int = 6, output_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, imu_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            imu_data: IMU data [batch_size, seq_len, 6] (ax, ay, az, gx, gy, gz)
        Returns:
            Encoded IMU features [batch_size, output_dim]
        """
        # Take the last time step or aggregate across sequence
        if imu_data.dim() == 3:
            imu_data = imu_data[:, -1]  # Use last time step
        
        encoded = self.encoder(imu_data)
        return self.norm(encoded)


class PressureEncoder(nn.Module):
    """Encode pressure/depth sensor data"""
    
    def __init__(self, input_dim: int = 1, output_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 4, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 2, output_dim)
        )
        
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, pressure_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pressure_data: Pressure data [batch_size, seq_len, 1] or [batch_size, 1]
        Returns:
            Encoded pressure features [batch_size, output_dim]
        """
        # Take the last time step or aggregate across sequence
        if pressure_data.dim() == 3:
            pressure_data = pressure_data[:, -1]  # Use last time step
        
        encoded = self.encoder(pressure_data)
        return self.norm(encoded)


class CrossModalAttention(nn.Module):
    """Cross-modal attention between different sensor modalities"""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.multihead_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: Query features [batch_size, 1, d_model]
            key: Key features [batch_size, num_keys, d_model]
            value: Value features [batch_size, num_keys, d_model]
        Returns:
            Attended features [batch_size, 1, d_model]
        """
        attn_output, attn_weights = self.multihead_attn(query, key, value)
        
        # Residual connection and normalization
        output = self.norm(query + self.dropout(attn_output))
        
        return output.squeeze(1)  # [batch_size, d_model]


class ModalityGate(nn.Module):
    """Gating mechanism to control contribution of each modality"""
    
    def __init__(self, d_model: int, num_modalities: int):
        super().__init__()
        
        self.gate = nn.Sequential(
            nn.Linear(d_model * num_modalities, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_modalities),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Concatenated features [batch_size, d_model * num_modalities]
        Returns:
            Gate weights [batch_size, num_modalities]
        """
        return self.gate(features)


class MultiModalFusion(nn.Module):
    """
    Multi-modal fusion layer that combines visual, IMU, and pressure data
    
    Uses cross-modal attention and gating mechanisms to effectively fuse
    different sensor modalities for underwater visual odometry
    """
    
    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 8,
        use_imu: bool = True,
        use_pressure: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.use_imu = use_imu
        self.use_pressure = use_pressure
        
        # Sensor encoders
        if use_imu:
            self.imu_encoder = IMUEncoder(
                input_dim=6, output_dim=d_model, dropout=dropout
            )
            
        if use_pressure:
            self.pressure_encoder = PressureEncoder(
                input_dim=1, output_dim=d_model, dropout=dropout
            )
        
        # Determine number of modalities
        self.num_modalities = 1  # Vision is always present
        if use_imu:
            self.num_modalities += 1
        if use_pressure:
            self.num_modalities += 1
            
        # Cross-modal attention layers
        if use_imu:
            self.vision_imu_attention = CrossModalAttention(d_model, num_heads, dropout)
            
        if use_pressure:
            self.vision_pressure_attention = CrossModalAttention(d_model, num_heads, dropout)
            
        # Modality gating
        if self.num_modalities > 1:
            self.modality_gate = ModalityGate(d_model, self.num_modalities)
            
        # Final fusion layer
        fusion_input_dim = d_model
        if use_imu and use_pressure:
            fusion_input_dim = d_model * 3  # vision + imu + pressure
        elif use_imu or use_pressure:
            fusion_input_dim = d_model * 2  # vision + one sensor
            
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
    def forward(
        self,
        visual_features: torch.Tensor,
        imu_data: Optional[torch.Tensor] = None,
        pressure_data: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of multi-modal fusion
        
        Args:
            visual_features: Visual features [batch_size, d_model]
            imu_data: IMU data [batch_size, seq_len, 6] or None
            pressure_data: Pressure data [batch_size, seq_len, 1] or None
            
        Returns:
            Fused features [batch_size, d_model]
        """
        batch_size = visual_features.shape[0]
        fused_features = [visual_features]
        
        # Process IMU data
        if self.use_imu and imu_data is not None:
            # Encode IMU data
            imu_features = self.imu_encoder(imu_data)  # [batch_size, d_model]
            
            # Cross-modal attention: vision attends to IMU
            visual_query = visual_features.unsqueeze(1)  # [batch_size, 1, d_model]
            imu_kv = imu_features.unsqueeze(1)  # [batch_size, 1, d_model]
            
            vision_imu_fused = self.vision_imu_attention(visual_query, imu_kv, imu_kv)
            fused_features.append(vision_imu_fused)
        
        # Process pressure data
        if self.use_pressure and pressure_data is not None:
            # Encode pressure data
            pressure_features = self.pressure_encoder(pressure_data)  # [batch_size, d_model]
            
            # Cross-modal attention: vision attends to pressure
            visual_query = visual_features.unsqueeze(1)  # [batch_size, 1, d_model]
            pressure_kv = pressure_features.unsqueeze(1)  # [batch_size, 1, d_model]
            
            vision_pressure_fused = self.vision_pressure_attention(visual_query, pressure_kv, pressure_kv)
            fused_features.append(vision_pressure_fused)
        
        # Concatenate all features
        if len(fused_features) > 1:
            concatenated = torch.cat(fused_features, dim=1)  # [batch_size, d_model * num_modalities]
            
            # Apply modality gating if multiple modalities
            if hasattr(self, 'modality_gate'):
                gate_weights = self.modality_gate(concatenated)  # [batch_size, num_modalities]
                
                # Apply gating
                gated_features = []
                for i, feature in enumerate(fused_features):
                    gated_feature = feature * gate_weights[:, i:i+1]
                    gated_features.append(gated_feature)
                
                concatenated = torch.cat(gated_features, dim=1)
        else:
            concatenated = fused_features[0]
        
        # Final fusion
        output = self.fusion_layer(concatenated)
        
        return output
    
    def get_modality_weights(
        self,
        visual_features: torch.Tensor,
        imu_data: Optional[torch.Tensor] = None,
        pressure_data: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get modality importance weights for analysis
        
        Returns:
            Gate weights [batch_size, num_modalities]
        """
        if not hasattr(self, 'modality_gate'):
            return None
            
        # Forward pass to get concatenated features
        with torch.no_grad():
            fused_features = [visual_features]
            
            if self.use_imu and imu_data is not None:
                imu_features = self.imu_encoder(imu_data)
                visual_query = visual_features.unsqueeze(1)
                imu_kv = imu_features.unsqueeze(1)
                vision_imu_fused = self.vision_imu_attention(visual_query, imu_kv, imu_kv)
                fused_features.append(vision_imu_fused)
            
            if self.use_pressure and pressure_data is not None:
                pressure_features = self.pressure_encoder(pressure_data)
                visual_query = visual_features.unsqueeze(1)
                pressure_kv = pressure_features.unsqueeze(1)
                vision_pressure_fused = self.vision_pressure_attention(visual_query, pressure_kv, pressure_kv)
                fused_features.append(vision_pressure_fused)
            
            concatenated = torch.cat(fused_features, dim=1)
            gate_weights = self.modality_gate(concatenated)
            
        return gate_weights
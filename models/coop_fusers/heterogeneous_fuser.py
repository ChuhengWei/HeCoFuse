import torch
from torch import nn
from torch.nn import functional as F
from typing import List

from mmdet3d.models.builder import COOPFUSERS

__all__ = ["HeterogeneousFuser"]


@COOPFUSERS.register_module()
class HeterogeneousFuser(nn.Module):
    """
    HeterogeneousFuser: Feature fusion module for heterogeneous cooperative perception systems.

    Args:
        in_channels (List[int]): List of input feature channel counts, typically [vehicle_channels, infrastructure_channels]
        out_channels (int): Number of output feature channels
        feature_dim (int, optional): Internal feature dimension, defaults to input channel count
        use_attention (bool, optional): Whether to use attention mechanism, defaults to False
        use_feature_completion (bool, optional): Whether to use feature completion, defaults to False
        dropout (float, optional): Dropout ratio, defaults to 0.1
        spatial_downsample (int, optional): Spatial downsampling factor, defaults to 2
    """

    def __init__(
            self,
            in_channels: List[int],
            out_channels: int,
            feature_dim=None,
            use_attention=True,  # Enable attention mechanism by default
            use_feature_completion=False,
            dropout: float = 0.1,
            spatial_downsample: int = 2  # New: spatial downsampling factor
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.use_attention = use_attention
        self.spatial_downsample = spatial_downsample

        # Print initialization info for debugging
        print(f"HeterogeneousFuser initialized with in_channels={in_channels}, out_channels={out_channels}")

        # Check if input channels are the same, this is a simple way to ensure features can be properly fused
        assert len(in_channels) == 2, "Expected exactly 2 input channels (vehicle and infrastructure)"
        assert in_channels[0] == in_channels[1], "Input channels must match for fusion"

        # Fusion channel count - use input channel count
        fusion_channels = in_channels[0]

        # Channel attention mechanism - learn a weight for each channel
        self.channel_weights = nn.Parameter(torch.ones(1, fusion_channels, 1, 1) * 0.5)

        # New: spatial attention module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(fusion_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Normalization layer
        self.norm = nn.GroupNorm(min(32, fusion_channels), fusion_channels)

        # Dropout layer (randomly drop features during training to improve robustness)
        self.dropout_layer = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # If output channels differ from fusion channels, add projection layer
        self.output_proj = None
        if out_channels != fusion_channels and out_channels != 0:
            self.output_proj = nn.Conv2d(fusion_channels, out_channels, kernel_size=1)
            print(f"Created output projection: {fusion_channels} -> {out_channels}")

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward propagation function

        Args:
            inputs (List[torch.Tensor]): List of input features
                                       Typically contains [vehicle_features, infrastructure_features]

        Returns:
            torch.Tensor: Fused features
        """
        # Check if input list is empty
        if not inputs or len(inputs) == 0:
            print("WARNING: No inputs provided to HeterogeneousFuser")
            return None

        # Single input case - return directly
        if len(inputs) == 1:
            output = inputs[0]
            if self.output_proj is not None:
                output = self.output_proj(output)
            return output

        # Get vehicle and infrastructure features
        vehicle_feat = inputs[0]
        roadside_feat = inputs[1]

        # Confirm channel dimensions match
        if vehicle_feat.shape[1] != roadside_feat.shape[1]:
            raise ValueError(f"Channel dimensions don't match: {vehicle_feat.shape[1]} vs {roadside_feat.shape[1]}")

        # Confirm spatial dimensions match
        if vehicle_feat.shape[2:] != roadside_feat.shape[2:]:
            # Adjust to larger feature map size
            target_h = max(vehicle_feat.shape[2], roadside_feat.shape[2])
            target_w = max(vehicle_feat.shape[3], roadside_feat.shape[3])

            if vehicle_feat.shape[2:] != (target_h, target_w):
                vehicle_feat = F.interpolate(
                    vehicle_feat, size=(target_h, target_w), mode='bilinear', align_corners=False)
                print(f"Resized vehicle features to {vehicle_feat.shape}")

            if roadside_feat.shape[2:] != (target_h, target_w):
                roadside_feat = F.interpolate(
                    roadside_feat, size=(target_h, target_w), mode='bilinear', align_corners=False)
                print(f"Resized roadside features to {roadside_feat.shape}")

        # Original feature shape
        original_shape = vehicle_feat.shape[2:]

        
        if self.spatial_downsample > 1 and (original_shape[0] > 32 or original_shape[1] > 32):
            target_h = original_shape[0] // self.spatial_downsample
            target_w = original_shape[1] // self.spatial_downsample

            # Downsampling processing
            vehicle_feat_down = F.adaptive_avg_pool2d(vehicle_feat, (target_h, target_w))
            roadside_feat_down = F.adaptive_avg_pool2d(roadside_feat, (target_h, target_w))

            # print(f"Downsampled features from {original_shape} to {vehicle_feat_down.shape[2:]}")
        else:
            
            vehicle_feat_down = vehicle_feat
            roadside_feat_down = roadside_feat

        # Use sigmoid to ensure weights are between 0 and 1 - channel attention
        alpha = torch.sigmoid(self.channel_weights)

        if self.use_attention:
            # New: compute spatial attention weights
            v_spatial_weights = self.spatial_attention(vehicle_feat_down)
            r_spatial_weights = self.spatial_attention(roadside_feat_down)

            # New: hierarchical fusion - channel attention first, then spatial attention
            # Channel weighting
            vehicle_weighted = vehicle_feat_down * alpha
            roadside_weighted = roadside_feat_down * (1 - alpha)

            # Spatial weighting
            vehicle_attention = vehicle_weighted * v_spatial_weights
            roadside_attention = roadside_weighted * r_spatial_weights

            # Fusion
            fused_feature = vehicle_attention + roadside_attention
        else:
            # Simple weighted combination of two features
            fused_feature = alpha * vehicle_feat_down + (1 - alpha) * roadside_feat_down

        # Apply normalization and activation
        fused_feature = self.norm(fused_feature)
        fused_feature = F.relu(fused_feature)

        # Apply dropout during training
        if self.training and self.dropout > 0:
            fused_feature = self.dropout_layer(fused_feature)

        # if downsampling was applied earlier, now restore original size
        if self.spatial_downsample > 1 and fused_feature.shape[2:] != original_shape:
            fused_feature = F.interpolate(
                fused_feature, size=original_shape, mode='bilinear', align_corners=False)
            # print(f"Upsampled fused features back to {original_shape}")

        # Apply output projection
        if self.output_proj is not None:
            fused_feature = self.output_proj(fused_feature)

        # Check and handle NaN
        if torch.isnan(fused_feature).any():
            print("WARNING: NaN values detected in output, replacing with zeros")
            fused_feature = torch.nan_to_num(fused_feature, nan=0.0)

        return fused_feature

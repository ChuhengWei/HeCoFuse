from typing import Any, Dict, List

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_neck,
    build_coop_fuser,
    build_fusion_model_headless,
    build_head
)
from mmdet3d.models import COOPFUSIONMODELS
from .bevfusion_coop import BEVFusionCoop


class FeatureAdapter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.adapter(x)


@COOPFUSIONMODELS.register_module()
class HeterogeneousBEVFusionCoop(BEVFusionCoop):
    """
    Extended BEVFusionCoop to support heterogeneous sensor configurations
    Can handle various configurations like L+L, C+C, LC+LC, L+C, C+L, LC+L, LC+C, L+LC, C+LC
    """

    def __init__(
            self,
            vehicle: Dict[str, Any],
            infrastructure: Dict[str, Any],
            coop_fuser: Dict[str, Any],
            decoder: Dict[str, Any],
            heads: Dict[str, Any],
            heterogeneous_config: Dict[str, Any] = None,  # New parameter to receive heterogeneous configuration
            **kwargs,
    ) -> None:

        # Handle heterogeneous configuration
        if heterogeneous_config is not None:
            # Get default sensor mode
            default_mode = heterogeneous_config.get("default_mode", "full")
            modes = heterogeneous_config.get("modes", {})

            # If specified mode exists, apply that mode's sensor configuration
            if default_mode in modes:
                mode_config = modes[default_mode]
                # Update vehicle sensor configuration
                if "vehicle" in mode_config:
                    vehicle["use_camera"] = mode_config["vehicle"].get("camera", False)
                    vehicle["use_lidar"] = mode_config["vehicle"].get("lidar", False)
                # Update infrastructure sensor configuration
                if "infrastructure" in mode_config:
                    infrastructure["use_camera"] = mode_config["infrastructure"].get("camera", False)
                    infrastructure["use_lidar"] = mode_config["infrastructure"].get("lidar", False)
                # print(f"DEBUG: Applied heterogeneous mode: {default_mode}")

        # Ensure at least one sensor is enabled
        if not any([vehicle.get("use_camera", False), vehicle.get("use_lidar", False)]):
            print("Warning: No vehicle sensors enabled, setting default to use lidar")
            vehicle["use_lidar"] = True

        if not any([infrastructure.get("use_camera", False), infrastructure.get("use_lidar", False)]):
            print("Warning: No infrastructure sensors enabled, setting default to use camera")
            infrastructure["use_camera"] = True

        # Set sensor configuration before calling parent constructor
        self.vehicle_sensors = []
        self.infra_sensors = []

        if vehicle.get("use_camera", False):
            self.vehicle_sensors.append("camera")
        if vehicle.get("use_lidar", False):
            self.vehicle_sensors.append("lidar")
        if infrastructure.get("use_camera", False):
            self.infra_sensors.append("camera")
        if infrastructure.get("use_lidar", False):
            self.infra_sensors.append("lidar")

        # print(f"DEBUG: Vehicle sensors: {self.vehicle_sensors}")
        # print(f"DEBUG: Infrastructure sensors: {self.infra_sensors}")

        # Call parent constructor after sensor list setup
        super().__init__(vehicle, infrastructure, coop_fuser, decoder, heads, **kwargs)

        # Other initialization (e.g., nodes initialization)
        self.nodes = nn.ModuleDict()

        # Initialize vehicle node
        if self.vehicle_sensors:
            self.nodes["vehicle"] = nn.ModuleDict({
                "fusion_model": build_fusion_model_headless(vehicle["fusion_model"]),
                "feature_adapter": FeatureAdapter(in_channels=64, out_channels=64)
            })

        # Initialize infrastructure node
        if self.infra_sensors:
            self.nodes["infrastructure"] = nn.ModuleDict({
                "fusion_model": build_fusion_model_headless(infrastructure["fusion_model"]),
                "feature_adapter": FeatureAdapter(in_channels=64, out_channels=64)
            })

        # Record sensor configuration for debugging
        self.sensor_config = f"V({''.join([s[0].upper() for s in self.vehicle_sensors])})+" + \
                             f"I({''.join([s[0].upper() for s in self.infra_sensors])})"
        # print(f"DEBUG: Initialized HeterogeneousBEVFusionCoop, sensor config: {self.sensor_config}")

        # Check and record feature dimensions to ensure proper subsequent configuration
        self.feature_channels = None
        if self.coop_fuser is not None:
            self.feature_channels = self.coop_fuser.out_channels
            # print(f"DEBUG: Cooperative fuser output channels: {self.feature_channels}")
        else:
            # Find node output channels
            if "vehicle" in self.nodes and self.nodes["vehicle"] is not None:
                try:
                    self.feature_channels = self.nodes["vehicle"]["fusion_model"].fuser.out_channels
                    # print(f"DEBUG: Vehicle node output channels: {self.feature_channels}")
                except:
                    print("Unable to determine vehicle node output channels")
            elif "infrastructure" in self.nodes and self.nodes["infrastructure"] is not None:
                try:
                    self.feature_channels = self.nodes["infrastructure"]["fusion_model"].fuser.out_channels
                    # print(f"DEBUG: Infrastructure node output channels: {self.feature_channels}")
                except:
                    print("Unable to determine infrastructure node output channels")

        # Record decoder configuration information
        try:
            self.out_channels = decoder["backbone"].get("out_channels", [64, 128, 256])
            # print(f"DEBUG: Decoder configured output channels: {self.out_channels}")

            # Check neck input channels
            neck_in_channels = decoder["neck"].get("in_channels", [64, 128, 256])
            # print(f"DEBUG: Neck expected input channels: {neck_in_channels}")

            # Get detection head configuration
            self.head_in_channels = 0
            for name, head_cfg in heads.items():
                if name == "object" and isinstance(head_cfg, dict):
                    self.head_in_channels = head_cfg.get("in_channels", 384)
                    print(f"DEBUG: Detection head expected input channels: {self.head_in_channels}")

                    # Try to get expected feature map size for the head
                    train_cfg = head_cfg.get("train_cfg", {})
                    if isinstance(train_cfg, dict) and "grid_size" in train_cfg:
                        grid_size = train_cfg["grid_size"]
                        if isinstance(grid_size, list) and len(grid_size) >= 2:
                            expected_h, expected_w = grid_size[0], grid_size[1]
                            out_size_factor = train_cfg.get("out_size_factor", 4)
                            self.expected_feat_h = expected_h // out_size_factor
                            self.expected_feat_w = expected_w // out_size_factor
                            # print(f"DEBUG: Detection head expected feature map size: {self.expected_feat_h}x{self.expected_feat_w}")
        except Exception as e:
            print(f"Error extracting configuration: {e}")

    def get_active_sensors(self):
        """Return current active sensor configuration for debugging"""
        return {
            "vehicle": {
                "camera": "camera" in self.vehicle_sensors,
                "lidar": "lidar" in self.vehicle_sensors
            },
            "infrastructure": {
                "camera": "camera" in self.infra_sensors,
                "lidar": "lidar" in self.infra_sensors
            }
        }

    def ensure_all_params_used(self, outputs):
        """Ensure all parameters participate in computation to prevent unused parameter errors in distributed training"""
        if self.training:
            # Create a very small loss term connected to all parameters
            dummy_loss = 0.0
            for name, param in self.named_parameters():
                if param.requires_grad:
                    dummy_loss = dummy_loss + 0.0 * param.sum()

            # Add this tiny loss to outputs
            if isinstance(outputs, dict):
                outputs['dummy_loss'] = dummy_loss * 0.0  # Coefficient is 0, doesn't affect actual training

        return outputs

    @auto_fp16(apply_to=("vehicle_img", "vehicle_points", "infrastructure_img", "infrastructure_points"))
    def forward_single(
            self,
            vehicle_img,
            vehicle_points,
            vehicle_lidar2camera,
            vehicle_lidar2image,
            vehicle_camera_intrinsics,
            vehicle_camera2lidar,
            vehicle_img_aug_matrix,
            vehicle_lidar_aug_matrix,
            vehicle2infrastructure,
            infrastructure_img,
            infrastructure_points,
            infrastructure_lidar2camera,
            infrastructure_lidar2image,
            infrastructure_camera_intrinsics,
            infrastructure_camera2lidar,
            infrastructure_img_aug_matrix,
            infrastructure_lidar_aug_matrix,
            metas,
            gt_masks_bev=None,
            gt_bboxes_3d=None,
            gt_labels_3d=None,
            **kwargs,
    ):
        # Print debug information
        # print("====== DEBUG INFO ======")
        # print(f"Vehicle sensors: {self.vehicle_sensors}")
        # print(f"Infrastructure sensors: {self.infra_sensors}")

        # Check input data
        # print(f"infrastructure_img is None: {infrastructure_img is None}")
        # if infrastructure_img is not None:
        #     print(f"infrastructure_img shape: {infrastructure_img.shape}")
        #
        # print(f"infrastructure_points is None: {infrastructure_points is None}")
        # if infrastructure_points is not None and isinstance(infrastructure_points, list):
        #     print(f"infrastructure_points length: {len(infrastructure_points)}")
        #     if len(infrastructure_points) > 0:
        #         print(f"First point cloud shape: {infrastructure_points[0].shape}")
        # print("========================")

        features = []
        batch_size = 0
        feature_device = None  # Track feature device

        # Process vehicle node
        vehicle_feature = None
        if "vehicle" in self.nodes and self.nodes["vehicle"] is not None:
            # Check if sensors are available
            v_has_camera = "camera" in self.vehicle_sensors and vehicle_img is not None and vehicle_img.numel() > 0
            v_has_lidar = "lidar" in self.vehicle_sensors and vehicle_points is not None and len(vehicle_points) > 0

            if v_has_camera or v_has_lidar:
                try:
                    # print(f"Vehicle sensor status: camera={v_has_camera}, lidar={v_has_lidar}")

                    # Create empty point cloud list (always create, regardless of usage)
                    batch_size_v = vehicle_img.shape[0] if vehicle_img is not None else 4
                    empty_v_points = []
                    for _ in range(batch_size_v):
                        device = vehicle_img.device if vehicle_img is not None else torch.device('cuda:0')
                        empty_point = torch.zeros((0, 5), device=device, dtype=torch.float32)
                        empty_v_points.append(empty_point)

                    # Use actual point cloud or empty point cloud
                    points_to_use = vehicle_points if v_has_lidar else empty_v_points

                    # Call vehicle fusion model
                    vehicle_feature, bs = self.nodes["vehicle"]["fusion_model"].forward(
                        vehicle_img if v_has_camera else None,
                        points_to_use,  # Use selected point cloud list
                        vehicle_lidar2camera,
                        vehicle_lidar2image,
                        vehicle_camera_intrinsics,
                        vehicle_camera2lidar,
                        vehicle_img_aug_matrix,
                        vehicle_lidar_aug_matrix,
                        vehicle2infrastructure,
                        "vehicle",
                        metas
                    )

                    # Check if features are valid
                    if vehicle_feature is not None:
                        # print(
                        #     f"Vehicle feature shape: {vehicle_feature.shape}, max: {vehicle_feature.max().item()}, min: {vehicle_feature.min().item()}")
                        feature_device = vehicle_feature.device
                        batch_size = bs
                    else:
                        print("Warning: Vehicle feature is None despite sensors being available")
                except Exception as e:
                    import traceback
                    print(f"Error processing vehicle node: {e}")
                    traceback.print_exc()
                    vehicle_feature = None

        # Process infrastructure node
        infra_feature = None
        if "infrastructure" in self.nodes and self.nodes["infrastructure"] is not None:
            # print("Starting infrastructure node processing")
            # Check if sensors are available
            i_has_camera = ("camera" in self.infra_sensors and
                            infrastructure_img is not None and
                            isinstance(infrastructure_img, torch.Tensor) and
                            infrastructure_img.numel() > 0)

            i_has_lidar = False
            if "lidar" in self.infra_sensors and infrastructure_points is not None:
                if isinstance(infrastructure_points, list):
                    if len(infrastructure_points) > 0:
                        i_has_lidar = True

            # print(f"i_has_camera: {i_has_camera}, i_has_lidar: {i_has_lidar}")

            if i_has_camera or i_has_lidar:
                try:
                    # print(f"Infrastructure sensor status: camera={i_has_camera}, lidar={i_has_lidar}")

                    # Create empty point cloud list (always create to ensure valid point cloud list)
                    batch_size_i = infrastructure_img.shape[0] if infrastructure_img is not None else 4
                    empty_i_points = []
                    for _ in range(batch_size_i):
                        device = infrastructure_img.device if infrastructure_img is not None else torch.device('cuda:0')
                        empty_point = torch.zeros((0, 5), device=device, dtype=torch.float32)
                        empty_i_points.append(empty_point)

                    # Use actual point cloud or empty point cloud
                    points_to_use = infrastructure_points if i_has_lidar else empty_i_points

                    # Call infrastructure fusion model
                    infra_feature, bs = self.nodes["infrastructure"]["fusion_model"].forward(
                        infrastructure_img if i_has_camera else None,
                        points_to_use,  # Always use valid point cloud list
                        infrastructure_lidar2camera,
                        infrastructure_lidar2image,
                        infrastructure_camera_intrinsics,
                        infrastructure_camera2lidar,
                        infrastructure_img_aug_matrix,
                        infrastructure_lidar_aug_matrix,
                        vehicle2infrastructure,
                        "infrastructure",
                        metas
                    )

                    # Check if features are valid
                    if infra_feature is not None:
                        # print(
                        #     f"Infrastructure feature shape: {infra_feature.shape}, max: {infra_feature.max().item()}, min: {infra_feature.min().item()}")
                        if feature_device is None:
                            feature_device = infra_feature.device
                        if batch_size == 0:
                            batch_size = bs
                    else:
                        print("Warning: Infrastructure feature is None despite sensors being available")
                except Exception as e:
                    import traceback
                    print(f"Error processing infrastructure node: {e}")
                    traceback.print_exc()
                    infra_feature = None

        # Collect valid features
        # Vehicle node feature processing
        if vehicle_feature is not None:
            # print(f"Processing vehicle feature with shape: {vehicle_feature.shape}")
            if len(self.vehicle_sensors) == 1:
                vehicle_feature = self.nodes["vehicle"]["feature_adapter"](vehicle_feature)
                # print(f"Vehicle feature shape after adaptation: {vehicle_feature.shape}")
            features.append(vehicle_feature)
            # print(f"Added vehicle feature to fusion list, current feature count: {len(features)}")
        else:
            print("Skipping vehicle feature because it is None")

        # Infrastructure node feature processing
        if infra_feature is not None:
            # print(f"Processing infrastructure feature with shape: {infra_feature.shape}")
            if len(self.infra_sensors) == 1:
                # print("Starting feature adaptation - infra_feature-1", infra_feature.shape)
                try:
                    infra_feature = self.nodes["infrastructure"]["feature_adapter"](infra_feature)
                    # print("After feature adaptation - infra_feature-2", infra_feature.shape)
                except Exception as e:
                    print(f"Infrastructure feature adapter error: {e}")
                    # If adapter fails, use original features

            features.append(infra_feature)
            # print(f"Added infrastructure feature to fusion list, current feature count: {len(features)}")
        else:
            print("Skipping infrastructure feature because it is None")

        # Determine device for subsequent tasks
        if feature_device is None:
            # Find an available device
            if vehicle_img is not None:
                feature_device = vehicle_img.device
            elif infrastructure_img is not None:
                feature_device = infrastructure_img.device
            elif vehicle_points is not None and len(vehicle_points) > 0:
                feature_device = vehicle_points[0].device
            elif infrastructure_points is not None and len(infrastructure_points) > 0:
                feature_device = infrastructure_points[0].device
            else:
                # Default to cuda:0
                feature_device = torch.device('cuda:0')
            # print(f"Using device: {feature_device}")

        # Use coop_fuser to fuse features
        if self.coop_fuser is not None and len(features) > 0:
            try:
                # print(f"Using {self.__class__.__name__} to fuse {len(features)} features")
                x = self.coop_fuser(features)
                # print(f"Fused feature shape: {x.shape}, max: {x.max().item()}, min: {x.min().item()}")
            except Exception as e:
                print(f"Cooperative fuser error: {e}")
                # Fallback to simple average fusion
                print("Falling back to simple average fusion")
                if len(features) > 1:
                    x = torch.stack(features).mean(dim=0)
                else:
                    x = features[0]
                print(f"Average fusion feature shape: {x.shape}")
        elif len(features) == 1:
            x = features[0]
            # print(f"Only one feature, using directly, shape: {x.shape}")
        else:
            # Create zero feature map
            print(f"Warning: No valid features detected from any node")

            # Ensure batch_size has a reasonable value
            if batch_size == 0:
                # Try to infer batch_size from metas
                if isinstance(metas, list):
                    batch_size = len(metas)
                else:
                    # Set a default value, usually 1
                    batch_size = 1
                # print(f"Using inferred batch size: {batch_size}")

            # Use known feature channel count, or default value
            channels = self.feature_channels if self.feature_channels is not None else 64
            # Create standard size BEV feature map
            x = torch.zeros((batch_size, channels, 200, 200), device=feature_device)
            print(f"Created zero feature map with shape: {x.shape}")

        # Add feature check to prevent NaN
        if torch.isnan(x).any():
            print("Warning: NaN values detected in fused features")
            # Replace NaNs with 0
            x = torch.nan_to_num(x, nan=0.0)
            print("Replaced NaN values with 0")

        # Continue processing
        try:
            # print(f"Decoder input shape: {x.shape}")

            # Process backbone
            backbone_out = self.decoder["backbone"](x)

            # Handle case where SECOND returns tuple or list
            multi_scale_features = []
            if isinstance(backbone_out, (tuple, list)):
                multi_scale_features = list(backbone_out)
                # print(f"Using backbone output directly: {[f.shape for f in multi_scale_features]}")
            else:
                # print(f"Shape after backbone: {backbone_out.shape}")
                multi_scale_features.append(backbone_out)

            # Process neck
            neck_out = self.decoder["neck"](multi_scale_features)
            shape_info = neck_out.shape if not isinstance(neck_out, (tuple, list)) else [f.shape for f in neck_out]
            # print(f"Shape after neck: {shape_info}")

            # If neck outputs multiple features, modify them
            if isinstance(neck_out, (tuple, list)):
                # Upsample all features to the same size
                x = neck_out[0]  # Use first feature, usually the highest resolution

                # Ensure feature size matches what Transformer decoder expects
                expected_size = (128, 128)  # Default value

                # Try to get expected feature map size from configuration
                if hasattr(self, 'expected_feat_h') and hasattr(self, 'expected_feat_w'):
                    expected_size = (self.expected_feat_h, self.expected_feat_w)

                if x.shape[2:] != expected_size:
                    # print(f"Adjusting feature size from {x.shape[2:]} to {expected_size}")
                    x = F.interpolate(x, size=expected_size, mode='bilinear', align_corners=False)
                    # print(f"Adjusted feature shape: {x.shape}")
            else:
                x = neck_out

                # Check feature map size
                expected_size = (128, 128)
                if hasattr(self, 'expected_feat_h') and hasattr(self, 'expected_feat_w'):
                    expected_size = (self.expected_feat_h, self.expected_feat_w)

                if x.shape[2:] != expected_size:
                    # print(f"Adjusting feature size from {x.shape[2:]} to {expected_size}")
                    x = F.interpolate(x, size=expected_size, mode='bilinear', align_corners=False)
                    # print(f"Adjusted feature shape: {x.shape}")

            # Check if channel count matches
            if hasattr(self, 'head_in_channels') and x.shape[1] != self.head_in_channels:
                print(f"Warning: Feature channel count ({x.shape[1]}) doesn't match head expected channels ({self.head_in_channels})")

                # Add channel adjustment if needed
                if x.shape[1] < self.head_in_channels:
                    # Increase channel count - create temporary convolution layer
                    channel_adjust = nn.Conv2d(x.shape[1], self.head_in_channels, kernel_size=1).to(x.device)
                    x = channel_adjust(x)
                    # print(f"Adjusted channel count to match head: {x.shape}")

        except Exception as e:
            import traceback
            print(f"Decoder processing error: {e}")
            traceback.print_exc()
            raise e

        
        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"Unsupported head type: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            # Ensure all parameters participate in computation
            outputs = self.ensure_all_params_used(outputs)
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"Unsupported head type: {type}")
            return outputs

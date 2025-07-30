from typing import Any, Dict, List

import torch
from mmcv.runner import auto_fp16, force_fp32, BaseModule
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS

__all__ = ["PseudoFusion"]


@FUSIONMODELS.register_module()
class PseudoFusion(BaseModule):
    """
    PseudoFusion: Simplified fusion module designed for single-sensor scenarios
    When a node has only one type of sensor (camera or lidar), it doesn't perform 
    real fusion but directly processes single sensor data
    """

    def __init__(
            self,
            encoders: Dict[str, Any] = None,
            fuser: Dict[str, Any] = None,
            output_channels: int = 64,
            **kwargs,
    ) -> None:
        super().__init__()

        # Set default output channels
        self.output_channels = output_channels

        # Initialize empty encoder dictionary
        self.encoders = nn.ModuleDict()

        # Check if encoder configuration exists
        if encoders is not None:
            # Add lidar encoder (if provided)
            if encoders.get("lidar") is not None:
                if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                    print("PseudoFusion: Using Voxelization")
                    voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
                else:
                    print("PseudoFusion: Using DynamicScatter")
                    voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])

                self.encoders["lidar"] = nn.ModuleDict(
                    {
                        "voxelize": voxelize_module,
                        "backbone": build_backbone(encoders["lidar"]["backbone"]),
                    }
                )
                self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

            # Add camera encoder (if provided)
            if encoders.get("camera") is not None:
                self.encoders["camera"] = nn.ModuleDict(
                    {
                        "backbone": build_backbone(encoders["camera"]["backbone"]),
                        "neck": build_neck(encoders["camera"]["neck"]),
                        "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                    }
                )

        # Detect which sensors are available in the module
        available_sensors = list(self.encoders.keys())
        self.single_sensor_mode = len(available_sensors) == 1
        if self.single_sensor_mode:
            self.sensor_type = available_sensors[0] if available_sensors else None
            print(f"PseudoFusion: Single sensor mode, using {self.sensor_type}")
        else:
            print(f"PseudoFusion: Multi-sensor mode, available sensors: {available_sensors}")

        # Adapters will be created dynamically when needed
        self.adapters = nn.ModuleDict()

    def init_weights(self) -> None:
        """Initialize weights"""
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
            self,
            x,
            points,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            vehicle2infrastructure,
            img_metas,
    ) -> torch.Tensor:
        """Extract camera features"""
        # If input is None, return None
        if x is None:
            print("PseudoFusion: Camera data is None, skipping camera feature extraction")
            return None

        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        # Handle case when points is None
        if points is None or len(points) == 0:
            # Create empty point cloud list to avoid errors
            # print("PseudoFusion: Point cloud data is empty, creating empty point cloud")
            empty_points = []
            for _ in range(B):
                empty_point = torch.zeros((0, 5), device=x.device, dtype=torch.float32)
                empty_points.append(empty_point)
            points = empty_points

        x = self.encoders["camera"]["vtransform"](
            self.training,
            x,
            points,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            vehicle2infrastructure,
            img_metas,
        )
        # print(f"PseudoFusion: Camera feature extraction completed, shape={x.shape}, dtype={x.dtype}")
        return x

    def extract_lidar_features(self, x) -> torch.Tensor:
        """Extract lidar features"""
        # If input is None or empty list, return None
        if x is None or not isinstance(x, list) or len(x) == 0:
            print("PseudoFusion: Lidar data is invalid, skipping lidar feature extraction")
            return None

        # Confirm each point cloud has data
        for i, point_cloud in enumerate(x):
            if point_cloud.shape[0] == 0:
                print(f"PseudoFusion: Warning - Point cloud {i} is empty")

        try:
            feats, coords, sizes = self.voxelize(x)

            # Check if coordinates are empty
            if coords.shape[0] == 0:
                print("PseudoFusion: Coordinates are empty after voxelization, cannot extract features")
                return None

            batch_size = coords[-1, 0] + 1
            x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
            # print(f"PseudoFusion: Lidar feature extraction completed, shape={x.shape}, dtype={x.dtype}")
            return x
        except Exception as e:
            print(f"PseudoFusion: Lidar feature extraction error - {e}")
            return None

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Voxelize point clouds"""
        feats, coords, sizes = [], [], []

        for k, res in enumerate(points):
            # Skip empty point clouds
            if res.shape[0] == 0:
                print(f"PseudoFusion: Skipping empty point cloud {k}")
                continue

            try:
                ret = self.encoders["lidar"]["voxelize"](res.to(torch.float32))
                if len(ret) == 3:
                    # Hard voxelization
                    f, c, n = ret
                else:
                    assert len(ret) == 2
                    f, c = ret
                    n = None
                feats.append(f)
                coords.append(F.pad(c, (1, 0), mode="constant", value=k))
                if n is not None:
                    sizes.append(n)
            except Exception as e:
                print(f"PseudoFusion: Error voxelizing point cloud {k} - {e}")

        # Check if there are valid features
        if len(feats) == 0:
            print("PseudoFusion: No valid voxel features")
            # Return empty tensors for subsequent processing
            device = points[0].device if len(points) > 0 else torch.device('cuda:0')
            return torch.zeros(0, 5, device=device), torch.zeros(0, 4, device=device), torch.zeros(0, device=device)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    def create_feature_adapter(self, in_channels, out_channels, dtype=None, device=None):
        """Dynamically create feature adapter, ensuring it matches input data type"""
        print(f"PseudoFusion: Creating feature adapter {in_channels} -> {out_channels}, dtype={dtype}")
        adapter = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Move adapter to correct device and data type
        if device is not None:
            adapter = adapter.to(device)
        if dtype is not None:
            adapter = adapter.to(dtype)

        return adapter

    def create_fallback_feature(self, batch_size, device, dtype=None):
        """Create a fallback feature for handling cases where all sensors are missing"""
        print(f"PseudoFusion: Creating fallback feature, shape=[{batch_size}, {self.output_channels}, 200, 200], dtype={dtype}")
        tensor = torch.zeros((batch_size, self.output_channels, 200, 200), device=device)
        if dtype is not None:
            tensor = tensor.to(dtype)
        return tensor

    @auto_fp16(apply_to=("img", "points", "vehicle2infrastructure"))
    def forward(
            self,
            img,
            points,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            vehicle2infrastructure,
            node,
            metas,
            gt_masks_bev=None,
            gt_bboxes_3d=None,
            gt_labels_3d=None,
            **kwargs,
    ):
        """Forward propagation function"""
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs, batch_size = self.forward_single(
                img,
                points,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                vehicle2infrastructure,
                node,
                metas,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs, batch_size

    @auto_fp16(apply_to=("img", "points", "vehicle2infrastructure"))
    def forward_single(
            self,
            img,
            points,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            vehicle2infrastructure,
            node,
            metas,
            gt_masks_bev=None,
            gt_bboxes_3d=None,
            gt_labels_3d=None,
            **kwargs,
    ):
        """Single forward pass"""
        # print(f"PseudoFusion: Processing node {node}")

        # Determine batch size
        if img is not None:
            batch_size = img.shape[0]
            device = img.device
            dtype = img.dtype
        elif points is not None and isinstance(points, list) and len(points) > 0:
            batch_size = len(points)
            device = points[0].device
            dtype = points[0].dtype
        else:
            batch_size = 1
            device = torch.device('cuda:0')
            dtype = torch.float32

        # Check input data validity
        has_camera = img is not None and img.numel() > 0
        has_lidar = points is not None and isinstance(points, list) and len(points) > 0 and any(
            p.shape[0] > 0 for p in points)

        # print(f"PseudoFusion: Data status - camera={has_camera}, lidar={has_lidar}")

        # Convert points to correct type (if valid)
        if has_lidar:
            vehicle2infrastructure = vehicle2infrastructure.to(torch.float32)
            for b in range(batch_size):
                points[b] = points[b].to(torch.float32)

        # Extract features
        camera_feature = None
        lidar_feature = None

        # Check which sensor types to actually process
        process_camera = has_camera and "camera" in self.encoders
        process_lidar = has_lidar and "lidar" in self.encoders

        # print(f"PseudoFusion: Processing sensors - camera={process_camera}, lidar={process_lidar}")

        # If in single sensor mode, only process one type of sensor
        if self.single_sensor_mode:
            if self.sensor_type == "camera" and process_camera:
                # print("PseudoFusion: Single sensor mode - extracting camera features")
                camera_feature = self.extract_camera_features(
                    img,
                    points if has_lidar else None,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    vehicle2infrastructure,
                    metas,
                )
            elif self.sensor_type == "lidar" and process_lidar:
                # print("PseudoFusion: Single sensor mode - extracting lidar features")
                lidar_feature = self.extract_lidar_features(points)
        else:
            # Multi-sensor mode
            if process_camera:
                # print("PseudoFusion: Multi-sensor mode - extracting camera features")
                camera_feature = self.extract_camera_features(
                    img,
                    points if has_lidar else None,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    vehicle2infrastructure,
                    metas,
                )

            if process_lidar:
                # print("PseudoFusion: Multi-sensor mode - extracting lidar features")
                lidar_feature = self.extract_lidar_features(points)

        # Select final output features
        if camera_feature is not None and lidar_feature is not None:
            # Both features exist, need to handle channel matching first
            # print(f"PseudoFusion: Camera feature shape={camera_feature.shape}, lidar feature shape={lidar_feature.shape}")

            # Get feature data type and device
            feature_device = camera_feature.device
            feature_dtype = camera_feature.dtype
            # print(f"PseudoFusion: Feature dtype={feature_dtype}, device={feature_device}")

            # Dynamically create adapters
            cam_channels = camera_feature.shape[1]
            lidar_channels = lidar_feature.shape[1]

            if cam_channels != self.output_channels:
                adapter_key = f"camera_{cam_channels}_{self.output_channels}"
                if adapter_key not in self.adapters:
                    self.adapters[adapter_key] = self.create_feature_adapter(
                        cam_channels, self.output_channels, dtype=feature_dtype, device=feature_device
                    )
                # Ensure adapter data type matches input
                self.adapters[adapter_key] = self.adapters[adapter_key].to(feature_dtype)
                camera_feature = self.adapters[adapter_key](camera_feature)

            if lidar_channels != self.output_channels:
                adapter_key = f"lidar_{lidar_channels}_{self.output_channels}"
                if adapter_key not in self.adapters:
                    self.adapters[adapter_key] = self.create_feature_adapter(
                        lidar_channels, self.output_channels, dtype=feature_dtype, device=feature_device
                    )
                # Ensure adapter data type matches input
                self.adapters[adapter_key] = self.adapters[adapter_key].to(feature_dtype)
                lidar_feature = self.adapters[adapter_key](lidar_feature)

            # Feature fusion
            # print("PseudoFusion: Fusing camera and lidar features")
            x = camera_feature + lidar_feature

        elif camera_feature is not None:
            # Only camera features
            # print(f"PseudoFusion: Using only camera features, shape={camera_feature.shape}")

            # Get feature data type and device
            feature_device = camera_feature.device
            feature_dtype = camera_feature.dtype
            # print(f"PseudoFusion: Feature dtype={feature_dtype}, device={feature_device}")

            cam_channels = camera_feature.shape[1]

            if cam_channels != self.output_channels:
                adapter_key = f"camera_{cam_channels}_{self.output_channels}"
                if adapter_key not in self.adapters:
                    self.adapters[adapter_key] = self.create_feature_adapter(
                        cam_channels, self.output_channels, dtype=feature_dtype, device=feature_device
                    )
                # Ensure adapter data type matches input
                self.adapters[adapter_key] = self.adapters[adapter_key].to(feature_dtype)
                camera_feature = self.adapters[adapter_key](camera_feature)

            x = camera_feature

        elif lidar_feature is not None:
            # Only lidar features
            # print(f"PseudoFusion: Using only lidar features, shape={lidar_feature.shape}")

            # Get feature data type and device
            feature_device = lidar_feature.device
            feature_dtype = lidar_feature.dtype
            # print(f"PseudoFusion: Feature dtype={feature_dtype}, device={feature_device}")

            lidar_channels = lidar_feature.shape[1]

            if lidar_channels != self.output_channels:
                adapter_key = f"lidar_{lidar_channels}_{self.output_channels}"
                if adapter_key not in self.adapters:
                    self.adapters[adapter_key] = self.create_feature_adapter(
                        lidar_channels, self.output_channels, dtype=feature_dtype, device=feature_device
                    )
                # Ensure adapter data type matches input
                self.adapters[adapter_key] = self.adapters[adapter_key].to(feature_dtype)
                lidar_feature = self.adapters[adapter_key](lidar_feature)

            x = lidar_feature

        else:
            # Neither feature exists, create a zero feature
            print("PseudoFusion: No valid features, creating fallback feature")
            # Try to determine data type
            if has_camera:
                dtype = img.dtype
            elif has_lidar and len(points) > 0:
                dtype = points[0].dtype
            else:
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            x = self.create_fallback_feature(batch_size, device, dtype=dtype)

        # Get output batch size
        batch_size = x.shape[0]
        # print(f"PseudoFusion: Output feature shape={x.shape}, dtype={x.dtype}")

        return x, batch_size

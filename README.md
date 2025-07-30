# HeCoFuse: Cross-Modal Complementary V2X Cooperative Perception with Heterogeneous Sensors

![CVPR 2025 DriveX Champion](https://img.shields.io/badge/CVPR2025-DriveX%20Champion-blue)

Official implementation of the CVPR 2025 paper:  
**HeCoFuse: Cross-Modal Complementary V2X Cooperative Perception with Heterogeneous Sensors**

> Chuheng Wei\*, Ziye Qin, Walter Zimmer, Guoyuan Wu, and Matthew J. Barth  
> \*Corresponding author: chuheng.wei@email.ucr.edu  
> 🏆 1st Place, **CVPR 2025 DriveX Challenge**

---

## 🧠 Introduction

HeCoFuse is a unified cooperative perception framework designed for real-world **heterogeneous V2X systems**, where vehicles and infrastructure may be equipped with different combinations of sensors (LiDAR-only, Camera-only, or both). Unlike previous methods assuming uniform sensor setups, HeCoFuse:

- Supports **9 heterogeneous sensor configurations** across vehicle–infrastructure nodes
- Introduces **Hierarchical Attention Fusion (HAF)** to adaptively weight features by modality and spatial reliability
- Proposes **Adaptive Spatial Resolution (ASR)** to dynamically adjust feature resolution, reducing computation by up to 45%
- Uses a **cooperative training strategy** to generalize across diverse configurations

> 📈 Achieved **43.37% 3D mAP** under L+LC setting and **43.22%** in LC+LC, surpassing CoopDet3D baseline.

---

## 🛠️ Installation

### Prerequisites

Please first install CoopDet3D following the official instructions:

1. **Install CoopDet3D**: Follow the installation guide at [CoopDet3D repository](https://github.com/tum-traffic-dataset/coopdet3d)
2. **Setup Dataset**: Download and prepare the TUMTraf-V2X dataset according to CoopDet3D requirements

### HeCoFuse Installation

After successfully setting up CoopDet3D, install HeCoFuse components:

```bash
# Clone HeCoFuse repository
git clone https://github.com/your-repo/hecofuse.git
cd hecofuse
```

---

## 📊 Dataset: TUMTraf-V2X

We use the [TUMTraf-V2X dataset](https://innovation-mobility.com/en/project-providentia/a9-dataset/#anchor_release_4), a real-world V2X dataset featuring:

- Synchronized vehicle and infrastructure data
- Multi-modal sensing (LiDAR & camera)
- 29K annotated 3D bounding boxes
- 8 object classes

```

**Note**: Please follow the [CoopDet3D installation guide](https://github.com/tum-traffic-dataset/coopdet3d) to properly install and setup the dataset.

---

## 🚀 Usage

### 1. Configuration Setup

Add the HeCoFuse configuration file to the CoopDet3D configs directory:

```bash
# Copy HeCoFuse.yaml to the appropriate config directory
cp HeCoFuse.yaml /path/to/coopdet3d/configs/tumtraf_v2x/det/transfusion/secfpn/cooperative/camera+lidar/yolov8/
```

### 2. Model Integration

Install HeCoFuse model components by copying the Python files to the corresponding MMDetection3D model subdirectories:

```bash
# Navigate to your CoopDet3D installation directory
cd /path/to/coopdet3d

# Copy model files to appropriate subdirectories under /mmdet3d/models/
# For example:
cp /path/to/hecofuse/models/fusion_models/*.py mmdet3d/models/coop_fusers/
cp /path/to/hecofuse/models/cooperative/*.py mmdet3d/models/coop_fusion_models/
cp /path/to/hecofuse/models/utils/*.py mmdet3d/models/fusion_models/

# Update __init__.py files in each subdirectory to register new modules
# Replace the existing __init__.py files with the updated versions provided
```

### 3. Directory Structure

After installation, your directory structure should look like:

```
/path/to/coopdet3d/
├── configs/
│   └── tumtraf_v2x/det/transfusion/secfpn/cooperative/camera+lidar/yolov8/
│       └── HeCoFuse.yaml
├── mmdet3d/
│   └── models/
│       ├── coop_fusers/
│       │   ├── __init__.py (updated)
│       │   └── heterogeneous_fuser.py
│       ├── coop_fusion_models/
│       │   ├── __init__.py (updated)
│       │   └── heterogeneous_bevfusion_coop.py
│       └── fusion_models/
│           ├── __init__.py (updated)
│           └── pseudo_fusion.py
```

### 4. Training

```bash
# Train HeCoFuse model
python tools/train.py configs/tumtraf_v2x/det/transfusion/secfpn/cooperative/camera+lidar/yolov8/HeCoFuse.yaml
```

### 5. Evaluation

```bash
# Evaluate trained model
python tools/test.py configs/tumtraf_v2x/det/transfusion/secfpn/cooperative/camera+lidar/yolov8/HeCoFuse.yaml /path/to/checkpoint.pth
```

---

## 🏗️ Architecture

### Hierarchical Attention Fusion (HAF)
- **Cross-modal attention** between LiDAR and camera features
- **Spatial reliability weighting** based on feature confidence
- **Adaptive feature selection** for optimal fusion

### Adaptive Spatial Resolution (ASR)
- **Dynamic resolution adjustment** based on scene complexity
- **Computational efficiency** with up to 45% reduction
- **Quality preservation** through intelligent downsampling

### Heterogeneous Configuration Support
Supports 9 different sensor configurations:
- **Vehicle**: L, C, LC
- **Infrastructure**: L, C, LC
- **Combinations**: L+L, L+C, L+LC, C+C, C+L, C+LC, LC+L, LC+C, LC+LC

---

## 📈 Results



---

## 📝 Citation

If you find HeCoFuse useful in your research, please consider citing:

```bibtex
@inproceedings{wei2025hecofuse,
  title={HeCoFuse: Cross-Modal Complementary V2X Cooperative Perception with Heterogeneous Sensors},
  author={Wei, Chuheng and Qin, Ziye and Zimmer, Walter and Wu, Guoyuan and Barth, Matthew J},
   booktitle={2025 IEEE 28th international conference on intelligent transportation systems (ITSC)},
  year={2025}
}
```

---

## 🤝 Acknowledgments

This work builds upon [CoopDet3D](https://github.com/tum-traffic-dataset/coopdet3d). We thank the authors for their excellent codebase and dataset.

---

## 📧 Contact

For questions and support, please contact:
- **Chuheng Wei**: chuheng.wei@email.ucr.edu

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

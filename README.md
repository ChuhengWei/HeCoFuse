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

## 📊 Dataset: TUMTraf-V2X

We use the [TUMTraf-V2X dataset](https://innovation-mobility.com/en/project-providentia/a9-dataset/#anchor_release_4), a real-world V2X dataset featuring:

- Synchronized vehicle and infrastructure data
- Multi-modal sensing (LiDAR & camera)
- 29K annotated 3D bounding boxes
- 8 object classes

To prepare the dataset:

```bash
# Download and organize dataset as follows
data/tumtraf/
    ├── lidar/
    ├── camera/
    └── annotations/

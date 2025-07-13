# HeCoFuse: Cross-Modal Complementary V2X Cooperative Perception with Heterogeneous Sensors

![CVPR 2025 DriveX Champion](https://img.shields.io/badge/CVPR2025-DriveX%20Champion-blue)

Official implementation of our CVPR 2025 paper:

> **HeCoFuse: Cross-Modal Complementary V2X Cooperative Perception with Heterogeneous Sensors**  
> Chuheng Wei*, Asta Chen*, Mathew Zhang, Edison Liu, Ziyan Wang, Matthew J. Barth  
> *Equal contribution

🏆 **Winner of the CVPR 2025 DriveX Challenge**  
📄 [Paper Coming Soon]  
📊 Based on [CoopDet3D](https://github.com/tum-traffic-dataset/coopdet3d)  
🗂 Supports [TUMTraf-V2X Dataset](https://innovation-mobility.com/en/project-providentia/a9-dataset/#anchor_release_4)

---

## 🔍 Overview

HeCoFuse introduces a **heterogeneous cooperative fusion framework** that enables perception across **multiple sensor modalities** (e.g., Lidar, Camera) and **multiple vehicles** in a V2X environment. Designed to tackle **real-world heterogeneity in sensor configurations**, our method leverages:

- **Cross-modal feature complementarity**
- **Hierarchical fusion at both local and cooperative levels**
- **Flexible adaptation to varying sensor types and communication conditions**

<div align="center">
  <img src="docs/framework_overview.png" width="600"/>
  <p><i>HeCoFuse supports diverse sensor combinations for robust V2X perception</i></p>
</div>

---

## 🗂 Dataset: TUMTraf-V2X

We use the [TUMTraf-V2X dataset](https://innovation-mobility.com/en/project-providentia/a9-dataset/#anchor_release_4), which provides synchronized multi-vehicle, multi-sensor data recorded in highway scenarios.  
To prepare the dataset:

```bash
# Create symlinks or copy TUMTraf-V2X data to:
data/tumtraf/

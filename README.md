# LiDAR Super-Resolution & Knowledge Distillation for Autonomous Driving

## 📌 Project Overview
This project implements a **Deep Learning-based LiDAR Super-Resolution (SR)** framework to enhance sparse 8-channel LiDAR data into high-density 128-channel representations. By integrating **Knowledge Distillation (KD)** and **Semantic Segmentation**, the system significantly improves object recognition performance in low-cost sensor environments.

* **Role:** Independent Researcher (Data Collection, Model Design, Evaluation)
* **Environment:** CARLA Simulator 0.9.16, PyTorch, YOLOv11, TensorRT
* **Key Achievement:** Reduced MAE by **68.4%** and improved Structural Similarity (SSIM) by **41.2%**.

---

## 📊 Interactive Performance Report
> **[Click Here to View the Interactive Dashboard]**
> (https://soyoungkim0327.github.io/lidar-research/)

---

## 🎥 Simulation Demo Videos
Here are the demonstration videos of the data collection and visualization system (located in `simulation/`):

| **1. Autopilot Mode** | **2. 1st Person View** | **3. Full Integration (North-Up)** |
| :---: | :---: | :---: |
| [**[▶️ Watch Demo 1]**](https://youtu.be/OOXVz8CxH-o) | [**[▶️ Watch Demo 2]**](https://youtu.be/zsbFbVcQ5rs) | [**[▶️ Watch Demo 3]**](https://youtu.be/gyIYqvA1HMU) |
| *Synced LiDAR & YOLO* | *Driver's Perspective* | *2D/3D Aligned Visualization* |

---

## 🛠️ Key Features

### 1. Advanced LiDAR Super-Resolution
* Developed `TinyRangeSR`, a lightweight residual CNN to upsample 8ch inputs to 128ch.
* Implemented **Spatial Feature Extraction** using `Conv2d` layers and bilinear interpolation.

### 2. Teacher-Student Knowledge Distillation
* **Teacher:** 128-channel High-Fidelity LiDAR (Semantic Mask Ground Truth).
* **Student:** 8-channel Sparse LiDAR.
* **Method:** Transferred geometric and semantic knowledge from Teacher to Student, enabling the 8ch sensor to recognize objects with "128ch-level intelligence."

### 3. Asynchronous Simulation Engine
* Built a robust data collection pipeline integrating **CARLA Simulator** and **YOLOv11**.
* Utilized **Multi-threading** and **Queue Management** to decouple sensor rendering from model inference, ensuring smooth simulation at high FPS.
* **North-Up Alignment:** Solved coordinate transformation challenges to align 3D LiDAR points with 2D BEV maps.

---

## 📂 Repository Structure

```text
├── src/               # Core implementation of SR and Recognition models
│   ├── recognition_step*.py   # Step-by-step training pipeline
│   └── recognition_lib...     # Model architecture & Dataloaders
├── simulation/        # CARLA data collection engine with async YOLO
│   ├── 1.demo_sync_autopilot...py   # (Demo 1 Code)
│   ├── 1.demo_sync_visual...py      # (Demo 2 Code)
│   └── 2.demo_sync_full_visual...py # (Demo 3 Code - Final)
├── utils/             # Visualization tools (3D Point Cloud & 2D Range Image)
└── docs/              # Installation guide & Research notes

# LiDAR Super-Resolution & Teacher–Student Distillation (Range-Image Pipeline)

## Project Overview
This repository explores a **LiDAR range-image super-resolution (SR)** workflow that upsamples sparse **8-channel** inputs toward **128-channel**-like representations, and evaluates SR outputs alongside a **teacher–student distillation** setup for downstream range-image segmentation.

**Environment:** CARLA Simulator 0.9.16, PyTorch, YOLOv11, TensorRT

> **Validation note (important):**  
> The public dashboard and the included HTML report summarize results from a specific run and a specific metric definition.  
> Some scores (especially “similarity” in long-range / sparse bins) can look overly optimistic depending on how empties are treated and how normalization is defined.  
> This repo therefore includes a verification script to **recompute metrics from raw NPY pairs** and to **check hallucinations / artifacts**.

---

## 📊 Interactive Performance Report
> **[Click Here to View the Interactive Dashboard]**  
> (https://soyoungkim0327.github.io/lidar-research/)

---

## 🎥 Simulation Demo Videos
Here are the demonstration videos of the data collection and visualization system (located in `simulation/`):

| **1. Autopilot Mode** | **2. 1st Person View** | **3. Full Integration (North-Up)** |
| :---: | :---: | :---: |
| [**[▶️ Watch Demo 1]**](https://youtu.be/OOXVz8CxH-o) | [**[▶️ Watch Demo 2]**](https://youtu.be/zsbFbVcQ5rs) | [**[▶️ Watch Demo 3]**](https://www.youtube.com/watch?v=bGTpCJeoDx0) |
| *YOLO Sensor Streaming* | *Driver's Perspective* | *Lidar 2D/3D Aligned Visualization* |

---

## 🛠️ Key Components

### 1) Lightweight LiDAR SR Model (TinyRangeSR)
- Implements `TinyRangeSR`, a compact residual CNN for range-image SR.
- Typical structure: **bilinear upsampling + 3× Conv2D(SiLU) + residual output**.

### 2) Distillation + Range-Image Segmentation (Teacher–Student)
- Uses a higher-fidelity (128ch) pathway as a **teacher signal** for training a student model.
- Typical losses used in this repo include **BCEWithLogitsLoss** (segmentation) and **logit/prob alignment** terms (distillation).  
  *(Exact formulation depends on the training step scripts.)*

### 3) CARLA Data Pipeline (Stability-Oriented)
- Integrates CARLA sensor streaming with model inference.
- Applies a decoupled/asynchronous design to reduce rendering stalls during inference-heavy runs.
- Includes coordinate handling utilities for aligned visualization (e.g., north-up orientation).

---

## ✅ What is “verified” vs “needs re-check”
- The HTML report (`docs/lidar_sr_report.html`) contains headline numbers and plots for one run.  
  These values should be treated as **run-specific** and **metric-definition-specific**.
- For a stronger claim, recompute from raw arrays:
  - distance-binned MAE
  - normalized similarity score (as defined)
  - hallucination checks (false returns / phantom-close / large errors)
  - optional conservative post-fix (teacher-free) for deployment safety

---

## 🔍 Audit / Verification Script (Recommended)
A single script that:
1) Parses the HTML report (internal consistency check)  
2) Recomputes metrics from raw NPY pairs  
3) Flags hallucinations / artifacts  
4) Optionally applies a conservative post-fix using baseline upsampling

### Run (HTML-only quick check)
```bash
python sr_verify_and_fix.py --project_root .
```

### Run (Recompute from pairs CSV)
```bash
python sr_verify_and_fix.py ^
  --data_root "C:\path\to\data_root" ^
  --pairs_csv "C:\path\to\distill_pairs_sr.csv" ^
  --bins "0,30,60,100"
```

### Run (Recompute + Apply fix + Export worst frames)
```bash
python sr_verify_and_fix.py ^
  --data_root "C:\path\to\data_root" ^
  --pairs_csv "C:\path\to\distill_pairs_sr.csv" ^
  --bins "0,30,60,100" ^
  --apply_fix 1 ^
  --export_topk 20
```

> Output:
- `sr_verify_rows.csv` (per-frame metrics)
- `sr_verify_summary.json` (weighted summary + bin summary)
- `topk_panels/` (optional PNG panels)

---

## 📌 Known Risk Pattern (Why “hallucination checks” matter)
Even when MAE improves, SR can still produce artifacts such as:
- **False returns:** teacher is empty (0) but SR outputs non-zero values
- **Phantom-close:** SR invents near-range obstacles (dangerous for autonomy)
- **Long-range score inflation:** sparse bins can look “too good” depending on normalization

This repo treats these as first-class checks, and the next step is to tighten training/evaluation so that improvements remain valid under those risk metrics.

---

## 📂 Repository Structure

```text
├── src/               # SR + recognition/segmentation training/eval scripts
│   ├── recognition_step*.py
│   └── recognition_lib... 
├── simulation/        # CARLA data collection engine + async YOLO integration
│   ├── 1.demo_sync_autopilot...py
│   ├── 1.demo_sync_visual...py
│   └── 2.demo_sync_full_visual...py
├── utils/             # Visualization tools (range image / point cloud / BEV)
└── docs/              # Notes and the HTML report
```

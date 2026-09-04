# 🚦 Smart Traffic Analysis

> **Real-time vehicle detection, tracking, counting & speed estimation from traffic videos — powered by YOLO & ByteTrack.**

<p align="center">

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLOv8-Ultralytics-111827?style=for-the-badge)
![ByteTrack](https://img.shields.io/badge/Tracking-ByteTrack-7C3AED?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge\&logo=opencv\&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

<p align="center">
  <b>Turn raw traffic footage into structured traffic intelligence.</b>
</p>

---

## ✨ Overview

**Smart Traffic Analysis** is a single-file Python computer vision application designed to extract useful traffic metrics directly from video footage.

The pipeline combines **Ultralytics YOLO** for object detection, **ByteTrack** for multi-object tracking, and classical geometric analysis for line-crossing and trajectory-based measurements.

From a single traffic video, the system can produce:

* 🚗 Detected and tracked road participants
* 🔢 Bidirectional traffic counts
* 🆔 Persistent tracker IDs
* 📏 Approximate object speed
* 🛑 Stationary vehicle detection
* 📈 Live traffic statistics
* 🎥 Annotated MP4 output
* 📄 Per-frame CSV data

No cloud service is required, and the pipeline is designed to run locally with CPU inference or GPU acceleration.

---

## 🎯 What It Does

For every processed frame, the pipeline follows this flow:

```text
                  ┌──────────────────┐
                  │   Input Video    │
                  │  or Webcam Feed  │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │   YOLO Detection │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ ByteTrack Object │
                  │     Tracking     │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Trajectory &     │
                  │ Speed Analysis   │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Line-Crossing    │
                  │    Counting      │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Dashboard &      │
                  │ Visualization    │
                  └────────┬─────────┘
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
        ┌────────────────┐   ┌────────────────┐
        │ Annotated MP4  │   │  Traffic CSV   │
        └────────────────┘   └────────────────┘
```

The complete pipeline is implemented as a sequential processing loop inside `SmartTrafficAnalyzer.run()`.

---

## 🔥 Features

| Feature                       | Description                                                 |
| ----------------------------- | ----------------------------------------------------------- |
| 🎯 **Object Detection**       | Detect road participants using Ultralytics YOLO             |
| 🧠 **Multi-Object Tracking**  | Track objects across frames using ByteTrack                 |
| ↔️ **Bidirectional Counting** | Count `A → B` and `B → A` line crossings                    |
| ⚡ **Speed Estimation**        | Estimate speed from tracked center-point displacement       |
| 🛑 **Stationary Detection**   | Identify objects with very low movement                     |
| 📊 **Live Dashboard**         | FPS, active tracks, crossings and object statistics         |
| 🎥 **Video Annotation**       | Bounding boxes, IDs, speed labels and trace trails          |
| 📄 **CSV Logging**            | Export structured per-frame tracking data                   |
| 🖥️ **Webcam Support**        | Process a webcam using a numeric source index               |
| ⚙️ **Configurable**           | Model, confidence, FPS, calibration and tracking parameters |

---

## 🧩 How It Works

### 1. Object Detection

The default configuration uses:

```text
yolov8n.pt
```

The model is pretrained on the **COCO dataset** and can be replaced with another YOLO-compatible model.

---

### 2. Multi-Object Tracking

Detected objects are passed to **ByteTrack** through the `supervision` library.

Each tracked object receives a stable `tracker_id`, allowing the system to maintain its trajectory across multiple frames.

---

### 3. Line-Crossing Detection

Traffic counting is based on user-defined virtual lines.

For each tracked object's center point, the system evaluates its position relative to the line using the sign of a 2D cross product.

This allows directional counts such as:

```text
A ─────────────────────────── B
        🚗 🚗 🚗

        A → B : 12
        B → A :  7
```

Each configured line maintains independent directional counters.

---

### 4. Speed Estimation

Speed is estimated from the displacement of an object's bounding-box center over a sliding trajectory window.

Without calibration:

```text
speed → px/s
```

With a `meters_per_pixel` calibration factor:

```text
speed → km/h
```

> ⚠️ Speed estimation is approximate and requires proper scene calibration to represent real-world speed.

---

### 5. Stationary Vehicle Detection

An object is marked as `STOPPED` when its average per-frame displacement falls below the configured threshold.

Default:

```text
stationary_speed_threshold = 1.5 px/frame
```

---

## 🖥️ Output

### Annotated Video

The generated MP4 contains:

* Bounding boxes
* Tracker IDs
* Estimated speed
* `STOPPED` status
* Object trace trails
* Virtual counting lines
* Live statistics dashboard

Example label:

```text
ID:17  34.21 px/s
```

or, when calibrated:

```text
ID:17  42.80 km/h
```

The dashboard displays:

```text
┌─────────────────────────────┐
│ FPS              28.4       │
│ Active Tracks      12       │
│ Stationary          2       │
│ Total Crossings    37       │
│                             │
│ Line-1                     │
│   A → B          21         │
│   B → A          16         │
└─────────────────────────────┘
```

---

### CSV Log

Every valid tracked detection produces one CSV row per frame.

| Column          | Description            |
| --------------- | ---------------------- |
| `frame`         | Zero-based frame index |
| `tracker_id`    | ByteTrack object ID    |
| `class_id`      | COCO class ID          |
| `confidence`    | Detection confidence   |
| `center_x`      | Bounding-box center X  |
| `center_y`      | Bounding-box center Y  |
| `speed`         | Estimated speed        |
| `is_stationary` | Stationary status      |

The CSV is written using `utf-8-sig` encoding.

---

## 🚀 Quick Start

### Requirements

* **Python 3.13**
* OpenCV-compatible Python environment
* CPU or CUDA-enabled PyTorch environment

The project has been developed and tested against CPython 3.13 on Windows.

---

### 1. Clone

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

**Windows PowerShell**

```powershell
.\venv\Scripts\Activate.ps1
```

**Windows CMD**

```bat
venv\Scripts\activate.bat
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run

```bash
python smart_traffic_analysis.py
```

On the first run, `yolov8n.pt` is downloaded automatically if it is not available locally.

---

## 🎬 Usage

### Process a custom video

```bash
python smart_traffic_analysis.py \
    --source path/to/video.mp4
```

### Use another YOLO model

```bash
python smart_traffic_analysis.py \
    --model yolov8s.pt
```

### Enable live preview

```bash
python smart_traffic_analysis.py \
    --show
```

### Process a webcam

```bash
python smart_traffic_analysis.py \
    --source 0 \
    --show
```

---

## ⚙️ Command-Line Options

| Argument   |                Default | Description                    |
| ---------- | ---------------------: | ------------------------------ |
| `--source` |            `input.mp4` | Input video or webcam index    |
| `--model`  |           `yolov8n.pt` | YOLO model weights             |
| `--output` | `output_annotated.mp4` | Annotated video path           |
| `--log`    |      `traffic_log.csv` | CSV output path                |
| `--conf`   |                 `0.35` | Detection confidence threshold |
| `--show`   |                `False` | Enable live preview            |

---

## 🔧 Configuration

Advanced behavior is controlled through `AnalysisConfig`.

| Parameter                    |                Default | Purpose                     |
| ---------------------------- | ---------------------: | --------------------------- |
| `model_path`                 |           `yolov8n.pt` | YOLO weights                |
| `video_source`               |            `input.mp4` | Video/webcam source         |
| `output_path`                | `output_annotated.mp4` | Video output                |
| `log_csv_path`               |      `traffic_log.csv` | CSV output                  |
| `confidence_threshold`       |                 `0.35` | Detection threshold         |
| `iou_threshold`              |                  `0.5` | YOLO NMS IoU threshold      |
| `target_classes`             |                 `None` | COCO classes to analyze     |
| `trajectory_length`          |                   `30` | Stored trajectory positions |
| `stationary_speed_threshold` |                  `1.5` | Stationary threshold        |
| `assumed_fps`                |                 `None` | Optional FPS override       |
| `meters_per_pixel`           |                 `None` | Pixel-to-meter calibration  |
| `show_live_preview`          |                `False` | OpenCV preview              |

---

## 📐 Traffic Lines

By default, the application creates one horizontal counting line at approximately **60% of the frame height**:

```text
                 Traffic Scene

──────────────────────────────────
                 ↑
              Line-1
                 ↓
──────────────────────────────────
```

To analyze multiple lanes or directions, add additional `TrafficLine` objects through `build_default_lines()`.

This allows configurations such as:

```text
Lane 1  ─────────────── Line-1
Lane 2  ─────────────── Line-2
Lane 3  ─────────────── Line-3
```

---

## 📁 Project Structure

```text
smart-traffic-analysis/
│
├── smart_traffic_analysis.py   # Main application
├── requirements.txt            # Dependencies
├── .gitignore
├── LICENSE
└── README.md
```

The project intentionally keeps the core implementation in a single Python file.

Generated artifacts such as videos, CSV logs, virtual environments and model weights are excluded from version control.

---

## 🧱 Tech Stack

```text
Python
  │
  ├── Ultralytics YOLO
  │       └── Object Detection
  │
  ├── supervision
  │       └── ByteTrack Tracking
  │
  ├── OpenCV
  │       └── Video I/O & Visualization
  │
  ├── NumPy
  │       └── Geometry & Trajectory Math
  │
  └── Pandas
          └── CSV Export
```

---

## 📊 Outputs at a Glance

```text
Input
  │
  │  traffic.mp4
  ▼
┌─────────────────────────────────┐
│       SMART TRAFFIC ANALYSIS    │
│                                 │
│  🚗 Detection                   │
│  🆔 Tracking                    │
│  ↔️  Counting                    │
│  ⚡ Speed                       │
│  🛑 Stationary Detection       │
└───────────────┬─────────────────┘
                │
        ┌───────┴───────┐
        ▼               ▼
   annotated.mp4   traffic_log.csv
```

---

## ⚠️ Limitations

### Speed Calibration

Speed is based on pixel displacement. Without a `meters_per_pixel` calibration factor, the result is only reported in `px/s` and should not be interpreted as real-world speed.

### Camera Perspective

The system does not perform perspective correction. Counting lines are defined directly in image coordinates, so their placement affects counting accuracy.

### Stationary Detection

The stationary threshold is heuristic and can be affected by:

* Camera vibration
* Frame rate
* Object distance
* Scene perspective

### Tracking Identity Switches

ByteTrack can experience identity switches during heavy occlusion or when objects leave and re-enter the scene.

The current crossing logic also uses a short cooldown window to reduce repeated crossing events.

### Single Default Line

The repository ships with one counting line. Multi-lane deployments require additional `TrafficLine` configuration.

### Processing Throughput

Frames are processed sequentially on a single thread without batching or streaming optimization, so throughput depends on the selected YOLO model and hardware.

---

## 🗺️ Roadmap

Potential future improvements include:

* [ ] YAML / JSON configuration
* [ ] Interactive line configuration
* [ ] Pixel-to-meter calibration tool
* [ ] Time-series traffic dashboard
* [ ] Batch video processing
* [ ] RTSP / HTTP streaming
* [ ] GPU batched inference
* [ ] Automated unit tests for geometry and tracking logic

---

## 🎯 Use Cases

The project can serve as a reference implementation for:

* 🚦 Traffic flow analysis
* 🛣️ Intersection studies
* 🚗 Vehicle counting
* 📈 Traffic statistics extraction
* ⚡ Approximate speed analysis
* 🎥 Computer vision experimentation
* 🧪 YOLO + tracking research projects

---

## 📄 License

This project is licensed under the **MIT License**.

See [`LICENSE`](LICENSE) for the full license text.

---

<p align="center">

### 🚦 From Traffic Video → Structured Data

**Detect · Track · Count · Measure**

</p>

<p align="center">
  <sub>Built with Python, YOLO, ByteTrack & OpenCV.</sub>
</p>

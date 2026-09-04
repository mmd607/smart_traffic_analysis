# Smart Traffic Analysis

A real-time computer vision pipeline for vehicle detection, tracking, counting, and speed estimation from traffic videos, built on top of YOLO and ByteTrack.

## Overview

Smart Traffic Analysis is a single-file Python application that processes a traffic video and produces both a visually annotated output video and a structured per-frame CSV log. It addresses the common problem of extracting quantitative traffic metrics from raw video footage without requiring cloud services or specialized hardware.

The system performs the following tasks on each frame of the input video:

1. **Detection** of road participants using an Ultralytics YOLO model (the repository is configured to use the pretrained `yolov8n` weights from the COCO dataset).
2. **Multi-object tracking** of those detections across frames using ByteTrack, exposed through the `supervision` library, which assigns a stable identifier to each object.
3. **Line-crossing counting** on user-defined virtual lines, with separate counters for each traversal direction (`A->B` and `B->A`), computed from the sign change of the 2D cross product of the line vector and the object center.
4. **Approximate speed estimation** computed from the displacement of each tracked object's bounding-box center over a sliding window of frames. The reported unit is `px/s` by default, and `km/h` if a `meters_per_pixel` calibration factor is provided.
5. **Stationary vehicle detection** based on the average per-frame displacement of an object's center over its trajectory history.
6. **Live dashboard overlay** drawn on each frame with current FPS, active track count, total crossings, per-line counters, and per-object labels containing the tracker ID, measured speed, and a `STOPPED` tag when applicable.

The intended use cases include traffic flow analysis, intersection studies, and as a reference implementation for combining modern detection and tracking models with classical geometric counting logic.

## Features

- Object detection with Ultralytics YOLO (any model compatible with `ultralytics.YOLO`).
- Multi-object tracking using ByteTrack via the `supervision` library.
- Bidirectional line-crossing counting on one or more user-defined virtual lines.
- Per-object speed estimation in `px/s`, optionally converted to `km/h` through pixel-to-meter calibration.
- Stationary / slow-moving object detection.
- Per-frame CSV log of all tracked detections.
- Annotated MP4 output with bounding boxes, trace trails, labels, and a live statistics panel.
- Optional live preview window during processing.
- Command-line interface for selecting input video, model path, output paths, confidence threshold, and live preview.

## Pipeline

```text
Input Video (file or webcam index)
        |
        v
  YOLO Object Detection
        |
        v
  ByteTrack Multi-Object Tracking
        |
        v
  Trajectory / Speed Analysis
        |
        v
  Line-Crossing Counting
        |
        v
  Dashboard Rendering
        |
        v
  Annotated MP4  +  Per-frame CSV log
```

The pipeline is implemented as a single sequential loop in `SmartTrafficAnalyzer.run()`. Each iteration reads one frame, runs inference, updates the tracker, updates per-object trajectory history, checks for line crossings, renders the annotated frame, and writes it to the output video.

## Project Structure

```text
smart-traffic-analysis/
├── smart_traffic_analysis.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

The project is intentionally a single-file application. Generated artifacts such as the annotated video, the CSV log, the virtual environment, and model weight files are excluded by `.gitignore`.

## Requirements

- Python 3.13. The project has been developed and tested only against CPython 3.13 on Windows. Earlier Python versions are not part of the tested matrix and may not work without dependency adjustments.
- A working OpenCV-compatible build of Python on the host system.
- For GPU-accelerated YOLO inference, an environment configured for `torch` / CUDA is recommended; CPU inference is also supported.

Main dependencies (see `requirements.txt`):

| Package       | Purpose                                                    |
| ------------- | ---------------------------------------------------------- |
| `ultralytics` | YOLO model loading and inference                           |
| `supervision` | ByteTrack tracker, detection containers, annotators        |
| `opencv-python` | Video I/O, image drawing, and live preview window       |
| `numpy`       | Numerical operations for trajectory and geometry math      |
| `pandas`      | CSV log export                                             |

## Installation

```bash
git clone <repository-url>
cd <repository-directory>
python -m venv venv
```

Activate the virtual environment:

- Windows PowerShell:

  ```powershell
  .\venv\Scripts\Activate.ps1
  ```

- Windows CMD:

  ```bat
  venv\Scripts\activate.bat
  ```

- Linux / macOS:

  ```bash
  source venv/bin/activate
  ```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

The first execution will automatically download the `yolov8n.pt` weights if they are not present locally, because Ultralytics resolves missing weight files transparently.

## Usage

The application is invoked through `smart_traffic_analysis.py`. It reads command-line arguments, builds an `AnalysisConfig`, defines a default counting line, and starts the analysis pipeline.

Basic run on the default input file (`input.mp4`):

```bash
python smart_traffic_analysis.py
```

Run on a custom video:

```bash
python smart_traffic_analysis.py --source path/to/video.mp4
```

Use a different YOLO weight file:

```bash
python smart_traffic_analysis.py --model yolov8s.pt
```

Enable the live preview window during processing:

```bash
python smart_traffic_analysis.py --show
```

Use a webcam (numeric source is treated as a camera index):

```bash
python smart_traffic_analysis.py --source 0 --show
```

Available command-line arguments:

| Argument  | Type   | Default              | Description                                  |
| --------- | ------ | -------------------- | -------------------------------------------- |
| `--source`| string | `input.mp4`          | Input video path or webcam index             |
| `--model` | string | `yolov8n.pt`         | YOLO model weights path                      |
| `--output`| string | `output_annotated.mp4` | Output annotated video path               |
| `--log`   | string | `traffic_log.csv`    | Output CSV log path                          |
| `--conf`  | float  | `0.35`               | Detection confidence threshold               |
| `--show`  | flag   | `False`              | Show live preview window while processing    |

To customize the counting lines or the analysis behavior beyond the CLI options, edit the `main()` function in `smart_traffic_analysis.py` to pass a custom list of `TrafficLine` objects and a custom `AnalysisConfig`.

## Configuration

The `AnalysisConfig` dataclass controls the runtime behavior of the pipeline.

| Parameter                    | Type            | Default              | Description                                                                                  |
| ---------------------------- | --------------- | -------------------- | -------------------------------------------------------------------------------------------- |
| `model_path`                 | `str`           | `"yolov8n.pt"`       | Path to the YOLO weights file                                                                |
| `video_source`               | `str`           | `"input.mp4"`        | Input video path or webcam index                                                             |
| `output_path`                | `str`           | `"output_annotated.mp4"` | Output annotated video path                                                               |
| `log_csv_path`               | `str`           | `"traffic_log.csv"`  | Output CSV log path                                                                          |
| `confidence_threshold`       | `float`         | `0.35`               | Minimum detection confidence                                                                 |
| `iou_threshold`              | `float`         | `0.5`                | NMS IoU threshold used by YOLO                                                               |
| `target_classes`             | `list[int] or None` | `None`           | COCO class IDs to consider; `None` means all classes                                         |
| `trajectory_length`          | `int`           | `30`                 | Number of recent positions retained per tracked object                                       |
| `stationary_speed_threshold` | `float`         | `1.5`                | Average per-frame displacement (px/frame) below which an object is flagged as `STOPPED`     |
| `assumed_fps`                | `float or None` | `None`               | Override FPS used for speed calculations; if `None`, the video FPS is used                   |
| `meters_per_pixel`           | `float or None` | `None`               | Pixel-to-meter scale; if set, speed is reported in `km/h`, otherwise in `px/s`              |
| `show_live_preview`          | `bool`          | `False`              | Open a live OpenCV window while processing                                                   |

The default counting line is defined by `build_default_lines()` and consists of a single horizontal line at 60% of the frame height, named `Line-1`. To analyze multiple lanes or directions, modify that function to return additional `TrafficLine` objects.

## Outputs

The pipeline produces two artifacts on each run:

### 1. Annotated MP4 video

Written to `output_path` (default `output_annotated.mp4`). The codec is `mp4v` and the frame size and FPS match the input video. Each frame contains:

- Bounding boxes for every detected object.
- Per-object label: `ID:<n> <speed><unit> [STOPPED]`.
- A colored trace trail showing the recent path of each tracked object.
- The configured counting line(s) drawn over the scene.
- A semi-transparent statistics panel in the top-left corner showing current FPS, number of active tracks, number of stationary objects, total crossings, and per-line directional counts.

### 2. Per-frame CSV log

Written to `log_csv_path` (default `traffic_log.csv`) with `utf-8-sig` encoding. One row is produced per frame for every object that received a valid `tracker_id` from ByteTrack; detections without a tracker ID are skipped. The columns are:

| Column         | Description                                                                                |
| -------------- | ------------------------------------------------------------------------------------------ |
| `frame`        | Zero-based index of the processed frame                                                    |
| `tracker_id`   | Stable identifier assigned by ByteTrack                                                    |
| `class_id`     | COCO class ID of the detection                                                             |
| `confidence`   | Detection confidence, rounded to 3 decimal places                                          |
| `center_x`     | X coordinate of the bounding-box center, in pixels                                          |
| `center_y`     | Y coordinate of the bounding-box center, in pixels                                          |
| `speed`        | Estimated speed in `px/s` or `km/h` depending on calibration, rounded to 2 decimal places |
| `is_stationary`| `True` if the object was classified as stationary on this frame, otherwise `False`        |

A final summary is also printed to standard output, listing the per-line directional counts and the output video path.

## Model

The application is configured to use `yolov8n.pt` from the Ultralytics distribution, which contains the YOLOv8 Nano model pretrained on the COCO dataset. This model is small and fast, which makes it suitable as a default for CPU-only environments. The class IDs it produces correspond to the COCO taxonomy, including the vehicle categories typically used in traffic analysis.

If the weight file is not present on disk when the program starts, Ultralytics will download it automatically to the current working directory. The repository does not include any custom-trained model weights.

Any other YOLO-compatible weights (for example `yolov8s.pt`, `yolov8m.pt`, or a custom fine-tuned model) can be selected through the `--model` argument or by changing `model_path` in `AnalysisConfig`.

## Limitations

- **Speed estimation is approximate.** It is derived from pixel displacement over a sliding window of frames. The result is meaningful only when combined with a calibrated `meters_per_pixel` factor. Without calibration the reported unit is `px/s`, which is not directly interpretable as a real-world speed.
- **Camera perspective is not modeled.** The system assumes a single static camera and does not correct for perspective distortion. Counting lines must be placed in the image plane manually, and their position directly affects counting accuracy.
- **Stationary detection is heuristic.** The threshold (`stationary_speed_threshold`) is in pixels per frame and is sensitive to camera vibration, frame rate, and object distance.
- **Tracking identity switches can occur.** Like any multi-object tracker, ByteTrack can reassign tracker IDs to different physical objects in heavy occlusion or when objects leave and re-enter the frame. The current line-crossing logic records a crossing event whenever the sign of the cross product changes between consecutive frames for a given `(tracker_id, line_name)` pair, with a small cooldown window of 5 frames. If an ID is reassigned to a different physical object, additional crossing events may be recorded for the new object that were not part of the original object's trajectory.
- **Default line is a single horizontal line.** The repository ships with a single counting line. Multi-lane or multi-direction deployments require code-level configuration of additional `TrafficLine` objects.
- **No batching or streaming optimization.** Frames are processed one at a time on a single thread, so throughput depends on the inference time of the chosen YOLO model.

## Future Work

- A configuration file (for example YAML or JSON) for counting lines, model parameters, and class filters, removing the need to edit source code for common adjustments. The class filter is already exposed through `AnalysisConfig.target_classes`; a file-based configuration layer would make it accessible without code changes.
- A calibration tool for `meters_per_pixel`, based on a known reference length in the scene.
- A dashboard for aggregating the CSV log into time-series charts of flow and speed.
- Support for batch processing of multiple videos in a single run.
- A real-time streaming mode that consumes an RTSP or HTTP video source.
- Optional GPU batched inference for higher throughput.
- A unit test suite covering the geometry, counting, and trajectory analysis modules.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for the full text.

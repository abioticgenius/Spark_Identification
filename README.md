# Detection and Tracking of Pantograph and Cable Contact Points with Spark Identification

## Overview

This project implements a **computer vision-based system** to detect, track, and analyze pantographs in railway videos. Using a trained **YOLOv8 model**, the system identifies pantograph components, tracks their movement, detects sparks, and logs all relevant information for performance and safety analysis. The solution is optimized for real-world deployment in railway monitoring systems.

---

## Features

* **YOLOv8 Object Detection**: Detects pantograph, pantobar, and cables in each video frame.
* **Movement Tracking**: Calculates horizontal and vertical displacement of the pantograph between frames.
* **Spark Detection**: Identifies sparks in the pantobar region by analyzing high-intensity (bright) spots.
* **Contact Point Analysis**: Computes distances between pantobar and cables to check for proper alignment.
* **CSV Logging**: Records timestamped movement data, detected sparks, and contact distances.
* **Annotated Video Output**: Saves the processed video with bounding boxes, annotations, and overlays.

---

## Setup

### Requirements

* Python 3.8+
* Required Libraries:

  ```bash
  pip install opencv-python-headless numpy ultralytics
  ```

### YOLO Model

* Ensure you have a **trained YOLOv8 model** for pantograph detection.
* Update the `model_path` variable in the script with the path to your weights file.

### File Paths

* **Input Video**: Set `input_video_path` to the location of your raw video.
* **Output Video**: Set `output_video_path` where the processed annotated video will be saved.
* **CSV File**: Set `csv_output_path` to define where the CSV log should be stored.

---

## Script Workflow

1. **Frame Capture**: Reads each frame from the input video.
2. **Object Detection**: Runs YOLOv8 inference to detect pantograph components.
3. **Movement Analysis**: Compares pantograph position across consecutive frames.
4. **Spark Detection**: Identifies bright pixel clusters within the pantobar region.
5. **Contact Analysis**: Measures distances between pantobar and overhead cables.
6. **Logging**: Writes structured data to a CSV file.
7. **Video Rendering**: Saves annotated video with bounding boxes, spark highlights, and metrics overlay.

---

## Usage

1. Update paths for model, input video, output video, and CSV inside the script.
2. Run the script:

   ```bash
   python pantograph_tracking.py
   ```
3. Outputs:

   * Processed video with annotations.
   * CSV log file containing movement, spark count, and distances.

---

## Notes

* Ensure YOLO is properly trained for pantograph, pantobar, and cable classes.
* Detection parameters may require tuning for different lighting or video conditions.
* Output files will be saved in paths configured in the script.

---

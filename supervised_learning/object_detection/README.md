# Object Detection with YOLO v3

This project focuses on implementing the YOLO v3 algorithm for object detection using Python and TensorFlow. The objective is to understand and apply various concepts related to computer vision and object detection, such as OpenCV, object detection algorithms, non-max suppression, anchor boxes, and mean Average Precision (mAP).

## Table of Contents

- [Learning Objectives](#learning-objectives)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Classes and Methods](#classes-and-methods)
  - [Yolo Class](#yolo-class)
    - [`__init__`](#__init__)
    - [`process_outputs`](#process_outputs)
    - [`filter_boxes`](#filter_boxes)
    - [`non_max_suppression`](#non_max_suppression)
- [Resources](#resources)
- [Acknowledgements](#acknowledgements)

## Requirements

- **Python Version:** Python 3.9
- **Operating System:** Ubuntu 20.04 LTS
- **Python Packages:**
  - `numpy` (version 1.25.2)
  - `tensorflow` (version 2.15)
  - `opencv-python` (version 4.9.0.80)
- **Files:**
  - `yolo.h5`: Pre-trained YOLO v3 Keras model.
  - `coco_classes.txt`: List of class names used by the COCO dataset.
  - `yolo_images.zip`: A set of images for testing the object detection.

## Project Structure


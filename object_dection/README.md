# Real-time Object Detection with YOLOv5-TFLite

This project implements real-time object detection using a YOLOv5 model converted to TensorFlow Lite format. It uses your computer's webcam to perform live object detection with support for multiple object classes.

## Features

- Real-time object detection using webcam feed
- TensorFlow Lite optimization for better performance
- Support for multiple object classes
- Non-maximum suppression for better detection accuracy
- Special handling for small objects (bottles, bananas, pens, etc.)
- Visual bounding box display with class labels and confidence scores

## Prerequisites

- Python 3.7 or higher
- Webcam
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository or download the source code
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have the following files in your project directory:
   - `yolov5.tflite` (YOLOv5 model in TFLite format)
   - `labels.txt` (List of class labels)

## Usage

Run the main script:
```bash
python app.py
```

- Press 'q' to quit the application
- The application will display the webcam feed with detected objects outlined by bounding boxes
- Each detection includes the class label and confidence score

## Model Details

The project uses a YOLOv5 model converted to TensorFlow Lite format for efficient inference. The model:
- Processes images at a specified input resolution
- Performs detection with configurable confidence thresholds
- Includes special handling for small objects
- Uses non-maximum suppression to remove duplicate detections

## Notes

- The confidence threshold can be adjusted in the code (default: 0.25)
- Small objects have a lower detection threshold for better sensitivity
- The model expects RGB input images normalized to [0, 1]

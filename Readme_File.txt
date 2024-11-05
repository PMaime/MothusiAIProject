
# Real-Time Object Detection with YOLOv4 Tiny

This Python script demonstrates real-time object detection using the YOLOv4 Tiny model in OpenCV. The script captures video from the default camera (webcam) and performs object detection on the frames, displaying the results in real-time.

## Features

- Utilizes the YOLOv4 Tiny model for efficient object detection
- Supports real-time video processing with a multi-threading approach
- Draws bounding boxes around detected objects and displays the class labels and confidence scores
- Allows for adjusting the confidence threshold for object detection

## Prerequisites

- Python 3.x
- OpenCV
- NumPy

You can install the required dependencies using pip:
    "pip install opencv-python numpy"

## Usage

1. Download the YOLOv4 Tiny configuration file, weights file, and class names file from the appropriate sources.
2. Update the following variables in the script with the correct file paths:
   - `cfg_file`
   - `weights_file`
   - `names_file`
3. Run the script:

   "python object_detection.py"
   
4. The script will start the video capture and display the real-time object detection results.
5. Press 'q' to exit the script.

## How it Works

1. The `VideoStream` class is used to capture video frames from the default camera in a separate thread, allowing for efficient frame processing.
2. The YOLOv4 Tiny model is loaded using the OpenCV `cv2.dnn.readNet()` function, and the layer names are obtained.
3. For each frame, the script prepares the frame for YOLOv4 Tiny detection by creating a blob and passing it through the network.
4. The detection results are processed, and bounding boxes are drawn around the detected objects, along with the class labels and confidence scores.
5. The processed frame is displayed in a window, and the loop continues until the user presses 'q' to exit.

## Customization

- Adjust the confidence threshold (`score_threshold`) to change the minimum confidence level for object detection.
- Modify the Non-Maximum Suppression (NMS) threshold (`nms_threshold`) to control the level of overlap between bounding boxes.
- Update the file paths for the YOLOv4 Tiny configuration, weights, and class names files as needed.

## Note

This script is designed to work with the YOLOv4 Tiny model. If you want to use a different YOLO model, you will need to update the model loading and processing steps accordingly.

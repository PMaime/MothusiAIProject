#Authors: Maime Pheello 
#         Mololi Rammalane
 
      
import cv2
import numpy as np
import time
from threading import Thread

class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
    
    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# Paths to YOLOv4 Tiny config, weights, and class names files
cfg_file = '/home/phillip/Downloads/yolov4-tiny.cfg'
weights_file = '/home/phillip/Downloads/yolov4-tiny.weights'
names_file = '/home/phillip/Downloads/coco.names'

# Load class names
with open(names_file, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load YOLOv4 Tiny model
net = cv2.dnn.readNet(weights_file, cfg_file)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Get YOLOv4 Tiny layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Set up the laptop's default camera (use src=0)
vs = VideoStream(0).start()
time.sleep(1.0)  # Allow camera sensor to warm up

while True:
    frame = vs.read()
    if frame is None:
        print("Failed to retrieve frame. Check the camera connection.")
        break  # Exit if no frame is found

    # Prepare frame for YOLOv4 Tiny
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Perform forward pass
    outs = net.forward(output_layers)

    # Process detection results
    confidences = []
    boxes = []
    class_ids = []
    height, width = frame.shape[:2]

    for output in outs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.1:  # Adjust threshold here
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.3, nms_threshold=0.4)

    # Draw bounding boxes
    if len(indices) > 0:
        for i in indices.flatten():  # Flatten the indices if they are not in a list format
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
vs.stop()
cv2.destroyAllWindows()

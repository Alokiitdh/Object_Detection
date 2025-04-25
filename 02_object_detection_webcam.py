# used for image/video capture and processing.
import cv2 

# A helper library built on top of OpenCV to simplify drawing elements like FPS, bounding boxes, etc.
import cvzone 

#to load and run YOLOv8 models
from ultralytics import YOLO
import math


# This opens your default webcam
# object cap allows you to read video frames in real time.
cap = cv2.VideoCapture(0) 

# setting the Height and weidth of the cam
# 3 is the property ID for width
#4 is the property ID for height

#This sets the camera resolution to 1280x720 pixels
cap.set(3,1280)
cap.set(4,720)


model = YOLO('yolov8n.pt') 

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    success, img = cap.read()
    results = model(img, stream = True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            
            # Bounding BOX
            x1,y1,x2,y2 = box.xyxy[0] # cordinates
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            # cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255), 3)
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1,y1,w,h))

            # Confidence
            conf = math.ceil((box.conf[0]*100))/100
            
            # Class Name
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{classNames[cls]}{conf}', (max(0, x1), max(35, y1)), scale=1.0, thickness=1)

    cv2.imshow('Image',img)
    cv2.waitKey(1)
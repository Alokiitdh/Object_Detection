# used for image/video capture and processing.
import cv2 
import numpy as np

# A helper library built on top of OpenCV to simplify drawing elements like FPS, bounding boxes, etc.
import cvzone 

#to load and run YOLOv8 models
from ultralytics import YOLO
import math

from sort import *


# cap.set(3,1280)
# cap.set(4,720)
# This opens your default webcam
# object cap allows you to read video frames in real time.
cap = cv2.VideoCapture(r'C:\Users\alokj\OneDrive\Documents\GitHub\Object_Detection\Dataset\people.mp4') 

model = YOLO(r'YOLO-Weights\yolov8n.pt') 

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

mask = cv2.imread(r'C:\Users\alokj\OneDrive\Documents\GitHub\Object_Detection\MAsk\mask-1.png')


## TRACKER ##


tracker = Sort(max_age=20, min_hits= 3, iou_threshold=0.3)


limitsup = [103, 161, 296, 161]
limitsdown = [527, 489, 735, 489]

totalCountup = []
totalCountdown = []

while True:
    success, img = cap.read()
    imgregion = cv2.bitwise_and(img, mask)
    imgGraphics = cv2.imread("graphics-1.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (730, 260))

    if not success or img is None:
        print("Failed to read frame or video has ended.")
        break
        ## instead om img we will send imgregion
    results = model(imgregion, stream = True)

    ### creating list 
    detection = np.empty((0,5))


    for r in results:
        boxes = r.boxes
        for box in boxes:
            
            # Bounding BOX
            x1,y1,x2,y2 = box.xyxy[0] # cordinates
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            # cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255), 3)
            w, h = x2-x1, y2-y1


            # Confidence
            conf = math.ceil((box.conf[0]*100))/100
            
            # Class Name
            cls = int(box.cls[0])
            ### NOw for detecting the only some vehicles like cars and only 
            ### we will dfine the rectangle on cars only
            current_class = classNames[cls]
            if current_class == 'person' and conf > 0.3:
                # cvzone.putTextRect(img, f'{current_class}{conf}', 
                #                    (max(0, x1), max(35, y1)), 
                #                    scale=0.6, thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1,y1,w,h), l=9, rt = 5)

                current_array = np.array([x1,y1,x2,y2,conf])
                detection = np.vstack([detection, current_array])

    resultsTracker = tracker.update(detection)
    
    cv2.line(img, (limitsup[0], limitsup[1]), (limitsup[2], limitsup[3]), (0,0, 255), 5)
    cv2.line(img, (limitsdown[0], limitsdown[1]), (limitsdown[2], limitsdown[3]), (0,0, 255), 5)
    
    
    for result in resultsTracker:
        x1,y1,x2,y2,id = result
        x1,y1,x2,y2, id= int(x1),int(y1),int(x2),int(y2), int(id)
        print(result)
        w, h = x2-x1, y2-y1
        cvzone.cornerRect(img, (x1,y1,w,h), l=9, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f'{id}', 
                                (max(0, x1), max(35, y1)), 
                                scale=2, thickness=3, offset=10)
        
        cx, cy = x1+w // 2, y1+h //2
        cv2.circle(img,(cx,cy), 5, (255,0,255), cv2.FILLED )
        
        if limitsup[0] <cx < limitsup[2] and limitsup[1] -15 < cy<limitsup[1]+ 15:
            if totalCountup.count(id) == 0:
                totalCountup.append(id)
                cv2.line(img, (limitsup[0], limitsup[1]), (limitsup[2], limitsup[3]), (0,255, 0), 5)
    
    # # cvzone.putTextRect(img, f'Count: {len(totalCount)}',(50,50))
        if limitsdown[0] <cx < limitsdown[2] and limitsdown[1] -15 < cy<limitsdown[1]+ 15:
            if totalCountdown.count(id) == 0:
                totalCountdown.append(id)
                cv2.line(img, (limitsdown[0], limitsdown[1]), (limitsdown[2], limitsdown[3]), (0,255, 0), 5)
    
    
    cv2.putText(img,str(len(totalCountup)),(929,345),cv2.FONT_HERSHEY_PLAIN,5,(139,195,75),7)  
    cv2.putText(img,str(len(totalCountdown)),(1191,345),cv2.FONT_HERSHEY_PLAIN,5,(50,50,230),7)                    
    cv2.imshow("Image", img)
    # cv2.imshow('ImageRegion',imgregion)
    cv2.waitKey(1)
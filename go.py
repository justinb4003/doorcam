import os
import sys
import cv2
import pygame
from time import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

pygame.mixer.init()
pygame.mixer.music.load("dogatdoor.wav")

rtsp_url = os.environ.get('CAM_RTSP_URL')

# Start capturing the video stream from the camera
print(rtsp_url)
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print('Error opening video stream or file')
    sys.exit()

# Now load the model.  This takes a while
model = YOLO('models/yolov8n.pt')


last_announce = time() - 20
while True:
    ret, data = cap.read()
    data = cv2.resize(data, (640, 480))
    results = model(data)
    for r in results:
        annotator = Annotator(data)
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            c = box.cls
            name = model.names[int(c)]
            print(name)
            annotator.box_label(b, name)
            curr_time = time()
            if curr_time - last_announce > 20:
                last_announce = curr_time
                if name == 'car' or name == 'truck':
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pass
    img = annotator.result()

    cv2.imshow('cam', img)
    cv2.waitKey(50)

    """
    for result in results:
        r = json.loads(result.tojson())
        for o in r:
            print(o['name'])
    """


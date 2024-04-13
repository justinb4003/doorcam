import os
import sys
import cv2
import pygame
import threading

from time import time
from threading import Lock
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


latest_frame = None
last_ret = None
lo = Lock()

pygame.mixer.init()
pygame.mixer.music.load("dogatdoor.wav")

rtsp_url = os.environ.get('CAM_RTSP_URL')

# Start capturing the video stream from the camera
print(rtsp_url)
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print('Error opening video stream or file')
    sys.exit()


def rtsp_cam_buffer(vcap):
    global latest_frame, lo, last_ret
    while True:
        with lo:
            last_ret, latest_frame = vcap.read()


t1 = threading.Thread(target=rtsp_cam_buffer,
                      args=(cap, ),
                      name="rtsp_read_thread")
t1.daemon = True
t1.start()

# Now load the model.  This takes a while
model = YOLO('models/yolov8n.pt')

announce_delay = 8
trigger_names = ['dog', 'cow']
last_announce = time() - announce_delay
while True:
    if (last_ret is not None) and (latest_frame is not None):
        ret = last_ret
        data = latest_frame
    else:
        continue
    data = cv2.resize(data, (1024, 720))
    results = model(data, verbose=False)
    for r in results:
        annotator = Annotator(data)
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            c = box.cls
            name = model.names[int(c)]
            annotator.box_label(b, name)
            curr_time = time()
            diff = int(curr_time - last_announce)
            if name in trigger_names:
                if diff > announce_delay:
                    last_announce = curr_time
                    pygame.mixer.music.play()
                else:
                    print(f"Too soon {announce_delay - diff} to go")
    img = annotator.result()

    cv2.imshow('cam', img)
    cv2.waitKey(50)

    """
    for result in results:
        r = json.loads(result.tojson())
        for o in r:
            print(o['name'])
    """


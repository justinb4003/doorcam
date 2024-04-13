import os
import sys
import cv2
import pygame
import threading

from time import time
from threading import Lock
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

pygame.mixer.init()
pygame.mixer.music.load("dogatdoor.wav")

rtsp_url = os.environ.get('CAM_RTSP_URL')
print('Monitoring: ', rtsp_url)
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print('Error opening video stream or file')
    sys.exit()



# Set up a thread that will read from the camera while we're
# doing other things; that way we've always got the latest frame
# from the stream
latest_frame = None
last_ret = None
lo = Lock()


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


# Now load the machine learning (vision) model.  This takes a while.
model = YOLO('models/yolov8n.pt')

# Number of seconds to wait between announcements
announce_delay = 8
# The classes that we'll trigger on
trigger_names = ['dog', 'cow']  # Leonard looks like a cow apparently

last_announce = time() - announce_delay
while True:
    # If we have a frame use it
    if (last_ret is not None) and (latest_frame is not None):
        ret = last_ret
        data = latest_frame
    # Otherwise we'll try again until the camera reading thread gives us
    # some data!
    else:
        continue
    # Resize the image so it displays nicer on the screen
    # The less data (pixels) we make the machine learning model process
    # the faster it goes but the less accurate it is.
    data = cv2.resize(data, (1024, 720))
    results = model(data, verbose=False)

    # Loop through every object found by the model
    for r in results:
        annotator = Annotator(data)
        boxes = r.boxes
        for box in boxes:
            # get box coordinates in (left, top, right, bottom) format
            b = box.xyxy[0]
            c = box.cls
            # 'c' when casted to an int is the index of the name in our
            # 'names' collection below
            name = model.names[int(c)]
            annotator.box_label(b, name)
            curr_time = time()
            diff = int(curr_time - last_announce)
            # Now check to see if we need to make an announcement
            if name in trigger_names:
                if diff > announce_delay:
                    last_announce = curr_time
                    pygame.mixer.music.play()
                else:
                    print(f"Too soon {announce_delay - diff} to go")
    img = annotator.result()
    cv2.imshow('cam', img)
    cv2.waitKey(25)


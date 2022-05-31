import cv2
import numpy as np
from imutils.video import VideoStream
import argparse
import imutils
import time

# from Trackers.centroid_tracker.centroid import Centroid_tracker
# from Detectors.detector import YOLO
from Detectors.YOLO import yolo
if __name__ == "__main__":
    # Add the arg parser
    

    # load the video
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    # Initiate object detector
    cfg_file = "Detectors/YOLO/darknet/cfg/yolo.cfg"
    weight_file = "Detectors/YOLO/darknet/yolov4.weights"
    namesfile = "Detectors/YOLO/darknet/data/coco.names"
    datafile = "Detectors/YOLO/darknet/cfg/coco.data"
    m=""
    class_names="Detectors/YOLO/darknet/data/coco.names"
    colors=""
    dect = yolo.Yolo(cfg_file, weight_file,namesfile, datafile)
    
    # Initiate tracker object
    # ot = Centroid_tracker()
    (H, W) = (None, None)

    # run the while loop
    while True:
        # read the next frame from the video stream and resize it
        
        # Read frame
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        # if the frame dimensions are None, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # run detection and get bbox
        dect.detect(frame)
        # run tracker update to get tracked tracks
        # display the output
        pass
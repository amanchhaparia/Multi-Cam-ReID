import cv2
import numpy as np
from imutils.video import VideoStream
import argparse
import imutils
import time

from Trackers.centroid_tracker.centroid import Centroid_tracker
# from Detectors.detector import YOLO
from Detectors.YOLO import yolo
if __name__ == "__main__":
    # Add the arg parser
    
    cap = cv2.VideoCapture(0)                             # Local Stored video detection - Set input video
    # load the video
    # vs = VideoStream(src="/home/dhruv/Downloads/traffic.mp4").start()
    # time.sleep(2.0)
    # Initiate object detector
    cfg_file = "/home/dhruv/Multi-Cam-ReID/Detectors/YOLO/darknet/cfg/yolov4.cfg"
    weight_file = "/home/dhruv/Multi-Cam-ReID/Detectors/YOLO/darknet/yolov4.weights"
    namesfile = "/home/dhruv/Multi-Cam-ReID/Detectors/YOLO/darknet/data/coco.names"
    datafile = "/home/dhruv/Multi-Cam-ReID/Detectors/YOLO/darknet/cfg/coco.data"
    m=""
    class_names="/home/dhruv/Multi-Cam-ReID/Detectors/YOLO/darknet/data/coco.names"
    colors=""
    dect = yolo.Yolo(cfg_file, weight_file,namesfile, datafile)
    
    # Initiate tracker object
    ot = Centroid_tracker()
    (H, W) = (None, None)

    # run the while loop
    # while True:
    #     # read the next frame from the video stream and resize it
        
    #     # Read frame
    #     frame = vs.read()
    #     frame = imutils.resize(frame, width=400)
    #     # if the frame dimensions are None, grab them
    #     # if W is None or H is None:
    #     #     (H, W) = frame.shape[:2]

    #     # run detection and get bbox
    #     a=dect.detect(frame)
    #     # run tracker update to get tracked tracks
    #     # display the output
    while(True):
        ret, frame = cap.read()                                 # Capture frame and return true if frame present
        # For Assertion Failed Error in OpenCV
        if not ret:                                                  # Check if frame present otherwise he break the while loop
            break
        # dect.detect(frame_read)
        detections,res = dect.detect(frame)
        cv2.imshow("result",res)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # run tracker update to get tracked tracks
        ot.update(detections, ot.nextID)
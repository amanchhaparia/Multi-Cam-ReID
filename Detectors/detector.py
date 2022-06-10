import cv2
import numpy as np
from YOLO.yolo import Yolo

cfg_file = "YOLO/darknet/cfg/yolov4.cfg"
weight_file = "YOLO/darknet/yolov4.weights"
namesfile = "YOLO/darknet/data/coco.names"
datafile = "YOLO/darknet/cfg/coco.data"
# class Detector():
#     def __init__(self) -> None:
#         pass
a = Yolo(cfg_file, weight_file, namesfile, datafile)
b = cv2.imread("/home/dhruv/Multi-Cam-ReID/Detectors/YOLO/darknet/data/dog.jpg")
a.detect(b)
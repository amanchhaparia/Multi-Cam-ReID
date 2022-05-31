from pydoc import classname
from turtle import color
from unicodedata import name
import numpy as np
import cv2
import matplotlib.pyplot as plt
# from utils import *
from .darknet import darknet

# cfg_file = "darknet/cfg/yolo.cfg"
# weight_file = "darknet/yolov4.weights"
# namesfile = "darknet/data/coco.names"
# datafile = "darknet/cfg/coco.data"
# m=""
# class_names="darknet/data/coco.names"
# colors=""

# cfg_file = "Detectors/YOLO/darknet/cfg/yolo.cfg"
# weight_file = "Detectors/YOLO/darknet/yolov4.weights"
# namesfile = "Detectors/YOLO/darknet/data/coco.names"
# datafile = "Detectors/YOLO/darknet/cfg/coco.data"
# m=""
# class_names="Detectors/YOLO/darknet/data/coco.names"
# colors=""

# cfg_file, datafile, namesfile, weight_file,m ,class_names,colors="","",'','','','',''
class Yolo():
    
    def __init__(self, cfgfile, weightfile, name_file, data_file) -> None:
        cfg_file, weight_file, namesfile ,datafile = cfgfile, weightfile, name_file, data_file
        self.load_model()

    def load_model(self):
        """
        Loads the YOLO darknet model
        """
        m, class_names, colors = darknet.load_network(cfg_file, datafile, weight_file)
        
    def detect(self, image):
        """
        Detects the person in the image.

        Args
        image: A ndarray bgr image.

        Returns a list of detections where each item in list contains [x1, y1, x2, y2, conf]
        """

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, (m.width, m.height))
        # detect the objects
        boxes = darknet.detect_image(m, class_names,image)
        # plot the image with the bounding boxes and corresponding object class labels
        darknet.draw_boxes(boxes, image, colors)

import numpy as np
import cv2
import matplotlib.pyplot as plt
from .darknet import darknet

class Yolo():

    cfg_file=" "
    weight_file =" "
    namesfile = ""
    datafile = ""
    m=""
    class_names=""
    colors=""

    def __init__(self, cfgfile, weightfile, name_file, data_file) -> None:
        self.cfg_file,self.weight_file, self.namesfile ,self.datafile = cfgfile, weightfile, name_file, data_file
        self.load_model()

    def load_model(self):
        """
        Loads the YOLO darknet model
        """
        self.m, self.class_names, self.colors = darknet.load_network(self.cfg_file, self.datafile, self.weight_file)
    
    def detect(self, image):
        """
        Detects the person in the image.

        Args
        image: A ndarray bgr image.

        Returns a list of detections where each item in list contains [x1, y1, x2, y2, conf]
        """

        width = darknet.network_width(self.m)
        height = darknet.network_height(self.m)
        darknet_image = darknet.make_image(width, height, 3)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        detections = darknet.detect_image(self.m, self.class_names, darknet_image)
        res=darknet.draw_boxes(detections, image, self.colors)
        # cv2.imshow("result",res)
        return detections, res
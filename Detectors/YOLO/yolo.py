import cv2
from .darknet import darknet
import numpy as np
import matplotlib.pyplot as plt
from ..tools import generate_detections as gdet
from ..detector import Detector
from ..tools import preprocessing

class Yolo():

    def __init__(self, cfgfile, weightfile, name_file, data_file):
        self.cfg_file,self.weight_file, self.namesfile ,self.datafile = cfgfile, weightfile, name_file, data_file
        self.load_model()

    def load_model(self):
        """
        Loads the YOLO darknet model
        """
        self.model, self.class_names, self.colors = darknet.load_network(self.cfg_file, self.datafile, self.weight_file)
    
    def detect(self, image):
        """
        Detects the person in the image.

        Args
        image: A ndarray bgr image.

        Returns 
        dect_list: A list of detections where each item in list contains [x1, y1, x2, y2, conf]
        """
        self.width = darknet.network_width(self.model)
        self.height = darknet.network_height(self.model)
        self.i_height = image.shape[0]
        self.i_width = image.shape[1]
        darknet_image = darknet.make_image(self.width, self.height, 3)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self.width, self.height),
                                interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        detections = darknet.detect_image(self.model, self.class_names, darknet_image)

        dect_list=[]
        for dect in detections:
            if(dect[0]== 'person'):
                bbox = self.convert2relative(dect[2])
                dect_list.append(bbox)
        return dect_list
    
    def detect_deepsort(self, image):
        """
        Detects the person in the image.

        Args
        image: A ndarray bgr image.

        Returns 
        dect_list: A list of detections where each item in list contains [x1, y1, x2, y2, conf]
        """
        self.width = darknet.network_width(self.model)
        self.height = darknet.network_height(self.model)
        self.i_height = image.shape[0]
        self.i_width = image.shape[1]
        darknet_image = darknet.make_image(self.width, self.height, 3)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self.width, self.height),
                                interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        detections = darknet.detect_image(self.model, self.class_names, darknet_image)
        # darknet.free_image(darknet_image)
        dect_list=[]
        bboxes = []
        scores = []
        names = []

        for dect in detections:
            if(dect[0]== 'person'):
                bbox = self.convert2wh(dect[2])
                dect_list.append(bbox)
                bboxes.append(bbox)
                names.append(dect[0])
                scores.append(dect[1])

        model_filename = './Trackers/deepsort/model_data/mars-small128.pb'
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        features = encoder(image, dect_list)
        detections = [Detector(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        # run non-maxima supression
        nms_max_overlap = 1.0
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]  
        return detections

    def convert2relative(self, bbox):
        """
        Helper function to calculate left, top, right,bottom co-ordinates.

        Args
        bbox : bounding box containing tuple of co-ordinates.

        Returns
        left,top,right,bottom co-ordinates.
        """

        x, y, w, h  = bbox
        _height = self.height
        _width = self.width
        x, y, w, h = x/_width, y/_height, w/_width, h/_height
        image_h, image_w = self.i_height, self.i_width
        orig_left    = int((x - w / 2.) * image_w)
        orig_right   = int((x + w / 2.) * image_w)
        orig_top     = int((y - h / 2.) * image_h)
        orig_bottom  = int((y + h / 2.) * image_h)

        if (orig_left < 0): orig_left = 0
        if (orig_right > image_w - 1): orig_right = image_w - 1
        if (orig_top < 0): orig_top = 0
        if (orig_bottom > image_h - 1): orig_bottom = image_h - 1

        return tuple((orig_left, orig_top, orig_right, orig_bottom))

    def convert2wh(self, bbox):
        """
        Helper function to calculate centroid point, width and height

        Args
        bbox : bounding box containing tuple of co-ordinates.

        Returns
        left,top,right,bottom co-ordinates.
        """
        bbox = self.convert2relative(bbox)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0]
        y = bbox[1]
        return tuple((x, y, w, h))

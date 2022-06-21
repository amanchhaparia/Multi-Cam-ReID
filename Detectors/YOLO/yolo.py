import cv2
from .darknet import darknet

class Yolo():

    def __init__(self, cfgfile, weightfile, name_file, data_file):
        self.cfg_file,self.weight_file, self.namesfile ,self.datafile = cfgfile, weightfile, name_file, data_file
        self.load_model()

    def load_model(self):
        """
        Loads the YOLO darknet model
        """
        self.model, self.class_names, self.colors = darknet.load_network(self.cfg_file, self.datafile, self.weight_file)
        return self.model
        
    def detect(self, image):
        """
        Detects the person in the image.

        Args
        image: A ndarray bgr image.

        Returns a list of detections where each item in list contains [x1, y1, x2, y2, conf]
        """

        self.width = darknet.network_width(self.model)
        self.height = darknet.network_height(self.model)
        darknet_image = darknet.make_image(self.width, self.height, 3)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self.width, self.height),
                                interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        detections = darknet.detect_image(self.model, self.class_names, darknet_image)
        dect_list=[]
        for dect in detections:
            if(dect[0]== 'person'):
                dect_list.append(dect[2])
        return dect_list

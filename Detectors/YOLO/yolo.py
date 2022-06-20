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

    def draw_box(self, detections, image):
        """
        Draws bbox around the detection and display "id" of each object

        Args
        detections: 
        image: A ndarray bgr image.

        Returns
        image: Annotated image
        """

        for bbox, id in detections:
            left, top, right, bottom = self.convert4cropping(image,bbox)
            cv2.rectangle(image, (left, top), (right, bottom), [255,0,0], 1)
            cv2.putText(image, f"id : {id} ", (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 2)
        return image
    
    def convert4cropping(self, image, bbox):
        """
        Calculates correct position for bbox according to image height and width

        Args
        image: A ndarray bgr image.
        bbox: A tuple consiting of bounding box coordinates

        Returns
        orig_left, orig_top, orig_right, orig_bottom: bbox co-ordinates
        """
        x, y, w, h = self.convert2relative(bbox)

        image_h, image_w, __ = image.shape

        orig_left    = int((x - w / 2.) * image_w)
        orig_right   = int((x + w / 2.) * image_w)
        orig_top     = int((y - h / 2.) * image_h)
        orig_bottom  = int((y + h / 2.) * image_h)

        if (orig_left < 0): orig_left = 0
        if (orig_right > image_w - 1): orig_right = image_w - 1
        if (orig_top < 0): orig_top = 0
        if (orig_bottom > image_h - 1): orig_bottom = image_h - 1

        return orig_left, orig_top, orig_right, orig_bottom

    def convert2relative(self, bbox):
        """
        Converts to relative coordinates for annotation of bbox on image

        Args
        bbox: A tuple consiting of bounding box coordinates

        Returns
        x/_width, y/_height, w/_width, h/_height: realtive coordinates
        """
        x, y, w, h  = bbox
        _height     = self.height
        _width      = self.width
        return x/_width, y/_height, w/_width, h/_height

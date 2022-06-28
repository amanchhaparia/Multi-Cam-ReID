from pwd import getpwuid
import cv2
from .darknet import darknet

# This 4 global variables stores shape  of model and image(frame) height and width so function like convert2relative
# can be called without calling Yolo class
m_height = 0 # stores model height 
m_width = 0 # stores model width
i_height = 0 # stores image height
i_width = 0 # stores image width

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
        global i_height, i_width, m_height, m_width
        self.width = darknet.network_width(self.model)
        self.height = darknet.network_height(self.model)
        m_height = self.height
        m_width = self.width
        i_height = image.shape[0]
        i_width = image.shape[1]
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
    
def convert2relative(bbox):
    """
    Helper function to calculate left,top,right,bottom co-ordinates.

    Args
    bbox : bounding box containing tuple of co-ordinates.

    Returns
    left,top,right,bottom co-ordinates.
    """

    x, y, w, h  = bbox
    _height = m_height
    _width = m_width
    x, y, w, h = x/_width, y/_height, w/_width, h/_height
    image_h, image_w = i_height, i_width
    orig_left    = int((x - w / 2.) * image_w)
    orig_right   = int((x + w / 2.) * image_w)
    orig_top     = int((y - h / 2.) * image_h)
    orig_bottom  = int((y + h / 2.) * image_h)

    if (orig_left < 0): orig_left = 0
    if (orig_right > image_w - 1): orig_right = image_w - 1
    if (orig_top < 0): orig_top = 0
    if (orig_bottom > image_h - 1): orig_bottom = image_h - 1

    return tuple((orig_left, orig_top, orig_right, orig_bottom))

import cv2
from Detectors.YOLO.darknet import darknet

def draw_box(detections, image ,model):
    for bbox, id in detections:
        left, top, right, bottom =convert4cropping(image,bbox,model)
        cv2.rectangle(image, (left, top), (right, bottom), [255,0,0], 1)
        cv2.putText(image, f"id : {id} ", (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 2)
    return image

def convert4cropping(image, bbox,model):
    x, y, w, h = convert2relative(bbox,model)

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

def convert2relative(bbox,model):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height = darknet.network_width(model)
    _width = darknet.network_height(model)
    return x/_width, y/_height, w/_width, h/_height
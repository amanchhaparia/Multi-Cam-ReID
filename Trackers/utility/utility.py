import cv2

def draw_box(detections, image):
    """
    Draws bbox around the detection and display "id" of each object

    Args
    detections : list of bbox of detected objects
    image : A ndarray bgr image.

    Returns
    image : Annotated image
    """
    
    for bbox, id in detections:
        left, top, right, bottom = bbox
        cv2.rectangle(image, (left, top), (right, bottom), [255,0,0], 1)
        cv2.putText(image, f"id : {id} ", (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 2)
    return image

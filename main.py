import cv2
from Trackers.centroid_tracker.centroid import Centroid_tracker
from Detectors.YOLO import yolo
from Detectors.YOLO.darknet import darknet
if __name__ == "__main__":
    
    # Add the arg parser
    
    # load the video
    vs = cv2.VideoCapture(0)

    cfg_file = "Detectors/YOLO/darknet/cfg/yolov4.cfg"
    weight_file = "Detectors/YOLO/darknet/yolov4.weights"
    namesfile = "Detectors/YOLO/darknet/data/coco.names"
    datafile = "Detectors/YOLO/darknet/cfg/coco.data"
    class_names="Detectors/YOLO/darknet/data/coco.names"

    # Initiate object detector
    dect = yolo.Yolo(cfg_file, weight_file,namesfile, datafile)
    
    # Initiate tracker object
    ot = Centroid_tracker()
    (H, W) = (None, None)

    # run the while loop
    while True:
        # read the next frame from the video stream and resize it
        
        # Read frame
        _, frame = vs.read()
        width = 400
        height = frame.shape[0] # keep original height
        dim = (width, height)
        resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        # if the frame dimensions are None, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # run detection and get bbox
        detections = dect.detect(frame)

        # run tracker update to get tracked tracks
        track = ot.update(detections)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

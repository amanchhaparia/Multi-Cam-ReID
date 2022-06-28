import cv2
from Trackers.centroid_tracker.centroid import Centroid_tracker
from Trackers.iou_tracker.iou import iou_tracker
from Detectors.YOLO import yolo
from Trackers.utility import utility
if __name__ == "__main__":
    
    # Add the arg parser
    
    # load the video
    vs = cv2.VideoCapture("/home/ppspr/Videos/campus4-c0.avi")

    cfg_file = "Detectors/YOLO/darknet/cfg/yolov4.cfg"
    weight_file = "Detectors/YOLO/darknet/yolov4.weights"
    namesfile = "Detectors/YOLO/darknet/data/coco.names"
    datafile = "Detectors/YOLO/darknet/cfg/coco.data"
    class_names="Detectors/YOLO/darknet/data/coco.names"

    # Initiate object detector
    dect = yolo.Yolo(cfg_file, weight_file,namesfile, datafile)
    
    # Initiate tracker object
    # ot = Centroid_tracker()
    ot = iou_tracker()

    # run the while loop
    while True:
        # Read frame 
        ret, frame = vs.read()
        if ret!=True:
            break
        frame = cv2.resize(frame,(640,480))
        # run detection and get bbox
        detections = dect.detect(frame)
        # run tracker update to get tracked tracks
        track_list = ot.update(detections)
        detections = []
        for track in track_list:
            detections.append((track.bbox, track.id))
        res = utility.draw_box(detections, frame)
        cv2.imshow("result",res)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
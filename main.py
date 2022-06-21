import cv2
from Trackers.centroid_tracker.centroid import Centroid_tracker
from Detectors.YOLO import yolo
from Trackers.utility.utility import draw_box
if __name__ == "__main__":
    
    # Add the arg parser
    
    # load the video
    vs = cv2.VideoCapture("/home/dhruv/Downloads/Single1.mp4")

    cfg_file = "Detectors/YOLO/darknet/cfg/yolov4-tiny.cfg"
    weight_file = "Detectors/YOLO/darknet/yolov4-tiny.weights"
    namesfile = "Detectors/YOLO/darknet/data/coco.names"
    datafile = "Detectors/YOLO/darknet/cfg/coco.data"
    class_names="Detectors/YOLO/darknet/data/coco.names"
    # Initiate object detector
    dect = yolo.Yolo(cfg_file, weight_file,namesfile, datafile)
    
    model = dect.load_model()
    # Initiate tracker object
    ot = Centroid_tracker()
    
    # run the while loop
    while True:
        # Read frame 
        ret, frame = vs.read()
        if ret!=True:
            break
        frame=cv2.resize(frame,(640,480))
        # run detection and get bbox
        detections = dect.detect(frame)
        # run tracker update to get tracked tracks
        track_list = ot.update(detections)
        detections = []
        for track in track_list:
            detections.append((track.bbox, track.id))
        res =draw_box(detections, frame,model)
        cv2.imshow("result",res)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
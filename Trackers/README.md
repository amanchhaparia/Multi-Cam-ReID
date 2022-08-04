## Trackers

Object trackers are the program which takes an initial set of object detections and develops a unique identification for each of the initial detections and then tracks the detected objects as they move around frames in a video.

General steps that every trackers has:

- Object detection, where the algorithm classifies and detects the object by creating a bounding box around it. 
- Assigning unique identification for each object (ID). 
- Tracking the detected object as it moves through frames while storing the relevant information.

Trackers which we have implemented:

- [Centroid Tracker](./centroid_tracker/)
- [IoU Tracker](./iou_tracker/)
- [IoU Predict Tracker](./iou_pred_tracker/)

We have a [Track](./track.py) module in track.py which has the basic attributes which tracker object needs to have.

We also have [utlilty](./utility/) file which will contain all basic functionality required by trackers, till now we have `draw_bbox()` which adds rectangle and text on the provided frame.
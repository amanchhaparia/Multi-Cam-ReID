## Trackers

Object trackers are the program which takes an initial set of object detections and develops a unique identification for each of the initial detections and then tracks the detected objects as they move around frames in a video.

<img src="../assets/tracker.gif" alt="IoU Predict flowchart" style="width:450px;"/>

</br>

**General steps that every tracker has:**

- Object detection, where the algorithm classifies and detects the object by creating a bounding box around it. 
- Assigning unique identification for each object (ID). 
- Tracking the detected object as it moves through frames while storing the relevant information.

### Trackers implemented:

- [Centroid Tracker](./centroid_tracker/)
- [IoU Tracker](./iou_tracker/)
- [IoU Predict Tracker](./iou_pred_tracker/)

[Track](./track.py) module in `track.py` contains the basic attributes which every tracker object needs to have.

[utlilty](./utility/) file contains all basic functionality required by trackers.

### Usage

**Detect and train objects in video**

```
- import any tracker module in main.py
- initiate `ot` with the imported tracker
- give path to video, weight, cfg file
- run main.py
```

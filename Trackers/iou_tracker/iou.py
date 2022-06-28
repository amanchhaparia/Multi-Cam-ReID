import numpy as np
from ..track import Track
from scipy.optimize import linear_sum_assignment as linear_assignment
from Detectors.YOLO import yolo

class iou_track(Track):
    def __init__(self, id, bbox, hits, miss):
        Track.__init__(self, id, bbox, hits, miss)

class iou_tracker():
    def __init__(self):
        self.tracks = []
        self.nextID = 0
        self.max_age = 10
        self.min_hits = 3

    def add_track(self, id, bbox):
        """
        Adding newly tracked objects  to track list.
        
        Args
        id : new id to be assigned 
        bbox : bounding box of the object
        centroid : centroid of the object

        Returns
        None
        """
        
        track = iou_track(id, bbox , 1, 0)
        self.tracks.append(track)
        self.nextID += 1
        print("added id ",track.id, "succesfully")

    def delete_track(self):
        """
        Deletes a centroid track object from tracks list if the max age is crossed.
        
        Args
        None

        Returns 
        None
        """

        for track in self.tracks:
            if(track.miss > self.max_age):
                print("deleted id ",track.id, "succesfully")
                self.tracks.remove(track)
    
    def update(self, detections):
        """
        Returns the active list of centroid track objects

        Args
        detections : list of bbox of detected objects

        Returns
        tracks : list of objects of class Track
        """

        if(detections == []):
            if(self.tracks == []):
                return self.tracks
            else:
                # increase every objects miss by 1
                for track in self.tracks:
                    track.miss +=1
                self.delete_track()
                return self.tracks
        if len(self.tracks) == 0:
            for i in range(0, len(detections)):
                self.add_track(self.nextID, detections[i])
            return self.tracks
        tracks = []
        for trk in self.tracks:
            tracks.append(trk.bbox)
        matched, unmatched_dets, unmatched_trks = self.assign_detections_to_trackers(tracks, detections, iou_thrd = 0.3)  
    
        for trk, det in matched:
            self.tracks[trk].bbox= detections[det]
            self.tracks[trk].hits += 1
            self.tracks[trk].miss = 0
        print(len(unmatched_dets),"unmatchdet")
        # Deal with unmatched detections      
        if len(unmatched_dets)>0:
            for idx in unmatched_dets:
                self.add_track(self.nextID , detections[idx])
        
        # Deal with unmatched tracks       
        if len(unmatched_trks)>0:
            print("unmat_trck",unmatched_trks)
            for trk_idx in unmatched_trks:
                self.tracks[trk_idx].miss += 1
            self.delete_track()
        result=[trk for trk in self.tracks if trk.hits>=self.min_hits]
        return result

    def assign_detections_to_trackers(self, tracks, detections, iou_thrd = 0.3):

        """
        From current list of trackers and new detections, output matched detections,
        unmatchted trackers, unmatched detections.

        Args
        tracks : list of bbox of tracks.
        detections : list of bbox of detected objects.
        iou_thrd : threshold iou value.

        Returns
        matched detections : list of index values of matching objects.
        unmatched trackers : list of index values of unmatched trackers.
        unmatched detections : list of index values of unmatched detections.

        """
        tracks_list = []
        detect = []
        if len(tracks)==0:
            tracks_list=np.zeros((1,4))
        else:
            for trk in tracks:
                tracks_list.append(list(yolo.convert2relative(trk)))
            tracks_list=np.array(tracks_list,dtype=np.float32)
        for det in detections:
            detect.append(list(yolo.convert2relative(bbox= det)))
        detect=np.array(detect,dtype=np.float32)
        IOU_mat = self.get_iou_matrix(tracks_list, detect)
        if(len(tracks_list)*len(detections) == 1 or len(tracks_list)*len(detections) == 0):
            IOU_mat = np.reshape(IOU_mat,(1,1))
        else: 
            IOU_mat = np.reshape(IOU_mat,(len(tracks_list),len(detections)))

        # Solve the maximizing the sum of IOU assignment problem using the
        # Hungarian algorithm (also known as Munkres algorithm)
        row , col = linear_assignment(-IOU_mat)  
        unmatched_trackers, unmatched_detections = [], []
        for t, track in enumerate(tracks_list):
            if(t not in row):
                unmatched_trackers.append(t)

        for d, det in enumerate(detections):
            if(d not in col):
                unmatched_detections.append(d)

        matches = []
    
        # For creating trackers we consider any detection with an 
        # overlap less than iou_thrd to signifiy the existence of 
        # an untracked object
        t=row
        d=col
        for i in range(len(row)):
            if(IOU_mat[t[i],d[i]]<iou_thrd):
                unmatched_trackers.append(t[i])
                unmatched_detections.append(d[i])
            else:
                matches.append((t[i],d[i]))
        
        if(len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        print("matches",matches)
        return matches, unmatched_detections, unmatched_trackers       
        
    def get_iou_matrix(self, box_arr1, box_arr2):
        """ 
        Given two arrays box1 , box2 where each row contains a bounding
        box defined as a list of four numbers: [x1,y1,x2,y2]
        It returns the Intersect of Union scores for each corresponding
        pair of boxes.

        Args
        box_arr1 : (numpy array) each row containing [x1,y1,x2,y2] coordinates
        box_arr2 : (numpy array) each row containing [x1,y1,x2,y2] coordinates
    
        Returns:
        (numpy array) The Intersect of Union scores for each pair of bounding boxes.
        """
        x11, y11, x12, y12 = np.split(box_arr1, 4, axis=1)
        x21, y21, x22, y22 = np.split(box_arr2, 4, axis=1)
        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))
        interArea = np.maximum((xB - xA + 1e-9), 0) * np.maximum((yB - yA + 1e-9), 0)
        boxAArea = (x12 - x11 + 1e-9) * (y12 - y11 + 1e-9)
        boxBArea = (x22 - x21 + 1e-9) * (y22 - y21 + 1e-9)
        iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
        return iou

    def convert2relative(self,bbox):
        """
        Helper function to calculate left,top,right,bottom co-ordinates.

        Args
        bbox : bounding box containing tuple of co-ordinates.

        Returns
        left,top,right,bottom co-ordinates.
        """
        x, y, w, h  = bbox
        _height = 416
        _width = 416
        x, y, w, h = x/_width,y/_height ,w/_width,h/_height
        image_h, image_w = 640,480
        orig_left    = int((x - w / 2.) * image_w)
        orig_right   = int((x + w / 2.) * image_w)
        orig_top     = int((y - h / 2.) * image_h)
        orig_bottom  = int((y + h / 2.) * image_h)

        if (orig_left < 0): orig_left = 0
        if (orig_right > image_w - 1): orig_right = image_w - 1
        if (orig_top < 0): orig_top = 0
        if (orig_bottom > image_h - 1): orig_bottom = image_h - 1

        return tuple((orig_left, orig_top, orig_right, orig_bottom))
        
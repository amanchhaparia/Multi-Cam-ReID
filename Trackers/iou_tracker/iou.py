import numpy as np
from ..track import Track
from scipy.optimize import linear_sum_assignment as linear_assignment

class iou_track(Track):
    def __init__(self, id, bbox, hits, miss):
        Track.__init__(self, id, bbox, hits, miss)

class iou_tracker():
    def __init__(self):
        self.tracks = []
        self.nextId = 0
        self.max_lost=10
        self.min_hits=3

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
        self.nextId += 1
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
            if(track.miss > self.max_lost):
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

        matched, unmatched_dets, unmatched_trks = self.assign_detections_to_trackers(detections, iou_thrd = 0.3)  
    
        for i in matched:
            self.tracks[i[0]].bbox= detections[i[1]]
            self.tracks[i[0]].hits += 1
            self.tracks[i[0]].miss = 0
        print(len(unmatched_dets),"unmatchdet")
        # Deal with unmatched detections      
        if len(unmatched_dets)>0:
            for idx in unmatched_dets:
                self.add_track(self.nextId , detections[idx])
        
        # Deal with unmatched tracks       
        if len(unmatched_trks)>0:
            print("unmat_trck",unmatched_trks)
            for trk_idx in unmatched_trks:
                self.tracks[trk_idx].miss += 1
            self.delete_track()
        result=[a for a in self.tracks if a.hits>=self.min_hits]
        return result

    def assign_detections_to_trackers(self , detections, iou_thrd = 0.3):

        """
        From current list of trackers and new detections, output matched detections,
        unmatchted trackers, unmatched detections.

        Args
        detections : list of bbox of detected objects
        iou_thrd : threshold iou value

        Returns
        matched detections : list of index values of matching objects.
        unmatched trackers : list of index values of unmatched trackers.
        unmatched detections : list of index values of unmatched detections.

        """
        
        IOU_mat= np.zeros((len(self.tracks),len(detections)),dtype=np.float32)
        for t,trk in enumerate(self.tracks):
            #trk = convert_to_cv2bbox(trk) 
            for d,det in enumerate(detections):
            #   det = convert_to_cv2bbox(det)
                IOU_mat[t,d] =self.box_iou2(trk.bbox,det) 
        
        print(IOU_mat)
        # Produces matches       
        # Solve the maximizing the sum of IOU assignment problem using the
        # Hungarian algorithm (also known as Munkres algorithm)
        
        row , col = linear_assignment(-IOU_mat)  
        print("row",row,"col",col)  
        unmatched_trackers, unmatched_detections = [], []
        for t, track in enumerate(self.tracks):
            if(t not in row):
                unmatched_trackers.append(t)
                print("t")

        for d, det in enumerate(detections):
            if(d not in col):
                unmatched_detections.append(d)
                print("d")

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
        
    def box_iou2(self ,a, b):
        """
        Helper function to calculate the ratio between intersection and the union of
        two boxes a and b

        Args
        a : bbox values of existing tracks
        b : bbox values of detections

        Returns
        iou : intersection over union value between bbox of tracks and detections.
        """

        a = self.convert2relative(a)
        b = self.convert2relative(b)
        w_intsec = np.maximum (0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
        h_intsec = np.maximum (0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
        s_intsec = w_intsec * h_intsec
        s_a = (a[2] - a[0])*(a[3] - a[1])
        s_b = (b[2] - b[0])*(b[3] - b[1])
        iou=float(s_intsec)/(s_a + s_b -s_intsec)
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

    
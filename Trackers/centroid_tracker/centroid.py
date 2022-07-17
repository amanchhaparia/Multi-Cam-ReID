import numpy as np
from ..track import Track
from scipy.spatial import distance as dist

class centroid_track(Track):
    def __init__(self, id, bbox, centroid, hits, miss):
        self.centroid = centroid
        Track.__init__(self, id, bbox, hits, miss)

class Centroid_tracker():
    def __init__(self):
        self.tracks = []
        self.nextID=0
        self.max_age = 10
        self.min_hits = 3
        self.thres_distance = 20

    def add_track(self, id, bbox, centroid):
        """
        Adding newly tracked objects to "tracks".
        
        Args
        id : new id to be assigned 
        bbox : bounding box of the object
        centroid : centroid of the object

        Returns
        None
        """
        
        track = centroid_track(id, bbox, centroid, 1, 0)
        self.tracks.append(track)
        self.nextID += 1
        print("added id ",track.id, "succesfully")

    def delete_track(self):
        """
        Deletes a centroid track object from "tracks" if the object is disappeared for more than "max_age".
        
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
        Returns the active list of centroid track objects.

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

        inputCentroids = np.zeros((len(detections), 2), dtype="int")

        for i in range(len(detections)):
            cX = int((detections[i][0]+ detections[i][1]) / 2.0)
            cY = int((detections[i][2] + detections[i][3]) / 2.0)
            inputCentroids[i] = (cX, cY)
        
        objectId = []
        objectCentroid = []
        if len(self.tracks) == 0:
            for i in range(0, len(inputCentroids)):
                self.add_track(self.nextID, detections[i], inputCentroids[i])
        else:      
            for track in self.tracks:
                objectId.append(track.id)
                objectCentroid.append(track.centroid) 
            matches, unmatched_tracks, unmatched_detections  = self.assign_detections_to_trackers(objectCentroid, inputCentroids)
            for trk, det in matches:
                self.tracks[trk].bbox = detections[det]
                self.tracks[trk].centroid = inputCentroids[det]
                self.tracks[trk].miss = 0
                self.tracks[trk].hits += 1
            # loop over the unused row indexes
            for row in unmatched_tracks: 
                # grab the object ID for the corresponding row index and increment the disappeared counter
                self.tracks[row].miss += 1
            for col in unmatched_detections:
                self.add_track(self.nextID, detections[col], inputCentroids[col])
        self.delete_track()
        result = [trk for trk in self.tracks if (trk.hits>=self.min_hits and trk.miss == 0)]
        return result

    def assign_detections_to_trackers(self,objectCentroid,inputCentroids):
        """
        Returns list of matched objects , unmatched tracks and unmatched detections.  
        
        Args
        objectCentroid : list of centroids of existing objects.
        inputCentroids : list of centroids of current detections.

        Returns
        unusedcols : index of newly detected tracks
        matched : list of index of matching objects 
        """
        D = dist.cdist(np.array(objectCentroid), inputCentroids)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        matched_tracks = set()
        matched_det = set()
        matches =[]
        # loop over the combination of the (row, column) index tuples
        for (row, col) in zip(rows, cols):
            # if we have already examined either the row or column value before, ignore it
            if row in matched_tracks or col in matched_det:
                continue
            matched_tracks.add(row)
            matched_det.add(col)
            matches.append((row,col))
        unmatched_tracks = set(range(0, D.shape[0])).difference(matched_tracks)
        unmatched_det = set(range(0, D.shape[1])).difference(matched_det)
        return matches, unmatched_tracks, unmatched_det

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
            D,unusedRows,unusedCols = self.check_track(objectCentroid,inputCentroids,detections)
            if D.shape[0] >= D.shape[1]:
            # loop over the unused row indexes
                for row in unusedRows: 
                    # grab the object ID for the corresponding row index and increment the disappeared counter
                    self.tracks[row].miss += 1
            else:
                for col in unusedCols:
                    self.add_track(self.nextID, detections[col], inputCentroids[col])
        self.delete_track()
        result=[a for a in self.tracks if a.hits>=self.min_hits]
        return result
        
    def check_track(self,objectCentroid,inputCentroids,detections):
        """
        Returns cost matrix and list of matched and unmatched tracks. 
        
        Args
        objectCentroid : list of centroids of existing objects.
        inputCentroids : list of centroids of current detections.
        detections : list of detections in current frame
        Returns
        D : cost matrix
        unusedrows : index of unmatched tracks
        unusedcols : index of newly detected tracks 
        """
        D = dist.cdist(np.array(objectCentroid), inputCentroids)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        usedRows = set()
        usedCols = set()
        # loop over the combination of the (row, column) index tuples
        for (row, col) in zip(rows, cols):
            # if we have already examined either the row or column value before, ignore it
            if row in usedRows or col in usedCols:
                continue
            self.tracks[row].bbox = detections[col]
            self.tracks[row].centroid = inputCentroids[col]
            self.tracks[row].miss = 0
            self.tracks[row].hits += 1
            # indicate that we have examined each of the row and column indexes, respectively
            usedRows.add(row)
            usedCols.add(col)
        unusedRows = set(range(0, D.shape[0])).difference(usedRows)
        unusedCols = set(range(0, D.shape[1])).difference(usedCols)
        return D,unusedRows,unusedCols

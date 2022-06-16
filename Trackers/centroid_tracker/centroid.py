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
        """Create a centroid track object and adds to tracks list
        """

        '''
        - add centroid to track list
        - id++
        - add instance from class track
        - if hits > min_hits add track
        '''

        track = centroid_track(id, bbox, centroid, 1, 0)
        self.tracks.append(track)
        self.nextID += 1
        print("added id ",track.id, "succesfully")

    def delete_track(self):
        """Deletes a centroid track object from tracks list if the max age is crossed.
        """
        for track in self.tracks:
            if(track.miss > self.max_age):
                print("deleted id ",track.id, "succesfully")
                self.tracks.remove(track)

    def update(self, detections):
        """Returns the active list of centroid track objects"""
        
        '''
        - check if bbox is empty, if its empty, increase miss and check if miss> max_age if it is delete_track and return
        - create centroid array, n calculate centroid of all tracks
        - if nextid is 0, means no objects are tracking, then add_track
        - else, find the min. distance of old centroid n new centroid
            - check whether the centroid is matched to any row or col, if not then this means this object was lost earlier and now found, and reset it miss to 0
            - check if input centroids is less than last time, so some objects are lost, and icrease their miss and check if it >max_age deregister it.
                - if not, then we new objects, so register them
        '''

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
                self.add_track(self.nextID, detections[i][2],inputCentroids[i])
        else:      
            for track in self.tracks:
                objectId.append(track.id)
                objectCentroid.append(track.centroid) 
            D = dist.cdist(np.array(objectCentroid), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()
            # loop over the combination of the (row, column) index tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or column value before, ignore it val
                if row in usedRows or col in usedCols:
                    continue
                self.tracks[row].centroid = inputCentroids[col]
                self.tracks[row].miss = 0
                # indicate that we have examined each of the row and column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            if D.shape[0] >= D.shape[1]:
            # loop over the unused row indexes
                for row in unusedRows: 
                    # grab the object ID for the corresponding row index and increment the disappeared counter
                    self.tracks[row].miss += 1
                    # check to see if the number of consecutive frames the object has been marked "disappeared" for warrants deregistering the object
                    if self.tracks[row].miss >= self.max_age:
                        self.delete_track()
                        return self.tracks
            else:
                for col in unusedCols:
                    self.add_track(self.nextID, detections[col], inputCentroids[col])
        return self.tracks

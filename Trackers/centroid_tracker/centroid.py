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
        # self.centroids = []

    def add_track(self, id, bbox, centroid):
        """Create a centroid track object and adds to tracks list
        """

        '''
        - add centroid to track list
        - id++
        - add instance from class track
        - if hits > min_hits add track
        '''

        track=centroid_track(id, bbox, centroid, 1, 0)
        self.tracks.append(track)
        self.nextID +=1

    def delete_track(self,id):
        """Deletes a centroid track object from tracks list if the max age is crossed.
        """
        del self.tracks[id]
        self.nextID -= 1

    def update(self, detections, id):
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

        if(detections==[] and self.tracks!=[]):
            print(id)
            self.tracks[max(0, id-1)].miss=self.tracks[max(0, id-1)].miss+1
            if(self.tracks[max(0, id-1)].miss>=self.max_age):
                self.delete_track(max(0, id-1))
            return self.tracks

        inputCentroids = np.zeros((max(1, len(detections)), 2), dtype="int")

        for i in range(0,len(detections)):
            cX = int((detections[i][2][0]+ detections[i][2][1]) / 2.0)
            cY = int((detections[i][2][2] + detections[i][2][3]) / 2.0)
            inputCentroids[i] = (cX, cY)
        objectId=[]
        objectCentroid=[]
        if len(self.tracks) == 0:
            for i in range(0, len(inputCentroids)):
                self.add_track(self.nextID, detections,inputCentroids[i])
        else:
           
            for i in range(0,len(self.tracks)):
                objectId.append(self.tracks[i].id)
                objectCentroid.append(self.tracks[i].centroid)
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
                print("ROW: ", row, "Object ID: ", objectId)
                oid = objectId[row]
                print("%%%%%%%% ",oid)
                print(len(self.tracks))
                self.tracks[min(len(self.tracks), oid )].centroid = inputCentroids[col]
                self.tracks[min(len(self.tracks), oid )].miss = 0
                # indicate that we have examined each of the row and column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            if D.shape[0] >= D.shape[1]:
            # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row index and increment the disappeared counter
                    print("ROW: ", row, "Object ID: ", objectId)
                    oid =objectId[row]
                    print("%%%%%%%% ",oid)
                    print(len(self.tracks))
                    self.tracks[min(len(self.tracks), oid )].miss += 1
                    # check to see if the number of consecutive frames the object has been marked "disappeared" for warrants deregistering the object
                    if self.tracks[min(len(self.tracks), oid )].miss >= self.max_age:
                        self.delete_track(oid)
            else:
                for col in unusedCols:
                    self.add_track(id,detections,inputCentroids[col])
        return self.tracks
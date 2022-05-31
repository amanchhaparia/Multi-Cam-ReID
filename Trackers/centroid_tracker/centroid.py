import numpy as np
from ..track import Track

class centroid_track(Track):
    def __init__(self, id, bbox, centroid):
        self.centroid = centroid
        Track.__init__(self, id, bbox)

class Centroid_tracker():
    def __init__(self):
        self.tracks = []
        self.max_age = 10
        self.min_hits = 3
        self.thres_distance = 20
        self.nextid = 0
        self.centroids = []

    def add_track(self, id, bbox, centroid):
        """Create a centroid track object and adds to tracks list
        """

        '''
        - add centroid to track list
        - id++
        - add instance from class track
        - if hits > min_hits add track
        '''


    def delete_track(self):
        """Deletes a centroid track object from tracks list if the max age is crossed.
        """

        '''
        - delete id from track
        '''
        pass

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

        pass

    
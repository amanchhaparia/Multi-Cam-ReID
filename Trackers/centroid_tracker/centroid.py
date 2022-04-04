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

    def add_track(self, id, bbox, centroid):
        """Create a centroid track object and adds to tracks list
        """
        pass

    def delete_track(self):
        """Deletes a centroid track object from tracks list if the max age is crossed.
        """
        pass

    def update(self, detections):
        """Returns the active list of centroid track objects"""
        pass

    
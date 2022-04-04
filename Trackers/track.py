
class Track():
    def __init__ (self, id, bbox):
        self.bbox = bbox
        self.id = id
        self.hits = 0
        self.miss = 0
        

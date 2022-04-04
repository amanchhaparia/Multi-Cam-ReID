import numpy as np

class Yolo():
    def __init__(self) -> None:
        pass

    def load_model(self):
        """
        Loads the YOLO darknet model
        """
        pass

    def detect(self, image):
        """
        Detects the person in the image.

        Args
        image: A ndarray bgr image.

        Returns a list of detections where each item in list contains [x1, y1, x2, y2, conf]
        """
        pass
import numpy as np

class TextBox:
    def __init__(self, box, score, **kwargs):
        self.box = box
        self.score = score
        if isinstance(self.box, list):
            self.box = np.array(self.box)
        self.text_type = None
        self.source = None
        self.layout_det = None
        self.speaker = None  # Add speaker attribute for nickname extraction
        self.related_layout_boxes = []

        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.x_min, self.y_min, self.x_max, self.y_max = self.box.tolist()

    @property
    def min_x(self): 
        return self.x_min 

    @property
    def min_y(self): 
        return self.y_min

    @property
    def max_x(self): 
        return self.x_max

    @property
    def max_y(self): 
        return self.y_max

    @property
    def center_x(self):
        return (self.x_min + self.x_max) / 2

    @property
    def center_y(self):
        return (self.y_min + self.y_max) / 2

    @property
    def height(self):
        return self.y_max - self.y_min

    @property
    def width(self):
        return self.x_max - self.x_min
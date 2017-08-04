import numpy as np

class Sentence:
    def __init__(self, e1, e2, r, w):
        self.entity1 = e1
        self.entity2 = e2
        self.relation = r
        self.words = w


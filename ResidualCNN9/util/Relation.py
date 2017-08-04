import numpy as np

class Relation:
    def __init__(self, name, id_):
        self.id = id_
        self.name = name
        self.number = 0

    def generate_vector(self, relationTotal):
        v = np.zeros(relationTotal)
        v[self.id] = 1
        self.vector = v

    def add_one(self):
        self.number += 1

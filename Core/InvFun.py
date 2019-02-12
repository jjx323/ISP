import numpy as np 
import matplotlib.pyplot as plt 
import fenics as fe


def trueScatterer(expre='0.0', de=0):
    return fe.Expression(expre, degree=de)


class Sample(object):
    def __init__(self, points=[(0, 0)]):
        self.points = points
        
    def setPoints(self, points=[(0, 0)]):
        self.points = points
        
    def sampling(self, sol):
        measures = np.zeros(len(self.points))
        i = 0
        for point in self.points:
            measures[i] = sol(point)
            i += 1
            
        return measures
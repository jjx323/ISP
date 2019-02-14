import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import fenics as fe


def genePoints(num=40, typem='full', domainPara={'xx': 2.0, 'yy': 2.0}):
    xl, xr = 0.0, domainPara['xx']
    yl, yr = 0.0, domainPara['yy']
    points = []
    numE = np.int(num/4)
    dy = yr/numE
    Yaxis = np.linspace(dy, yr-dy, numE)
    dx = xr/numE
    Xaxis = np.linspace(dx, xr-dx, numE)
    if typem == 'full':
        for i in range(numE):
            points.append((xl, Yaxis[i]))
        for i in range(numE):
            points.append((xr, Yaxis[i]))
        for i in range(numE):
            points.append((Xaxis[i], yl))
        for i in range(numE):
            points.append((Xaxis[i], yr))
    if typem == 'yr':
        for i in range(numE):
            points.append((Xaxis[i], yr))
    return points


def my_draw3D(fun, ax):
    NMX, NMY = 200, 200
    xx = np.linspace(0, ax[0], NMX)
    yy = np.linspace(0, ax[1], NMY)
    M = np.zeros((NMX, NMY))
    for i in range(NMX):
        for j in range(NMY):
            M[i,j] = fun((xx[i], yy[j]))
    XX, YY = np.meshgrid(xx, yy)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    fig1 = ax.plot_surface(XX, YY, M)
    return fig1


def trueScatterer(expre='0.0', de=0):
    return fe.Expression(expre, degree=de)


def initScatterer(V, method='Zero'):
    if method == 'Zero':
        return fe.interpolate(fe.Constant(0.0), V) 


class Regu(object):
    def __init__(self, typem='L2_1'):
        self.type = typem
        
    def eva(self, sol):
        if self.type == 'L2_0':
            energy = 0.5*fe.inner(sol, sol)*fe.dx
        elif self.type == 'L2_1':
            energy = 0.5*fe.inner(fe.grad(sol), fe.grad(sol))*fe.dx
            
        E = fe.assemble(energy)
        return E
    
    def evaGrad(self, sol):
        if self.type == 'L2_0':
            grad = sol
        elif self.type == 'L2_1':
            grad = fe.div(fe.grad(sol))
    
        return grad


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
    



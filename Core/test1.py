import numpy as np 
import matplotlib.pyplot as plt 
import fenics as fe

from HSolver import *
from InvFun import *


# specify basic parameters
domain_para = {'nx': 100, 'ny': 100, 'dPML': 0.15, 'xx': 2.0, 'yy': 2.0, \
				'sig0': 1.5, 'p': 2.3}

domain = Domain(domain_para)
domain.geneMesh()

theta_all = np.linspace(0, 2*np.pi, 10)
kappa_all = [1.0, 2.0, 3.0] #4.0, 5.0, 6.0]
Ntheta, Nkappa = len(theta_all), len(kappa_all)
NN = Ntheta*Nkappa
NS = 400    # number of measuring points
points = genePoints(NS, 'full', domain_para)
measure = Sample(points)
equ_para = {'kappa': 2.0, 'theta': 0.0}

qFunStr = 'exp(-(pow(x[0]-1, 2) + pow(x[1]-1, 2)))'
q_fun = trueScatterer(qFunStr, 3)

# init the scatterer
Fsol = Helmholtz(domain, equ_para)

uincR = fe.Expression('cos(kappa*(x[0]*cos(theta)+x[1]*sin(theta)))', degree=3, \
	                          kappa=equ_para['kappa'], theta=equ_para['theta'])
uincI = fe.Expression('sin(kappa*(x[0]*cos(theta)+x[1]*sin(theta)))', degree=3, \
	                          kappa=equ_para['kappa'], theta=equ_para['theta'])
fR = -(equ_para['kappa']**2)*q_fun*uincR
fI = -(equ_para['kappa']**2)*q_fun*uincI
# solve equation
Fsol.geneForwardMatrix('full', q_fun, fR, fI)
#start = time.time()
Fsol.solve()
uR, uI = Fsol.uReal, Fsol.uImag

Vreal = Fsol.getFunctionSpace('real')
uRp = fe.project(uR, Vreal)
uIp = fe.project(uI, Vreal)

ax = [2, 2]
NMX, NMY = 200, 200
xx = np.linspace(0, ax[0], NMX)
yy = np.linspace(0, ax[1], NMY)
errorR = np.zeros((NMX, NMY))
errorI = np.zeros((NMX, NMY))
uRM = np.zeros((NMX, NMY))
uIM = np.zeros((NMX, NMY))
for i in range(NMX):
    for j in range(NMY):
        errorR[i,j] = np.abs(uRp((xx[i], yy[j])) - uR((xx[i], yy[j])))
        errorI[i,j] = np.abs(uIp((xx[i], yy[j])) - uI((xx[i], yy[j])))
        uRM[i,j] = uR((xx[i], yy[j]))
        uIM[i,j] = uI((xx[i], yy[j]))
er1R = np.max(errorR)
er1I = np.max(errorI)
er2R = np.max(uRM)
er2I = np.max(uIM)
        
        
        
        
    